//! Voice activity gating.
//!
//! A [`VoiceActivityDetector`] scores fixed-size frames. The [`Segmenter`] turns those scores into
//! [`SpeechSegment`]s with hysteresis, a pre-roll so the first consonant is kept, and a hangover so a
//! short pause inside a sentence does not split it.

mod energy;
#[cfg(feature = "onnx")]
mod silero;

pub use energy::EnergyVad;
#[cfg(feature = "onnx")]
pub use silero::SileroVad;

use std::collections::VecDeque;

use crate::audio::SAMPLE_RATE;

/// Scores frames of 16 kHz audio for speech.
pub trait VoiceActivityDetector: Send {
    /// Samples per frame. [`Segmenter`] always passes exactly this many.
    fn frame_size(&self) -> usize;
    /// Speech probability of the next frame, from 0 to 1. Detectors can keep state between frames.
    fn probability(&mut self, frame: &[f32]) -> f32;
    /// Forgets state from earlier audio, at the start of a dictation.
    fn reset(&mut self);
}

#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// A frame at or above this probability is speech.
    pub threshold: f32,
    /// A frame below this probability is silence. Frames between the two thresholds keep the state.
    pub release_threshold: f32,
    /// Silence that ends an utterance.
    pub min_silence_ms: u32,
    /// Utterances with less speech than this are dropped, for example a click or a cough.
    pub min_speech_ms: u32,
    /// Audio kept before the first speech frame.
    pub pre_roll_ms: u32,
    /// Audio kept after the last speech frame.
    pub post_roll_ms: u32,
    /// An utterance this long is emitted even without a pause, so latency and memory stay bounded.
    pub max_segment_ms: u32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            release_threshold: 0.35,
            min_silence_ms: 600,
            min_speech_ms: 200,
            pre_roll_ms: 250,
            post_roll_ms: 150,
            max_segment_ms: 20_000,
        }
    }
}

/// One utterance, ready for recognition.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Mono 16 kHz samples, with pre-roll and post-roll.
    pub samples: Vec<f32>,
    /// Index of the first sample since the dictation started.
    pub start_sample: u64,
    /// True when the segment was cut at [`SegmenterConfig::max_segment_ms`] and speech goes on.
    pub truncated: bool,
}

impl SpeechSegment {
    pub fn duration_seconds(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }
}

enum State {
    Silence,
    Speech {
        samples: Vec<f32>,
        start_sample: u64,
        /// Speech frames since the utterance started, across max-length cuts.
        speech_frames: usize,
        /// Speech frames since the last max-length cut. A part without any is only silence.
        part_speech_frames: usize,
        silence_frames: usize,
    },
}

/// Turns frame scores into speech segments.
pub struct Segmenter<V> {
    vad: V,
    config: SegmenterConfig,
    pending: Vec<f32>,
    pre_roll: VecDeque<f32>,
    processed: u64,
    state: State,
}

impl<V: VoiceActivityDetector> Segmenter<V> {
    pub fn new(vad: V, config: SegmenterConfig) -> Self {
        Self {
            vad,
            config,
            pending: Vec::new(),
            pre_roll: VecDeque::new(),
            processed: 0,
            state: State::Silence,
        }
    }

    fn frames_for(&self, ms: u32) -> usize {
        (ms as usize * SAMPLE_RATE as usize / 1000).div_ceil(self.vad.frame_size())
    }

    fn samples_for(ms: u32) -> usize {
        ms as usize * SAMPLE_RATE as usize / 1000
    }

    /// Takes the next 16 kHz samples and returns the utterances that ended in them.
    pub fn push(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        self.pending.extend_from_slice(samples);
        let frame_size = self.vad.frame_size();
        let mut segments = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= frame_size {
            let frame: Vec<f32> = self.pending[offset..offset + frame_size].to_vec();
            offset += frame_size;
            if let Some(segment) = self.step(&frame) {
                segments.push(segment);
            }
        }
        self.pending.drain(..offset);
        segments
    }

    /// Ends the dictation. Speech in progress becomes a segment if it is long enough.
    pub fn finish(&mut self) -> Option<SpeechSegment> {
        let tail = std::mem::take(&mut self.pending);
        let segment = match std::mem::replace(&mut self.state, State::Silence) {
            State::Speech {
                mut samples,
                start_sample,
                speech_frames,
                part_speech_frames,
                silence_frames,
            } => {
                samples.extend_from_slice(&tail);
                self.close(
                    samples,
                    start_sample,
                    speech_frames,
                    part_speech_frames,
                    silence_frames,
                    false,
                )
            }
            State::Silence => None,
        };
        self.vad.reset();
        self.pre_roll.clear();
        self.processed = 0;
        segment
    }

    fn step(&mut self, frame: &[f32]) -> Option<SpeechSegment> {
        let probability = self.vad.probability(frame);
        let frame_start = self.processed;
        self.processed += frame.len() as u64;
        let min_silence = self.frames_for(self.config.min_silence_ms);
        let max_samples = Self::samples_for(self.config.max_segment_ms);

        match &mut self.state {
            State::Silence => {
                if probability >= self.config.threshold {
                    let mut samples: Vec<f32> = self.pre_roll.drain(..).collect();
                    let start_sample = frame_start - samples.len() as u64;
                    samples.extend_from_slice(frame);
                    self.state = State::Speech {
                        samples,
                        start_sample,
                        speech_frames: 1,
                        part_speech_frames: 1,
                        silence_frames: 0,
                    };
                } else {
                    self.pre_roll.extend(frame);
                    let keep = Self::samples_for(self.config.pre_roll_ms);
                    while self.pre_roll.len() > keep {
                        self.pre_roll.pop_front();
                    }
                }
                None
            }
            State::Speech {
                samples,
                speech_frames,
                part_speech_frames,
                silence_frames,
                start_sample,
            } => {
                samples.extend_from_slice(frame);
                if probability >= self.config.threshold {
                    *speech_frames += 1;
                    *part_speech_frames += 1;
                    *silence_frames = 0;
                } else if probability < self.config.release_threshold || *silence_frames > 0 {
                    // Between the thresholds, a pause that has started keeps counting.
                    *silence_frames += 1;
                }

                if *silence_frames >= min_silence {
                    let State::Speech {
                        samples,
                        start_sample,
                        speech_frames,
                        part_speech_frames,
                        silence_frames,
                    } = std::mem::replace(&mut self.state, State::Silence)
                    else {
                        unreachable!()
                    };
                    return self.close(
                        samples,
                        start_sample,
                        speech_frames,
                        part_speech_frames,
                        silence_frames,
                        false,
                    );
                }
                if samples.len() >= max_samples {
                    let samples = std::mem::take(samples);
                    let start = *start_sample;
                    *start_sample += samples.len() as u64;
                    // Speech goes on, so the utterance's counts carry over: the tail of a long
                    // utterance is never dropped as too short, and a pause in progress still ends
                    // the utterance on time.
                    let (speech, part_speech, silence) =
                        (*speech_frames, *part_speech_frames, *silence_frames);
                    *part_speech_frames = 0;
                    return self.close(samples, start, speech, part_speech, silence, true);
                }
                None
            }
        }
    }

    fn close(
        &self,
        mut samples: Vec<f32>,
        start_sample: u64,
        speech_frames: usize,
        part_speech_frames: usize,
        silence_frames: usize,
        truncated: bool,
    ) -> Option<SpeechSegment> {
        // A part after a max-length cut that holds only the end of a pause has nothing to say.
        if part_speech_frames == 0 {
            return None;
        }
        // Trailing silence beyond the post-roll carries no speech and only slows the decoder down.
        let trailing = silence_frames * self.vad.frame_size();
        let keep_trailing = Self::samples_for(self.config.post_roll_ms).min(trailing);
        samples.truncate(samples.len() - (trailing - keep_trailing).min(samples.len()));
        if speech_frames < self.frames_for(self.config.min_speech_ms) && !truncated {
            return None;
        }
        Some(SpeechSegment {
            samples,
            start_sample,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores a frame as speech when its first sample is 1.
    struct Scripted;

    impl VoiceActivityDetector for Scripted {
        fn frame_size(&self) -> usize {
            160
        }
        fn probability(&mut self, frame: &[f32]) -> f32 {
            frame[0]
        }
        fn reset(&mut self) {}
    }

    fn audio(pattern: &[(f32, usize)]) -> Vec<f32> {
        pattern
            .iter()
            .flat_map(|(value, frames)| std::iter::repeat_n(*value, frames * 160))
            .collect()
    }

    fn config() -> SegmenterConfig {
        SegmenterConfig {
            min_silence_ms: 100,
            min_speech_ms: 50,
            pre_roll_ms: 20,
            post_roll_ms: 10,
            max_segment_ms: 10_000,
            ..Default::default()
        }
    }

    #[test]
    fn emits_one_segment_per_utterance_with_pre_and_post_roll() {
        let mut segmenter = Segmenter::new(Scripted, config());
        // 10 ms frames: 30 silent, 20 speech, 5 silent (a pause), 10 speech, 20 silent.
        let segments = segmenter.push(&audio(&[
            (0.0, 30),
            (1.0, 20),
            (0.0, 5),
            (1.0, 10),
            (0.0, 20),
        ]));
        assert_eq!(segments.len(), 1);
        let segment = &segments[0];
        assert_eq!(segment.start_sample, 28 * 160);
        // 2 pre-roll frames, 35 frames of speech and pause, 1 post-roll frame.
        assert_eq!(segment.samples.len(), (2 + 35 + 1) * 160);
    }

    #[test]
    fn drops_blips_shorter_than_min_speech() {
        let mut segmenter = Segmenter::new(Scripted, config());
        assert!(segmenter
            .push(&audio(&[(0.0, 5), (1.0, 2), (0.0, 30)]))
            .is_empty());
    }

    #[test]
    fn finish_flushes_speech_in_progress() {
        let mut segmenter = Segmenter::new(Scripted, config());
        assert!(segmenter.push(&audio(&[(1.0, 10)])).is_empty());
        let segment = segmenter.finish().expect("segment");
        assert_eq!(segment.samples.len(), 10 * 160);
    }

    #[test]
    fn cuts_long_speech_at_max_length() {
        let mut segmenter = Segmenter::new(
            Scripted,
            SegmenterConfig {
                max_segment_ms: 500,
                ..config()
            },
        );
        let segments = segmenter.push(&audio(&[(1.0, 120)]));
        assert_eq!(segments.len(), 2);
        assert!(segments
            .iter()
            .all(|s| s.truncated && s.samples.len() == 50 * 160));
        assert_eq!(segments[1].start_sample, 50 * 160);
    }

    #[test]
    fn a_cut_during_a_pause_emits_no_silent_segment() {
        let mut segmenter = Segmenter::new(
            Scripted,
            SegmenterConfig {
                max_segment_ms: 500,
                ..config()
            },
        );
        // The cut lands 5 frames into a 10-frame pause, which then ends the utterance.
        let segments = segmenter.push(&audio(&[(1.0, 45), (0.0, 30)]));
        assert_eq!(segments.len(), 1);
        assert!(segments[0].truncated);
        // 45 speech frames and 1 post-roll frame; the rest of the pause is trimmed.
        assert_eq!(segments[0].samples.len(), 46 * 160);
        assert!(segmenter.finish().is_none());

        // The same when the dictation ends during the pause.
        let segments = segmenter.push(&audio(&[(1.0, 45), (0.0, 8)]));
        assert_eq!(segments.len(), 1);
        assert!(segmenter.finish().is_none());

        // Speech after the pause is its own part.
        let segments = segmenter.push(&audio(&[(1.0, 45), (0.0, 8), (1.0, 10), (0.0, 30)]));
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[1].start_sample, 50 * 160);
        assert!(!segments[1].truncated);
    }
}
