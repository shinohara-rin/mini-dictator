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
    /// Samples before the first speech frame. Shorter than the configured pre-roll when speech
    /// starts again soon after the last segment closed, and 0 after a max-length cut.
    pub pre_roll: usize,
    /// Samples after the last speech frame.
    pub post_roll: usize,
}

impl SpeechSegment {
    /// The sample where speech starts, since the dictation started.
    pub fn speech_start(&self) -> u64 {
        self.start_sample + self.pre_roll.min(self.samples.len()) as u64
    }

    /// The sample where speech ends, since the dictation started.
    pub fn speech_end(&self) -> u64 {
        let end = self.start_sample + self.samples.len() as u64;
        end.saturating_sub(self.post_roll as u64)
            .max(self.speech_start())
    }

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
        /// Pre-roll samples at the start of `samples`.
        pre_roll: usize,
    },
}

/// Turns frame scores into speech segments.
pub struct Segmenter<V> {
    vad: V,
    config: SegmenterConfig,
    /// See [`Segmenter::set_sound_playing`].
    sound_playing: bool,
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
            sound_playing: false,
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

    /// While on, the samples pushed may hold a sound the app played, such as a cue that sounds a
    /// little like a voice. Their frames cannot start an utterance or count as speech, but keep
    /// one in progress going. The pre-roll grows over them, so speech that starts during the sound
    /// keeps its first word once the sound ends.
    pub fn set_sound_playing(&mut self, on: bool) {
        self.sound_playing = on;
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

    /// The first sample that a segment still to come can start at: the open utterance's start,
    /// or the oldest pre-roll sample.
    pub fn earliest_start(&self) -> u64 {
        match &self.state {
            State::Speech { start_sample, .. } => *start_sample,
            State::Silence => self.processed - self.pre_roll.len() as u64,
        }
    }

    /// The utterance in progress: its first sample, its samples so far, and the silence at its
    /// end in milliseconds.
    pub fn open(&self) -> Option<(u64, usize, u32)> {
        match &self.state {
            State::Speech {
                samples,
                start_sample,
                silence_frames,
                ..
            } => Some((
                *start_sample,
                samples.len(),
                (*silence_frames * self.vad.frame_size() * 1000 / SAMPLE_RATE as usize) as u32,
            )),
            State::Silence => None,
        }
    }

    /// The segment the utterance in progress would make if its pause went on until it ended:
    /// the same samples, as long as no speech comes first.
    pub fn preview(&self) -> Option<SpeechSegment> {
        let State::Speech {
            samples,
            start_sample,
            speech_frames,
            part_speech_frames,
            silence_frames,
            pre_roll,
        } = &self.state
        else {
            return None;
        };
        self.close(
            samples.clone(),
            *start_sample,
            *speech_frames,
            *part_speech_frames,
            *silence_frames,
            false,
        )
        .map(|segment| SpeechSegment {
            pre_roll: *pre_roll,
            ..segment
        })
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
                pre_roll,
            } => {
                // In a pause, the samples too few for a frame are more of it, beyond the post-roll.
                if silence_frames == 0 {
                    samples.extend_from_slice(&tail);
                }
                self.close(
                    samples,
                    start_sample,
                    speech_frames,
                    part_speech_frames,
                    silence_frames,
                    false,
                )
                .map(|segment| SpeechSegment {
                    pre_roll,
                    ..segment
                })
            }
            State::Silence => None,
        };
        self.vad.reset();
        self.pre_roll.clear();
        self.processed = 0;
        segment
    }

    fn step(&mut self, frame: &[f32]) -> Option<SpeechSegment> {
        let mut probability = self.vad.probability(frame);
        if self.sound_playing {
            // Between the thresholds: neither speech nor a new pause.
            probability =
                probability.min((self.config.threshold + self.config.release_threshold) / 2.0);
        }
        let frame_start = self.processed;
        self.processed += frame.len() as u64;
        let min_silence = self.frames_for(self.config.min_silence_ms);
        let max_samples = Self::samples_for(self.config.max_segment_ms);

        match &mut self.state {
            State::Silence => {
                if probability >= self.config.threshold {
                    let mut samples: Vec<f32> = self.pre_roll.drain(..).collect();
                    let start_sample = frame_start - samples.len() as u64;
                    let pre_roll = samples.len();
                    samples.extend_from_slice(frame);
                    self.state = State::Speech {
                        samples,
                        start_sample,
                        speech_frames: 1,
                        part_speech_frames: 1,
                        silence_frames: 0,
                        pre_roll,
                    };
                } else {
                    self.pre_roll.extend(frame);
                    let keep = Self::samples_for(self.config.pre_roll_ms);
                    while !self.sound_playing && self.pre_roll.len() > keep {
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
                pre_roll,
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
                        pre_roll,
                    } = std::mem::replace(&mut self.state, State::Silence)
                    else {
                        unreachable!()
                    };
                    return self
                        .close(
                            samples,
                            start_sample,
                            speech_frames,
                            part_speech_frames,
                            silence_frames,
                            false,
                        )
                        .map(|segment| SpeechSegment {
                            pre_roll,
                            ..segment
                        });
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
                    // The next part continues the speech, with no pre-roll of its own.
                    let lead = std::mem::take(pre_roll);
                    return self
                        .close(samples, start, speech, part_speech, silence, true)
                        .map(|segment| SpeechSegment {
                            pre_roll: lead,
                            ..segment
                        });
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
            pre_roll: 0,
            post_roll: keep_trailing,
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
    fn records_the_pre_roll_and_post_roll_each_segment_kept() {
        let mut segmenter = Segmenter::new(Scripted, config());
        // The second utterance starts one frame after the first one closed.
        let segments = segmenter.push(&audio(&[
            (0.0, 5),
            (1.0, 10),
            (0.0, 10),
            (0.0, 1),
            (1.0, 10),
            (0.0, 10),
        ]));
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].pre_roll, segments[0].post_roll), (320, 160));
        assert_eq!((segments[1].pre_roll, segments[1].post_roll), (160, 160));
        // Eleven silent frames separate the two utterances.
        let pause = segments[1].speech_start() - segments[0].speech_end();
        assert_eq!(pause, 11 * 160);
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
    fn earliest_start_trails_by_the_pre_roll_or_holds_the_open_utterance() {
        let mut segmenter = Segmenter::new(Scripted, config());
        segmenter.push(&audio(&[(0.0, 30)]));
        assert_eq!(segmenter.earliest_start(), 28 * 160);
        segmenter.push(&audio(&[(1.0, 5), (0.0, 1)]));
        assert_eq!(segmenter.earliest_start(), 28 * 160);
    }

    #[test]
    fn the_preview_of_an_utterance_is_the_segment_its_pause_ends() {
        let mut segmenter = Segmenter::new(Scripted, config());
        assert!(segmenter.preview().is_none());
        segmenter.push(&audio(&[(0.0, 5), (1.0, 20), (0.0, 3)]));
        assert_eq!(segmenter.open().map(|(_, _, silence)| silence), Some(30));
        let preview = segmenter.preview().expect("an utterance in progress");
        let segments = segmenter.push(&audio(&[(0.0, 30)]));
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sample, preview.start_sample);
        assert_eq!(segments[0].samples, preview.samples);
        assert_eq!(segments[0].pre_roll, preview.pre_roll);
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

    #[test]
    fn a_sound_the_app_plays_cannot_start_an_utterance() {
        let mut segmenter = Segmenter::new(Scripted, config());
        segmenter.set_sound_playing(true);
        assert!(segmenter.push(&audio(&[(0.0, 2), (1.0, 20)])).is_empty());
        segmenter.set_sound_playing(false);
        assert!(segmenter.push(&audio(&[(0.0, 20)])).is_empty());
        assert!(segmenter.finish().is_none());
    }

    #[test]
    fn speech_under_a_sound_goes_on_and_keeps_its_start() {
        // Speech already going on when the sound starts carries through it.
        let mut segmenter = Segmenter::new(Scripted, config());
        assert!(segmenter.push(&audio(&[(0.0, 2), (1.0, 5)])).is_empty());
        segmenter.set_sound_playing(true);
        assert!(segmenter.push(&audio(&[(1.0, 15)])).is_empty());
        segmenter.set_sound_playing(false);
        let segments = segmenter.push(&audio(&[(1.0, 5), (0.0, 20)]));
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speech_start(), 2 * 160);

        // Speech that starts while the sound plays begins once it ends, with the sound's frames
        // as pre-roll, so its first word is kept.
        let mut segmenter = Segmenter::new(Scripted, config());
        assert!(segmenter.push(&audio(&[(0.0, 10)])).is_empty());
        segmenter.set_sound_playing(true);
        assert!(segmenter.push(&audio(&[(0.0, 3), (1.0, 6)])).is_empty());
        segmenter.set_sound_playing(false);
        let segments = segmenter.push(&audio(&[(1.0, 5), (0.0, 20)]));
        assert_eq!(segments.len(), 1);
        // The configured pre-roll before the sound, then all of it.
        assert_eq!(segments[0].start_sample, (10 - 2) * 160);
        assert_eq!(segments[0].speech_start(), 19 * 160);
    }
}
