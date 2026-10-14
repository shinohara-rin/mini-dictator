//! The verbose cues, drawn here as tiny WAV files when the app starts: no sound files to ship or
//! license, and each cue is a few kilobytes in memory. Also the limiter that merges a burst of
//! the same cue.

use std::time::{Duration, Instant};

const RATE: u32 = 44_100;

/// A pipeline moment with a sound of its own in the verbose set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Jingle {
    /// The model loaded: a rising major arpeggio.
    Ready,
    /// A segment went to the recognizer: a little "gulp".
    Heard,
    /// The uncertainty check found unclear words: a wobbly "hm-hm?".
    Unclear,
    /// A segment was decoded to nothing, so nothing will be typed: a deflating "pfff".
    Nothing,
    /// Segments are queueing up and the check is skipped: a "zip-zip-zip" speeding up.
    CatchingUp,
    /// Text was typed: a bell sparkle.
    Typed,
    /// A step failed but the dictation goes on: "uh-oh".
    Warning,
    /// The dictation or the model failed: a low "bonk".
    Error,
}

impl Jingle {
    pub const ALL: [Jingle; 8] = [
        Jingle::Ready,
        Jingle::Heard,
        Jingle::Unclear,
        Jingle::Nothing,
        Jingle::CatchingUp,
        Jingle::Typed,
        Jingle::Warning,
        Jingle::Error,
    ];

    /// The cues that Silero VAD scores as speech when the microphone picks them up from the
    /// speakers (over 0.5 for 200 ms or more at moderate levels). The bells, arpeggio, zips and
    /// bonk stay under 0.45, so they never hold back speech detection.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn sounds_like_voice(self) -> bool {
        matches!(
            self,
            Jingle::Heard | Jingle::Unclear | Jingle::Nothing | Jingle::Warning
        )
    }

    /// The cue as a 16-bit mono WAV file.
    pub fn wav(self) -> Vec<u8> {
        wav(&self.samples())
    }

    fn samples(self) -> Vec<f32> {
        let mut out = Vec::new();
        match self {
            Jingle::Ready => {
                for (i, hz) in [659.3, 830.6, 987.8, 1318.5].into_iter().enumerate() {
                    let ms = if i == 3 { 260.0 } else { 110.0 };
                    voice(
                        &mut out,
                        Voice::new(i as f32 * 70.0, ms, hz).wave(Wave::Triangle),
                    );
                }
            }
            Jingle::Heard => voice(&mut out, Voice::new(0.0, 90.0, 380.0).glide(780.0)),
            Jingle::Unclear => {
                let hm = |at, hz: f32, to| {
                    Voice::new(at, 150.0, hz)
                        .glide(to)
                        .wave(Wave::Triangle)
                        .vibrato(0.04, 11.0)
                };
                voice(&mut out, hm(0.0, 440.0, 400.0));
                voice(&mut out, hm(170.0, 470.0, 720.0).ms(220.0));
            }
            Jingle::Nothing => {
                let pfff = Voice::new(0.0, 260.0, 520.0)
                    .glide(160.0)
                    .wave(Wave::Triangle);
                voice(&mut out, pfff.vibrato(0.02, 23.0).decay(4.0));
            }
            Jingle::CatchingUp => {
                for (i, at) in [0.0, 60.0, 105.0, 140.0].into_iter().enumerate() {
                    let hz = 900.0 + 180.0 * i as f32;
                    let zip = Voice::new(at, 32.0, hz).glide(hz * 1.5).wave(Wave::Soft);
                    voice(&mut out, zip.gain(0.5));
                }
            }
            Jingle::Typed => {
                for (at, hz) in [(0.0, 1568.0), (45.0, 2093.0), (90.0, 2637.0)] {
                    let bell = Voice::new(at, 320.0, hz).decay(9.0);
                    voice(&mut out, bell.gain(0.6));
                    // A quiet inharmonic partial makes it ring like a bell, not a beep.
                    voice(&mut out, bell.freq(hz * 2.76).gain(0.12).decay(16.0));
                }
            }
            Jingle::Warning => {
                voice(&mut out, Voice::new(0.0, 130.0, 523.3).wave(Wave::Soft));
                voice(&mut out, Voice::new(170.0, 200.0, 392.0).wave(Wave::Soft));
            }
            Jingle::Error => {
                voice(
                    &mut out,
                    Voice::new(0.0, 220.0, 240.0).glide(110.0).decay(6.0),
                );
                let buzz = Voice::new(0.0, 220.0, 120.0).glide(90.0).wave(Wave::Soft);
                voice(&mut out, buzz.gain(0.35).decay(8.0));
            }
        }
        normalize(&mut out, 0.8);
        out
    }
}

#[derive(Clone, Copy)]
enum Wave {
    Sine,
    Triangle,
    /// A sine pushed into soft clipping: buzzier, still gentle.
    Soft,
}

/// One note: where it starts, how long it lasts, and a pitch that can glide and wobble.
#[derive(Clone, Copy)]
struct Voice {
    at_ms: f32,
    ms: f32,
    hz: f32,
    to_hz: f32,
    wave: Wave,
    gain: f32,
    /// Exponential decay rate across the note; 0 keeps the level until the release.
    decay: f32,
    vibrato_depth: f32,
    vibrato_hz: f32,
}

impl Voice {
    fn new(at_ms: f32, ms: f32, hz: f32) -> Self {
        Voice {
            at_ms,
            ms,
            hz,
            to_hz: hz,
            wave: Wave::Sine,
            gain: 1.0,
            decay: 3.0,
            vibrato_depth: 0.0,
            vibrato_hz: 0.0,
        }
    }
    fn ms(self, ms: f32) -> Self {
        Voice { ms, ..self }
    }
    fn freq(self, hz: f32) -> Self {
        Voice {
            hz,
            to_hz: hz,
            ..self
        }
    }
    fn glide(self, to_hz: f32) -> Self {
        Voice { to_hz, ..self }
    }
    fn wave(self, wave: Wave) -> Self {
        Voice { wave, ..self }
    }
    fn gain(self, gain: f32) -> Self {
        Voice { gain, ..self }
    }
    fn decay(self, decay: f32) -> Self {
        Voice { decay, ..self }
    }
    fn vibrato(self, depth: f32, hz: f32) -> Self {
        Voice {
            vibrato_depth: depth,
            vibrato_hz: hz,
            ..self
        }
    }
}

/// Mixes the voice into `out`, growing it as needed.
fn voice(out: &mut Vec<f32>, v: Voice) {
    let start = (v.at_ms / 1000.0 * RATE as f32) as usize;
    let len = (v.ms / 1000.0 * RATE as f32) as usize;
    if out.len() < start + len {
        out.resize(start + len, 0.0);
    }
    let attack = (0.004 * RATE as f32) as usize;
    let release = (0.012 * RATE as f32) as usize;
    let mut phase = 0.0f32;
    for i in 0..len {
        let t = i as f32 / len as f32;
        let seconds = i as f32 / RATE as f32;
        // Glides are exponential, so they sound even across the range.
        let mut hz = v.hz * (v.to_hz / v.hz).powf(t);
        hz *= 1.0 + v.vibrato_depth * (std::f32::consts::TAU * v.vibrato_hz * seconds).sin();
        phase = (phase + hz / RATE as f32).fract();
        let sine = (std::f32::consts::TAU * phase).sin();
        let sample = match v.wave {
            Wave::Sine => sine,
            Wave::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
            Wave::Soft => (2.5 * sine).tanh(),
        };
        let fade_in = (i as f32 / attack as f32).min(1.0);
        let fade_out = ((len - 1 - i) as f32 / release as f32).min(1.0);
        let envelope = fade_in.min(fade_out) * (-v.decay * t).exp();
        out[start + i] += sample * envelope * v.gain;
    }
}

fn normalize(samples: &mut [f32], peak: f32) {
    let max = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if max > 0.0 {
        let scale = peak / max;
        samples.iter_mut().for_each(|s| *s *= scale);
    }
}

fn wav(samples: &[f32]) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&RATE.to_le_bytes());
    out.extend_from_slice(&(RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    out
}

/// Merges repeats of one cue, so a queue of segments decoding back to back makes one gulp rather
/// than a wall of them. Different cues always play, even on top of each other: each one tells a
/// different state, and the sounds are the app's only window into the pipeline.
pub struct Limiter {
    last_of: Vec<(Jingle, Instant)>,
}

/// The same cue no more often than this.
const SAME_GAP: Duration = Duration::from_millis(500);

impl Limiter {
    pub fn new() -> Self {
        Limiter {
            last_of: Vec::new(),
        }
    }

    /// True when `jingle` may play at `now`, which then counts as its last play. Errors always
    /// play: they are rare, and the one that matters.
    pub fn allow(&mut self, jingle: Jingle, now: Instant) -> bool {
        if jingle != Jingle::Error
            && self
                .last_of
                .iter()
                .any(|(j, at)| *j == jingle && now.saturating_duration_since(*at) < SAME_GAP)
        {
            return false;
        }
        self.last_of.retain(|(j, _)| *j != jingle);
        self.last_of.push((jingle, now));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_jingle_is_a_short_clean_wav() {
        for jingle in Jingle::ALL {
            let samples = jingle.samples();
            let seconds = samples.len() as f32 / RATE as f32;
            assert!((0.05..0.6).contains(&seconds), "{jingle:?}: {seconds} s");
            let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!((peak - 0.8).abs() < 1e-3, "{jingle:?}: peak {peak}");
            // Fades in and out, so it does not click.
            assert!(
                samples[0].abs() < 0.05,
                "{jingle:?} starts at {}",
                samples[0]
            );
            let last = *samples.last().unwrap();
            assert!(last.abs() < 0.05, "{jingle:?} ends at {last}");

            let wav = jingle.wav();
            assert_eq!(&wav[..4], b"RIFF");
            assert_eq!(&wav[8..16], b"WAVEfmt ");
            assert_eq!(wav.len(), 44 + samples.len() * 2);
            assert_eq!(
                u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize,
                wav.len() - 8
            );
        }
    }

    #[test]
    fn jingles_sound_different() {
        let all: Vec<Vec<f32>> = Jingle::ALL.iter().map(|j| j.samples()).collect();
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                assert_ne!(
                    all[i],
                    all[j],
                    "{:?} = {:?}",
                    Jingle::ALL[i],
                    Jingle::ALL[j]
                );
            }
        }
    }

    #[test]
    fn limiter_merges_repeats_only() {
        let t0 = Instant::now();
        let ms = |n| t0 + Duration::from_millis(n);
        let mut limiter = Limiter::new();
        assert!(limiter.allow(Jingle::Heard, ms(0)));
        // A different cue plays at once.
        assert!(limiter.allow(Jingle::Unclear, ms(10)));
        assert!(limiter.allow(Jingle::Typed, ms(20)));
        // The same cue again too soon is merged into the first.
        assert!(!limiter.allow(Jingle::Heard, ms(300)));
        assert!(limiter.allow(Jingle::Heard, ms(600)));
        assert!(limiter.allow(Jingle::Error, ms(610)));
        assert!(limiter.allow(Jingle::Error, ms(620)));
    }
}
