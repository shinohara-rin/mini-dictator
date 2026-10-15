//! End to end: a mock microphone plays LibriSpeech utterances in real time into `run_dictation`,
//! which gates them with Silero VAD, decodes them with the real recognizer, checks uncertainty, and
//! types into a sink. The typed text must match the corpus transcripts.
//!
//! It needs the models and ONNX Runtime, so it is ignored by default. `scripts/e2e.sh` downloads
//! them and runs it; by hand:
//!
//! ```text
//! MD_E2E_MODEL=<model dir> MD_E2E_VAD=<silero_vad.onnx> MD_E2E_ORT=<libonnxruntime> \
//!   cargo test --release -p mini-dictator-core --features onnx --test e2e -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, SearchConfig};
use mini_dictator_core::audio::{AudioChunk, SAMPLE_RATE};
use mini_dictator_core::context::NoContext;
use mini_dictator_core::pipeline::{run_dictation, Engine, EngineConfig, Event, TextSink};
use mini_dictator_core::rewrite::PassThrough;
use mini_dictator_core::rng::SplitMix64;
use mini_dictator_core::vad::SileroVad;

/// The rate and channels of a usual built-in microphone, so the resampler and downmix run too.
const MIC_RATE: u32 = 48_000;
const MIC_CHANNELS: u16 = 2;
/// Audio per callback, as a CoreAudio or WASAPI input stream delivers it.
const BLOCK: Duration = Duration::from_millis(10);
/// The share of reference words the dictation may get wrong. Qwen3-ASR 0.6B scores about 2% on
/// LibriSpeech test-clean; this leaves room for VAD cuts without hiding a broken pipeline.
const MAX_WORD_ERROR_RATE: f32 = 0.08;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/librispeech")
}

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}; see the top of crates/core/tests/e2e.rs"))
}

/// The utterances in file order, with their transcripts.
fn utterances() -> Vec<(Vec<f32>, String)> {
    let transcripts = std::fs::read_to_string(data_dir().join("transcripts.txt")).unwrap();
    transcripts
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (id, text) = line.split_once(' ').unwrap();
            let mut reader = hound::WavReader::open(data_dir().join(format!("{id}.wav"))).unwrap();
            let spec = reader.spec();
            assert_eq!((spec.sample_rate, spec.channels), (SAMPLE_RATE, 1), "{id}");
            let samples = reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect();
            (samples, text.to_owned())
        })
        .collect()
}

/// What the microphone hears: each utterance after a pause, at 16 kHz, over a faint noise floor.
fn recording(utterances: &[(Vec<f32>, String)]) -> Vec<f32> {
    let silence = |seconds: f32| vec![0.0; (seconds * SAMPLE_RATE as f32) as usize];
    let mut audio = silence(0.5);
    for (samples, _) in utterances {
        audio.extend(samples);
        audio.extend(silence(1.5));
    }
    let mut rng = SplitMix64::new(7);
    for sample in &mut audio {
        *sample += (rng.next_f64() as f32 - 0.5) * 2e-3;
    }
    audio
}

/// 16 kHz mono to interleaved [`MIC_RATE`] with [`MIC_CHANNELS`], by linear interpolation.
fn to_mic_format(audio: &[f32]) -> Vec<f32> {
    let factor = (MIC_RATE / SAMPLE_RATE) as usize;
    let mut out = Vec::with_capacity(audio.len() * factor * MIC_CHANNELS as usize);
    for (i, &sample) in audio.iter().enumerate() {
        let next = audio.get(i + 1).copied().unwrap_or(sample);
        for step in 0..factor {
            let value = sample + (next - sample) * step as f32 / factor as f32;
            out.extend(std::iter::repeat_n(value, MIC_CHANNELS as usize));
        }
    }
    out
}

/// Plays `audio` like a microphone: one block every [`BLOCK`], stamped when it is sent. The
/// channel closes after the last block, as when the user lets go of the hotkey.
fn microphone(audio: Vec<f32>) -> (Receiver<AudioChunk>, thread::JoinHandle<Instant>) {
    let (sender, receiver) = mpsc::channel();
    let player = thread::spawn(move || {
        let frames = (MIC_RATE as f64 * BLOCK.as_secs_f64()) as usize;
        let started = Instant::now();
        for (n, block) in audio.chunks(frames * MIC_CHANNELS as usize).enumerate() {
            // Paced from the start, so a late wake-up does not slow the whole recording.
            let due = started + BLOCK * (n as u32 + 1);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }
            let chunk = AudioChunk {
                samples: block.to_vec(),
                sample_rate: MIC_RATE,
                channels: MIC_CHANNELS,
                captured: Instant::now(),
                sound_playing: false,
            };
            if sender.send(chunk).is_err() {
                break;
            }
        }
        Instant::now()
    });
    (receiver, player)
}

struct Collect(Vec<String>);

impl TextSink for Collect {
    fn insert(&mut self, text: &str) -> mini_dictator_core::Result<()> {
        self.0.push(text.to_owned());
        Ok(())
    }
}

/// Lowercase words without punctuation; a hyphen splits a word.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Word-level edit distance.
fn word_errors(reference: &[String], hypothesis: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = diagonal + usize::from(r != h);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(diagonal + 1);
        }
    }
    row[hypothesis.len()]
}

#[test]
fn word_errors_count_edits() {
    let w = |s: &str| words(s);
    assert_eq!(word_errors(&w("a b c"), &w("a b c")), 0);
    assert_eq!(word_errors(&w("a b c"), &w("a x c d")), 2);
    assert_eq!(word_errors(&w("a b c"), &w("")), 3);
    assert_eq!(
        w("Flour-fattened sauce, he said."),
        w("flour fattened SAUCE HE SAID")
    );
}

#[test]
#[ignore = "needs the models and ONNX Runtime; run scripts/e2e.sh"]
fn dictates_librispeech_from_a_mock_microphone() {
    init_runtime(Some(&env_path("MD_E2E_ORT"))).unwrap();
    let threads = thread::available_parallelism().map_or(4, |n| n.get());
    let recognizer =
        asr::load(&env_path("MD_E2E_MODEL"), threads, SearchConfig::default()).unwrap();
    let vad = SileroVad::load(&env_path("MD_E2E_VAD")).unwrap();
    let engine = Engine {
        config: EngineConfig {
            monte_carlo: recognizer.monte_carlo(),
            ..EngineConfig::default()
        },
        asr: recognizer,
        context: Box::new(NoContext),
        memory: Default::default(),
        rewriter: Box::new(PassThrough),
    };

    let utterances = utterances();
    let audio = recording(&utterances);
    println!(
        "playing {:.1} s of audio, {} utterances",
        audio.len() as f32 / SAMPLE_RATE as f32,
        utterances.len()
    );
    let (microphone, player) = microphone(to_mic_format(&audio));
    let mut sink = Collect(Vec::new());
    let mut events = Vec::new();
    let text = run_dictation(&engine, vad, microphone, &mut sink, &mut |event| {
        if let Event::Transcribed {
            plain,
            unclear_spans,
            behind_seconds,
            ..
        } = &event
        {
            println!("transcribed: {plain:?} ({unclear_spans} unclear, behind {behind_seconds:?})");
        }
        events.push(event);
    })
    .unwrap();
    let released = player.join().unwrap();
    println!("text: {text}");
    println!(
        "finished {:.2} s after the microphone closed",
        released.elapsed().as_secs_f32()
    );

    let warnings: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, Event::Warning(_)))
        .collect();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(matches!(events.last(), Some(Event::Finished)), "{events:?}");
    let transcribed = events
        .iter()
        .filter(|e| matches!(e, Event::Transcribed { .. }))
        .count();
    assert!(
        transcribed >= utterances.len(),
        "the VAD found {transcribed} segments in {} utterances",
        utterances.len()
    );
    assert_eq!(sink.0.concat(), text);

    let reference: Vec<String> = utterances.iter().flat_map(|(_, t)| words(t)).collect();
    let errors = word_errors(&reference, &words(&text));
    let rate = errors as f32 / reference.len() as f32;
    println!(
        "word error rate {:.1}% ({errors} of {} words)",
        rate * 100.0,
        reference.len()
    );
    assert!(
        rate <= MAX_WORD_ERROR_RATE,
        "word error rate {:.1}% is above {:.0}%\nreference: {}\ntyped:     {text}",
        rate * 100.0,
        MAX_WORD_ERROR_RATE * 100.0,
        reference.join(" ")
    );
}
