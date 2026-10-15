//! Times the recognizer on the LibriSpeech clips of the end-to-end test: model load, Silero VAD,
//! and per clip the encode, the clean decode, and the noisy decodes of the uncertainty check.
//!
//! ```text
//! cargo run --release -p mini-dictator-core --features onnx --example bench -- \
//!   --ort <libonnxruntime> --model <dir> --vad <silero_vad.onnx> [--runs 3] [--threads N] \
//!   [--json out.json] [--baseline base.json]
//! ```
//!
//! Each figure is the median of `--runs`, after one warm-up decode. The early stop is off, so the
//! check always runs every noisy decode and its time does not depend on what was heard. With
//! `--baseline`, a Markdown table compares each figure with that earlier run. CI compares a pull
//! request with its base on the same runner; shared runners are noisy, so only large changes mean
//! anything.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, Bias, SearchConfig};
use mini_dictator_core::uncertainty;
use mini_dictator_core::vad::{Segmenter, SegmenterConfig, SileroVad};

/// A change smaller than this share is within the noise of a shared runner.
const NOTABLE: f64 = 0.15;

struct Args {
    ort: PathBuf,
    model: PathBuf,
    vad: PathBuf,
    runs: usize,
    threads: usize,
    json: Option<PathBuf>,
    baseline: Option<PathBuf>,
}

fn parse() -> Args {
    let mut args = std::env::args().skip(1);
    let (mut ort, mut model, mut vad, mut json, mut baseline) = (None, None, None, None, None);
    let mut runs = 3;
    let mut threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--ort" => ort = Some(PathBuf::from(value)),
            "--model" => model = Some(PathBuf::from(value)),
            "--vad" => vad = Some(PathBuf::from(value)),
            "--runs" => runs = value.parse().expect("--runs"),
            "--threads" => threads = value.parse().expect("--threads"),
            "--json" => json = Some(PathBuf::from(value)),
            "--baseline" => baseline = Some(PathBuf::from(value)),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        ort: ort.expect("--ort is required"),
        model: model.expect("--model is required"),
        vad: vad.expect("--vad is required"),
        runs: runs.max(1),
        threads,
        json,
        baseline,
    }
}

fn clips() -> Vec<(String, Vec<f32>)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/librispeech");
    let transcripts = std::fs::read_to_string(dir.join("transcripts.txt")).unwrap();
    transcripts
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(|id| {
            let wav = std::fs::read(dir.join(format!("{id}.wav"))).unwrap();
            (id.to_owned(), pcm16(&wav))
        })
        .collect()
}

/// The samples of a 16-bit PCM WAV file. Read by hand so the benchmark builds on a base branch
/// without the test's dependencies.
fn pcm16(wav: &[u8]) -> Vec<f32> {
    let mut at = 12;
    while at + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
        if &wav[at..at + 4] == b"data" {
            return wav[at + 8..(at + 8 + size).min(wav.len())]
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                .collect();
        }
        at += 8 + size + size % 2;
    }
    panic!("no data chunk");
}

fn median(mut values: Vec<Duration>) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1000.0
}

/// Named figures in milliseconds, in report order.
type Figures = Vec<(String, f64)>;

fn main() {
    let args = parse();
    init_runtime(Some(&args.ort)).unwrap();
    let mut figures: Figures = Vec::new();

    let started = Instant::now();
    let recognizer = asr::load(&args.model, args.threads, SearchConfig::default()).unwrap();
    figures.push((
        "model load".into(),
        started.elapsed().as_secs_f64() * 1000.0,
    ));
    let mut monte_carlo = recognizer.monte_carlo();
    monte_carlo.first_samples = 0;
    let bias = Bias::default();
    let clips = clips();

    let all: Vec<f32> = clips.iter().flat_map(|(_, s)| s.iter().copied()).collect();
    let vad: Vec<Duration> = (0..args.runs)
        .map(|_| {
            let mut segmenter = Segmenter::new(
                SileroVad::load(&args.vad).unwrap(),
                SegmenterConfig::default(),
            );
            let started = Instant::now();
            for block in all.chunks(160) {
                segmenter.push(block);
            }
            segmenter.finish();
            started.elapsed()
        })
        .collect();
    figures.push((
        format!("VAD, {:.0} s of audio", all.len() as f64 / 16_000.0),
        median(vad),
    ));

    let warm = recognizer.encode(&clips[0].1).unwrap();
    recognizer.decode_one(&warm, &bias).unwrap();

    let (mut total_encode, mut total_decode, mut total_check) = (0.0, 0.0, 0.0);
    for (id, samples) in &clips {
        let (mut encode, mut decode, mut check) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..args.runs {
            let started = Instant::now();
            let encoder = recognizer.encode(samples).unwrap();
            let encoded = started.elapsed();
            let reference = recognizer.decode_one(&encoder, &bias).unwrap();
            let decoded = started.elapsed();
            uncertainty::estimate(
                recognizer.as_ref(),
                &encoder,
                &reference,
                &bias,
                &monte_carlo,
            )
            .unwrap();
            encode.push(encoded);
            decode.push(decoded - encoded);
            check.push(started.elapsed() - decoded);
        }
        let seconds = samples.len() as f64 / 16_000.0;
        let (encode, decode, check) = (median(encode), median(decode), median(check));
        total_encode += encode;
        total_decode += decode;
        total_check += check;
        figures.push((format!("{id} ({seconds:.1} s) encode"), encode));
        figures.push((format!("{id} ({seconds:.1} s) decode"), decode));
        figures.push((
            format!(
                "{id} ({seconds:.1} s) check, {} noisy decodes",
                monte_carlo.samples
            ),
            check,
        ));
    }
    figures.push(("all clips encode".into(), total_encode));
    figures.push(("all clips decode".into(), total_decode));
    figures.push(("all clips check".into(), total_check));
    let audio_ms = all.len() as f64 / 16.0;
    println!(
        "{} threads, {} runs; realtime factor {:.2} without the check, {:.2} with it",
        args.threads,
        args.runs,
        (total_encode + total_decode) / audio_ms,
        (total_encode + total_decode + total_check) / audio_ms,
    );

    if let Some(path) = &args.json {
        std::fs::write(path, to_json(&figures)).unwrap();
    }
    let baseline = args
        .baseline
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| from_json(&text));
    print!("{}", table(&figures, baseline.as_ref()));
}

fn to_json(figures: &Figures) -> String {
    let mut json = serde_json::Map::new();
    for (name, ms) in figures {
        json.insert(name.clone(), serde_json::json!((ms * 10.0).round() / 10.0));
    }
    serde_json::to_string_pretty(&json).unwrap() + "\n"
}

fn from_json(text: &str) -> Figures {
    let value: serde_json::Map<String, serde_json::Value> = serde_json::from_str(text).unwrap();
    value
        .into_iter()
        .filter_map(|(name, ms)| Some((name, ms.as_f64()?)))
        .collect()
}

fn table(figures: &Figures, baseline: Option<&Figures>) -> String {
    let mut out = String::new();
    match baseline {
        Some(_) => {
            out += "| | base (ms) | this PR (ms) | change |\n| --- | ---: | ---: | ---: |\n";
        }
        None => out += "| | ms |\n| --- | ---: |\n",
    }
    for (name, ms) in figures {
        let Some(baseline) = baseline else {
            let _ = writeln!(out, "| {name} | {ms:.0} |");
            continue;
        };
        match baseline.iter().find(|(n, _)| n == name) {
            Some((_, before)) => {
                let change = ms / before - 1.0;
                let mark = if change > NOTABLE {
                    " 🔺"
                } else if change < -NOTABLE {
                    " 🟢"
                } else {
                    ""
                };
                let _ = writeln!(
                    out,
                    "| {name} | {before:.0} | {ms:.0} | {:+.0}%{mark} |",
                    change * 100.0
                );
            }
            None => {
                let _ = writeln!(out, "| {name} | | {ms:.0} | new |");
            }
        }
    }
    out
}
