//! `md`: runs the dictation core on a WAV file and prints each step.
//!
//! ```text
//! md --model <dir> --wav <file> [--vad silero_vad.onnx | --whole] [--ort libonnxruntime.so] [--threads 4]
//!    [--hotword <phrase>]... [--screen <text file>] [--beam 4] [--boost 1.5]
//!    [--samples N] [--sigma X] [--no-early-stop] [--seed N]   (defaults depend on the model)
//!    [--rewrite-url <base url> --rewrite-model <name>]   (API key from MD_API_KEY)
//!    [--language <code>]   (streaming NeMo models; default auto)
//!    [--live [--debug]]   (play the file into the dictation pipeline in real time)
//! ```
//!
//! The model directory holds a Qwen3-ASR export or a transducer; `--beam` and `--boost` apply to
//! transducers. `--live` runs the pipeline as the app does, so a streaming model encodes while the
//! file plays, and it reports how long the text took after the audio ended.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, SearchConfig};
use mini_dictator_core::audio::{AudioChunk, Resampler};
use mini_dictator_core::context::{
    rank_terms, ContextProvider, ContextSource, Region, ScreenContext, TextBlock,
};
use mini_dictator_core::join::{Joiner, Piece};
use mini_dictator_core::pipeline::{
    run_dictation, Engine, EngineConfig, Event, SessionContext, TextSink, Timings, Transcript,
};
use mini_dictator_core::rewrite::{OpenAiCompatible, PassThrough, Rewriter};
use mini_dictator_core::uncertainty;
use mini_dictator_core::vad::{
    EnergyVad, Segmenter, SileroVad, SpeechSegment, VoiceActivityDetector,
};

struct Args {
    model: PathBuf,
    wav: PathBuf,
    vad: Option<PathBuf>,
    ort: Option<PathBuf>,
    threads: usize,
    hotwords: Vec<String>,
    screen: Option<PathBuf>,
    search: SearchConfig,
    samples: Option<usize>,
    sigma: Option<f32>,
    early_stop: bool,
    seed: Option<u64>,
    /// Decode the file as one segment, without VAD.
    whole: bool,
    rewrite_url: Option<String>,
    rewrite_model: Option<String>,
    live: bool,
    /// Print the pipeline's debug events in `--live` mode.
    debug: bool,
    partial_ms: Option<u32>,
    early_commit_ms: Option<u32>,
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut wav, mut vad, mut ort, mut screen, mut rewrite_url, mut rewrite_model) =
        (None, None, None, None, None, None, None);
    let mut hotwords = Vec::new();
    let mut threads = 4;
    let mut search = SearchConfig::default();
    let (mut samples, mut sigma, mut seed, mut early_stop) = (None, None, None, true);
    let (mut whole, mut live, mut debug) = (false, false, false);
    let (mut partial_ms, mut early_commit_ms) = (None, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        let number = |v: String| v.parse::<f32>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--wav" => wav = Some(PathBuf::from(value()?)),
            "--vad" => vad = Some(PathBuf::from(value()?)),
            "--ort" => ort = Some(PathBuf::from(value()?)),
            "--threads" => threads = number(value()?)? as usize,
            "--screen" => screen = Some(PathBuf::from(value()?)),
            "--hotword" => hotwords.push(value()?),
            "--beam" => search.beam = number(value()?)? as usize,
            "--boost" => search.hotword_boost = number(value()?)?,
            "--blank-penalty" => search.blank_penalty = number(value()?)?,
            "--samples" => samples = Some(number(value()?)? as usize),
            "--sigma" => sigma = Some(number(value()?)?),
            "--seed" => seed = Some(number(value()?)? as u64),
            "--no-early-stop" => early_stop = false,
            "--whole" => whole = true,
            "--live" => live = true,
            "--debug" => debug = true,
            "--partial-ms" => partial_ms = Some(number(value()?)? as u32),
            "--early-commit-ms" => early_commit_ms = Some(number(value()?)? as u32),
            "--language" => search.language = Some(value()?),
            "--rewrite-url" => rewrite_url = Some(value()?),
            "--rewrite-model" => rewrite_model = Some(value()?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if whole && live {
        return Err("--whole and --live do not go together: live mode always uses a VAD".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        wav: wav.ok_or("--wav is required")?,
        vad,
        ort,
        threads,
        hotwords,
        screen,
        search,
        samples,
        sigma,
        early_stop,
        seed,
        whole,
        rewrite_url,
        rewrite_model,
        live,
        debug,
        partial_ms,
        early_commit_ms,
    })
}

fn read_wav(path: &PathBuf) -> Result<AudioChunk, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };
    Ok(AudioChunk {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        captured: Instant::now(),
        sound_playing: false,
    })
}

fn segments<V: VoiceActivityDetector>(
    vad: V,
    engine_config: &EngineConfig,
    samples: &[f32],
) -> Vec<SpeechSegment> {
    let mut segmenter = Segmenter::new(vad, engine_config.segmenter.clone());
    let mut found = Vec::new();
    for block in samples.chunks(1600) {
        found.extend(segmenter.push(block));
    }
    found.extend(segmenter.finish());
    found
}

/// Plays `audio` into [`run_dictation`] in 100 ms blocks at real-time pace, printing each event
/// with the time since playback started.
/// The screen given on the command line, as live dictation would read it.
struct Fixed(ScreenContext);

impl ContextProvider for Fixed {
    fn snapshot(&self) -> mini_dictator_core::Result<ScreenContext> {
        Ok(self.0.clone())
    }
}

fn live<V: VoiceActivityDetector>(
    engine: &Engine,
    vad: V,
    audio: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    struct Print;
    impl TextSink for Print {
        fn insert(&mut self, _: &str) -> mini_dictator_core::Result<()> {
            Ok(())
        }
    }
    let (sender, receiver) = std::sync::mpsc::channel();
    let started = Instant::now();
    let duration = Duration::from_secs_f64(audio.len() as f64 / 16_000.0);
    let feeder = std::thread::spawn(move || {
        let mut sent = 0;
        for block in audio.chunks(1600) {
            // Each block is due when its last sample would have been recorded.
            sent += block.len();
            let due = started + Duration::from_secs_f64(sent as f64 / 16_000.0);
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            let chunk = AudioChunk {
                samples: block.to_vec(),
                sample_rate: 16_000,
                channels: 1,
                captured: Instant::now(),
                sound_playing: false,
            };
            if sender.send(chunk).is_err() {
                return;
            }
        }
    });
    let at = |label: &str| println!("[{:6.2}s] {label}", started.elapsed().as_secs_f64());
    run_dictation(
        engine,
        vad,
        receiver,
        &mut Print,
        &mut |event| match event {
            Event::Transcribed {
                plain,
                unclear_spans,
                behind_seconds,
                ..
            } => at(&format!(
                "transcribed ({unclear_spans} unclear{}): {plain}",
                behind_seconds.map_or(String::new(), |s| format!(", {s:.1} s behind"))
            )),
            Event::Inserted { text } => at(&format!("inserted: {text}")),
            Event::Partial { text } => at(&format!("hearing: {text}")),
            Event::Warning(message) => at(&format!("warning: {message}")),
            Event::Debug(text) => at(&text),
            _ => {}
        },
    )?;
    feeder.join().ok();
    println!(
        "audio ended at {:.2}s, done at {:.2}s: {} after the audio",
        duration.as_secs_f64(),
        started.elapsed().as_secs_f64(),
        ms(started.elapsed().saturating_sub(duration)),
    );
    Ok(())
}

fn ms(duration: Duration) -> String {
    format!("{:.0} ms", duration.as_secs_f64() * 1000.0)
}

fn main() {
    let args = match parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\n\nusage: md --model <dir> --wav <file> [--vad <onnx> | --whole] [--ort <lib>] [--threads N] [--hotword <phrase>]... [--screen <txt>] [--beam N] [--boost X] [--samples N] [--sigma X] [--rewrite-url URL --rewrite-model NAME] [--language CODE] [--live [--debug] [--partial-ms N] [--early-commit-ms N]]");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    init_runtime(args.ort.as_deref())?;
    let started = Instant::now();
    let recognizer = asr::load(&args.model, args.threads, args.search.clone())?;
    eprintln!(
        "loaded {} in {}",
        args.model.display(),
        ms(started.elapsed())
    );

    let rewriter: Box<dyn Rewriter> = match (&args.rewrite_url, &args.rewrite_model) {
        (Some(url), Some(model)) => Box::new(OpenAiCompatible {
            base_url: url.clone(),
            api_key: std::env::var("MD_API_KEY").ok(),
            model: model.clone(),
            timeout: Duration::from_secs(20),
        }),
        _ => Box::new(PassThrough),
    };
    let mut monte_carlo = recognizer.monte_carlo();
    monte_carlo.samples = args.samples.unwrap_or(monte_carlo.samples);
    monte_carlo.relative_sigma = args.sigma.unwrap_or(monte_carlo.relative_sigma);
    monte_carlo.seed = args.seed.unwrap_or(monte_carlo.seed);
    if !args.early_stop {
        monte_carlo.first_samples = 0;
    }
    let defaults = EngineConfig::default();
    let config = EngineConfig {
        monte_carlo,
        debug: std::sync::Arc::new(args.debug.into()),
        partial_ms: args.partial_ms.unwrap_or(defaults.partial_ms),
        early_commit_ms: args.early_commit_ms.unwrap_or(defaults.early_commit_ms),
        ..defaults
    };
    let mut engine = Engine {
        asr: recognizer,
        context: Box::new(mini_dictator_core::context::NoContext),
        memory: Default::default(),
        rewriter,
        config,
    };

    let mut blocks: Vec<TextBlock> = args
        .hotwords
        .iter()
        .map(|h| TextBlock {
            region: Region::Window,
            text: format!("「{h}」"),
        })
        .collect();
    if let Some(path) = &args.screen {
        blocks.extend(
            std::fs::read_to_string(path)?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| TextBlock {
                    region: Region::Window,
                    text: l.to_owned(),
                }),
        );
    }
    let screen = ScreenContext {
        app: None,
        blocks,
        source: ContextSource::Accessibility,
    };
    let hotwords: Vec<String> = rank_terms(&screen, engine.config.max_hotwords)
        .into_iter()
        .map(|t| t.phrase)
        .collect();
    let bias = engine.asr.bias(&hotwords);
    if !hotwords.is_empty() {
        println!("hotwords: {}", bias.phrases.join(", "));
    }
    let mut session = SessionContext {
        screen,
        hotwords,
        bias,
        inserted: String::new(),
        joiner: Joiner::default(),
    };

    let audio = read_wav(&args.wav)?.to_model_input(&mut Resampler::default());
    if args.live {
        // The pipeline reads the screen itself when dictation starts.
        engine.context = Box::new(Fixed(session.screen));
        return match &args.vad {
            Some(path) => live(&engine, SileroVad::load(path)?, audio),
            None => live(&engine, EnergyVad::default(), audio),
        };
    }
    let found = match &args.vad {
        _ if args.whole => vec![SpeechSegment {
            samples: audio,
            start_sample: 0,
            truncated: false,
            pre_roll: 0,
            post_roll: 0,
        }],
        Some(path) => segments(SileroVad::load(path)?, &engine.config, &audio),
        None => segments(EnergyVad::default(), &engine.config, &audio),
    };

    for segment in &found {
        let started = Instant::now();
        let encoder = engine.asr.encode(&segment.samples)?;
        let encoded = started.elapsed();
        let reference = engine.asr.decode_one(&encoder, &session.bias)?;
        let searched = started.elapsed();
        let uncertainty = uncertainty::estimate(
            engine.asr.as_ref(),
            &encoder,
            &reference,
            &session.bias,
            &engine.config.monte_carlo,
        )?;
        let total = started.elapsed();
        let marked = uncertainty::format_for_llm(&reference, &uncertainty.analysis.spans);
        println!(
            "\n[{:.2}s +{:.2}s] {} encoder frames: encode {}, decode {}, {} noisy decodes {}{}",
            segment.start_sample as f32 / 16_000.0,
            segment.duration_seconds(),
            encoder.frames,
            ms(encoded),
            ms(searched - encoded),
            uncertainty.samples.len(),
            ms(total - searched),
            if uncertainty.stopped_early {
                " (stopped early)"
            } else {
                ""
            },
        );
        let transcript = Transcript {
            reference,
            uncertainty,
            marked,
            timings: Timings {
                encode: encoded,
                decode: searched - encoded,
                check: total - searched,
            },
        };
        println!("  plain:  {}", transcript.reference.text());
        println!("  marked: {}", transcript.marked);
        let pause_seconds = session.joiner.pause_before(segment);
        let mut text = transcript.reference.text();
        let mut rewritten = false;
        if args.rewrite_url.is_some() {
            let request = engine.rewrite_request(&transcript, &session, pause_seconds);
            let started = Instant::now();
            match engine.rewriter.rewrite(&request) {
                Ok(Some(reply)) => {
                    println!("  rewrite ({}): {reply}", ms(started.elapsed()));
                    text = reply;
                    rewritten = true;
                }
                Ok(None) => {}
                Err(error) => println!("  rewrite failed: {error}"),
            }
        }
        if !text.trim().is_empty() {
            let piece = Piece {
                rewritten,
                text: &text,
                pause_seconds,
                names: &session.hotwords,
            };
            let joined = session
                .joiner
                .join(&session.inserted, &piece, &engine.config.join);
            session.joiner.typed(segment);
            session.inserted.push_str(&joined);
        }
    }
    session.inserted += &session.joiner.finish();
    println!("\ntext: {}", session.inserted);
    Ok(())
}
