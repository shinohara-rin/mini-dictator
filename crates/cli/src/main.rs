//! `md`: runs the dictation core on a WAV file and prints each step.
//!
//! ```text
//! md --model <dir> --wav <file> [--vad silero_vad.onnx | --whole] [--ort libonnxruntime.so] [--threads 4]
//!    [--hotword <phrase>]... [--screen <text file>] [--beam 4] [--boost 1.5]
//!    [--samples N] [--sigma X] [--no-early-stop] [--seed N]   (defaults depend on the model)
//!    [--rewrite-url <base url> --rewrite-model <name> [--rewrite-audio]]   (API key from MD_API_KEY)
//!    [--llama-server <binary> --rewrite-gguf <model.gguf> [--rewrite-mmproj <mmproj.gguf>]]
//! ```
//!
//! `--llama-server` runs the built-in rewriter: llama.cpp's server with a local model such as
//! Gemma 4 E2B, which also hears each segment when given its audio projector.
//!
//! The model directory holds a Qwen3-ASR export or a transducer; `--beam` and `--boost` apply to
//! transducers.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, SearchConfig};
use mini_dictator_core::audio::{AudioChunk, Resampler};
use mini_dictator_core::context::{rank_terms, ContextSource, Region, ScreenContext, TextBlock};
use mini_dictator_core::join::{Joiner, Piece};
use mini_dictator_core::pipeline::{Engine, EngineConfig, SessionContext, Timings, Transcript};
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
    /// Attach each segment's audio to the endpoint request.
    rewrite_audio: bool,
    llama_server: Option<PathBuf>,
    rewrite_gguf: Option<PathBuf>,
    rewrite_mmproj: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut wav, mut vad, mut ort, mut screen, mut rewrite_url, mut rewrite_model) =
        (None, None, None, None, None, None, None);
    let mut hotwords = Vec::new();
    let mut threads = 4;
    let mut search = SearchConfig::default();
    let (mut samples, mut sigma, mut seed, mut early_stop) = (None, None, None, true);
    let mut whole = false;
    let (mut rewrite_audio, mut llama_server, mut rewrite_gguf, mut rewrite_mmproj) =
        (false, None, None, None);
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
            "--rewrite-url" => rewrite_url = Some(value()?),
            "--rewrite-model" => rewrite_model = Some(value()?),
            "--rewrite-audio" => rewrite_audio = true,
            "--llama-server" => llama_server = Some(PathBuf::from(value()?)),
            "--rewrite-gguf" => rewrite_gguf = Some(PathBuf::from(value()?)),
            "--rewrite-mmproj" => rewrite_mmproj = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}")),
        }
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
        rewrite_audio,
        llama_server,
        rewrite_gguf,
        rewrite_mmproj,
    })
}

#[cfg(unix)]
fn builtin_rewriter(args: &Args) -> Result<Option<Box<dyn Rewriter>>, String> {
    use mini_dictator_core::rewrite::{Builtin, LlamaServer};
    let Some(binary) = &args.llama_server else {
        return Ok(None);
    };
    let model = args
        .rewrite_gguf
        .clone()
        .ok_or("--llama-server needs --rewrite-gguf")?;
    let started = Instant::now();
    let rewriter = Builtin::new(LlamaServer {
        binary: binary.clone(),
        model,
        mmproj: args.rewrite_mmproj.clone(),
        log: Some(std::env::temp_dir().join("md-llama-server.log")),
    });
    rewriter.start().map_err(|e| e.to_string())?;
    eprintln!("started llama-server in {}", ms(started.elapsed()));
    Ok(Some(Box::new(rewriter)))
}

#[cfg(not(unix))]
fn builtin_rewriter(args: &Args) -> Result<Option<Box<dyn Rewriter>>, String> {
    match args.llama_server {
        Some(_) => Err("the built-in rewriter needs a Unix system".into()),
        None => Ok(None),
    }
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

fn ms(duration: Duration) -> String {
    format!("{:.0} ms", duration.as_secs_f64() * 1000.0)
}

fn main() {
    let args = match parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\n\nusage: md --model <dir> --wav <file> [--vad <onnx> | --whole] [--ort <lib>] [--threads N] [--hotword <phrase>]... [--screen <txt>] [--beam N] [--boost X] [--samples N] [--sigma X] [--rewrite-url URL --rewrite-model NAME [--rewrite-audio]] [--llama-server BIN --rewrite-gguf GGUF [--rewrite-mmproj GGUF]]");
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
            send_audio: args.rewrite_audio,
        }),
        _ => match builtin_rewriter(&args)? {
            Some(rewriter) => rewriter,
            None => Box::new(PassThrough),
        },
    };
    let rewrites = args.rewrite_url.is_some() || args.llama_server.is_some();
    let mut monte_carlo = recognizer.monte_carlo();
    monte_carlo.samples = args.samples.unwrap_or(monte_carlo.samples);
    monte_carlo.relative_sigma = args.sigma.unwrap_or(monte_carlo.relative_sigma);
    monte_carlo.seed = args.seed.unwrap_or(monte_carlo.seed);
    if !args.early_stop {
        monte_carlo.first_samples = 0;
    }
    let config = EngineConfig {
        monte_carlo,
        ..EngineConfig::default()
    };
    let engine = Engine {
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
        if rewrites {
            let request = engine.rewrite_request(segment, &transcript, &session, pause_seconds);
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
