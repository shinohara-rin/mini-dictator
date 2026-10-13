//! `md`: runs the dictation core on a WAV file and prints each step.
//!
//! ```text
//! md --model <dir> --wav <file> [--vad silero_vad.onnx] [--ort libonnxruntime.so]
//!    [--hotword <phrase>]... [--screen <text file>] [--beam 4] [--boost 1.5]
//!    [--samples 12] [--sigma 0.5] [--no-early-stop]
//!    [--rewrite-url <base url> --rewrite-model <name>]   (API key from MD_API_KEY)
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::{init_runtime, OnnxTransducer, TransducerFiles};
use mini_dictator_core::asr::{Recognizer, SearchConfig};
use mini_dictator_core::audio::{AudioChunk, Resampler};
use mini_dictator_core::context::{rank_terms, ContextSource, Region, ScreenContext, TextBlock};
use mini_dictator_core::pipeline::{Engine, EngineConfig, SessionContext, Transcript};
use mini_dictator_core::rewrite::{OpenAiCompatible, PassThrough, Rewriter};
use mini_dictator_core::uncertainty::{self, MonteCarloConfig};
use mini_dictator_core::vad::{
    EnergyVad, Segmenter, SileroVad, SpeechSegment, VoiceActivityDetector,
};

struct Args {
    model: PathBuf,
    wav: PathBuf,
    vad: Option<PathBuf>,
    ort: Option<PathBuf>,
    hotwords: Vec<String>,
    screen: Option<PathBuf>,
    search: SearchConfig,
    monte_carlo: MonteCarloConfig,
    rewrite_url: Option<String>,
    rewrite_model: Option<String>,
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut wav, mut vad, mut ort, mut screen, mut rewrite_url, mut rewrite_model) =
        (None, None, None, None, None, None, None);
    let mut hotwords = Vec::new();
    let mut search = SearchConfig::default();
    let mut monte_carlo = MonteCarloConfig::default();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        let number = |v: String| v.parse::<f32>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--wav" => wav = Some(PathBuf::from(value()?)),
            "--vad" => vad = Some(PathBuf::from(value()?)),
            "--ort" => ort = Some(PathBuf::from(value()?)),
            "--screen" => screen = Some(PathBuf::from(value()?)),
            "--hotword" => hotwords.push(value()?),
            "--beam" => search.beam = number(value()?)? as usize,
            "--boost" => search.hotword_boost = number(value()?)?,
            "--blank-penalty" => search.blank_penalty = number(value()?)?,
            "--samples" => monte_carlo.samples = number(value()?)? as usize,
            "--sigma" => monte_carlo.relative_sigma = number(value()?)?,
            "--seed" => monte_carlo.seed = number(value()?)? as u64,
            "--no-early-stop" => monte_carlo.first_samples = 0,
            "--rewrite-url" => rewrite_url = Some(value()?),
            "--rewrite-model" => rewrite_model = Some(value()?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        wav: wav.ok_or("--wav is required")?,
        vad,
        ort,
        hotwords,
        screen,
        search,
        monte_carlo,
        rewrite_url,
        rewrite_model,
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
            eprintln!("{message}\n\nusage: md --model <dir> --wav <file> [--vad <onnx>] [--ort <lib>] [--hotword <phrase>]... [--screen <txt>] [--beam N] [--boost X] [--samples N] [--sigma X] [--rewrite-url URL --rewrite-model NAME]");
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
    let files = TransducerFiles::discover(&args.model, true)?;
    let started = Instant::now();
    let (model, symbols) = OnnxTransducer::load(&files, 2)?;
    eprintln!(
        "loaded {} in {}",
        files.encoder.display(),
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
    let config = EngineConfig {
        monte_carlo: args.monte_carlo.clone(),
        ..EngineConfig::default()
    };
    let engine = Engine {
        recognizer: Recognizer::new(model, symbols, args.search.clone()),
        context: Box::new(mini_dictator_core::context::NoContext),
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
    let graph = engine
        .recognizer
        .context_graph(&hotwords, engine.recognizer.search.hotword_boost);
    if !hotwords.is_empty() {
        println!("hotwords: {}", hotwords.join(", "));
    }
    let mut session = SessionContext {
        screen,
        hotwords,
        graph: (!graph.is_empty()).then_some(graph),
        inserted: String::new(),
    };

    let audio = read_wav(&args.wav)?.to_model_input(&mut Resampler::default());
    let found = match &args.vad {
        Some(path) => segments(SileroVad::load(path)?, &engine.config, &audio),
        None => segments(EnergyVad::default(), &engine.config, &audio),
    };

    for segment in &found {
        let started = Instant::now();
        let encoder = engine.recognizer.encode(&segment.samples)?;
        let encoded = started.elapsed();
        let reference = engine.recognizer.search(&encoder, session.graph.as_ref())?;
        let searched = started.elapsed();
        let uncertainty = uncertainty::estimate(
            &engine.recognizer,
            &encoder,
            &reference,
            session.graph.as_ref(),
            &engine.config.monte_carlo,
        )?;
        let total = started.elapsed();
        let marked = uncertainty::format_for_llm(&reference, &uncertainty.analysis.spans);
        println!(
            "\n[{:.2}s +{:.2}s] {} encoder frames: encode {}, search {}, {} noisy decodes {}{}",
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
        };
        println!("  plain:  {}", transcript.reference.text());
        println!("  marked: {}", transcript.marked);
        if args.rewrite_url.is_some() {
            let request = engine.rewrite_request(&transcript, &session);
            let started = Instant::now();
            match engine.rewriter.rewrite(&request) {
                Ok(text) => {
                    println!("  rewrite ({}): {text}", ms(started.elapsed()));
                    session.inserted.push_str(&text);
                    session.inserted.push(' ');
                }
                Err(error) => println!("  rewrite failed: {error}"),
            }
        }
    }
    Ok(())
}
