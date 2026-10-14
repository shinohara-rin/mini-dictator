//! Measures speculative decoding and a teacher-forced uncertainty pass on Qwen3-ASR.
//!
//! cargo run --release -p mini-dictator-cli --example spec -- <model dir> <libonnxruntime> <wav>...

use std::path::Path;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::qwen::lab::TextDrafter;
use mini_dictator_core::asr::qwen::{QwenAsr, QwenConfig, QwenFiles};
use mini_dictator_core::asr::{Asr, Bias, EncoderOutput};
use mini_dictator_core::audio::{AudioChunk, Resampler};
use mini_dictator_core::rng::SplitMix64;
use mini_dictator_core::uncertainty::{dimension_scale, perturb};

fn ms(d: Duration) -> String {
    format!("{:.0}ms", d.as_secs_f64() * 1000.0)
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    };
    AudioChunk {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        captured: Instant::now(),
    }
    .to_model_input(&mut Resampler::default())
}

/// What a streaming transducer might give: lowercase, no punctuation, every `every`th word wrong.
fn transducer_like(text: &str, every: usize) -> String {
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| {
            let w: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect();
            if every > 0 && i % every == every - 1 {
                "uh".to_owned()
            } else {
                w
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    init_runtime(Some(Path::new(&args[1]))).unwrap();
    let threads = std::env::var("THREADS").map_or(4, |t| t.parse().unwrap());
    let asr = QwenAsr::load(
        &QwenFiles::discover(Path::new(&args[0])).unwrap(),
        threads,
        QwenConfig::default(),
    )
    .unwrap();
    let hotwords: Vec<String> = std::env::var("HOTWORDS")
        .map(|h| h.split(',').map(|s| s.trim().to_owned()).collect())
        .unwrap_or_default();
    let bias = asr.bias(&hotwords);
    let quick = std::env::var("QUICK").is_ok();

    for wav in &args[2..] {
        let samples = read_wav(wav);
        let encoder = asr.encode(&samples).unwrap();
        println!(
            "\n=== {wav}: {:.1}s, {} audio tokens, prompt {} tokens",
            samples.len() as f32 / 16000.0,
            encoder.frames,
            asr.lab_prompt_len(encoder.frames, &bias)
        );
        // Warm up.
        asr.lab_decode(&encoder, &bias, None).unwrap();
        let greedy = asr.lab_decode(&encoder, &bias, None).unwrap();
        let text = asr.lab_text(&greedy.ids);
        let header = asr.tokenizer().decode(
            &greedy.ids[..greedy
                .ids
                .iter()
                .position(|id| Some(*id) == asr.tokenizer().added("<asr_text>"))
                .unwrap_or(0)],
        );
        let language = header.trim().strip_prefix("language ").unwrap_or("English").to_owned();
        println!("text: {text}");
        println!(
            "greedy: {} tokens, {} calls, prefill {}, total {} ({:.0} ms/token after prefill)",
            greedy.ids.len(),
            greedy.calls,
            ms(greedy.first_call),
            ms(greedy.total),
            (greedy.total - greedy.first_call).as_secs_f64() * 1000.0 / (greedy.calls - 1).max(1) as f64
        );

        if !quick {
            let prefill = asr.lab_prefill_cost(&encoder, &bias).unwrap();
            let costs: Vec<String> = [1, 2, 4, 8, 16, 32, 64]
                .iter()
                .map(|l| {
                    let t = (0..3)
                        .map(|_| asr.lab_step_cost(&encoder, &bias, *l).unwrap())
                        .min()
                        .unwrap();
                    format!("L={l}: {}", ms(t))
                })
                .collect();
            println!("prefill alone {}; one call of L tokens: {}", ms(prefill), costs.join(", "));
        }

        let mut drafts: Vec<(String, TextDrafter)> = vec![
            ("exact, whole".into(), TextDrafter::new(&asr, &language, &text, 512)),
            ("exact, block 8".into(), TextDrafter::new(&asr, &language, &text, 8)),
        ];
        for fraction in [0.6, 0.8] {
            let cut = (samples.len() as f32 * fraction) as usize;
            let partial = asr.encode(&samples[..cut]).unwrap();
            let partial = asr.lab_text(&asr.lab_decode(&partial, &bias, None).unwrap().ids);
            drafts.push((
                format!("partial {fraction} [{partial}]"),
                TextDrafter::new(&asr, &language, &partial, 16),
            ));
        }
        for every in [0, 8, 4] {
            let rough = transducer_like(&text, every);
            for block in [8, 16] {
                drafts.push((
                    format!("lowercase, 1 in {every} wrong, block {block} [{rough}]"),
                    TextDrafter::new(&asr, &language, &rough, block),
                ));
            }
        }
        // A cased, punctuated draft with some words wrong, as a streaming model with punctuation
        // (Nemotron) might give, and a draft of the header alone.
        for every in [8, 4] {
            let rough: String = text
                .split_whitespace()
                .enumerate()
                .map(|(i, w)| if i % every == every - 1 { "uh" } else { w })
                .collect::<Vec<_>>()
                .join(" ");
            drafts.push((
                format!("cased, 1 in {every} wrong, block 8 [{rough}]"),
                TextDrafter::new(&asr, &language, &rough, 8),
            ));
        }
        let mut header_only = TextDrafter::new(&asr, &language, "", 8);
        header_only.header_only = true;
        drafts.push(("header only".into(), header_only));
        if let Ok(only) = std::env::var("ONLY") {
            drafts.retain(|(name, _)| only.split(',').any(|o| name.starts_with(o)));
        }
        for (name, drafter) in &drafts {
            let run = asr.lab_decode(&encoder, &bias, Some(drafter)).unwrap();
            let same = run.ids == greedy.ids;
            println!(
                "  {name}\n    -> {} calls, kept {}/{} drafted, {} ({:.2}x), {}",
                run.calls,
                run.accepted,
                run.proposed,
                ms(run.total),
                greedy.total.as_secs_f64() / run.total.as_secs_f64(),
                if same {
                    "same output".to_owned()
                } else {
                    format!("DIFFERS: {}", asr.lab_text(&run.ids))
                }
            );
        }

        // Uncertainty: 4 noisy free-running decodes against one teacher-forced pass.
        let scale = dimension_scale(&encoder);
        let mut rng = SplitMix64::new(7);
        let noisy: Vec<EncoderOutput> = (0..4)
            .map(|_| perturb(&encoder, &scale, 0.5, &mut rng))
            .collect();
        let started = Instant::now();
        let free = asr.decode(&noisy, &bias).unwrap();
        let free_time = started.elapsed();
        let started = Instant::now();
        let refs: Vec<&EncoderOutput> = noisy.iter().collect();
        let scored = asr.lab_score(&refs, &bias, &greedy.ids).unwrap();
        let forced_time = started.elapsed();
        println!("check: 4 free noisy decodes {}, 1 teacher-forced pass of 4 rows {}", ms(free_time), ms(forced_time));
        for h in &free {
            println!("    free:   {}", h.text());
        }
        let started = Instant::now();
        let clean = asr.lab_score(&[&encoder], &bias, &greedy.ids).unwrap();
        println!("    clean teacher-forced pass alone {}", ms(started.elapsed()));
        let mut flagged = Vec::new();
        for (p, id) in greedy.ids.iter().enumerate() {
            let flips = scored.iter().filter(|row| row[p].best != *id).count();
            let worst = scored
                .iter()
                .map(|row| row[p].target_log_prob)
                .fold(f32::INFINITY, f32::min);
            if flips > 0 || worst < (0.5f32).ln() {
                flagged.push(format!(
                    "{:?} flips {flips}/4, p(ref) clean {:.2} noisy min {:.2}, alt {:?}",
                    asr.tokenizer().decode(&[*id]),
                    clean[0][p].target_log_prob.exp(),
                    worst.exp(),
                    scored
                        .iter()
                        .filter(|row| row[p].best != *id)
                        .map(|row| asr.tokenizer().decode(&[row[p].best]))
                        .collect::<Vec<_>>()
                ));
            }
        }
        for f in flagged {
            println!("    forced: {f}");
        }
        let _ = Bias::default();
    }
}
