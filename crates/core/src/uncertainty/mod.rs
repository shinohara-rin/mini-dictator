//! ASR uncertainty by Monte Carlo over the encoder output.
//!
//! The encoder runs once. The search then runs again on copies of the encoder output with Gaussian
//! noise, and [`analysis`] compares those decodes with the clean one. Spans where the decodes
//! disagree are where the recognizer is unsure, and the rewrite model reads them as
//! `[unclear: ...]` markers with the other readings.
//!
//! Noise on the encoder output costs only decoder and joiner calls, which are small next to the
//! encoder, and it keeps every frame at its time, so decodes compare by frame. AIRI's probe adds
//! noise to the audio instead and decodes each copy from scratch.

pub mod analysis;

pub use analysis::{analyze, format_for_llm, Analysis, AnalysisOptions, SpanKind, UncertainSpan};

use crate::asr::{ContextGraph, EncoderOutput, Hypothesis, Recognizer, TransducerModel};
use crate::rng::SplitMix64;
use crate::Result;

#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Noisy decodes when the early stop does not apply. 0 turns uncertainty off.
    pub samples: usize,
    /// Noisy decodes that run first. The others run only when one of these disagrees with the
    /// clean decode. 0, or `samples` or more, turns the early stop off.
    pub first_samples: usize,
    /// Noise standard deviation as a fraction of each encoder dimension's standard deviation over
    /// the utterance. Dimensions differ in scale, so a single absolute level would drown some and
    /// miss others.
    pub relative_sigma: f32,
    pub seed: u64,
    pub analysis: AnalysisOptions,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            samples: 12,
            first_samples: 4,
            relative_sigma: 1.0,
            seed: 0x6d69_6e69,
            analysis: AnalysisOptions::default(),
        }
    }
}

/// The clean decode with its noisy decodes and their comparison.
#[derive(Debug, Clone)]
pub struct Uncertainty {
    pub samples: Vec<Hypothesis>,
    pub analysis: Analysis,
    /// True when the first decodes all matched the clean one, so the others did not run.
    pub stopped_early: bool,
}

/// Standard deviation of each encoder dimension over the frames of one utterance.
pub fn dimension_scale(encoder: &EncoderOutput) -> Vec<f32> {
    let n = encoder.frames.max(1) as f32;
    let mut mean = vec![0.0f32; encoder.dim];
    for t in 0..encoder.frames {
        for (m, v) in mean.iter_mut().zip(encoder.frame(t)) {
            *m += v / n;
        }
    }
    let mut variance = vec![0.0f32; encoder.dim];
    for t in 0..encoder.frames {
        for ((s, v), m) in variance.iter_mut().zip(encoder.frame(t)).zip(&mean) {
            *s += (v - m).powi(2) / n;
        }
    }
    variance.into_iter().map(f32::sqrt).collect()
}

/// A copy of `encoder` with independent Gaussian noise of `relative_sigma * scale[d]` on dimension `d`.
pub fn perturb(
    encoder: &EncoderOutput,
    scale: &[f32],
    relative_sigma: f32,
    rng: &mut SplitMix64,
) -> EncoderOutput {
    let mut noise = vec![0.0f32; encoder.data.len()];
    rng.fill_gaussian(&mut noise);
    let data = encoder
        .data
        .iter()
        .zip(noise)
        .enumerate()
        .map(|(i, (v, n))| v + n * relative_sigma * scale[i % encoder.dim])
        .collect();
    EncoderOutput {
        frames: encoder.frames,
        dim: encoder.dim,
        data,
    }
}

/// True when every sample has the reference's words in order, each at its reference time.
fn agrees(
    reference: &Hypothesis,
    samples: &[Hypothesis],
    duration: f32,
    options: &AnalysisOptions,
) -> bool {
    let words = |h: &Hypothesis| {
        h.tokens
            .iter()
            .map(|t| analysis::normalize_token(&t.text))
            .filter(|k| !k.is_empty())
            .collect::<Vec<_>>()
    };
    let expected = words(reference);
    samples.iter().all(|s| words(s) == expected)
        && analyze(reference, samples, duration, options)
            .tokens
            .iter()
            .all(|t| t.by_time == 1.0)
}

/// Decodes noisy copies of `encoder` with the recognizer's search and the same hotwords as the
/// reference, so a hotword cannot look unstable just because the samples lack its boost.
pub fn estimate<M: TransducerModel>(
    recognizer: &Recognizer<M>,
    encoder: &EncoderOutput,
    reference: &Hypothesis,
    graph: Option<&ContextGraph>,
    config: &MonteCarloConfig,
) -> Result<Uncertainty> {
    let duration = encoder.frames as f32 * recognizer.model().frame_seconds();
    if config.samples == 0 {
        return Ok(Uncertainty {
            samples: Vec::new(),
            analysis: analyze(reference, &[], duration, &config.analysis),
            stopped_early: false,
        });
    }
    let scale = dimension_scale(encoder);
    let mut rng = SplitMix64::new(config.seed);
    let mut decode = |count: usize, samples: &mut Vec<Hypothesis>| -> Result<()> {
        let noisy: Vec<EncoderOutput> = (0..count)
            .map(|_| perturb(encoder, &scale, config.relative_sigma, &mut rng))
            .collect();
        samples.extend(recognizer.search_batch(&noisy, graph)?);
        Ok(())
    };

    let early = config.first_samples > 0 && config.first_samples < config.samples;
    let mut samples = Vec::with_capacity(config.samples);
    decode(
        if early {
            config.first_samples
        } else {
            config.samples
        },
        &mut samples,
    )?;
    let stopped_early = early && agrees(reference, &samples, duration, &config.analysis);
    if early && !stopped_early {
        decode(config.samples - config.first_samples, &mut samples)?;
    }
    let analysis = analyze(reference, &samples, duration, &config.analysis);
    Ok(Uncertainty {
        samples,
        analysis,
        stopped_early,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::testing::{symbols, ScriptedModel};
    use crate::asr::SearchConfig;

    fn recognizer() -> Recognizer<ScriptedModel> {
        Recognizer::new(
            ScriptedModel { vocab: 5 },
            symbols(&["<blk>", "▁A", "▁B", "▁C", "▁D"]),
            SearchConfig {
                beam: 1,
                ..SearchConfig::default()
            },
        )
    }

    #[test]
    fn confident_frames_stop_early() {
        let recognizer = recognizer();
        let encoder = recognizer.model().encoder(&[0, 1, 0, 2, 0, 3, 0], 12.0);
        let reference = recognizer.search(&encoder, None).unwrap();
        let result = estimate(
            &recognizer,
            &encoder,
            &reference,
            None,
            &MonteCarloConfig {
                relative_sigma: 0.3,
                ..MonteCarloConfig::default()
            },
        )
        .unwrap();
        assert!(result.stopped_early);
        assert!(result.analysis.spans.is_empty());
        assert_eq!(result.samples.len(), 4);
    }

    #[test]
    fn a_close_call_becomes_an_unclear_span() {
        let recognizer = recognizer();
        let mut encoder = recognizer.model().encoder(&[0, 1, 0, 2, 0, 3, 0], 12.0);
        // Frame 3 barely prefers B over D.
        encoder.data[3 * 5 + 4] = 11.9;
        let reference = recognizer.search(&encoder, None).unwrap();
        assert_eq!(reference.text(), "A B C");
        let result = estimate(
            &recognizer,
            &encoder,
            &reference,
            None,
            &MonteCarloConfig {
                samples: 40,
                relative_sigma: 0.3,
                ..MonteCarloConfig::default()
            },
        )
        .unwrap();
        assert!(!result.stopped_early);
        let spans = &result.analysis.spans;
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!((spans[0].token_start, spans[0].token_end), (1, 2));
        assert_eq!(
            format_for_llm(&reference, spans),
            "A [unclear: \"B\", also heard \"D\"] C"
        );
    }
}
