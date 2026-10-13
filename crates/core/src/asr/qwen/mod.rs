//! Qwen3-ASR through ONNX Runtime, in the export that sherpa-onnx ships
//! (`sherpa-onnx-qwen3-asr-0.6B-int8-*`).
//!
//! The model is an audio encoder in front of a Qwen3 LLM:
//!
//! - `conv_frontend.onnx` downsamples Whisper mel features by 8, to 12.5 audio tokens per second.
//! - `encoder.onnx` turns them into audio features in the LLM's embedding space. These are the
//!   [`EncoderOutput`] that uncertainty adds noise to.
//! - `decoder.onnx` runs the LLM on a chat prompt whose `<|audio_pad|>` tokens stand for the audio
//!   features. It returns the key/value cache entries of the tokens it read, and the caller keeps
//!   the cache.
//!
//! The LLM reads hotwords as text in its system prompt, and it gives no token times.

mod tokenizer;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorRef};

pub use tokenizer::BpeTokenizer;

use super::onnx::load_session;
use super::{split_words, Asr, Bias, EncoderOutput, Hypothesis, TimedToken};
use crate::features::WhisperMel;
use crate::uncertainty::MonteCarloConfig;
use crate::{Error, Result};

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

/// Mel frames the frontend reads as one chunk. The export bakes it in.
const CHUNK_FRAMES: usize = 100;
const MEL_BINS: usize = 128;

/// The files of one Qwen3-ASR export.
#[derive(Debug, Clone)]
pub struct QwenFiles {
    pub conv_frontend: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    /// Directory with `vocab.json`, `merges.txt`, and `tokenizer_config.json`.
    pub tokenizer: PathBuf,
}

impl QwenFiles {
    /// True when `dir` looks like a Qwen3-ASR export rather than a transducer.
    pub fn is_qwen(dir: &Path) -> bool {
        dir.join("conv_frontend.onnx").is_file()
    }

    /// Finds the files in a model directory, preferring the int8 encoder and decoder.
    pub fn discover(dir: &Path) -> Result<Self> {
        let pick = |stem: &str| -> Result<PathBuf> {
            [format!("{stem}.int8.onnx"), format!("{stem}.onnx")]
                .into_iter()
                .map(|name| dir.join(name))
                .find(|path| path.is_file())
                .ok_or_else(|| Error::Model(format!("no {stem}.onnx in {}", dir.display())))
        };
        Ok(Self {
            conv_frontend: pick("conv_frontend")?,
            encoder: pick("encoder")?,
            decoder: pick("decoder")?,
            tokenizer: dir.join("tokenizer"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct QwenConfig {
    /// Tokens of hotwords in the system prompt. Terms come ranked, so the budget keeps the best ones,
    /// and every prompt token costs a decoder pass in each Monte Carlo sample.
    pub hotword_tokens: usize,
    /// Tokens decoded per second of audio before a decode stops, on top of [`Self::extra_tokens`].
    /// Fast speech runs at about 5 tokens per second.
    pub tokens_per_second: f32,
    pub extra_tokens: usize,
    /// Bytes of key/value cache and logits that one batched decode may use. Larger batches run in parts.
    pub batch_bytes: usize,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            hotword_tokens: 96,
            tokens_per_second: 10.0,
            extra_tokens: 24,
            batch_bytes: 1 << 30,
        }
    }
}

/// Token ids of the chat template.
struct Template {
    im_start: u32,
    im_end: u32,
    audio_start: u32,
    audio_end: u32,
    audio_pad: u32,
    asr_text: u32,
    newline: Vec<u32>,
    user: Vec<u32>,
    assistant: Vec<u32>,
}

impl Template {
    fn new(tokenizer: &BpeTokenizer) -> Result<Self> {
        let added = |name: &str| {
            tokenizer
                .added(name)
                .ok_or_else(|| Error::Model(format!("tokenizer has no {name}")))
        };
        Ok(Self {
            im_start: added("<|im_start|>")?,
            im_end: added("<|im_end|>")?,
            audio_start: added("<|audio_start|>")?,
            audio_end: added("<|audio_end|>")?,
            audio_pad: added("<|audio_pad|>")?,
            asr_text: added("<asr_text>")?,
            newline: tokenizer.encode("\n"),
            user: tokenizer.encode("user\n"),
            assistant: tokenizer.encode("assistant\n"),
        })
    }
}

/// Shapes the decoder's inputs reveal.
struct DecoderShape {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
}

pub struct QwenAsr {
    frontend: Mutex<Session>,
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: BpeTokenizer,
    template: Template,
    mel: WhisperMel,
    shape: DecoderShape,
    pub config: QwenConfig,
}

impl QwenAsr {
    pub fn load(files: &QwenFiles, threads: usize, config: QwenConfig) -> Result<Self> {
        let tokenizer = BpeTokenizer::load(&files.tokenizer)?;
        let template = Template::new(&tokenizer)?;
        let decoder = load_session(&files.decoder, threads)?;
        let layers = decoder
            .inputs()
            .iter()
            .filter(|input| input.name().starts_with("cache_key_"))
            .count();
        let dims: Vec<i64> = decoder
            .inputs()
            .iter()
            .find(|input| input.name() == "cache_key_0")
            .and_then(|input| input.dtype().tensor_shape())
            .map(|shape| shape.iter().copied().collect())
            .unwrap_or_default();
        let (kv_heads, head_dim) = match dims.as_slice() {
            [_, _, heads, dim] if *heads > 0 && *dim > 0 => (*heads as usize, *dim as usize),
            _ => {
                return Err(Error::Model(format!(
                    "unexpected decoder cache shape {dims:?} in {}",
                    files.decoder.display()
                )))
            }
        };
        Ok(Self {
            frontend: Mutex::new(load_session(&files.conv_frontend, threads)?),
            encoder: Mutex::new(load_session(&files.encoder, threads)?),
            decoder: Mutex::new(decoder),
            tokenizer,
            template,
            mel: WhisperMel::new(MEL_BINS),
            shape: DecoderShape {
                layers,
                kv_heads,
                head_dim,
            },
            config,
        })
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// The prompt for `audio_tokens` audio features, with the hotwords in the system turn.
    fn prompt(&self, audio_tokens: usize, phrases: &[String]) -> Vec<i64> {
        let t = &self.template;
        let mut ids = vec![t.im_start];
        ids.extend(
            self.tokenizer
                .encode(&format!("system\n{}", hotword_text(phrases))),
        );
        ids.push(t.im_end);
        ids.extend(&t.newline);
        ids.push(t.im_start);
        ids.extend(&t.user);
        ids.push(t.audio_start);
        ids.extend(std::iter::repeat_n(t.audio_pad, audio_tokens));
        ids.extend([t.audio_end, t.im_end]);
        ids.extend(&t.newline);
        ids.push(t.im_start);
        ids.extend(&t.assistant);
        ids.into_iter().map(i64::from).collect()
    }

    /// Greedy decoding of rows whose audio has the same length, in one batch.
    fn generate(
        &self,
        prompt: &[i64],
        rows: &[&EncoderOutput],
        max_new: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        let batch = rows.len();
        let frames = rows[0].frames;
        let dim = rows[0].dim;
        let context = prompt.len();
        let total = context + max_new;
        let width = self.shape.kv_heads * self.shape.head_dim;
        let mut cache: Vec<Vec<f32>> = (0..2 * self.shape.layers)
            .map(|_| vec![0.0; batch * total * width])
            .collect();
        let audio: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.data[..frames * dim].iter().copied())
            .collect();
        let eos = self.template.im_end;

        let mut generated: Vec<Vec<(u32, f32)>> = vec![Vec::new(); batch];
        let mut done = vec![false; batch];
        let mut input: Vec<i64> = prompt.repeat(batch);
        let mut start = 0;
        let mut length = context;
        loop {
            let picks = self.step(
                &input,
                batch,
                length,
                start,
                &audio,
                (frames, dim),
                &mut cache,
                total,
            )?;
            start += length;
            for (b, (id, log_prob)) in picks.iter().copied().enumerate() {
                if done[b] {
                    continue;
                }
                if id == eos {
                    done[b] = true;
                    continue;
                }
                generated[b].push((id, log_prob));
                if is_degenerate(&generated[b]) {
                    trim_degenerate_tail(&mut generated[b]);
                    done[b] = true;
                }
            }
            if done.iter().all(|d| *d) || start >= total {
                break;
            }
            // Finished rows keep stepping with the end token; their output is ignored.
            input = picks
                .iter()
                .zip(&done)
                .map(|((id, _), finished)| i64::from(if *finished { eos } else { *id }))
                .collect();
            length = 1;
        }
        Ok(generated)
    }

    /// Runs the decoder on `length` new tokens per row at cache position `start`, stores their
    /// cache entries, and returns each row's most likely next token with its log probability.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        input: &[i64],
        batch: usize,
        length: usize,
        start: usize,
        audio: &[f32],
        (frames, dim): (usize, usize),
        cache: &mut [Vec<f32>],
        total: usize,
    ) -> Result<Vec<(u32, f32)>> {
        let shape = &self.shape;
        let width = shape.kv_heads * shape.head_dim;
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            (
                "input_ids".into(),
                Tensor::from_array(([batch, length], input.to_vec()))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "audio_features".into(),
                TensorRef::from_array_view(([batch, frames, dim], audio))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array(([batch, length], vec![1i64; batch * length]))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "cache_position".into(),
                Tensor::from_array((
                    [length],
                    (start as i64..(start + length) as i64).collect::<Vec<_>>(),
                ))
                .map_err(model_error)?
                .into(),
            ),
        ];
        let dims = [batch, total, shape.kv_heads, shape.head_dim];
        for (layer, pair) in cache.chunks(2).enumerate() {
            inputs.push((
                format!("cache_key_{layer}"),
                TensorRef::from_array_view((dims, pair[0].as_slice()))
                    .map_err(model_error)?
                    .into(),
            ));
            inputs.push((
                format!("cache_value_{layer}"),
                TensorRef::from_array_view((dims, pair[1].as_slice()))
                    .map_err(model_error)?
                    .into(),
            ));
        }

        let mut session = self.decoder.lock().expect("decoder lock");
        let outputs = session.run(inputs).map_err(model_error)?;
        let (logits_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let vocab = logits_shape[2] as usize;
        let steps = logits_shape[1] as usize;
        let picks = (0..batch)
            .map(|b| {
                let last = (b * steps + steps - 1) * vocab;
                pick(&logits[last..last + vocab])
            })
            .collect();

        for (layer, pair) in cache.chunks_mut(2).enumerate() {
            for (kind, buffer) in ["key", "value"].iter().zip(pair.iter_mut()) {
                let (delta_shape, delta) = outputs[format!("{kind}_delta_{layer}").as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(model_error)?;
                let rows = (delta_shape[1] as usize).min(total - start);
                for b in 0..batch {
                    let from = b * delta_shape[1] as usize * width;
                    let to = (b * total + start) * width;
                    buffer[to..to + rows * width]
                        .copy_from_slice(&delta[from..from + rows * width]);
                }
            }
        }
        Ok(picks)
    }

    /// Words of the generated text after its `language X<asr_text>` header. Uncertainty compares
    /// decodes word by word, because BPE can split one word differently in two decodes.
    fn to_hypothesis(&self, generated: &[(u32, f32)]) -> Hypothesis {
        let header = generated
            .iter()
            .take(16)
            .position(|(id, _)| *id == self.template.asr_text)
            .filter(|end| {
                let ids: Vec<u32> = generated[..*end].iter().map(|(id, _)| *id).collect();
                self.tokenizer.decode(&ids).starts_with("language ")
            })
            .map_or(0, |end| end + 1);
        let body = &generated[header..];
        let mut bytes = Vec::new();
        let mut starts = Vec::with_capacity(body.len());
        for (id, _) in body {
            starts.push(bytes.len());
            bytes.extend_from_slice(self.tokenizer.bytes(*id));
        }
        let text = String::from_utf8_lossy(&bytes);
        let tokens = split_words(&text)
            .into_iter()
            .map(|range| {
                // The token where the word starts, and the tokens that start inside it.
                let first = starts
                    .partition_point(|s| *s <= range.start)
                    .saturating_sub(1);
                let log_prob = body
                    .iter()
                    .zip(&starts)
                    .filter(|(_, s)| range.contains(s))
                    .map(|((_, p), _)| p)
                    .sum();
                TimedToken {
                    id: body.get(first).map_or(0, |(id, _)| *id),
                    text: text[range].to_owned(),
                    time: None,
                    log_prob,
                }
            })
            .collect();
        Hypothesis { tokens }
    }

    fn max_new_tokens(&self, frames: usize) -> usize {
        (frames as f32 * self.frame_seconds() * self.config.tokens_per_second) as usize
            + self.config.extra_tokens
    }
}

/// Hotwords as the system prompt reads them.
fn hotword_text(phrases: &[String]) -> String {
    phrases.join(", ")
}

/// Most likely token and its log probability. Logits far below the best add nothing measurable to
/// the normalizer, so they skip the exponential.
fn pick(logits: &[f32]) -> (u32, f32) {
    let (best, max) = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .fold(
            (0, f32::NEG_INFINITY),
            |acc, (i, v)| if v > acc.1 { (i, v) } else { acc },
        );
    let sum: f32 = logits
        .iter()
        .map(|v| v - max)
        .filter(|d| *d > -30.0)
        .map(f32::exp)
        .sum();
    (best as u32, -sum.ln())
}

/// Tokens a greedy decode stuck in a loop cycles over: at most this many distinct ids in the last
/// [`LOOP_WINDOW`]. Real speech varies more than that, even repeated lyrics.
const LOOP_DISTINCT: usize = 3;
const LOOP_WINDOW: usize = 64;

fn loop_ids(generated: &[(u32, f32)]) -> Vec<u32> {
    let mut distinct = Vec::new();
    for (id, _) in &generated[generated.len() - LOOP_WINDOW..] {
        if !distinct.contains(id) {
            distinct.push(*id);
        }
    }
    distinct
}

fn is_degenerate(generated: &[(u32, f32)]) -> bool {
    generated.len() >= LOOP_WINDOW && loop_ids(generated).len() <= LOOP_DISTINCT
}

/// Drops the loop: trailing tokens from the loop's ids, looking back no further than the window.
fn trim_degenerate_tail(generated: &mut Vec<(u32, f32)>) {
    let ids = loop_ids(generated);
    let floor = generated.len() - LOOP_WINDOW;
    let mut keep = generated.len();
    while keep > floor && ids.contains(&generated[keep - 1].0) {
        keep -= 1;
    }
    generated.truncate(keep);
}

/// Audio tokens the frontend makes from `frames` mel frames: three stride-2 convolutions on each
/// chunk of [`CHUNK_FRAMES`].
pub fn audio_token_count(frames: usize) -> usize {
    let after_convs = |n: usize| (0..3).fold(n, |x: usize, _| x.div_ceil(2));
    let full = frames / CHUNK_FRAMES;
    let rest = frames % CHUNK_FRAMES;
    full * after_convs(CHUNK_FRAMES) + if rest > 0 { after_convs(rest) } else { 0 }
}

impl Asr for QwenAsr {
    /// Keeps the leading phrases that fit the prompt's hotword budget.
    fn bias(&self, phrases: &[String]) -> Bias {
        let mut kept: Vec<String> = Vec::new();
        for phrase in phrases {
            kept.push(phrase.clone());
            if self.tokenizer.encode(&hotword_text(&kept)).len() > self.config.hotword_tokens {
                kept.pop();
                break;
            }
        }
        Bias {
            phrases: kept,
            graph: None,
        }
    }

    fn encode(&self, samples: &[f32]) -> Result<EncoderOutput> {
        let features = self.mel.compute(samples);
        let empty = EncoderOutput {
            frames: 0,
            dim: 0,
            data: Vec::new(),
        };
        if features.frames < 2 {
            return Ok(empty);
        }
        let input = Tensor::from_array(([1, features.frames, features.dim], features.data))
            .map_err(model_error)?;
        let (conv_shape, conv) = {
            let mut session = self.frontend.lock().expect("frontend lock");
            let outputs = session
                .run(ort::inputs!["input_features" => input])
                .map_err(model_error)?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(model_error)?;
            (shape.iter().copied().collect::<Vec<i64>>(), data.to_vec())
        };
        let tokens = conv_shape[1] as usize;
        let valid = audio_token_count(features.frames).min(tokens);
        let mask: Vec<bool> = (0..tokens).map(|i| i < valid).collect();
        let conv = Tensor::from_array((conv_shape, conv)).map_err(model_error)?;
        let mask = Tensor::from_array(([1, tokens], mask)).map_err(model_error)?;
        let mut session = self.encoder.lock().expect("encoder lock");
        let outputs = session
            .run(ort::inputs!["input_features" => conv, "feature_attention_mask" => mask])
            .map_err(model_error)?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let dim = shape[2] as usize;
        // The encoder zeroes padded positions. Silence at the end is not speech either.
        let speech = (0..valid)
            .rev()
            .find(|t| data[t * dim..(t + 1) * dim].iter().any(|v| v.abs() > 1e-6))
            .map_or(0, |t| t + 1);
        if speech == 0 {
            return Ok(empty);
        }
        Ok(EncoderOutput {
            frames: speech,
            dim,
            data: data[..speech * dim].to_vec(),
        })
    }

    fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
        let mut results = vec![Hypothesis::default(); encoders.len()];
        // Rows batch only with rows of the same audio length, since they share the prompt.
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, encoder) in encoders.iter().enumerate() {
            if encoder.frames > 0 {
                groups.entry(encoder.frames).or_default().push(index);
            }
        }
        let vocab = self.tokenizer.vocab_size();
        for (frames, indices) in groups {
            let prompt = self.prompt(frames, &bias.phrases);
            let max_new = self.max_new_tokens(frames);
            let cache_bytes = 2
                * self.shape.layers
                * (prompt.len() + max_new)
                * self.shape.kv_heads
                * self.shape.head_dim
                * 4;
            // The first pass returns logits for every prompt position.
            let row_bytes = cache_bytes + prompt.len() * vocab * 4;
            let rows_per_call = (self.config.batch_bytes / row_bytes).max(1);
            for part in indices.chunks(rows_per_call) {
                let rows: Vec<&EncoderOutput> = part.iter().map(|i| &encoders[*i]).collect();
                for (index, generated) in part.iter().zip(self.generate(&prompt, &rows, max_new)?) {
                    results[*index] = self.to_hypothesis(&generated);
                }
            }
        }
        Ok(results)
    }

    fn frame_seconds(&self) -> f32 {
        0.08
    }

    /// Every noisy decode reruns the whole LLM, prompt included, so there are fewer of them than for a
    /// transducer. Audio features also need less noise than transducer encoder output to move the
    /// decode where the model is unsure.
    fn monte_carlo(&self) -> MonteCarloConfig {
        MonteCarloConfig {
            samples: 4,
            first_samples: 2,
            relative_sigma: 0.5,
            ..MonteCarloConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_audio_tokens_per_chunk() {
        assert_eq!(audio_token_count(100), 13);
        assert_eq!(audio_token_count(508), 66);
        assert_eq!(audio_token_count(672), 87);
        assert_eq!(audio_token_count(1), 1);
    }

    #[test]
    fn trims_a_repetition_loop() {
        let mut generated: Vec<(u32, f32)> = (0..10).map(|i| (i, 0.0)).collect();
        generated.extend((0..70).map(|i| (100 + i % 2, 0.0)));
        assert!(is_degenerate(&generated));
        trim_degenerate_tail(&mut generated);
        // The window reaches back 64 tokens, so 6 tokens of the loop stay.
        assert_eq!(generated.len(), 16);
        assert!(!is_degenerate(
            &(0..80).map(|i| (i, 0.0)).collect::<Vec<_>>()
        ));
    }

    #[test]
    fn picks_the_best_logit_with_its_probability() {
        let (id, log_prob) = pick(&[0.0, 2.0f32.ln(), 0.0, f32::NAN]);
        assert_eq!(id, 1);
        assert!((log_prob - 0.5f32.ln()).abs() < 1e-5);
    }
}
