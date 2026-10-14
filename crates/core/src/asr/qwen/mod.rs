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
use super::{
    is_cjk, is_punctuation, split_words, Asr, Bias, EncoderOutput, Hypothesis, TimedToken,
};
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
    /// Tokens that end the answer: the end of the turn or of the text.
    stops: [u32; 2],
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
        let im_end = added("<|im_end|>")?;
        Ok(Self {
            im_start: added("<|im_start|>")?,
            im_end,
            stops: [im_end, added("<|endoftext|>")?],
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
            let picks: Vec<(u32, f32)> = self
                .step(
                    &input,
                    batch,
                    length,
                    start,
                    &audio,
                    (frames, dim),
                    &mut cache,
                    total,
                    length - 1,
                )?
                .into_iter()
                .map(|row| row[0])
                .collect();
            start += length;
            for (b, (id, log_prob)) in picks.iter().copied().enumerate() {
                if done[b] {
                    continue;
                }
                // A stop stays in the output, so a later pass knows the decode ended on one;
                // the transcript leaves it out with the other added tokens.
                generated[b].push((id, log_prob));
                if self.template.stops.contains(&id) {
                    done[b] = true;
                    continue;
                }
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

    /// Decodes noisy rows whose clean decode generated `reference`, in far fewer calls than
    /// [`Self::generate`]. One call reads the prompt and the reference in every row, which gives
    /// each row's own pick after every reference token. Where a row's pick differs from the
    /// reference, that row decodes on its own until the next word begins, and the sample is the
    /// reference with that word replaced.
    ///
    /// A full greedy decode of the row would also condition everything after a replaced word on
    /// it, so this is an approximation, but it keeps whole words for the readings the uncertainty
    /// analysis compares. The int8 decoder quantizes the activations of a whole call together, so picks
    /// in the scoring call can differ slightly from one-token calls, about as much as other rows
    /// in a batch change them.
    fn generate_noisy(
        &self,
        prompt: &[i64],
        rows: &[&EncoderOutput],
        reference: &[u32],
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        let batch = rows.len();
        let (frames, dim) = (rows[0].frames, rows[0].dim);
        let context = prompt.len();
        let stops = |id: u32| self.template.stops.contains(&id);
        // Past the reference, rows are checked for words it lacks only when the clean decode
        // stopped there, not when it ran out of tokens or a loop was cut off.
        let stopped = reference.last().is_some_and(|id| stops(*id));
        let reference = &reference[..reference.len() - usize::from(stopped)];
        let length = reference.len();
        let total = context + length + LOCAL_TOKENS + 1;
        let width = self.shape.kv_heads * self.shape.head_dim;
        let mut cache: Vec<Vec<f32>> = (0..2 * self.shape.layers)
            .map(|_| vec![0.0; batch * total * width])
            .collect();
        let audio: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.data[..frames * dim].iter().copied())
            .collect();
        let mut input: Vec<i64> = prompt.to_vec();
        input.extend(reference.iter().map(|id| i64::from(*id)));
        // After the prompt and after each reference token: what each row picks next.
        let scored = self.step(
            &input.repeat(batch),
            batch,
            input.len(),
            0,
            &audio,
            (frames, dim),
            &mut cache,
            total,
            context - 1,
        )?;
        // The noisy rows keep the reference's `language X<asr_text>` header, if it has one; any
        // other text before the marker is transcript, as `transcript` keeps it.
        let header = reference
            .iter()
            .take(16)
            .position(|id| *id == self.template.asr_text)
            .filter(|end| {
                let ids: Vec<u32> = reference[..*end]
                    .iter()
                    .copied()
                    .filter(|id| !self.tokenizer.is_added(*id))
                    .collect();
                is_language_header(&self.tokenizer.decode(&ids))
            })
            .map_or(0, |end| end + 1);
        let pick_at = |b: usize, i: usize| scored[b][i];
        let differs = |b: usize, i: usize| {
            let pick = pick_at(b, i).0;
            if i < header {
                false
            } else if i == length {
                stopped && !stops(pick)
            } else {
                pick != reference[i]
            }
        };

        let mut stretches: Vec<Vec<Stretch>> = vec![Vec::new(); batch];
        let mut resume = vec![0usize; batch];
        let slab = total * width;
        let row_audio = frames * dim;
        for i in 0..=length {
            let group: Vec<usize> = (0..batch)
                .filter(|b| i >= resume[*b] && differs(*b, i))
                .collect();
            if group.is_empty() {
                continue;
            }
            let mut finished: Vec<(usize, Stretch)> = Vec::new();
            let mut open: Vec<(usize, Vec<(u32, f32)>)> = Vec::new();
            for b in group {
                let mut row = vec![pick_at(b, i)];
                match stretch_end(
                    &self.tokenizer,
                    &self.template.stops,
                    &mut row,
                    reference,
                    i,
                ) {
                    Some(end) => finished.push((
                        b,
                        Stretch {
                            start: i,
                            tokens: row,
                            end,
                        },
                    )),
                    None => open.push((b, row)),
                }
            }
            if !open.is_empty() {
                // The cache of each open row holds the prompt and the reference before `i`.
                let mut local_cache: Vec<Vec<f32>> = cache
                    .iter()
                    .map(|buffer| {
                        open.iter()
                            .flat_map(|(b, _)| buffer[b * slab..(b + 1) * slab].iter().copied())
                            .collect()
                    })
                    .collect();
                let local_audio: Vec<f32> = open
                    .iter()
                    .flat_map(|(b, _)| audio[b * row_audio..(b + 1) * row_audio].iter().copied())
                    .collect();
                let mut ends: Vec<Option<Option<usize>>> = vec![None; open.len()];
                let mut position = context + i;
                while ends.iter().any(Option::is_none) && position + 1 < total {
                    let input: Vec<i64> = open
                        .iter()
                        .zip(&ends)
                        .map(|((_, row), end)| {
                            i64::from(match (end, row.last()) {
                                (None, Some((id, _))) => *id,
                                _ => self.template.im_end,
                            })
                        })
                        .collect();
                    let picks = self.step(
                        &input,
                        open.len(),
                        1,
                        position,
                        &local_audio,
                        (frames, dim),
                        &mut local_cache,
                        total,
                        0,
                    )?;
                    position += 1;
                    for (o, (_, row)) in open.iter_mut().enumerate() {
                        if ends[o].is_none() {
                            row.push(picks[o][0]);
                            ends[o] = stretch_end(
                                &self.tokenizer,
                                &self.template.stops,
                                row,
                                reference,
                                i,
                            );
                        }
                    }
                }
                for ((b, row), end) in open.into_iter().zip(ends) {
                    let end = end.unwrap_or(Some(next_word(&self.tokenizer, reference, i)));
                    finished.push((
                        b,
                        Stretch {
                            start: i,
                            tokens: row,
                            end,
                        },
                    ));
                }
            }
            for (b, stretch) in finished {
                resume[b] = stretch.end.unwrap_or(usize::MAX);
                stretches[b].push(stretch);
            }
        }

        Ok((0..batch)
            .map(|b| splice(reference, &scored[b], &stretches[b]))
            .collect())
    }

    /// Runs the decoder on `length` new tokens per row at cache position `start`, stores their
    /// cache entries, and returns each row's most likely next token with its log probability after
    /// each new token from `keep_from` on.
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
        keep_from: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
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
                (keep_from..steps)
                    .map(|p| {
                        let at = (b * steps + p) * vocab;
                        pick(&logits[at..at + vocab])
                    })
                    .collect()
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

    /// Words of the generated text after its `language X<asr_text>` header, without added tokens.
    /// Uncertainty compares decodes word by word, because BPE can split one word differently in
    /// two decodes.
    fn to_hypothesis(&self, generated: &[(u32, f32)]) -> Hypothesis {
        let body = transcript(&self.tokenizer, self.template.asr_text, generated);
        let mut bytes = Vec::new();
        let mut starts = Vec::with_capacity(body.len());
        for (id, _) in &body {
            starts.push(bytes.len());
            bytes.extend_from_slice(self.tokenizer.bytes(*id));
        }
        let (text, offsets) = decode_lossy(&bytes);
        let starts: Vec<usize> = starts.iter().map(|s| offsets[*s]).collect();
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
        Hypothesis {
            tokens,
            ids: generated.iter().map(|(id, _)| *id).collect(),
        }
    }

    /// Decodes rows in batches of the same audio length, which share a prompt: greedily, or
    /// starting from the ids of a reference decode.
    fn decode_rows(
        &self,
        encoders: &[EncoderOutput],
        bias: &Bias,
        reference: Option<&[u32]>,
    ) -> Result<Vec<Hypothesis>> {
        let mut results = vec![Hypothesis::default(); encoders.len()];
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
            // The first call returns logits for every position it reads.
            let first_call = prompt.len() + reference.map_or(0, <[u32]>::len);
            let row_bytes = cache_bytes + first_call * vocab * 4;
            let rows_per_call = (self.config.batch_bytes / row_bytes).max(1);
            for part in indices.chunks(rows_per_call) {
                let rows: Vec<&EncoderOutput> = part.iter().map(|i| &encoders[*i]).collect();
                let generated = match reference {
                    Some(ids) if ids.iter().any(|id| !self.template.stops.contains(id)) => {
                        self.generate_noisy(&prompt, &rows, ids)?
                    }
                    _ => self.generate(&prompt, &rows, max_new)?,
                };
                for (index, generated) in part.iter().zip(generated) {
                    results[*index] = self.to_hypothesis(&generated);
                }
            }
        }
        Ok(results)
    }

    fn max_new_tokens(&self, frames: usize) -> usize {
        (frames as f32 * self.frame_seconds() * self.config.tokens_per_second) as usize
            + self.config.extra_tokens
    }
}

/// The generated tokens after the `language X<asr_text>` header, without added tokens. Chat and
/// audio markers are never text, and the model sometimes opens its answer with `<|im_start|>`
/// before the header.
fn transcript(
    tokenizer: &BpeTokenizer,
    asr_text: u32,
    generated: &[(u32, f32)],
) -> Vec<(u32, f32)> {
    let tokens: Vec<(u32, f32)> = generated
        .iter()
        .copied()
        .filter(|(id, _)| *id == asr_text || !tokenizer.is_added(*id))
        .collect();
    let header = tokens
        .iter()
        .take(16)
        .position(|(id, _)| *id == asr_text)
        .filter(|end| {
            let ids: Vec<u32> = tokens[..*end].iter().map(|(id, _)| *id).collect();
            is_language_header(&tokenizer.decode(&ids))
        })
        .map_or(0, |end| end + 1);
    tokens[header..]
        .iter()
        .copied()
        .filter(|(id, _)| *id != asr_text)
        .collect()
}

/// True for the metadata Qwen3-ASR writes before `<asr_text>`: a `language English` line, or
/// `language None` for audio without speech. Another line can come before it, such as the role of
/// a turn the model started on its own (`assistant`).
fn is_language_header(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        line.get(..8)
            .is_some_and(|word| word.eq_ignore_ascii_case("language"))
            && line[8..].starts_with(char::is_whitespace)
    })
}

/// Decodes UTF-8 like [`String::from_utf8_lossy`], with the text offset of every byte offset, so
/// token starts still point at their text when a malformed sequence turns into U+FFFD.
fn decode_lossy(bytes: &[u8]) -> (String, Vec<usize>) {
    let mut text = String::with_capacity(bytes.len());
    let mut offsets = Vec::with_capacity(bytes.len() + 1);
    for chunk in bytes.utf8_chunks() {
        offsets.extend((0..chunk.valid().len()).map(|i| text.len() + i));
        text.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            offsets.extend(std::iter::repeat_n(text.len(), chunk.invalid().len()));
            text.push(char::REPLACEMENT_CHARACTER);
        }
    }
    offsets.push(text.len());
    (text, offsets)
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

/// Where a stretch a noisy row decodes on its own, from reference token `start`, takes the
/// reference back: `Some(Some(j))` to go on from reference token `j` (`start` itself when the
/// stretch only adds punctuation), `Some(None)` once the row stopped, or `None` while the
/// stretch goes on. It ends at the end of its first word,
/// whose next token it drops, or earlier where it meets the reference again: two tokens that
/// match in a row, or one matching token that starts a word, as when the row skips a comma.
fn stretch_end(
    tokenizer: &BpeTokenizer,
    stops: &[u32],
    row: &mut Vec<(u32, f32)>,
    reference: &[u32],
    start: usize,
) -> Option<Option<usize>> {
    let (last, _) = *row.last()?;
    if stops.contains(&last) {
        row.pop();
        return Some(None);
    }
    // Back on the reference: two tokens in a row match it, or one whole word does at most one
    // reference word on, as when the row skips a comma or reads two characters as one word. One
    // common word such as ` the` alone can match further on by chance.
    let following_word = next_word(tokenizer, reference, next_word(tokenizer, reference, start));
    let window = (start + 1)..reference.len().min(start + REJOIN_WINDOW + 1);
    let matched = window.into_iter().find(|j| {
        reference[*j] == last
            && ((*j <= following_word
                && starts_word(
                    tokenizer,
                    Some(reference[*j - 1]),
                    last,
                    &reference[*j + 1..],
                )
                && tokenizer.bytes(last).len() >= 2)
                || (row.len() >= 2 && reference[*j - 1] == row[row.len() - 2].0))
    });
    if let Some(j) = matched {
        return Some(Some(j + 1));
    }
    let whole = |tokens: &[(u32, f32)]| {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|(id, _)| tokenizer.bytes(*id).iter().copied())
            .collect();
        std::str::from_utf8(&bytes).is_ok()
    };
    let text = |id: u32| String::from_utf8_lossy(tokenizer.bytes(id)).into_owned();
    let punctuation = |id: u32| {
        let text = text(id);
        !text.trim().is_empty() && text.trim().chars().all(is_punctuation)
    };
    // Punctuation where the reference has a word: the row likely puts it before that word.
    if row.len() == 1 && punctuation(last) && reference.get(start).is_some_and(|r| !punctuation(*r))
    {
        return Some(Some(start));
    }
    // The word ended when the next one begins: the row's word stands for the reference's. A
    // character split over tokens shows whether it begins a word only once its last byte is in.
    let ids: Vec<u32> = row.iter().map(|(id, _)| *id).collect();
    let next = (ids.len().saturating_sub(4).max(1)..ids.len()).find(|k| {
        starts_word(tokenizer, Some(ids[*k - 1]), ids[*k], &ids[*k + 1..]) && whole(&row[..*k])
    });
    if let Some(k) = next {
        row.truncate(k);
        return Some(Some(next_word(tokenizer, reference, start)));
    }
    // A word too long to finish here: it stands for the reference's word all the same.
    (row.len() >= LOCAL_TOKENS).then(|| Some(next_word(tokenizer, reference, start)))
}

/// The first reference token after `start` that begins a word, or the reference's end.
fn next_word(tokenizer: &BpeTokenizer, reference: &[u32], start: usize) -> usize {
    ((start + 1)..reference.len())
        .find(|j| {
            starts_word(
                tokenizer,
                Some(reference[*j - 1]),
                reference[*j],
                &reference[*j + 1..],
            )
        })
        .unwrap_or(reference.len())
}

/// True when a token begins a word: after a space, or with punctuation or a CJK character,
/// which are words of their own, unless it is an apostrophe or hyphen going on from the word in
/// the `previous` token. The `following` tokens complete a character the token only begins.
fn starts_word(
    tokenizer: &BpeTokenizer,
    previous: Option<u32>,
    id: u32,
    following: &[u32],
) -> bool {
    let bytes = tokenizer.bytes(id);
    if bytes.first() == Some(&b' ') {
        return true;
    }
    let mut head = bytes.to_vec();
    for id in following.iter().take(3) {
        head.extend_from_slice(tokenizer.bytes(*id));
    }
    let first = valid_prefix(&head).chars().next();
    // An apostrophe or hyphen right after a letter joins the word, as in `John's`, the way
    // `split_words` reads it.
    let joins = matches!(first, Some('\'' | '’' | '-'))
        && previous.is_some_and(|p| {
            valid_prefix(tokenizer.bytes(p))
                .chars()
                .next_back()
                .is_some_and(|c| !c.is_whitespace() && !is_punctuation(c) && !is_cjk(c))
        });
    !joins && first.is_some_and(|c| is_punctuation(c) || is_cjk(c))
}

/// The bytes up to the first that isn't valid UTF-8, as text.
fn valid_prefix(bytes: &[u8]) -> &str {
    match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => std::str::from_utf8(&bytes[..error.valid_up_to()]).unwrap_or(""),
    }
}

/// Tokens a noisy row decodes on its own from reference token `start`, and the reference token it
/// goes on from after them (`None`: the row stopped).
#[derive(Debug, Clone)]
struct Stretch {
    start: usize,
    tokens: Vec<(u32, f32)>,
    end: Option<usize>,
}

/// A noisy row's sample: the reference, with the row's own log probabilities, and its stretches
/// in place of the reference tokens they cover.
fn splice(reference: &[u32], scored: &[(u32, f32)], stretches: &[Stretch]) -> Vec<(u32, f32)> {
    let own = |from: usize, to: usize| {
        reference[from..to]
            .iter()
            .zip(&scored[from..to])
            .map(|(id, (_, log_prob))| (*id, *log_prob))
    };
    let mut row = Vec::with_capacity(reference.len());
    let mut at = 0;
    for stretch in stretches {
        row.extend(own(at, stretch.start));
        row.extend(stretch.tokens.iter().copied());
        match stretch.end {
            Some(end) => at = end,
            None => return row,
        }
    }
    row.extend(own(at.min(reference.len()), reference.len()));
    row
}

/// Tokens a noisy row decodes on its own where it leaves the reference, at most, and how far ahead
/// in the reference it may come back. Most disagreements are one word.
const LOCAL_TOKENS: usize = 8;
const REJOIN_WINDOW: usize = 12;

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
        self.decode_rows(encoders, bias, None)
    }

    /// Scores the reference under every noisy copy in one call, and decodes on only from where a
    /// copy's pick differs (see [`QwenAsr::generate_noisy`]).
    fn decode_noisy(
        &self,
        encoders: &[EncoderOutput],
        bias: &Bias,
        reference: &Hypothesis,
    ) -> Result<Vec<Hypothesis>> {
        self.decode_rows(encoders, bias, Some(&reference.ids))
    }

    fn frame_seconds(&self) -> f32 {
        0.08
    }

    /// Every noisy decode reads the whole prompt and transcript again, so there are fewer of them
    /// than for a transducer. Audio features also need less noise than transducer encoder output to move the
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

    #[test]
    fn parses_the_transcript_past_chat_tokens() {
        let tokenizer = tokenizer::tests::tokenizer();
        let (im_start, asr_text) = (301, 302);
        let text = |generated: &[u32]| {
            let generated: Vec<(u32, f32)> = generated.iter().map(|id| (*id, 0.0)).collect();
            let ids: Vec<u32> = transcript(&tokenizer, asr_text, &generated)
                .iter()
                .map(|(id, _)| *id)
                .collect();
            tokenizer.decode(&ids)
        };
        let encode = |text: &str| tokenizer.encode(text);
        // One token per byte here, so the header is short enough for the 16-token window.
        let header = encode("language De");
        let body = encode("ab ab");
        let with = |parts: &[&[u32]]| parts.concat();

        assert_eq!(text(&with(&[&header, &[asr_text], &body])), "ab ab");
        // The model opened with a new turn before the header, with or without a role line (a
        // short stand-in for `assistant` here).
        assert_eq!(
            text(&with(&[&[im_start], &header, &[asr_text], &body])),
            "ab ab"
        );
        assert_eq!(
            text(&with(&[
                &[im_start],
                &encode("ab\n"),
                &header,
                &[asr_text],
                &body
            ])),
            "ab ab"
        );
        // A marker inside the text.
        assert_eq!(
            text(&with(&[&header, &[asr_text], &body, &[im_start]])),
            "ab ab"
        );
        // No header: everything is text.
        assert_eq!(text(&body), "ab ab");
        // Silence.
        assert_eq!(text(&with(&[&encode("language None"), &[asr_text]])), "");
    }

    #[test]
    fn ends_a_noisy_stretch_where_it_rejoins_or_its_word_ends() {
        let t = tokenizer::tests::tokenizer();
        let stops = [300];
        let row = |ids: &[u32]| ids.iter().map(|id| (*id, 0.0)).collect::<Vec<(u32, f32)>>();
        let ids = |text: &str| t.encode(text);
        let (space, comma) = (ids(" ")[0], ids(",")[0]);
        let reference = ids("ab ab ab");
        assert_eq!(reference, vec![256, 257, 257]);

        // Inside a word, and a token that may begin a longer word, both go on.
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&ids("x")), &reference, 0),
            None
        );
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&[258]), &reference, 1),
            None
        );
        // A word for a word, once the next word begins.
        let mut swapped = row(&[258, space]);
        assert_eq!(
            stretch_end(&t, &stops, &mut swapped, &reference, 1),
            Some(Some(2))
        );
        assert_eq!(swapped, row(&[258]));
        // Its word ended: the next word's first token goes, and the reference's next word follows.
        let mut ended = row(&[ids("x")[0], ids("y")[0], space]);
        assert_eq!(
            stretch_end(&t, &stops, &mut ended, &reference, 0),
            Some(Some(1))
        );
        assert_eq!(ended, row(&ids("xy")));
        // An apostrophe after a letter goes on with the word, as in `John's`.
        let apostrophe = ids("'")[0];
        assert_eq!(
            stretch_end(
                &t,
                &stops,
                &mut row(&[ids("x")[0], apostrophe]),
                &reference,
                0
            ),
            None
        );
        // The row stopped.
        let mut stopped = row(&[ids("x")[0], 300]);
        assert_eq!(
            stretch_end(&t, &stops, &mut stopped, &reference, 0),
            Some(None)
        );
        assert_eq!(stopped, row(&ids("x")));
        // Out of tokens: the long word stands for one reference word, not for as many tokens.
        let long = row(&ids("xxxxxxxx"));
        assert_eq!(
            stretch_end(&t, &stops, &mut long.clone(), &reference, 0),
            Some(Some(1))
        );
        // A comma before a word comes in front of it.
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&[comma]), &reference, 1),
            Some(Some(1))
        );
        // A character split over tokens begins a word once all of it is in.
        let cjk = ids("中");
        assert_eq!(cjk.len(), 3);
        let mut split = row(&[ids("x")[0], cjk[0]]);
        assert_eq!(stretch_end(&t, &stops, &mut split, &reference, 0), None);
        let mut split = row(&[ids("x")[0], cjk[0], cjk[1], cjk[2]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut split, &reference, 0),
            Some(Some(1))
        );
        assert_eq!(split, row(&ids("x")));
        // One word can rejoin the reference one word on, but further on it may match by chance.
        let near = [256, 258, 257];
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&[257]), &near, 0),
            Some(Some(3))
        );
        let far = [256, 258, 258, 257];
        assert_eq!(stretch_end(&t, &stops, &mut row(&[257]), &far, 0), None);
        // It skipped a comma and is back on the reference.
        let reference = ids("ab, ab c");
        assert_eq!(reference, vec![256, comma, 257, 258]);
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&[257]), &reference, 1),
            Some(Some(3))
        );
    }

    #[test]
    fn splices_stretches_into_the_reference() {
        let reference = [1, 2, 3, 4, 5];
        let scored: Vec<(u32, f32)> = (0..6).map(|i| (0, -(i as f32))).collect();
        let stretch = |start, tokens: &[(u32, f32)], end| Stretch {
            start,
            tokens: tokens.to_vec(),
            end,
        };
        assert_eq!(
            splice(&reference, &scored, &[stretch(1, &[(9, -9.0)], Some(3))]),
            vec![(1, 0.0), (9, -9.0), (4, -3.0), (5, -4.0)]
        );
        // A row that stopped early, and one that went on past the reference.
        assert_eq!(
            splice(&reference, &scored, &[stretch(2, &[], None)]),
            vec![(1, 0.0), (2, -1.0)]
        );
        assert_eq!(
            splice(&reference, &scored, &[stretch(5, &[(7, -7.0)], Some(5))]).len(),
            6
        );
    }

    #[test]
    fn recognizes_the_language_header() {
        assert!(is_language_header("language English"));
        assert!(is_language_header("Language None"));
        assert!(is_language_header("language Chinese\n"));
        assert!(is_language_header("assistant\nlanguage German"));
        assert!(!is_language_header("languages are fun"));
        assert!(!is_language_header("a language German"));
        assert!(!is_language_header("lang"));
        assert!(!is_language_header("言語は日本語"));
    }

    #[test]
    fn maps_byte_offsets_through_lossy_decoding() {
        // A cut-off "中" (E4 B8 AD), then " hi", then a whole "中".
        let bytes = [0xE4, 0xB8, b' ', b'h', b'i', 0xE4, 0xB8, 0xAD];
        let (text, offsets) = decode_lossy(&bytes);
        assert_eq!(text, String::from_utf8_lossy(&bytes));
        assert_eq!(text, "\u{FFFD} hi中");
        assert_eq!(offsets.len(), bytes.len() + 1);
        assert_eq!(&text[offsets[2]..offsets[5]], " hi");
        assert_eq!(&text[offsets[5]..offsets[8]], "中");
        assert_eq!(offsets[1], 0);
    }
}
