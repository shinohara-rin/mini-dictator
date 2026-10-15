//! Qwen3-ASR: an audio encoder in front of a Qwen3 LLM.
//!
//! The encoder turns Whisper mel features into audio features in the LLM's embedding space, 12.5
//! per second. These are the [`EncoderOutput`] that uncertainty adds noise to. The LLM runs on a chat
//! prompt whose `<|audio_pad|>` tokens stand for the audio features, reads hotwords as text in its
//! system prompt, and gives no token times.
//!
//! Two runtimes run the networks, behind [`QwenModel`]:
//!
//! - [`ort::OrtQwen`] runs the int8 ONNX export that sherpa-onnx ships, on the CPU.
//! - [`mlx::MlxQwen`] runs an MLX checkpoint such as `mlx-community/Qwen3-ASR-0.6B-4bit`, on the GPU
//!   of Apple silicon.
//!
//! [`QwenAsr`] holds what they share: the tokenizer and prompt, greedy decoding, the noisy check,
//! and reading the transcript out of the answer.

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;
#[cfg(feature = "onnx")]
pub mod ort;
mod tokenizer;

use std::collections::BTreeMap;
use std::path::Path;

pub use tokenizer::BpeTokenizer;

use super::{
    is_cjk, is_punctuation, split_words, Asr, Bias, EncoderOutput, Hypothesis, TimedToken,
};
use crate::features::{Features, WhisperMel};
use crate::uncertainty::MonteCarloConfig;
use crate::{Error, Result};

/// Mel frames the encoder reads as one chunk.
const CHUNK_FRAMES: usize = 100;
const MEL_BINS: usize = 128;

/// A token the decoder picked, with its log probability.
pub type Pick = (u32, f32);

/// The networks of Qwen3-ASR in one runtime: the audio encoder, and the LLM decoder with its
/// key/value cache.
pub trait QwenModel: Send + Sync {
    /// The key/value cache of a batch of rows.
    type Cache;

    /// Audio features for Whisper mel features of 128 bins, one row per audio token
    /// ([`audio_token_count`]).
    fn encode(&self, features: Features) -> Result<EncoderOutput>;

    /// An empty cache for `batch` rows of at most `total` tokens.
    fn cache(&self, batch: usize, total: usize) -> Self::Cache;

    /// A cache of the given rows of `cache`, holding its first `length` positions.
    fn fork(&self, cache: &Self::Cache, rows: &[usize], length: usize) -> Self::Cache;

    /// Bytes the cache of one row takes for `tokens` tokens.
    fn cache_bytes(&self, tokens: usize) -> usize;

    /// Runs the decoder on `length` new tokens per row at cache position `start`, stores their
    /// cache entries, and returns each row's most likely next token with its log probability after
    /// each new token from `keep_from` on. `input` holds the rows one after another; where it has
    /// `<|audio_pad|>` tokens, the decoder reads each row's `audio` features of `(frames, dim)`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        input: &[i64],
        length: usize,
        start: usize,
        audio: &[f32],
        shape: (usize, usize),
        cache: &mut Self::Cache,
        keep_from: usize,
    ) -> Result<Vec<Vec<Pick>>>;
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

pub struct QwenAsr<M> {
    model: M,
    tokenizer: BpeTokenizer,
    template: Template,
    mel: WhisperMel,
    pub config: QwenConfig,
}

#[cfg(feature = "onnx")]
impl QwenAsr<ort::OrtQwen> {
    pub fn load_onnx(files: &ort::QwenFiles, threads: usize, config: QwenConfig) -> Result<Self> {
        Self::new(
            ort::OrtQwen::load(files, threads)?,
            &files.tokenizer,
            config,
        )
    }
}

impl<M: QwenModel> QwenAsr<M> {
    /// Qwen3-ASR on `model`, with the tokenizer files in `tokenizer`.
    pub fn new(model: M, tokenizer: &Path, config: QwenConfig) -> Result<Self> {
        let tokenizer = BpeTokenizer::load(tokenizer)?;
        let template = Template::new(&tokenizer)?;
        Ok(Self {
            model,
            tokenizer,
            template,
            mel: WhisperMel::new(MEL_BINS),
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
        let mut cache = self.model.cache(batch, total);
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
                .model
                .step(
                    &input,
                    length,
                    start,
                    &audio,
                    (frames, dim),
                    &mut cache,
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
        let mut cache = self.model.cache(batch, total);
        let audio: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.data[..frames * dim].iter().copied())
            .collect();
        let mut input: Vec<i64> = prompt.to_vec();
        input.extend(reference.iter().map(|id| i64::from(*id)));
        // After the prompt and after each reference token: what each row picks next.
        let scored = self.model.step(
            &input.repeat(batch),
            input.len(),
            0,
            &audio,
            (frames, dim),
            &mut cache,
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
                let rows: Vec<usize> = open.iter().map(|(b, _)| *b).collect();
                let mut local_cache = self.model.fork(&cache, &rows, context + i);
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
                    let picks = self.model.step(
                        &input,
                        1,
                        position,
                        &local_audio,
                        (frames, dim),
                        &mut local_cache,
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
            let cache_bytes = self.model.cache_bytes(prompt.len() + max_new);
            // The first call returns logits for every position it reads.
            let first_call = prompt.len() + reference.map_or(0, <[u32]>::len);
            let row_bytes = cache_bytes + first_call * vocab * 4;
            let rows_per_call = (self.config.batch_bytes / row_bytes).max(1);
            for part in indices.chunks(rows_per_call) {
                let rows: Vec<&EncoderOutput> = part.iter().map(|i| &encoders[*i]).collect();
                let generated = match reference {
                    Some(ids) => self.generate_noisy(&prompt, &rows, ids)?,
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
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn pick(logits: &[f32]) -> Pick {
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
    // Past the reference's end, everything up to the stop is speech it may have missed.
    if start >= reference.len() {
        return (row.len() >= LOCAL_TOKENS).then_some(Some(reference.len()));
    }
    // The start of the row as whole characters: a stretch can begin inside a character the
    // reference began before `start`, so its bytes come first.
    let continues = |id: u32| {
        tokenizer
            .bytes(id)
            .first()
            .is_some_and(|b| b & 0xC0 == 0x80)
    };
    let lead = (0..start.min(reference.len()))
        .rev()
        .find(|j| !continues(reference[*j]))
        .unwrap_or(start.min(reference.len()));
    let whole = |tokens: &[(u32, f32)]| {
        let bytes: Vec<u8> = reference[lead..start.min(reference.len())]
            .iter()
            .copied()
            .chain(tokens.iter().map(|(id, _)| *id))
            .flat_map(|id| tokenizer.bytes(id).iter().copied())
            .collect();
        std::str::from_utf8(&bytes).is_ok()
    };
    let ids: Vec<u32> = row.iter().map(|(id, _)| *id).collect();
    // Only punctuation, which can take several tokens, even for one character.
    let punctuation = |ids: &[u32]| {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|id| tokenizer.bytes(*id).iter().copied())
            .collect();
        std::str::from_utf8(&bytes)
            .is_ok_and(|text| !text.trim().is_empty() && text.trim().chars().all(is_punctuation))
    };
    // The reference's word ends after token `end - 1`.
    let word_ends = |end: usize| {
        end == reference.len()
            || starts_word(
                tokenizer,
                &reference[..end],
                reference[end],
                &reference[end + 1..],
            )
    };
    let room = ids.len() < LOCAL_TOKENS;
    // Punctuation where the reference has a word: the tokens after it show whether the row puts
    // it in front of that word, an insertion, once they make the whole word.
    // The reference's first character from `start`, which can take several tokens.
    let head: Vec<u8> = reference[start.min(reference.len())..]
        .iter()
        .take(4)
        .flat_map(|id| tokenizer.bytes(*id).iter().copied())
        .collect();
    let word_here = valid_prefix(&head)
        .chars()
        .find(|c| !c.is_whitespace())
        .is_some_and(|c| !is_punctuation(c));
    let marks = if word_here {
        (1..=ids.len())
            .rev()
            .find(|k| punctuation(&ids[..*k]))
            .unwrap_or(0)
    } else {
        0
    };
    if marks == ids.len() && room {
        return None;
    }
    // Punctuation or a word of its own, then the reference's word from its start: the row
    // inserted them in front of it, once the whole reference word follows.
    let inserted = (1..ids.len()).find(|k| {
        (*k == marks || starts_word(tokenizer, &ids[..*k], ids[*k], &ids[*k + 1..]))
            && whole(&row[..*k])
            && reference[start..].starts_with(&ids[*k..])
    });
    if let Some(k) = inserted {
        if word_ends(start + ids.len() - k) {
            row.truncate(k);
            return Some(Some(start));
        }
        if room {
            return None;
        }
    }
    // Back on the reference: two tokens in a row match it, or one whole word does at most one
    // reference word on, as when the row skips a comma or reads two characters as one word. One
    // common word such as ` the` alone can match further on by chance.
    let following_word = next_word(tokenizer, reference, next_word(tokenizer, reference, start));
    let window = (start + 1)..reference.len().min(start + REJOIN_WINDOW + 1);
    let matched = window.into_iter().find(|j| {
        reference[*j] == last
            && ((*j <= following_word
                && starts_word(tokenizer, &reference[..*j], last, &reference[*j + 1..])
                && tokenizer.bytes(last).len() >= 2)
                || (row.len() >= 2 && reference[*j - 1] == row[row.len() - 2].0))
    });
    // It rejoins where a whole character of the reference ends, so a shared first byte is not
    // taken for the reference's character. Out of room part way through one, it does not rejoin.
    if let Some(j) = matched {
        if whole(row) {
            return Some(Some(j + 1));
        }
        if room {
            return None;
        }
    }
    // The word ended when the next one begins: the row's word stands for the reference's. A
    // character split over tokens shows whether it begins a word only once its last byte is in,
    // so the boundary is the first one the newest token decides, after any punctuation in front.
    let begins = |k: usize, to: usize| starts_word(tokenizer, &ids[..k], ids[k], &ids[k + 1..to]);
    let last_index = ids.len() - 1;
    let next = (ids.len().saturating_sub(4).max(marks + 1)..ids.len()).find(|k| {
        begins(*k, ids.len()) && (*k == last_index || !begins(*k, last_index)) && whole(&row[..*k])
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
                &reference[..*j],
                reference[*j],
                &reference[*j + 1..],
            )
        })
        .unwrap_or(reference.len())
}

/// True when a token begins a word: after whitespace, or with punctuation or a CJK character,
/// which are words of their own, unless it is an apostrophe or hyphen going on from the word in
/// the `previous` token. The `following` tokens complete a character the token only begins.
fn starts_word(tokenizer: &BpeTokenizer, before: &[u32], id: u32, following: &[u32]) -> bool {
    let mut head = tokenizer.bytes(id).to_vec();
    for id in following.iter().take(3) {
        head.extend_from_slice(tokenizer.bytes(*id));
    }
    let first = valid_prefix(&head).chars().next();
    // An apostrophe or hyphen right after a letter joins the word, as in `John's`, the way
    // `split_words` reads it. The letter can be split over the tokens `before`, as `é` can.
    let tail: Vec<u8> = before[before.len().saturating_sub(4)..]
        .iter()
        .flat_map(|id| tokenizer.bytes(*id).iter().copied())
        .collect();
    let joins = matches!(first, Some('\'' | '’' | '-'))
        && String::from_utf8_lossy(&tail)
            .chars()
            .next_back()
            .is_some_and(|c| {
                c == char::REPLACEMENT_CHARACTER
                    || !(c.is_whitespace() || is_punctuation(c) || is_cjk(c))
            });
    !joins && first.is_some_and(|c| c.is_whitespace() || is_punctuation(c) || is_cjk(c))
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

impl<M: QwenModel> Asr for QwenAsr<M> {
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
        let encoded = self.model.encode(features)?;
        let dim = encoded.dim;
        // The ONNX encoder zeroes padded positions. Silence at the end is not speech either.
        let speech = (0..encoded.frames)
            .rev()
            .find(|t| encoded.frame(*t).iter().any(|v| v.abs() > 1e-6))
            .map_or(0, |t| t + 1);
        if speech == 0 {
            return Ok(empty);
        }
        let mut data = encoded.data;
        data.truncate(speech * dim);
        Ok(EncoderOutput {
            frames: speech,
            dim,
            data,
        })
    }

    fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
        self.decode_rows(encoders, bias, None)
    }

    /// Scores the reference under every noisy copy in one call, and decodes on only from where a
    /// copy's pick differs (see [`QwenAsr::generate_noisy`]). An empty reference, such as a
    /// `language None` decode of what the model took for silence, gets full noisy decodes: its
    /// header would hold every copy to the same reading.
    fn decode_noisy(
        &self,
        encoders: &[EncoderOutput],
        bias: &Bias,
        reference: &Hypothesis,
    ) -> Result<Vec<Hypothesis>> {
        let ids = (!reference.tokens.is_empty()).then_some(reference.ids.as_slice());
        self.decode_rows(encoders, bias, ids)
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
        // Also after a letter split over tokens.
        let accent = ids("é");
        assert_eq!(accent.len(), 2);
        let mut split = row(&[accent[0], accent[1], apostrophe]);
        assert_eq!(stretch_end(&t, &stops, &mut split, &reference, 0), None);
        // But a hyphen after a split CJK character begins a word of its own.
        let (hyphen, zh) = (ids("-")[0], ids("中"));
        let mut cjk_hyphen = row(&[ids("x")[0], zh[0], zh[1], zh[2], hyphen]);
        assert_eq!(
            stretch_end(&t, &stops, &mut cjk_hyphen, &reference, 0),
            Some(Some(1))
        );
        assert_eq!(cjk_hyphen, row(&[ids("x")[0], zh[0], zh[1], zh[2]]));
        // A newline ends the word like a space.
        let mut newline = row(&[ids("x")[0], ids("y")[0], ids("\n")[0]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut newline, &reference, 0),
            Some(Some(1))
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
        // A comma before a word comes in front of it, once the word follows; before a stop, it
        // ends the row.
        assert_eq!(
            stretch_end(&t, &stops, &mut row(&[comma]), &reference, 1),
            None
        );
        let mut inserted = row(&[comma, 257]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &reference, 1),
            Some(Some(1))
        );
        assert_eq!(inserted, row(&[comma]));
        // Also punctuation split over tokens, or more than one mark.
        let wide = ids("，");
        assert_eq!(wide.len(), 3);
        let mut inserted = row(&[wide[0], wide[1], wide[2], 257]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &reference, 1),
            Some(Some(1))
        );
        assert_eq!(inserted, row(&wide));
        let mut inserted = row(&[comma, comma, 257]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &reference, 1),
            Some(Some(1))
        );
        // A character split over tokens has to match all of it.
        let han = ids("中");
        let split_reference = [256, han[0], han[1], han[2]];
        let mut partial = row(&[comma, han[0]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut partial, &split_reference, 1),
            None
        );
        let mut inserted = row(&[comma, han[0], han[1], han[2]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &split_reference, 1),
            Some(Some(1))
        );
        // The whole reference word has to follow the punctuation, not only its first token.
        let (x, y) = (ids("x")[0], ids("y")[0]);
        let long_word = [256, 257, x, 258];
        let mut begun = row(&[comma, 257]);
        assert_eq!(stretch_end(&t, &stops, &mut begun, &long_word, 1), None);
        let mut inserted = row(&[comma, 257, x]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &long_word, 1),
            Some(Some(1))
        );
        // Punctuation where the reference has punctuation split over tokens replaces it.
        let split_mark = [256, wide[0], wide[1], wide[2], 258];
        let mut swapped = row(&[comma, comma]);
        assert_eq!(
            stretch_end(&t, &stops, &mut swapped, &split_mark, 1),
            Some(Some(4))
        );
        // A stretch from inside a character the reference began: its first bytes count.
        let mid = [256, han[0], han[1], han[2], 258];
        let other = ids("丫");
        assert_eq!(other[..2], han[..2]);
        let mut swapped = row(&[other[2], 257]);
        assert_eq!(
            stretch_end(&t, &stops, &mut swapped, &mid, 3),
            Some(Some(4))
        );
        assert_eq!(swapped, row(&[other[2]]));
        // Past the reference, punctuation and words go on to the stop.
        let mut added = row(&[comma, 258]);
        assert_eq!(stretch_end(&t, &stops, &mut added, &reference, 3), None);
        let mut added = row(&[comma, 258, 300]);
        assert_eq!(
            stretch_end(&t, &stops, &mut added, &reference, 3),
            Some(None)
        );
        assert_eq!(added, row(&[comma, 258]));
        // A word inserted before the reference's word.
        let mut inserted = row(&[258, 257]);
        assert_eq!(
            stretch_end(&t, &stops, &mut inserted, &reference, 1),
            Some(Some(1))
        );
        assert_eq!(inserted, row(&[258]));
        // A rejoin on a split character waits for all of it.
        let split_later = [256, 258, han[0], han[1], han[2]];
        let mut partial = row(&[y, han[0], han[1]]);
        assert_eq!(stretch_end(&t, &stops, &mut partial, &split_later, 1), None);
        let mut rejoined = row(&[y, han[0], han[1], han[2]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut rejoined, &split_later, 1),
            Some(Some(5))
        );
        // Out of room part way through it, the row's word stands for the reference's instead.
        let mut capped = row(&[y, x, y, x, y, x, han[0], han[1]]);
        assert_eq!(
            stretch_end(&t, &stops, &mut capped, &split_later, 1),
            Some(Some(2))
        );
        let mut ended = row(&[comma, 300]);
        assert_eq!(
            stretch_end(&t, &stops, &mut ended, &reference, 1),
            Some(None)
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
