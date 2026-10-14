//! Research prototypes for faster Qwen decoding: speculative decoding against a text draft, and a
//! teacher-forced pass that scores a transcript under noisy audio in one decoder call. Not used by
//! the app.

use std::time::{Duration, Instant};

use super::*;

/// One decode and what it cost.
#[derive(Debug, Clone, Default)]
pub struct LabRun {
    /// Generated ids, the `language X<asr_text>` header included, without the stop token.
    pub ids: Vec<u32>,
    pub calls: usize,
    pub first_call: Duration,
    pub total: Duration,
    /// Draft tokens offered to the decoder, and how many it kept.
    pub proposed: usize,
    pub accepted: usize,
}

/// Per position of a scored transcript: the decoder's own pick, and the log probability of the
/// transcript's token there.
#[derive(Debug, Clone, Copy)]
pub struct Scored {
    pub best: u32,
    pub best_log_prob: f32,
    pub target_log_prob: f32,
}

/// Proposes the rest of a text as tokens, following the decode through casing and punctuation
/// differences, and picking the draft up again after a word it got wrong.
pub struct TextDrafter {
    header: Vec<u32>,
    text: String,
    /// The text lowercased, with punctuation dropped and spaces collapsed, and the byte offset in
    /// `text` just after each kept char.
    norm: Vec<char>,
    ends: Vec<usize>,
    pub block: usize,
    pub header_only: bool,
}

fn normalize(text: &str) -> (Vec<char>, Vec<usize>) {
    let mut norm = Vec::new();
    let mut ends = Vec::new();
    for (offset, c) in text.char_indices() {
        let end = offset + c.len_utf8();
        if c.is_alphanumeric() {
            for lower in c.to_lowercase() {
                norm.push(lower);
                ends.push(end);
            }
        } else if c.is_whitespace() && norm.last().is_some_and(|l| *l != ' ') {
            norm.push(' ');
            ends.push(offset);
        }
    }
    (norm, ends)
}

impl TextDrafter {
    pub fn new(asr: &QwenAsr, language: &str, text: &str, block: usize) -> Self {
        let mut header = asr.tokenizer.encode(&format!("language {language}"));
        header.push(asr.template.asr_text);
        let (norm, ends) = normalize(text);
        Self {
            header,
            text: text.to_owned(),
            norm,
            ends,
            block,
            header_only: false,
        }
    }

    pub fn propose(&self, asr: &QwenAsr, generated: &[u32]) -> Vec<u32> {
        let asr_text = asr.template.asr_text;
        let (mut out, body) = match generated.iter().position(|id| *id == asr_text) {
            Some(end) => (Vec::new(), &generated[end + 1..]),
            None if self.header.starts_with(generated) => {
                (self.header[generated.len()..].to_vec(), &[][..])
            }
            None => return Vec::new(),
        };
        if self.header_only {
            return out;
        }
        if let Some(rest) = self.rest(&asr.tokenizer.decode(body)) {
            out.extend(asr.tokenizer.encode(rest));
        }
        out.truncate(self.block);
        out
    }

    /// The draft text after what was emitted, or `None` when the draft lost the decode.
    fn rest(&self, emitted: &str) -> Option<&str> {
        let (said, _) = normalize(emitted);
        let said: Vec<char> = said.into_iter().collect();
        let trimmed = said.len() - said.iter().rev().take_while(|c| **c == ' ').count();
        let said = &said[..trimmed];
        if said.is_empty() {
            return Some(self.text.trim_start());
        }
        let common = said.iter().zip(&self.norm).take_while(|(a, b)| a == b).count();
        let cursor = if common == said.len() {
            common
        } else {
            // Pick the draft up again after the last few emitted chars, near where it lost the decode.
            let tail = &said[said.len().saturating_sub(6)..];
            let from = common.saturating_sub(12);
            let to = (common + 60).min(self.norm.len());
            (from..to.saturating_sub(tail.len()).max(from))
                .find(|i| self.norm[*i..].starts_with(tail))
                .map(|i| i + tail.len())?
        };
        if cursor == 0 {
            return Some(&self.text);
        }
        self.ends
            .get(cursor - 1)
            .map(|end| &self.text[*end..])
            .filter(|rest| !rest.is_empty())
    }
}

impl QwenAsr {
    /// The decoder on `input`, with logits kept from position `keep_from` of the call on.
    #[allow(clippy::too_many_arguments)]
    fn lab_call(
        &self,
        input: &[i64],
        batch: usize,
        start: usize,
        audio: &[f32],
        (frames, dim): (usize, usize),
        cache: &mut [Vec<f32>],
        total: usize,
        keep_from: usize,
        targets: &[Option<u32>],
    ) -> Result<Vec<Vec<Scored>>> {
        let length = input.len() / batch;
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
        // A decoder edited to take `logits_from` runs the LM head only from there on.
        let sliced = session.inputs().iter().any(|i| i.name() == "logits_from");
        if sliced {
            inputs.push((
                "logits_from".into(),
                Tensor::from_array(([1usize], vec![keep_from as i64]))
                    .map_err(model_error)?
                    .into(),
            ));
        }
        let outputs = session.run(inputs).map_err(model_error)?;
        let offset = if sliced { keep_from } else { 0 };
        let (logits_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let vocab = logits_shape[2] as usize;
        let steps = logits_shape[1] as usize;
        let scored = (0..batch)
            .map(|b| {
                (keep_from - offset..steps)
                    .map(|p| {
                        let row = &logits[(b * steps + p) * vocab..(b * steps + p + 1) * vocab];
                        let (best, best_log_prob) = pick(row);
                        let max = row[best as usize];
                        let lse = max - best_log_prob;
                        let target_log_prob = targets
                            .get(p + offset - keep_from)
                            .copied()
                            .flatten()
                            .map_or(f32::NAN, |t| row[t as usize] - lse);
                        Scored {
                            best,
                            best_log_prob,
                            target_log_prob,
                        }
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
        Ok(scored)
    }

    fn lab_cache(&self, batch: usize, total: usize) -> Vec<Vec<f32>> {
        let width = self.shape.kv_heads * self.shape.head_dim;
        (0..2 * self.shape.layers)
            .map(|_| vec![0.0; batch * total * width])
            .collect()
    }

    /// Greedy decoding, or speculative decoding when `drafter` is given. Speculative decoding
    /// feeds the draft after the last token, keeps the draft up to the decoder's first
    /// disagreement, and takes the decoder's own token there.
    pub fn lab_decode(
        &self,
        encoder: &EncoderOutput,
        bias: &Bias,
        drafter: Option<&TextDrafter>,
    ) -> Result<LabRun> {
        let started = Instant::now();
        let prompt = self.prompt(encoder.frames, &bias.phrases);
        let context = prompt.len();
        let max_new = self.max_new_tokens(encoder.frames);
        let block = drafter.map_or(0, |d| d.block);
        let total = context + max_new + block + 1;
        let mut cache = self.lab_cache(1, total);
        let audio = &encoder.data[..encoder.frames * encoder.dim];
        let fd = (encoder.frames, encoder.dim);
        let mut run = LabRun::default();

        let mut generated: Vec<u32> = Vec::new();
        let mut input = prompt.clone();
        let mut start = 0;
        let mut keep_from = context - 1;
        loop {
            let draft = drafter.map_or_else(Vec::new, |d| d.propose(self, &generated));
            input.extend(draft.iter().map(|id| i64::from(*id)));
            let scored =
                self.lab_call(&input, 1, start, audio, fd, &mut cache, total, keep_from, &[])?;
            run.calls += 1;
            if run.calls == 1 {
                run.first_call = started.elapsed();
            }
            let picks: Vec<u32> = scored[0].iter().map(|s| s.best).collect();
            let kept = draft
                .iter()
                .zip(&picks)
                .take_while(|(d, p)| d == p)
                .count();
            run.proposed += draft.len();
            run.accepted += kept;
            start += input.len() - draft.len() + kept;
            let mut stopped = false;
            for id in draft[..kept].iter().chain(std::iter::once(&picks[kept])) {
                if self.template.stops.contains(id) {
                    stopped = true;
                    break;
                }
                generated.push(*id);
            }
            if stopped || generated.len() >= max_new {
                break;
            }
            input = vec![i64::from(*generated.last().unwrap())];
            keep_from = 0;
        }
        run.ids = generated;
        run.total = started.elapsed();
        Ok(run)
    }

    /// Scores `ids` (a decode's output, header included) under each encoder output in one batched
    /// call: the prompt and the whole transcript go in together.
    pub fn lab_score(
        &self,
        encoders: &[&EncoderOutput],
        bias: &Bias,
        ids: &[u32],
    ) -> Result<Vec<Vec<Scored>>> {
        let prompt = self.prompt(encoders[0].frames, &bias.phrases);
        let context = prompt.len();
        let mut row: Vec<i64> = prompt;
        row.extend(ids.iter().map(|id| i64::from(*id)));
        let total = row.len();
        let batch = encoders.len();
        let mut cache = self.lab_cache(batch, total);
        let audio: Vec<f32> = encoders
            .iter()
            .flat_map(|e| e.data[..e.frames * e.dim].iter().copied())
            .collect();
        let mut targets: Vec<Option<u32>> = ids.iter().copied().map(Some).collect();
        targets.push(Some(self.template.im_end));
        self.lab_call(
            &row.repeat(batch),
            batch,
            0,
            &audio,
            (encoders[0].frames, encoders[0].dim),
            &mut cache,
            total,
            context - 1,
            &targets,
        )
    }

    /// Time of one decoder call that reads `length` new tokens after the prompt.
    pub fn lab_step_cost(&self, encoder: &EncoderOutput, bias: &Bias, length: usize) -> Result<Duration> {
        let prompt = self.prompt(encoder.frames, &bias.phrases);
        let context = prompt.len();
        let total = context + length + 1;
        let mut cache = self.lab_cache(1, total);
        let audio = &encoder.data[..encoder.frames * encoder.dim];
        let fd = (encoder.frames, encoder.dim);
        self.lab_call(&prompt, 1, 0, audio, fd, &mut cache, total, context - 1, &[])?;
        let input = vec![i64::from(self.template.asr_text); length];
        let started = Instant::now();
        self.lab_call(&input, 1, context, audio, fd, &mut cache, total, 0, &[])?;
        Ok(started.elapsed())
    }

    /// Time of the prompt prefill alone.
    pub fn lab_prefill_cost(&self, encoder: &EncoderOutput, bias: &Bias) -> Result<Duration> {
        let prompt = self.prompt(encoder.frames, &bias.phrases);
        let total = prompt.len() + 1;
        let mut cache = self.lab_cache(1, total);
        let audio = &encoder.data[..encoder.frames * encoder.dim];
        let started = Instant::now();
        self.lab_call(
            &prompt,
            1,
            0,
            audio,
            (encoder.frames, encoder.dim),
            &mut cache,
            total,
            prompt.len() - 1,
            &[],
        )?;
        Ok(started.elapsed())
    }

    pub fn lab_text(&self, ids: &[u32]) -> String {
        let generated: Vec<(u32, f32)> = ids.iter().map(|id| (*id, 0.0)).collect();
        self.to_hypothesis(&generated).text()
    }

    pub fn lab_prompt_len(&self, frames: usize, bias: &Bias) -> usize {
        self.prompt(frames, &bias.phrases).len()
    }
}
