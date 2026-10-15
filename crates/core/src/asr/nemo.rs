//! Cache-aware streaming NeMo transducers, such as NVIDIA's Nemotron ASR Streaming, as sherpa-onnx
//! exports them.
//!
//! The encoder is a FastConformer that keeps attention and convolution caches between chunks, so
//! it encodes audio as it arrives: each call reads `window_size` feature frames (a pre-encode
//! overlap plus `chunk_shift` new frames) and emits one encoder frame per 80 ms. The prediction
//! network is an LSTM, so unlike the stateless decoders that [`super::Recognizer`] drives, each
//! hypothesis carries its own decoder state, and a frame can emit several tokens.
//!
//! The multilingual export takes a language prompt, and writes a language tag such as `<en-US>`
//! after the text, which [`NemoTransducer`] drops.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Mutex;

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorRef};

use super::onnx::{load_session, TransducerFiles};
use super::{
    Asr, Bias, ContextGraph, ContextState, EncoderOutput, EncoderStream, Hypothesis, SearchConfig,
    SymbolTable, TimedToken, Tokenizer,
};
use crate::audio::SAMPLE_RATE;
use crate::features::{Fbank, FbankConfig};
use crate::uncertainty::MonteCarloConfig;
use crate::{Error, Result};

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

/// Tokens one frame may emit before the search moves on, as in NeMo and sherpa-onnx.
const MAX_SYMBOLS_PER_FRAME: usize = 10;

/// True when the encoder session is a cache-aware streaming NeMo encoder.
pub fn is_streaming_encoder(session: &Session) -> bool {
    session
        .inputs()
        .iter()
        .any(|input| input.name() == "cache_last_channel")
}

/// Shapes and settings from the encoder's metadata.
#[derive(Debug, Clone)]
struct Meta {
    feature_dim: usize,
    /// Feature frames per encoder call.
    window: usize,
    /// New feature frames per encoder call; the rest of the window overlaps the previous call.
    shift: usize,
    subsampling: usize,
    /// `cache_last_channel` without its batch dimension.
    channel_cache: [usize; 3],
    /// `cache_last_time` without its batch dimension.
    time_cache: [usize; 3],
    pred_layers: usize,
    pred_hidden: usize,
    vocab_size: usize,
}

impl Meta {
    fn read(session: &Session) -> Result<Self> {
        let metadata = session.metadata().map_err(model_error)?;
        let read = |key: &str| -> Result<usize> {
            metadata
                .custom(key)
                .and_then(|v| v.trim().parse().ok())
                .ok_or_else(|| Error::Model(format!("encoder metadata has no {key}")))
        };
        Ok(Self {
            feature_dim: read("feat_dim")?,
            window: read("window_size")?,
            shift: read("chunk_shift")?,
            subsampling: read("subsampling_factor")?,
            channel_cache: [
                read("cache_last_channel_dim1")?,
                read("cache_last_channel_dim2")?,
                read("cache_last_channel_dim3")?,
            ],
            time_cache: [
                read("cache_last_time_dim1")?,
                read("cache_last_time_dim2")?,
                read("cache_last_time_dim3")?,
            ],
            pred_layers: read("pred_rnn_layers")?,
            pred_hidden: read("pred_hidden")?,
            // NeMo leaves the blank out of its vocabulary size.
            vocab_size: read("vocab_size")? + 1,
        })
    }

    /// Feature frames the encoder reads before the new frames of a call.
    fn overlap(&self) -> usize {
        self.window - self.shift
    }

    fn frame_samples(&self) -> usize {
        self.subsampling * FEATURE_SHIFT
    }
}

/// Samples per feature frame.
const FEATURE_SHIFT: usize = SAMPLE_RATE as usize / 100;

/// A streaming NeMo transducer with its tokens.
pub struct NemoTransducer {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joiner: Mutex<Session>,
    decoder_inputs: Vec<String>,
    meta: Meta,
    symbols: SymbolTable,
    /// `<en-US>` and the like, which the multilingual model writes after the text.
    language_tags: HashSet<u32>,
    /// The language prompt, for a multilingual encoder.
    prompt: Option<i64>,
    fbank: Fbank,
    pub search: SearchConfig,
}

impl NemoTransducer {
    /// Loads the model around an encoder session that [`is_streaming_encoder`] accepted.
    /// `language` is a code such as `en`, `ja-JP` or `auto`; `None` means `auto`.
    pub fn load(
        encoder: Session,
        files: &TransducerFiles,
        search: SearchConfig,
        language: Option<&str>,
    ) -> Result<Self> {
        let meta = Meta::read(&encoder)?;
        let multilingual = encoder
            .inputs()
            .iter()
            .any(|input| input.name() == "prompt_index");
        let prompt = if multilingual {
            Some(language_prompt(&encoder, language.unwrap_or("auto"))?)
        } else {
            None
        };
        let decoder = load_session(&files.decoder, 1)?;
        let decoder_inputs = decoder
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<Vec<_>>();
        if decoder_inputs.len() != 4 {
            return Err(Error::Model(format!(
                "expected 4 decoder inputs (targets, lengths, two LSTM states), found {}",
                decoder_inputs.len()
            )));
        }
        let symbols = SymbolTable::load(&files.tokens)?;
        if symbols.len() != meta.vocab_size
            || symbols.id("<blk>") != Some(meta.vocab_size as u32 - 1)
        {
            return Err(Error::Model(format!(
                "{} does not match the encoder: expected {} symbols ending in <blk>",
                files.tokens.display(),
                meta.vocab_size
            )));
        }
        let language_tags = (0..symbols.len() as u32)
            .filter(|&id| is_language_tag(symbols.text(id)))
            .collect();
        Ok(Self {
            fbank: Fbank::new(FbankConfig::nemo(meta.feature_dim)),
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            joiner: Mutex::new(load_session(&files.joiner, 1)?),
            decoder_inputs,
            meta,
            symbols,
            language_tags,
            prompt,
            search,
        })
    }

    fn blank(&self) -> u32 {
        self.meta.vocab_size as u32 - 1
    }

    /// Starts encoding a new stream of audio.
    pub fn stream(&self) -> NemoStream<'_> {
        let [c1, c2, c3] = self.meta.channel_cache;
        let [t1, t2, t3] = self.meta.time_cache;
        NemoStream {
            model: self,
            samples: Vec::new(),
            dropped_samples: 0,
            features: Vec::new(),
            dropped_features: 0,
            feature_frames: 0,
            processed: 0,
            channel_cache: vec![0.0; c1 * c2 * c3],
            time_cache: vec![0.0; t1 * t2 * t3],
            channel_cache_len: 0,
            frames: Vec::new(),
            dropped_frames: 0,
            finished: false,
            taken: 0,
        }
    }

    /// Runs the encoder on one window of features, `[window, feature_dim]` row-major.
    fn encode_window(&self, features: &[f32], stream: &mut NemoStream<'_>) -> Result<Vec<f32>> {
        let meta = &self.meta;
        let (window, dim) = (meta.window, meta.feature_dim);
        // The encoder reads `[batch, dim, time]`.
        let mut transposed = vec![0.0f32; window * dim];
        for t in 0..window {
            for d in 0..dim {
                transposed[d * window + t] = features[t * dim + d];
            }
        }
        let [c1, c2, c3] = meta.channel_cache;
        let [t1, t2, t3] = meta.time_cache;
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            (
                "audio_signal".into(),
                Tensor::from_array(([1, dim, window], transposed))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "length".into(),
                Tensor::from_array(([1], vec![window as i64]))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "cache_last_channel".into(),
                TensorRef::from_array_view(([1, c1, c2, c3], stream.channel_cache.as_slice()))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "cache_last_time".into(),
                TensorRef::from_array_view(([1, t1, t2, t3], stream.time_cache.as_slice()))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "cache_last_channel_len".into(),
                Tensor::from_array(([1], vec![stream.channel_cache_len]))
                    .map_err(model_error)?
                    .into(),
            ),
        ];
        if let Some(prompt) = self.prompt {
            inputs.push((
                "prompt_index".into(),
                Tensor::from_array(([1], vec![prompt]))
                    .map_err(model_error)?
                    .into(),
            ));
        }
        let mut session = self.encoder.lock().expect("encoder lock");
        let outputs = session.run(inputs).map_err(model_error)?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (_, lengths) = outputs[1]
            .try_extract_tensor::<i64>()
            .map_err(model_error)?;
        let (encoder_dim, time) = (shape[1] as usize, shape[2] as usize);
        let frames = (lengths[0].max(0) as usize).min(time);
        let mut encoded = Vec::with_capacity(frames * encoder_dim);
        for t in 0..frames {
            encoded.extend((0..encoder_dim).map(|d| data[d * time + t]));
        }
        let (_, channel) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (_, time_cache) = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (_, channel_len) = outputs[4]
            .try_extract_tensor::<i64>()
            .map_err(model_error)?;
        stream.channel_cache.copy_from_slice(channel);
        stream.time_cache.copy_from_slice(time_cache);
        stream.channel_cache_len = channel_len[0];
        Ok(encoded)
    }

    /// Runs the prediction network on each `(last token, state)`, returning each output and new state.
    fn predict(&self, items: &[(u32, &DecoderState)]) -> Result<Vec<(Vec<f32>, DecoderState)>> {
        // sherpa-onnx exports the second LSTM state with a fixed batch of 1.
        items
            .iter()
            .map(|item| Ok(self.predict_batch(std::slice::from_ref(item))?.remove(0)))
            .collect()
    }

    fn predict_batch(
        &self,
        items: &[(u32, &DecoderState)],
    ) -> Result<Vec<(Vec<f32>, DecoderState)>> {
        let (layers, hidden) = (self.meta.pred_layers, self.meta.pred_hidden);
        let batch = items.len();
        let stack = |pick: fn(&DecoderState) -> &[f32]| {
            let mut stacked = vec![0.0f32; layers * batch * hidden];
            for (b, (_, state)) in items.iter().enumerate() {
                let source = pick(state);
                for l in 0..layers {
                    stacked[(l * batch + b) * hidden..(l * batch + b + 1) * hidden]
                        .copy_from_slice(&source[l * hidden..(l + 1) * hidden]);
                }
            }
            stacked
        };
        let names = &self.decoder_inputs;
        let inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            (
                names[0].clone(),
                Tensor::from_array((
                    [batch, 1],
                    items.iter().map(|(t, _)| *t as i32).collect::<Vec<_>>(),
                ))
                .map_err(model_error)?
                .into(),
            ),
            (
                names[1].clone(),
                Tensor::from_array(([batch], vec![1i32; batch]))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                names[2].clone(),
                Tensor::from_array(([layers, batch, hidden], stack(|s| &s.h)))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                names[3].clone(),
                Tensor::from_array(([layers, batch, hidden], stack(|s| &s.c)))
                    .map_err(model_error)?
                    .into(),
            ),
        ];
        let mut session = self.decoder.lock().expect("decoder lock");
        let outputs = session.run(inputs).map_err(model_error)?;
        let (_, out) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (_, h) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (_, c) = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let unstack = |stacked: &[f32], b: usize| {
            let mut state = vec![0.0f32; layers * hidden];
            for l in 0..layers {
                state[l * hidden..(l + 1) * hidden].copy_from_slice(
                    &stacked[(l * batch + b) * hidden..(l * batch + b + 1) * hidden],
                );
            }
            state
        };
        // The output is `[batch, hidden, 1]`, so each row is contiguous.
        Ok((0..batch)
            .map(|b| {
                (
                    out[b * hidden..(b + 1) * hidden].to_vec(),
                    DecoderState {
                        h: unstack(h, b),
                        c: unstack(c, b),
                    },
                )
            })
            .collect())
    }

    /// Joiner logits for pairs of an encoder frame and a prediction network output.
    fn join(&self, pairs: &[(&[f32], &[f32])]) -> Result<Vec<Vec<f32>>> {
        let batch = pairs.len();
        let encoder_dim = pairs.first().map_or(0, |p| p.0.len());
        let hidden = self.meta.pred_hidden;
        let encoder: Vec<f32> = pairs.iter().flat_map(|p| p.0.iter().copied()).collect();
        let decoder: Vec<f32> = pairs.iter().flat_map(|p| p.1.iter().copied()).collect();
        let mut session = self.joiner.lock().expect("joiner lock");
        let outputs = session
            .run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(([batch, encoder_dim, 1], encoder)).map_err(model_error)?,
                "decoder_outputs" => Tensor::from_array(([batch, hidden, 1], decoder)).map_err(model_error)?,
            ])
            .map_err(model_error)?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let vocab = self.meta.vocab_size;
        Ok(data.chunks(vocab).map(<[f32]>::to_vec).collect())
    }

    fn initial_state(&self) -> DecoderState {
        let size = self.meta.pred_layers * self.meta.pred_hidden;
        DecoderState {
            h: vec![0.0; size],
            c: vec![0.0; size],
        }
    }

    /// Greedy search, with up to [`MAX_SYMBOLS_PER_FRAME`] tokens per frame.
    fn greedy(&self, encoder: &EncoderOutput) -> Result<Vec<(u32, usize, f32)>> {
        let blank = self.blank();
        let (mut out, mut state) = self.predict(&[(blank, &self.initial_state())])?.remove(0);
        let mut tokens = Vec::new();
        for t in 0..encoder.frames {
            for _ in 0..MAX_SYMBOLS_PER_FRAME {
                let mut logits = self.join(&[(encoder.frame(t), &out)])?.remove(0);
                log_softmax(&mut logits, blank as usize, self.search.blank_penalty);
                let (best, log_prob) = argmax(&logits);
                if best == blank {
                    break;
                }
                tokens.push((best, t, log_prob));
                (out, state) = self.predict(&[(best, &state)])?.remove(0);
            }
        }
        Ok(tokens)
    }

    /// Modified beam search with several tokens per frame and optional hotword biasing, after
    /// sherpa-onnx's search for these models. A hypothesis that emits a token stays on its frame,
    /// and the search always expands the hypotheses furthest behind. Hypotheses with the same
    /// tokens at the same place keep the better alignment.
    fn beam_search(
        &self,
        encoder: &EncoderOutput,
        graph: Option<&ContextGraph>,
    ) -> Result<Vec<(u32, usize, f32)>> {
        let blank = self.blank();
        let beam = self.search.beam.max(1);
        let mut hyps = vec![BeamHyp {
            ys: Vec::new(),
            tokens: Vec::new(),
            log_prob: 0.0,
            context_state: ContextState::default(),
            state: Rc::new(self.initial_state()),
            out: None,
            frame: 0,
            symbols: 0,
        }];

        loop {
            let frame = hyps.iter().map(|h| h.frame).min().unwrap_or(encoder.frames);
            if frame >= encoder.frames {
                break;
            }
            // Hypotheses whose last token the prediction network has not read yet.
            let pending: Vec<usize> = (0..hyps.len())
                .filter(|&i| {
                    let h = &hyps[i];
                    h.frame == frame && h.symbols < MAX_SYMBOLS_PER_FRAME && h.out.is_none()
                })
                .collect();
            if !pending.is_empty() {
                let items: Vec<(u32, &DecoderState)> = pending
                    .iter()
                    .map(|&i| {
                        (
                            hyps[i].ys.last().copied().unwrap_or(blank),
                            hyps[i].state.as_ref(),
                        )
                    })
                    .collect();
                for (&i, (out, state)) in pending.iter().zip(self.predict(&items)?) {
                    hyps[i].out = Some(Rc::new(out));
                    hyps[i].state = Rc::new(state);
                }
            }
            let expand: Vec<usize> = (0..hyps.len())
                .filter(|&i| hyps[i].frame == frame && hyps[i].symbols < MAX_SYMBOLS_PER_FRAME)
                .collect();
            let pairs: Vec<(&[f32], &[f32])> = expand
                .iter()
                .map(|&i| {
                    (
                        encoder.frame(frame),
                        hyps[i].out.as_deref().expect("predicted").as_slice(),
                    )
                })
                .collect();
            let logits = if pairs.is_empty() {
                Vec::new()
            } else {
                self.join(&pairs)?
            };

            let mut candidates: Vec<BeamHyp> = Vec::new();
            let mut expansions = expand.iter().zip(logits).peekable();
            for (i, hyp) in hyps.iter().enumerate() {
                if hyp.frame > frame {
                    candidates.push(hyp.clone());
                    continue;
                }
                if hyp.symbols >= MAX_SYMBOLS_PER_FRAME {
                    let mut next = hyp.clone();
                    next.frame += 1;
                    next.symbols = 0;
                    candidates.push(next);
                    continue;
                }
                let (_, mut row) = expansions.next_if(|(&j, _)| j == i).expect("expanded");
                log_softmax(&mut row, blank as usize, self.search.blank_penalty);
                // Hotword boosts come after the acoustic pruning, so the next token of a started
                // hotword joins the candidates however the acoustics rank it.
                let mut tokens = top_k(&row, beam);
                if let Some(graph) = graph {
                    for token in graph.continuations(hyp.context_state) {
                        if !tokens.contains(&(token as usize)) {
                            tokens.push(token as usize);
                        }
                    }
                }
                for token in tokens {
                    let mut next = hyp.clone();
                    next.log_prob += row[token];
                    if token as u32 == blank {
                        next.frame += 1;
                        next.symbols = 0;
                    } else {
                        let token = token as u32;
                        next.ys.push(token);
                        next.tokens.push((token, frame, row[token as usize]));
                        // The prediction network reads the new token at this hypothesis's next
                        // expansion, if it survives the pruning.
                        next.out = None;
                        next.symbols += 1;
                        if let Some(graph) = graph {
                            let (delta, state) = graph.step(hyp.context_state, token);
                            next.log_prob += delta;
                            next.context_state = state;
                        }
                    }
                    candidates.push(next);
                }
            }

            let mut best: HashMap<(Vec<u32>, usize, usize), BeamHyp> = HashMap::new();
            for candidate in candidates {
                let key = (candidate.ys.clone(), candidate.frame, candidate.symbols);
                match best.get(&key) {
                    Some(existing) if existing.log_prob >= candidate.log_prob => {}
                    _ => {
                        best.insert(key, candidate);
                    }
                }
            }
            hyps = best.into_values().collect();
            hyps.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            hyps.truncate(beam);
        }

        let finish = |h: &BeamHyp| h.log_prob + graph.map_or(0.0, |g| g.finish(h.context_state));
        Ok(hyps
            .into_iter()
            .max_by(|a, b| finish(a).total_cmp(&finish(b)))
            .map(|h| h.tokens)
            .unwrap_or_default())
    }

    fn to_hypothesis(&self, tokens: &[(u32, usize, f32)]) -> Hypothesis {
        let frame_seconds = self.frame_seconds();
        Hypothesis {
            tokens: tokens
                .iter()
                .filter(|(id, _, _)| !self.language_tags.contains(id))
                .map(|&(id, frame, log_prob)| TimedToken {
                    id,
                    text: self.symbols.text(id).to_owned(),
                    time: Some(frame as f32 * frame_seconds),
                    log_prob,
                })
                .collect(),
            ids: Vec::new(),
        }
    }
}

impl Asr for NemoTransducer {
    fn bias(&self, phrases: &[String]) -> Bias {
        let tokenizer = Tokenizer::new(&self.symbols);
        let sequences: Vec<Vec<u32>> = phrases
            .iter()
            .filter_map(|phrase| tokenizer.encode(phrase))
            .collect();
        let graph = ContextGraph::new(&sequences, self.search.hotword_boost);
        Bias {
            phrases: phrases.to_vec(),
            graph: (!graph.is_empty()).then_some(graph),
        }
    }

    /// Encodes a whole segment as a stream of its own.
    fn encode(&self, samples: &[f32]) -> Result<EncoderOutput> {
        let mut stream = self.stream();
        stream.accept(samples)?;
        stream.finish()?;
        // Nothing comes before the segment, so only its end moves by the delay.
        Ok(stream.slice(0, samples.len() as u64 + stream.delay()))
    }

    fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
        encoders
            .iter()
            .map(|encoder| {
                let tokens = if self.search.beam <= 1 && bias.graph.is_none() {
                    self.greedy(encoder)?
                } else {
                    self.beam_search(encoder, bias.graph.as_ref())?
                };
                Ok(self.to_hypothesis(&tokens))
            })
            .collect()
    }

    fn frame_seconds(&self) -> f32 {
        self.meta.frame_samples() as f32 / SAMPLE_RATE as f32
    }

    /// Each noisy decode costs about as much as the clean one, which is what the user waits for
    /// once a segment ends, so the check takes few samples. Noise of one standard deviation, as
    /// zipformers take, scrambles whole sentences of this encoder; 0.3 leaves clear speech alone.
    fn monte_carlo(&self) -> MonteCarloConfig {
        MonteCarloConfig {
            samples: 4,
            first_samples: 2,
            relative_sigma: 0.3,
            ..MonteCarloConfig::default()
        }
    }

    fn stream(&self) -> Option<Box<dyn EncoderStream + '_>> {
        Some(Box::new(NemoTransducer::stream(self)))
    }
}

/// The two LSTM states of the prediction network, each `[layers, hidden]`.
#[derive(Debug, Clone)]
struct DecoderState {
    h: Vec<f32>,
    c: Vec<f32>,
}

#[derive(Clone)]
struct BeamHyp {
    ys: Vec<u32>,
    tokens: Vec<(u32, usize, f32)>,
    /// Acoustic log probability plus the hotword bonus so far.
    log_prob: f32,
    context_state: ContextState,
    /// The prediction network state after every token but, while `out` is `None`, the last.
    state: Rc<DecoderState>,
    out: Option<Rc<Vec<f32>>>,
    frame: usize,
    /// Tokens emitted at `frame`.
    symbols: usize,
}

/// Encodes one stream of audio as it arrives. It keeps only the audio, features and encoder frames
/// that later work can still read, so a long dictation does not grow it; the `dropped_*` counts
/// keep indices absolute.
pub struct NemoStream<'a> {
    model: &'a NemoTransducer,
    samples: Vec<f32>,
    dropped_samples: usize,
    features: Vec<f32>,
    dropped_features: usize,
    feature_frames: usize,
    /// Feature frames the encoder has moved past.
    processed: usize,
    channel_cache: Vec<f32>,
    time_cache: Vec<f32>,
    channel_cache_len: i64,
    /// Encoder frames not yet taken, row-major.
    frames: Vec<f32>,
    dropped_frames: usize,
    finished: bool,
    /// Where the frames of the last segment taken end, in samples.
    taken: u64,
}

impl NemoStream<'_> {
    fn encoder_dim(&self) -> usize {
        self.model.meta.channel_cache[2]
    }

    /// Encoder frames so far, counting the dropped ones.
    fn encoder_frames(&self) -> usize {
        self.dropped_frames + self.frames.len() / self.encoder_dim()
    }

    fn sample_count(&self) -> usize {
        self.dropped_samples + self.samples.len()
    }

    /// Computes the features that the audio so far allows, `limit` frames at most, and runs the
    /// encoder on every full window.
    fn advance(&mut self, limit: usize) -> Result<()> {
        let fbank = &self.model.fbank;
        let shift = FEATURE_SHIFT;
        let ready = fbank.complete_frames(self.sample_count()).min(limit);
        if ready > self.feature_frames {
            // The kept audio starts a whole number of frames in, and at least one frame before the
            // first new one, so frames count from it as from the start and none reflects there.
            let features = fbank.compute_range(
                &self.samples,
                self.feature_frames - self.dropped_samples / shift,
                ready - self.feature_frames,
            );
            self.features.extend_from_slice(&features.data);
            self.feature_frames = ready;
        }
        let meta = self.model.meta.clone();
        let dim = meta.feature_dim;
        while self.processed + meta.window <= self.feature_frames {
            let first = self.processed - self.dropped_features;
            let window = self.features[first * dim..(first + meta.window) * dim].to_vec();
            let model = self.model;
            let encoded = model.encode_window(&window, self)?;
            self.frames.extend_from_slice(&encoded);
            self.processed += meta.shift;
        }

        let features = self.processed - self.dropped_features;
        self.features.drain(..features * dim);
        self.dropped_features = self.processed;
        let samples = self.feature_frames.saturating_sub(1) * shift - self.dropped_samples;
        self.samples.drain(..samples);
        self.dropped_samples += samples;
        Ok(())
    }

    /// How long after a sound the model can write its token: about one chunk.
    fn delay(&self) -> u64 {
        (self.model.meta.shift * FEATURE_SHIFT) as u64
    }

    /// The encoder frames that overlap samples `start..end`.
    fn slice(&self, start: u64, end: u64) -> EncoderOutput {
        let (dim, count) = (self.encoder_dim(), self.encoder_frames());
        let frame_samples = self.model.meta.frame_samples() as u64;
        let first = (0..count)
            .skip(self.dropped_frames)
            .find(|&i| self.frame_start(i) + frame_samples > start)
            .unwrap_or(count);
        let last = (first..count)
            .take_while(|&i| self.frame_start(i) < end)
            .last()
            .map_or(first, |i| i + 1);
        EncoderOutput {
            frames: last - first,
            dim,
            data: self.frames
                [(first - self.dropped_frames) * dim..(last - self.dropped_frames) * dim]
                .to_vec(),
        }
    }

    /// The first sample of encoder frame `index`: each window's new frames follow its overlap.
    fn frame_start(&self, index: usize) -> u64 {
        let meta = &self.model.meta;
        ((meta.overlap() + index * meta.subsampling) * FEATURE_SHIFT) as u64
    }
}

impl EncoderStream for NemoStream<'_> {
    fn accept(&mut self, samples: &[f32]) -> Result<()> {
        if self.finished {
            return Err(Error::Model("audio after the stream finished".into()));
        }
        self.samples.extend_from_slice(samples);
        self.advance(usize::MAX)
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        // Pad with silence so that the last window covers every frame of the audio, then run one
        // more chunk, where the model writes the tokens of the audio's end (see `take`). Later
        // windows would hold only padding.
        let meta = self.model.meta.clone();
        let real = self.model.fbank.frame_count(self.sample_count());
        let last_window = real.saturating_sub(meta.window).div_ceil(meta.shift) * meta.shift;
        self.samples.resize(
            self.samples.len() + (meta.window + meta.shift) * FEATURE_SHIFT,
            0.0,
        );
        self.advance(last_window + meta.window + meta.shift)
    }

    fn take(&mut self, start: u64, end: u64) -> Option<EncoderOutput> {
        // A token comes after its sound, so a segment's first token is never early, while its
        // last ones can come up to `delay` after its end. The frames before `taken` went to the
        // previous segment, whose last tokens they hold.
        let end = end + self.delay();
        if !self.finished && self.frame_start(self.encoder_frames()) < end {
            return None;
        }
        let frames = self.slice(start.max(self.taken), end);
        self.taken = end;
        // Later segments start at `taken` or after it.
        self.discard(end);
        Some(frames)
    }

    fn discard(&mut self, before: u64) {
        let (dim, frame_samples) = (self.encoder_dim(), self.model.meta.frame_samples() as u64);
        let done = (self.dropped_frames..self.encoder_frames())
            .take_while(|&i| self.frame_start(i) + frame_samples <= before)
            .count();
        self.frames.drain(..done * dim);
        self.dropped_frames += done;
    }
}

/// The prompt index of `language` from the encoder's `prompt_dictionary`, matching `ja` to
/// `ja-JP` and ignoring case.
fn language_prompt(encoder: &Session, language: &str) -> Result<i64> {
    let metadata = encoder.metadata().map_err(model_error)?;
    let auto = metadata
        .custom("auto_prompt_id")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(101);
    let dictionary: HashMap<String, i64> = metadata
        .custom("prompt_dictionary")
        .and_then(|v| serde_json::from_str(&v).ok())
        .unwrap_or_default();
    let wanted = language.trim().replace('_', "-").to_lowercase();
    if wanted.is_empty() || wanted == "auto" {
        return Ok(auto);
    }
    let mut entries: Vec<(&String, &i64)> = dictionary.iter().collect();
    entries.sort();
    entries
        .iter()
        .find(|(key, _)| key.to_lowercase() == wanted)
        .or_else(|| {
            entries
                .iter()
                .find(|(key, _)| key.to_lowercase().split('-').next() == Some(wanted.as_str()))
        })
        .map(|(_, id)| **id)
        .ok_or_else(|| Error::Model(format!("the model has no language {language:?}")))
}

/// `<en>` or `<en-US>`.
fn is_language_tag(symbol: &str) -> bool {
    let Some(inner) = symbol.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else {
        return false;
    };
    let lower = |s: &str| s.chars().all(|c| c.is_ascii_lowercase());
    match inner.split_once('-') {
        // `<unk>` and `<blk>` have three letters, so a bare tag has two.
        None => inner.len() == 2 && lower(inner),
        Some((language, region)) => {
            (2..=3).contains(&language.len())
                && lower(language)
                && region.len() == 2
                && region.chars().all(|c| c.is_ascii_uppercase())
        }
    }
}

fn log_softmax(logits: &mut [f32], blank: usize, blank_penalty: f32) {
    logits[blank] -= blank_penalty;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln() + max;
    logits.iter_mut().for_each(|v| *v -= sum);
}

fn argmax(values: &[f32]) -> (u32, f32) {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i as u32, v)
            } else {
                best
            }
        })
}

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    let k = k.min(values.len());
    order.select_nth_unstable_by(k - 1, |a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(k);
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_language_tags() {
        assert!(is_language_tag("<en-US>"));
        assert!(is_language_tag("<ja>"));
        assert!(!is_language_tag("<unk>"));
        assert!(!is_language_tag("<blk>"));
        assert!(!is_language_tag("▁en"));
    }

    #[test]
    fn top_k_picks_the_largest() {
        let mut picked = top_k(&[0.1, 3.0, -1.0, 2.0], 2);
        picked.sort();
        assert_eq!(picked, vec![1, 3]);
    }
}
