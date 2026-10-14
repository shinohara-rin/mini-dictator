//! Speech recognition.
//!
//! The pipeline drives any [`Asr`]: it encodes a segment once, then decodes the encoder output, and
//! uncertainty decodes noisy copies of it. Two backends implement it:
//!
//! - [`Recognizer`] runs a transducer. The core runs its search itself instead of calling a packaged
//!   recognizer, so hotwords can score partial matches inside the beam search ([`ContextGraph`]). A
//!   [`TransducerModel`] has the three networks of a sherpa-onnx transducer: the encoder, the
//!   stateless decoder (prediction network), and the joiner.
//! - [`nemo::NemoTransducer`] runs a cache-aware streaming NeMo transducer such as Nemotron ASR
//!   Streaming. It encodes audio while the user speaks ([`Asr::stream`]), so a segment is ready to
//!   decode almost as soon as it ends.
//! - [`qwen::QwenAsr`] runs Qwen3-ASR, an audio encoder in front of an LLM that reads hotwords in its
//!   prompt. It covers about 30 languages.

mod context_graph;
#[cfg(feature = "onnx")]
pub mod nemo;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
pub mod qwen;
mod search;
mod tokens;

pub use context_graph::{ContextGraph, ContextState};
pub use search::{greedy_search, modified_beam_search, modified_beam_search_batch, SearchConfig};
pub use tokens::{SymbolTable, Tokenizer};

use crate::features::{Fbank, FbankConfig, Features};
use crate::uncertainty::MonteCarloConfig;
use crate::Result;

/// Encoder frames of one utterance, in row-major order.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    pub frames: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl EncoderOutput {
    pub fn frame(&self, index: usize) -> &[f32] {
        &self.data[index * self.dim..(index + 1) * self.dim]
    }
}

/// The three networks of a stateless transducer.
///
/// Batched calls let the search run every beam, and every Monte Carlo sample, in one model call.
pub trait TransducerModel: Send + Sync {
    /// Tokens of left context that the decoder reads.
    fn context_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn blank_id(&self) -> u32 {
        0
    }
    /// Seconds between two encoder frames. Zipformer subsamples 10 ms features by 4.
    fn frame_seconds(&self) -> f32 {
        0.04
    }
    fn encode(&self, features: &Features) -> Result<EncoderOutput>;
    /// Decoder outputs for a batch of token contexts, each [`Self::context_size`] long.
    fn decode(&self, contexts: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>;
    /// Joiner logits for pairs of an encoder frame and a decoder output.
    fn join(&self, encoder: &[&[f32]], decoder: &[&[f32]]) -> Result<Vec<Vec<f32>>>;
}

/// One recognized token on the clock of its utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedToken {
    pub id: u32,
    pub text: String,
    /// Seconds from the start of the utterance. `None` when the model gives no token times.
    pub time: Option<f32>,
    /// Log probability of the token.
    pub log_prob: f32,
}

/// The tokens of one decode, in emission order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hypothesis {
    pub tokens: Vec<TimedToken>,
}

impl Hypothesis {
    pub fn text(&self) -> String {
        join_tokens(self.tokens.iter().map(|token| token.text.as_str()))
    }
}

/// Hotwords prepared for one dictation. Each backend uses the form it supports: a transducer
/// biases its beam search with the graph, and an LLM decoder reads the phrases in its prompt.
#[derive(Debug, Clone, Default)]
pub struct Bias {
    pub phrases: Vec<String>,
    pub graph: Option<ContextGraph>,
}

/// A speech recognizer that the pipeline drives.
///
/// Encoding and decoding are separate calls, because uncertainty decodes noisy copies of one
/// encoder output.
pub trait Asr: Send + Sync {
    fn bias(&self, phrases: &[String]) -> Bias;
    fn encode(&self, samples: &[f32]) -> Result<EncoderOutput>;
    /// Decodes each encoder output with the same bias, batching the model calls where it can.
    fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>>;
    /// Seconds of audio per encoder frame.
    fn frame_seconds(&self) -> f32;

    /// Uncertainty settings that suit this model's decoding cost and encoder output.
    fn monte_carlo(&self) -> MonteCarloConfig {
        MonteCarloConfig::default()
    }

    fn decode_one(&self, encoder: &EncoderOutput, bias: &Bias) -> Result<Hypothesis> {
        let mut decoded = self.decode(std::slice::from_ref(encoder), bias)?;
        Ok(decoded.pop().unwrap_or_default())
    }

    /// An encoder that reads audio as it arrives, for models that can. `None` makes the pipeline
    /// encode each segment once it ends.
    fn stream(&self) -> Option<Box<dyn EncoderStream + '_>> {
        None
    }
}

/// Encodes one dictation's audio while it is recorded.
///
/// The stream reads every sample since the dictation started, speech and silence alike, and the
/// pipeline takes each segment's frames out of it once they are ready. Sample positions count
/// from the first accepted sample.
pub trait EncoderStream {
    /// Adds 16 kHz samples and encodes what they complete.
    fn accept(&mut self, samples: &[f32]) -> Result<()>;
    /// Encodes the rest after the last sample.
    fn finish(&mut self) -> Result<()>;
    /// The encoder frames to decode for the segment of samples `start..end`, or `None` until the
    /// stream has encoded them. A streaming model writes a token some time after it hears the
    /// sound, so the frames run past the segment's end, and the next segment's frames begin
    /// where these end. Segments come in order.
    fn take(&mut self, start: u64, end: u64) -> Option<EncoderOutput>;
    /// No segment will start before sample `before`, so the stream may drop what lies there.
    fn discard(&mut self, before: u64) {
        let _ = before;
    }
}

/// Loads the recognizer in a sherpa-onnx model directory: Qwen3-ASR when the directory has its
/// files, otherwise a transducer searched with `search`, streaming when its encoder is. ONNX Runtime must be initialized first.
#[cfg(feature = "onnx")]
pub fn load(dir: &std::path::Path, threads: usize, search: SearchConfig) -> Result<Box<dyn Asr>> {
    if qwen::QwenFiles::is_qwen(dir) {
        let files = qwen::QwenFiles::discover(dir)?;
        let model = qwen::QwenAsr::load(&files, threads, qwen::QwenConfig::default())?;
        return Ok(Box::new(model));
    }
    let files = onnx::TransducerFiles::discover(dir, true)?;
    let encoder = onnx::load_session(&files.encoder, threads)?;
    if nemo::is_streaming_encoder(&encoder) {
        let language = search.language.clone();
        let model = nemo::NemoTransducer::load(encoder, &files, search, language.as_deref())?;
        Ok(Box::new(model))
    } else {
        let (model, symbols) = onnx::OnnxTransducer::with_encoder(encoder, &files)?;
        Ok(Box::new(Recognizer::new(model, symbols, search)))
    }
}

/// Joins model tokens into text. The BPE word marker `▁` becomes a space, and CJK characters join
/// without spaces.
pub fn join_tokens<'a>(tokens: impl IntoIterator<Item = &'a str>) -> String {
    let joined: String = tokens.into_iter().collect();
    let spaced = joined.replace('▁', " ");
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Byte ranges of the words of `text`. Whitespace before a word belongs to it. A CJK character is a
/// word of its own, and so is a punctuation mark, except an apostrophe or hyphen inside a word.
pub fn split_words(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut words = Vec::new();
    let mut start = 0;
    let mut inside = false;
    for (index, c) in text.char_indices() {
        if c.is_whitespace() {
            if inside {
                words.push(start..index);
                start = index;
                inside = false;
            }
            continue;
        }
        let joins = matches!(c, '\'' | '’' | '-') && inside;
        if is_cjk(c) || (is_punctuation(c) && !joins) {
            if inside {
                words.push(start..index);
                start = index;
            }
            words.push(start..index + c.len_utf8());
            start = index + c.len_utf8();
            inside = false;
            continue;
        }
        inside = true;
    }
    if inside {
        words.push(start..text.len());
    }
    words
}

/// Han, kana, and Hangul, which are written without spaces between words.
pub fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF)
}

pub fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '，' | '。'
                | '！'
                | '？'
                | '；'
                | '：'
                | '、'
                | '「'
                | '」'
                | '『'
                | '』'
                | '《'
                | '》'
                | '（'
                | '）'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '…'
                | '—'
                | '–'
                | '¿'
                | '¡'
                | '«'
                | '»'
                | '・'
                | '؟'
                | '،'
        )
}

/// Raw search output before token ids become text.
#[derive(Debug, Clone, Default)]
pub struct SearchResult {
    /// `(token id, encoder frame, log probability)` in emission order.
    pub tokens: Vec<(u32, usize, f32)>,
    pub score: f32,
}

/// Decodes speech segments with a transducer, a symbol table, and optional hotwords.
pub struct Recognizer<M> {
    model: M,
    symbols: SymbolTable,
    fbank: Fbank,
    pub search: SearchConfig,
}

impl<M: TransducerModel> Recognizer<M> {
    pub fn new(model: M, symbols: SymbolTable, search: SearchConfig) -> Self {
        Self {
            model,
            symbols,
            fbank: Fbank::new(FbankConfig::default()),
            search,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    /// Builds the hotword graph for this model's vocabulary. Phrases the tokenizer cannot spell are skipped.
    pub fn context_graph(&self, phrases: &[String], boost: f32) -> ContextGraph {
        let tokenizer = Tokenizer::new(&self.symbols);
        let sequences: Vec<Vec<u32>> = phrases
            .iter()
            .filter_map(|phrase| tokenizer.encode(phrase))
            .collect();
        ContextGraph::new(&sequences, boost)
    }

    pub fn encode(&self, samples: &[f32]) -> Result<EncoderOutput> {
        self.model.encode(&self.fbank.compute(samples))
    }

    /// Runs the configured search on encoder output.
    pub fn search(
        &self,
        encoder: &EncoderOutput,
        graph: Option<&ContextGraph>,
    ) -> Result<Hypothesis> {
        let result = if self.search.beam <= 1 && graph.is_none() {
            greedy_search(&self.model, encoder, &self.search)?
        } else {
            modified_beam_search(&self.model, encoder, graph, &self.search)?
        };
        Ok(self.to_hypothesis(&result))
    }

    /// Runs the search on several encoder outputs in lockstep, batching the model calls.
    pub fn search_batch(
        &self,
        encoders: &[EncoderOutput],
        graph: Option<&ContextGraph>,
    ) -> Result<Vec<Hypothesis>> {
        let results = modified_beam_search_batch(&self.model, encoders, graph, &self.search)?;
        Ok(results.iter().map(|r| self.to_hypothesis(r)).collect())
    }

    pub fn recognize(&self, samples: &[f32], graph: Option<&ContextGraph>) -> Result<Hypothesis> {
        self.search(&self.encode(samples)?, graph)
    }

    pub fn to_hypothesis(&self, result: &SearchResult) -> Hypothesis {
        let frame_seconds = self.model.frame_seconds();
        Hypothesis {
            tokens: result
                .tokens
                .iter()
                .map(|&(id, frame, log_prob)| TimedToken {
                    id,
                    text: self.symbols.text(id).to_owned(),
                    time: Some(frame as f32 * frame_seconds),
                    log_prob,
                })
                .collect(),
        }
    }
}

impl<M: TransducerModel> Asr for Recognizer<M> {
    fn bias(&self, phrases: &[String]) -> Bias {
        let graph = self.context_graph(phrases, self.search.hotword_boost);
        Bias {
            phrases: phrases.to_vec(),
            graph: (!graph.is_empty()).then_some(graph),
        }
    }

    fn encode(&self, samples: &[f32]) -> Result<EncoderOutput> {
        Recognizer::encode(self, samples)
    }

    fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
        match encoders {
            [single] => Ok(vec![self.search(single, bias.graph.as_ref())?]),
            _ => self.search_batch(encoders, bias.graph.as_ref()),
        }
    }

    fn frame_seconds(&self) -> f32 {
        self.model.frame_seconds()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<&str> {
        split_words(text).into_iter().map(|r| &text[r]).collect()
    }

    #[test]
    fn splits_words_punctuation_and_cjk() {
        assert_eq!(
            words("Okay, Charles. I'm Rap-Tourium!"),
            vec!["Okay", ",", " Charles", ".", " I'm", " Rap-Tourium", "!"]
        );
        assert_eq!(words("我流氓了。"), vec!["我", "流", "氓", "了", "。"]);
        assert_eq!(words(" 3.14 "), vec![" 3", ".", "14"]);
        assert!(words("  ").is_empty());
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! A transducer whose joiner reads the answer from the encoder frame, so search and
    //! uncertainty tests need no model files.

    use super::*;

    /// Encoder frames are one-hot rows of logits. The decoder output is ignored.
    pub struct ScriptedModel {
        pub vocab: usize,
    }

    impl ScriptedModel {
        /// Builds encoder output where frame `t` scores `script[t]` with logit `margin` and every
        /// other token with 0.
        pub fn encoder(&self, script: &[u32], margin: f32) -> EncoderOutput {
            let mut data = vec![0.0; script.len() * self.vocab];
            for (t, id) in script.iter().enumerate() {
                data[t * self.vocab + *id as usize] = margin;
            }
            EncoderOutput {
                frames: script.len(),
                dim: self.vocab,
                data,
            }
        }
    }

    impl TransducerModel for ScriptedModel {
        fn context_size(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn encode(&self, _: &Features) -> Result<EncoderOutput> {
            unimplemented!()
        }
        fn decode(&self, contexts: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
            Ok(contexts.iter().map(|_| vec![0.0; self.vocab]).collect())
        }
        fn join(&self, encoder: &[&[f32]], _: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
            Ok(encoder.iter().map(|frame| frame.to_vec()).collect())
        }
    }

    pub fn symbols(words: &[&str]) -> SymbolTable {
        SymbolTable::from_lines(words.iter().enumerate().map(|(i, w)| format!("{w} {i}")))
    }
}
