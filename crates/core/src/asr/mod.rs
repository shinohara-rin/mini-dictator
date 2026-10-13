//! Transducer speech recognition.
//!
//! The core runs the transducer search itself instead of calling a packaged recognizer, because two
//! features need the parts of the model:
//!
//! - Hotword biasing scores partial matches inside the beam search ([`ContextGraph`]).
//! - Uncertainty decodes the same encoder output many times with noise, so it needs the encoder
//!   output on its own ([`crate::uncertainty`]).
//!
//! A [`TransducerModel`] has the three networks of a sherpa-onnx transducer: the encoder, the
//! stateless decoder (prediction network), and the joiner.

mod context_graph;
#[cfg(feature = "onnx")]
pub mod onnx;
mod search;
mod tokens;

pub use context_graph::{ContextGraph, ContextState};
pub use search::{greedy_search, modified_beam_search, modified_beam_search_batch, SearchConfig};
pub use tokens::{SymbolTable, Tokenizer};

use crate::features::{Fbank, FbankConfig, Features};
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
    /// Encoder frame that emitted the token.
    pub frame: usize,
    /// Seconds from the start of the utterance.
    pub time: f32,
    /// Log probability of the token at its frame.
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

/// Joins model tokens into text. The BPE word marker `▁` becomes a space, and CJK characters join
/// without spaces.
pub fn join_tokens<'a>(tokens: impl IntoIterator<Item = &'a str>) -> String {
    let joined: String = tokens.into_iter().collect();
    let spaced = joined.replace('▁', " ");
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
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
                    frame,
                    time: frame as f32 * frame_seconds,
                    log_prob,
                })
                .collect(),
        }
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
