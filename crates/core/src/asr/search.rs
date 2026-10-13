use std::collections::HashMap;

use super::{ContextGraph, ContextState, EncoderOutput, SearchResult, TransducerModel};
use crate::Result;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Hypotheses kept per frame. 1 with no hotwords runs greedy search.
    pub beam: usize,
    /// Subtracted from the blank logit. A positive value makes the model emit more tokens, which
    /// exposes speech that it would otherwise delete.
    pub blank_penalty: f32,
    /// Log-score bonus per hotword token.
    pub hotword_boost: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam: 4,
            blank_penalty: 0.0,
            hotword_boost: 1.5,
        }
    }
}

fn log_softmax(logits: &mut [f32], blank: usize, blank_penalty: f32) {
    logits[blank] -= blank_penalty;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln() + max;
    logits.iter_mut().for_each(|v| *v -= sum);
}

fn log_add(a: f32, b: f32) -> f32 {
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// Greedy search with at most one symbol per frame, as sherpa-onnx does for offline transducers.
pub fn greedy_search<M: TransducerModel + ?Sized>(
    model: &M,
    encoder: &EncoderOutput,
    config: &SearchConfig,
) -> Result<SearchResult> {
    let blank = model.blank_id();
    let mut context = vec![blank; model.context_size()];
    let mut decoder = model.decode(std::slice::from_ref(&context))?.remove(0);
    let mut result = SearchResult::default();
    for t in 0..encoder.frames {
        let mut logits = model.join(&[encoder.frame(t)], &[&decoder])?.remove(0);
        log_softmax(&mut logits, blank as usize, config.blank_penalty);
        let (best, log_prob) =
            logits
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |acc, (i, v)| {
                    if *v > acc.1 {
                        (i, *v)
                    } else {
                        acc
                    }
                });
        result.score += log_prob;
        if best as u32 != blank {
            result.tokens.push((best as u32, t, log_prob));
            context.remove(0);
            context.push(best as u32);
            decoder = model.decode(std::slice::from_ref(&context))?.remove(0);
        }
    }
    Ok(result)
}

#[derive(Debug, Clone)]
struct Hyp {
    /// Emitted token ids, after `context_size` leading blanks.
    ys: Vec<u32>,
    tokens: Vec<(u32, usize, f32)>,
    log_prob: f32,
    context_score: f32,
    context_state: ContextState,
}

impl Hyp {
    fn score(&self) -> f32 {
        self.log_prob + self.context_score
    }
}

/// Modified beam search (at most one symbol per frame) with optional hotword biasing.
///
/// Hypotheses with the same tokens merge, and their probabilities add, as in icefall.
pub fn modified_beam_search<M: TransducerModel + ?Sized>(
    model: &M,
    encoder: &EncoderOutput,
    graph: Option<&ContextGraph>,
    config: &SearchConfig,
) -> Result<SearchResult> {
    Ok(modified_beam_search_batch(model, std::slice::from_ref(encoder), graph, config)?.remove(0))
}

/// [`modified_beam_search`] over several encoder outputs at once.
///
/// Every frame makes one decoder call and one joiner call for all hypotheses of all utterances,
/// which matters for Monte Carlo, where a dozen noisy copies of one utterance decode together.
/// Decoder outputs are computed once per distinct token context.
pub fn modified_beam_search_batch<M: TransducerModel + ?Sized>(
    model: &M,
    encoders: &[EncoderOutput],
    graph: Option<&ContextGraph>,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let blank = model.blank_id();
    let context_size = model.context_size();
    let beam = config.beam.max(1);
    let initial = Hyp {
        ys: vec![blank; context_size],
        tokens: Vec::new(),
        log_prob: 0.0,
        context_score: 0.0,
        context_state: 0,
    };
    let mut streams: Vec<Vec<Hyp>> = vec![vec![initial]; encoders.len()];
    let frames = encoders.iter().map(|e| e.frames).max().unwrap_or(0);

    for t in 0..frames {
        // Utterances that already ended keep their hypotheses.
        let active: Vec<usize> = (0..encoders.len())
            .filter(|&i| t < encoders[i].frames)
            .collect();
        let mut context_index: HashMap<&[u32], usize> = HashMap::new();
        let mut contexts: Vec<Vec<u32>> = Vec::new();
        let mut rows: Vec<(usize, usize, usize)> = Vec::new();
        for &stream in &active {
            for (index, hyp) in streams[stream].iter().enumerate() {
                let context = &hyp.ys[hyp.ys.len() - context_size..];
                let slot = *context_index.entry(context).or_insert_with(|| {
                    contexts.push(context.to_vec());
                    contexts.len() - 1
                });
                rows.push((stream, index, slot));
            }
        }
        let decoder = model.decode(&contexts)?;
        let frame_refs: Vec<&[f32]> = rows
            .iter()
            .map(|&(stream, _, _)| encoders[stream].frame(t))
            .collect();
        let decoder_refs: Vec<&[f32]> = rows
            .iter()
            .map(|&(_, _, slot)| decoder[slot].as_slice())
            .collect();
        let logits = model.join(&frame_refs, &decoder_refs)?;

        let mut candidates: Vec<Vec<(f32, usize, u32, f32)>> = vec![Vec::new(); encoders.len()];
        for (&(stream, index, _), mut row) in rows.iter().zip(logits) {
            log_softmax(&mut row, blank as usize, config.blank_penalty);
            let mut order: Vec<usize> = (0..row.len()).collect();
            let top = beam.min(row.len());
            order.select_nth_unstable_by(top - 1, |a, b| row[*b].total_cmp(&row[*a]));
            for &token in &order[..top] {
                candidates[stream].push((
                    streams[stream][index].log_prob + row[token],
                    index,
                    token as u32,
                    row[token],
                ));
            }
        }

        for &stream in &active {
            let hyps = &streams[stream];
            let mut next: HashMap<Vec<u32>, Hyp> = HashMap::new();
            for &(log_prob, index, token, token_log_prob) in &candidates[stream] {
                let parent = &hyps[index];
                let mut hyp = parent.clone();
                hyp.log_prob = log_prob;
                if token != blank {
                    hyp.ys.push(token);
                    hyp.tokens.push((token, t, token_log_prob));
                    if let Some(graph) = graph {
                        let (delta, state) = graph.step(parent.context_state, token);
                        hyp.context_score += delta;
                        hyp.context_state = state;
                    }
                }
                match next.get_mut(&hyp.ys) {
                    Some(existing) => existing.log_prob = log_add(existing.log_prob, hyp.log_prob),
                    None => {
                        next.insert(hyp.ys.clone(), hyp);
                    }
                }
            }
            let mut merged: Vec<Hyp> = next.into_values().collect();
            merged.sort_by(|a, b| b.score().total_cmp(&a.score()));
            merged.truncate(beam);
            streams[stream] = merged;
        }
    }

    let finish = |hyp: &Hyp| hyp.score() + graph.map_or(0.0, |g| g.finish(hyp.context_state));
    Ok(streams
        .into_iter()
        .map(|hyps| {
            let best = hyps
                .into_iter()
                .max_by(|a, b| finish(a).total_cmp(&finish(b)))
                .expect("the beam is never empty");
            SearchResult {
                score: finish(&best),
                tokens: best.tokens,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::testing::ScriptedModel;
    use super::*;

    #[test]
    fn greedy_emits_the_scripted_tokens() {
        let model = ScriptedModel { vocab: 5 };
        let encoder = model.encoder(&[0, 2, 0, 3, 3, 0], 5.0);
        let result = greedy_search(&model, &encoder, &SearchConfig::default()).unwrap();
        let ids: Vec<(u32, usize)> = result.tokens.iter().map(|t| (t.0, t.1)).collect();
        assert_eq!(ids, vec![(2, 1), (3, 3), (3, 4)]);
    }

    #[test]
    fn beam_search_agrees_with_greedy_on_clear_input() {
        let model = ScriptedModel { vocab: 5 };
        let encoder = model.encoder(&[0, 2, 0, 3, 4, 0], 6.0);
        let greedy = greedy_search(&model, &encoder, &SearchConfig::default()).unwrap();
        let beam = modified_beam_search(&model, &encoder, None, &SearchConfig::default()).unwrap();
        assert_eq!(
            greedy.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
            beam.tokens.iter().map(|t| t.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batch_matches_one_by_one() {
        let model = ScriptedModel { vocab: 5 };
        let encoders = [
            model.encoder(&[0, 2, 0, 3], 6.0),
            model.encoder(&[1, 0, 4, 4, 0, 2], 6.0),
        ];
        let batch =
            modified_beam_search_batch(&model, &encoders, None, &SearchConfig::default()).unwrap();
        for (encoder, result) in encoders.iter().zip(&batch) {
            let single =
                modified_beam_search(&model, encoder, None, &SearchConfig::default()).unwrap();
            assert_eq!(single.tokens, result.tokens);
        }
    }

    #[test]
    fn hotword_boost_flips_a_close_call() {
        let model = ScriptedModel { vocab: 5 };
        // Frame 1 prefers token 2 slightly over token 4. The hotword "4 3" should win once boosted.
        let mut encoder = model.encoder(&[0, 2, 3, 0], 4.0);
        encoder.data[5 + 4] = 3.6;
        let plain = modified_beam_search(&model, &encoder, None, &SearchConfig::default()).unwrap();
        assert_eq!(
            plain.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![2, 3]
        );

        let graph = ContextGraph::new(&[vec![4, 3]], 1.0);
        let biased =
            modified_beam_search(&model, &encoder, Some(&graph), &SearchConfig::default()).unwrap();
        assert_eq!(
            biased.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
            vec![4, 3]
        );
    }

    #[test]
    fn blank_penalty_exposes_a_weak_token() {
        let model = ScriptedModel { vocab: 5 };
        let mut encoder = model.encoder(&[0, 0, 0], 4.0);
        encoder.data[5 + 2] = 3.0;
        let config = SearchConfig {
            blank_penalty: 2.0,
            ..SearchConfig::default()
        };
        assert!(greedy_search(&model, &encoder, &SearchConfig::default())
            .unwrap()
            .tokens
            .is_empty());
        assert_eq!(
            greedy_search(&model, &encoder, &config).unwrap().tokens[0].0,
            2
        );
    }
}
