use std::collections::{HashMap, VecDeque};

/// A hotword trie with Aho-Corasick failure links, after the context graph of icefall.
///
/// Each hypothesis in the beam carries a [`ContextState`]. A token that extends a partial hotword
/// earns `boost`. A token that breaks a partial match takes back the bonus that the broken prefix
/// earned, so a hypothesis cannot collect bonuses from prefixes it never finishes. A finished
/// hotword keeps `boost` for each of its tokens, also when it overlaps another hotword: with `A B`
/// and `B C`, the tokens `A B C` finish both.
///
/// Source: Yang et al., "Fast and parallel decoding for transducer" and icefall's `ContextGraph`,
/// https://github.com/k2-fsa/icefall/blob/master/icefall/context_graph.py
#[derive(Debug, Clone)]
pub struct ContextGraph {
    nodes: Vec<Node>,
    boost: f32,
}

#[derive(Debug, Clone, Default)]
struct Node {
    children: HashMap<u32, usize>,
    fail: usize,
    depth: usize,
    is_end: bool,
    /// Total length of the hotwords that end here as a proper suffix of this node's tokens.
    suffix_ends: usize,
}

/// Position of one hypothesis in the [`ContextGraph`]. The default is the root.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContextState {
    /// The longest hotword prefix that ends the hypothesis.
    node: usize,
    /// The last tokens of that prefix whose bonus is provisional: a break takes it back. The
    /// others already count toward a finished hotword.
    pending: usize,
}

impl ContextGraph {
    /// Builds the graph from hotwords spelled in token ids. `boost` is the log-score bonus per token.
    pub fn new(phrases: &[Vec<u32>], boost: f32) -> Self {
        let mut nodes = vec![Node::default()];
        for phrase in phrases.iter().filter(|p| !p.is_empty()) {
            let mut current = 0;
            for &token in phrase {
                current = match nodes[current].children.get(&token) {
                    Some(&next) => next,
                    None => {
                        let depth = nodes[current].depth + 1;
                        nodes.push(Node {
                            depth,
                            ..Node::default()
                        });
                        let next = nodes.len() - 1;
                        nodes[current].children.insert(token, next);
                        next
                    }
                };
            }
            nodes[current].is_end = true;
        }

        // Breadth-first, so a node's failure target is final before its children need it.
        let mut queue: VecDeque<usize> = nodes[0].children.values().copied().collect();
        while let Some(node) = queue.pop_front() {
            let children: Vec<(u32, usize)> =
                nodes[node].children.iter().map(|(t, c)| (*t, *c)).collect();
            for (token, child) in children {
                let mut fail = nodes[node].fail;
                let target = loop {
                    if let Some(&next) = nodes[fail].children.get(&token) {
                        break next;
                    }
                    if fail == 0 {
                        break 0;
                    }
                    fail = nodes[fail].fail;
                };
                nodes[child].fail = target;
                nodes[child].suffix_ends = nodes[target].suffix_ends
                    + if nodes[target].is_end {
                        nodes[target].depth
                    } else {
                        0
                    };
                queue.push_back(child);
            }
        }
        Self { nodes, boost }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    fn goto(&self, mut node: usize, token: u32) -> usize {
        loop {
            if let Some(&next) = self.nodes[node].children.get(&token) {
                return next;
            }
            if node == 0 {
                return 0;
            }
            node = self.nodes[node].fail;
        }
    }

    /// The score change and the next state after `token`.
    pub fn step(&self, state: ContextState, token: u32) -> (f32, ContextState) {
        let node = self.goto(state.node, token);
        let next = &self.nodes[node];
        // The new prefix keeps the pending tokens that it still covers, and adds this one.
        let mut pending = if node == 0 {
            0
        } else {
            state.pending.min(next.depth - 1) + 1
        };
        let mut gain = pending as f32 - state.pending as f32 + next.suffix_ends as f32;
        if next.is_end {
            // The prefix is a whole hotword: its pending bonus becomes final, and its other tokens,
            // which counted toward an overlapping hotword, count again for this one. The state
            // stays here, so the end of this hotword can start another.
            gain += (next.depth - pending) as f32;
            pending = 0;
        }
        (gain * self.boost, ContextState { node, pending })
    }

    /// The tokens that extend a hotword this state has started, so a search can weigh them even
    /// when the acoustics alone would prune them. Hotword starts are left to the acoustics.
    pub fn continuations(&self, state: ContextState) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut node = state.node;
        while node != 0 {
            tokens.extend(self.nodes[node].children.keys());
            node = self.nodes[node].fail;
        }
        tokens.sort_unstable();
        tokens.dedup();
        tokens
    }

    /// The score change at the end of the utterance: an unfinished prefix gives its bonus back.
    pub fn finish(&self, state: ContextState) -> f32 {
        -(state.pending as f32) * self.boost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuations_follow_started_hotwords_only() {
        let graph = ContextGraph::new(&[vec![1, 2, 3], vec![2, 4]], 1.0);
        let start = ContextState::default();
        assert!(graph.continuations(start).is_empty());
        let (_, after_1) = graph.step(start, 1);
        assert_eq!(graph.continuations(after_1), vec![2]);
        // `1 2` also starts `2 4`.
        let (_, after_12) = graph.step(after_1, 2);
        assert_eq!(graph.continuations(after_12), vec![3, 4]);
    }

    fn walk(graph: &ContextGraph, tokens: &[u32]) -> f32 {
        let mut state = ContextState::default();
        let mut total = 0.0;
        for &token in tokens {
            let (delta, next) = graph.step(state, token);
            total += delta;
            state = next;
        }
        total + graph.finish(state)
    }

    #[test]
    fn finished_hotword_keeps_its_bonus() {
        let graph = ContextGraph::new(&[vec![1, 2, 3]], 1.5);
        assert_eq!(walk(&graph, &[9, 1, 2, 3, 9]), 4.5);
    }

    #[test]
    fn broken_prefix_gives_its_bonus_back() {
        let graph = ContextGraph::new(&[vec![1, 2, 3]], 1.5);
        assert_eq!(walk(&graph, &[1, 2, 9]), 0.0);
        assert_eq!(walk(&graph, &[1, 2]), 0.0);
    }

    #[test]
    fn failure_link_keeps_an_overlapping_prefix() {
        // After 1 2, the token 4 breaks "1 2 3" but "2 4" continues from the shared token 2.
        let graph = ContextGraph::new(&[vec![1, 2, 3], vec![2, 4]], 1.0);
        assert_eq!(walk(&graph, &[1, 2, 4]), 2.0);
    }

    #[test]
    fn overlapping_hotwords_both_finish() {
        let graph = ContextGraph::new(&[vec![1, 2], vec![2, 3]], 1.0);
        assert_eq!(walk(&graph, &[1, 2, 3]), 4.0);
        assert_eq!(walk(&graph, &[2, 3]), 2.0);
        // A longer overlap: "2 3" ends "1 2 3" and starts "2 3 4".
        let graph = ContextGraph::new(&[vec![1, 2, 3], vec![2, 3, 4]], 1.0);
        assert_eq!(walk(&graph, &[1, 2, 3, 4]), 6.0);
        // The second hotword breaks, so only the first keeps its bonus.
        assert_eq!(walk(&graph, &[1, 2, 3, 9]), 3.0);
    }

    #[test]
    fn a_hotword_inside_a_longer_one_counts_on_its_own() {
        let graph = ContextGraph::new(&[vec![1, 2], vec![1, 2, 3, 4]], 1.0);
        assert_eq!(walk(&graph, &[1, 2, 3, 4]), 6.0);
        // The longer one breaks after 3, which gives back only the bonus of 3.
        assert_eq!(walk(&graph, &[1, 2, 3, 9]), 2.0);
        assert_eq!(walk(&graph, &[1, 2, 3]), 2.0);
        // A hotword that ends as a suffix of a longer prefix.
        let graph = ContextGraph::new(&[vec![1, 2, 3], vec![2]], 1.0);
        assert_eq!(walk(&graph, &[1, 2, 3]), 4.0);
        assert_eq!(walk(&graph, &[1, 2, 9]), 1.0);
    }

    #[test]
    fn the_score_never_counts_a_bonus_twice_on_the_way() {
        // The running score is the final bonuses plus the pending prefix, at every step.
        let graph = ContextGraph::new(&[vec![1, 2], vec![2, 3]], 1.0);
        let mut state = ContextState::default();
        let mut running = Vec::new();
        let mut total = 0.0;
        for token in [1, 2, 3] {
            let (delta, next) = graph.step(state, token);
            total += delta;
            state = next;
            running.push(total);
        }
        assert_eq!(running, [1.0, 2.0, 4.0]);
    }
}
