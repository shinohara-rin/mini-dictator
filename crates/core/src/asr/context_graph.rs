use std::collections::{HashMap, VecDeque};

/// A hotword trie with Aho-Corasick failure links, after the context graph of icefall.
///
/// Each hypothesis in the beam carries a [`ContextState`]. A token that extends a partial hotword
/// earns `boost`. A token that breaks a partial match takes back the bonus that the broken prefix
/// earned, so a hypothesis cannot collect bonuses from prefixes it never finishes. A finished
/// hotword keeps its bonus.
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
}

/// Position of one hypothesis in the [`ContextGraph`]. The root is state 0.
pub type ContextState = usize;

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
                queue.push_back(child);
            }
        }
        Self { nodes, boost }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    fn goto(&self, mut state: ContextState, token: u32) -> ContextState {
        loop {
            if let Some(&next) = self.nodes[state].children.get(&token) {
                return next;
            }
            if state == 0 {
                return 0;
            }
            state = self.nodes[state].fail;
        }
    }

    /// The score change and the next state after `token`.
    pub fn step(&self, state: ContextState, token: u32) -> (f32, ContextState) {
        let next = self.goto(state, token);
        let delta = (self.nodes[next].depth as f32 - self.nodes[state].depth as f32) * self.boost;
        // A finished hotword keeps its bonus: the next state is the root, so no later step takes it back.
        let next = if self.nodes[next].is_end { 0 } else { next };
        (delta, next)
    }

    /// The score change at the end of the utterance: an unfinished prefix gives its bonus back.
    pub fn finish(&self, state: ContextState) -> f32 {
        -(self.nodes[state].depth as f32) * self.boost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(graph: &ContextGraph, tokens: &[u32]) -> f32 {
        let mut state = 0;
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
}
