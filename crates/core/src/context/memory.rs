//! A short memory of the names on screen, so a dictation can use a name that was visible a moment
//! ago in another window or tab.
//!
//! A platform shell samples the screen every few seconds while no dictation runs and passes each
//! snapshot's ranked terms to [`TermMemory::observe`]. A term earns weight for the time it stays on
//! screen, its rank score times the seconds since the previous sample, and the weight halves every
//! [`MemoryConfig::half_life`]. So a name read for a minute outweighs one that flashed by, and
//! both fade within a few minutes. Only terms are kept, never the screen text, and only in memory.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{Term, TermForm};

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub half_life: Duration,
    /// A longer gap between samples counts as this long, so time asleep or locked earns nothing.
    pub max_gap: Duration,
    /// 0 remembers nothing.
    pub max_terms: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            half_life: Duration::from_secs(60),
            max_gap: Duration::from_secs(10),
            max_terms: 200,
        }
    }
}

/// Weight below which a term is forgotten: about four half-lives after a single glance.
const FORGET_WEIGHT: f32 = 0.02;

#[derive(Debug, Clone)]
struct Entry {
    phrase: String,
    form: TermForm,
    weight: f32,
    updated: Instant,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<String, Entry>,
    last_sample: Option<Instant>,
    /// Screen reading is off: nothing is kept and nothing is added.
    off: bool,
}

/// Recent screen terms with screen-time weights. Shared between the sampler and dictations.
#[derive(Debug, Default)]
pub struct TermMemory {
    pub config: MemoryConfig,
    state: Mutex<State>,
}

impl TermMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            state: Mutex::default(),
        }
    }

    fn decay(&self, weight: f32, elapsed: Duration) -> f32 {
        weight * 0.5f32.powf(elapsed.as_secs_f32() / self.config.half_life.as_secs_f32())
    }

    /// Adds one snapshot's terms, weighted by the time since the previous snapshot. A term on
    /// screen without a break settles near 1.44 times its rank score (1 / ln 2).
    pub fn observe(&self, terms: &[Term], now: Instant) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.off {
            return;
        }
        let gap = state
            .last_sample
            .map_or(Duration::ZERO, |last| now.saturating_duration_since(last))
            .min(self.config.max_gap);
        state.last_sample = Some(now);
        let share = gap.as_secs_f32() / self.config.half_life.as_secs_f32();
        if share == 0.0 {
            return;
        }
        for term in terms {
            let entry = state
                .entries
                .entry(term.phrase.to_lowercase())
                .or_insert_with(|| Entry {
                    phrase: term.phrase.clone(),
                    form: term.form,
                    weight: 0.0,
                    updated: now,
                });
            let weight = self.decay(entry.weight, now.saturating_duration_since(entry.updated));
            entry.weight = weight + term.score * share;
            entry.updated = now;
        }
        self.prune(&mut state, now);
    }

    fn prune(&self, state: &mut State, now: Instant) {
        if self.config.max_terms == 0 {
            state.entries.clear();
            return;
        }
        let current = |e: &Entry| self.decay(e.weight, now.saturating_duration_since(e.updated));
        state.entries.retain(|_, e| current(e) >= FORGET_WEIGHT);
        if state.entries.len() > self.config.max_terms {
            // Exactly the heaviest `max_terms`, even when weights tie at the cut.
            let mut ranked: Vec<(String, f32)> = state
                .entries
                .iter()
                .map(|(key, e)| (key.clone(), current(e)))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (key, _) in &ranked[self.config.max_terms..] {
                state.entries.remove(key);
            }
        }
    }

    /// The remembered terms with their current weights as scores, highest first.
    pub fn recall(&self, now: Instant) -> Vec<Term> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut terms: Vec<Term> = state
            .entries
            .values()
            .map(|e| Term {
                phrase: e.phrase.clone(),
                score: self.decay(e.weight, now.saturating_duration_since(e.updated)),
                form: e.form,
            })
            .filter(|t| t.score >= FORGET_WEIGHT)
            .collect();
        terms.sort_by(|a, b| b.score.total_cmp(&a.score));
        terms
    }

    /// Follows the screen reading switch. Off forgets everything and ignores later observations
    /// until it is on again. The switch and the memory change under one lock, so a snapshot read
    /// before the switch went off cannot land after the memory was cleared.
    pub fn set_enabled(&self, enabled: bool) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = State {
            off: !enabled,
            ..if enabled {
                std::mem::take(&mut *state)
            } else {
                State::default()
            }
        };
    }
}

/// The current screen's terms and the remembered ones in one ranking, at most `limit`. A term in
/// both adds its scores. On a tie the current screen comes first.
pub fn merge_terms(current: Vec<Term>, recalled: Vec<Term>, limit: usize) -> Vec<Term> {
    let mut merged: Vec<Term> = Vec::with_capacity(current.len() + recalled.len());
    let mut index: HashMap<String, usize> = HashMap::new();
    for term in current.into_iter().chain(recalled) {
        match index.get(&term.phrase.to_lowercase()) {
            Some(&i) => merged[i].score += term.score,
            None => {
                index.insert(term.phrase.to_lowercase(), merged.len());
                merged.push(term);
            }
        }
    }
    // A stable sort keeps the current screen's order on ties.
    merged.sort_by(|a, b| b.score.total_cmp(&a.score));
    merged.truncate(limit);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(phrase: &str, score: f32) -> Term {
        Term {
            phrase: phrase.into(),
            score,
            form: TermForm::Name,
        }
    }

    fn phrases(terms: &[Term]) -> Vec<&str> {
        terms.iter().map(|t| t.phrase.as_str()).collect()
    }

    #[test]
    fn screen_time_outweighs_a_glance() {
        let memory = TermMemory::default();
        let start = Instant::now();
        // Airicraft stays on screen for 50 s; Zustand shows in one sample at the end.
        for i in 0..=10 {
            let mut terms = vec![term("Airicraft", 1.0)];
            if i == 10 {
                terms.push(term("Zustand", 2.0));
            }
            memory.observe(&terms, start + Duration::from_secs(5 * i));
        }
        let recalled = memory.recall(start + Duration::from_secs(50));
        assert_eq!(phrases(&recalled), vec!["Airicraft", "Zustand"]);
        // About 1 - 0.5^(50/60) of the 1.44 a term settles at.
        assert!((recalled[0].score - 0.63).abs() < 0.05, "{recalled:?}");
    }

    #[test]
    fn terms_fade_and_are_forgotten() {
        let memory = TermMemory::default();
        let start = Instant::now();
        memory.observe(&[], start);
        memory.observe(&[term("Airicraft", 1.0)], start + Duration::from_secs(5));
        let fresh = memory.recall(start + Duration::from_secs(5))[0].score;
        let later = memory.recall(start + Duration::from_secs(65))[0].score;
        assert!((later - fresh / 2.0).abs() < 1e-4, "{fresh} {later}");
        assert!(memory.recall(start + Duration::from_secs(600)).is_empty());
    }

    #[test]
    fn a_long_gap_earns_no_more_than_max_gap() {
        let memory = TermMemory::default();
        let start = Instant::now();
        memory.observe(&[], start);
        // The Mac slept for an hour between the two samples.
        let woke = start + Duration::from_secs(3600);
        memory.observe(&[term("Airicraft", 1.0)], woke);
        let score = memory.recall(woke)[0].score;
        assert!((score - 10.0 / 60.0).abs() < 1e-4, "{score}");
    }

    #[test]
    fn keeps_only_the_heaviest_terms() {
        let memory = TermMemory::new(MemoryConfig {
            max_terms: 2,
            ..MemoryConfig::default()
        });
        let start = Instant::now();
        memory.observe(&[], start);
        memory.observe(
            &[term("A1", 1.0), term("B2", 3.0), term("C3", 2.0)],
            start + Duration::from_secs(5),
        );
        let recalled = memory.recall(start + Duration::from_secs(5));
        assert_eq!(phrases(&recalled), vec!["B2", "C3"]);
        // Ties at the cut still leave exactly `max_terms`.
        memory.observe(
            &[term("D4", 9.0), term("E5", 9.0), term("F6", 9.0)],
            start + Duration::from_secs(10),
        );
        assert_eq!(memory.recall(start + Duration::from_secs(10)).len(), 2);
        memory.set_enabled(false);
        assert!(memory.recall(start + Duration::from_secs(5)).is_empty());
        // A snapshot read before the switch went off lands after it: still nothing.
        memory.observe(&[term("A1", 1.0)], start + Duration::from_secs(10));
        assert!(memory.recall(start + Duration::from_secs(10)).is_empty());
        memory.set_enabled(true);
        memory.observe(&[term("A1", 1.0)], start + Duration::from_secs(15));
        memory.observe(&[term("A1", 1.0)], start + Duration::from_secs(20));
        assert_eq!(
            phrases(&memory.recall(start + Duration::from_secs(20))),
            vec!["A1"]
        );
    }

    #[test]
    fn merging_adds_scores_and_prefers_the_current_screen_on_ties() {
        let merged = merge_terms(
            vec![term("Kubernetes", 1.0), term("ArgoCD", 0.5)],
            vec![term("argocd", 1.0), term("Zustand", 1.0)],
            3,
        );
        assert_eq!(phrases(&merged), vec!["ArgoCD", "Kubernetes", "Zustand"]);
        assert_eq!(merged[0].score, 1.5);
    }
}
