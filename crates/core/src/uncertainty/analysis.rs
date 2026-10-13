//! Compares a reference decode with perturbed decodes and finds the spans where they disagree.
//!
//! Ported from the ASR uncertainty probe of AIRI (`stage-ui/libs/voice/asr-uncertainty/analysis.ts`).
//! There the perturbations change the audio, so token times are mapped back to the clip clock. Here
//! the noise goes on the encoder output, whose frames never move, so times compare directly.

use std::collections::{BTreeSet, HashMap};

use crate::asr::{join_tokens, Hypothesis, TimedToken};

/// How often the perturbed decodes reproduce one reference token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenAgreement {
    /// Fraction of samples that emit the same token near the same time, or match it in the alignment
    /// within [`AnalysisOptions::aligned_tolerance_seconds`].
    pub by_time: f32,
    /// Fraction of samples whose edit-distance alignment matches this token. An early insertion
    /// can shift it for the rest of the utterance, so spans use [`Self::by_time`].
    pub by_text: f32,
}

/// Disagreement of the samples around one point of the utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineBin {
    pub start: f32,
    /// Mean multiset Jaccard distance between each sample's tokens and the reference tokens near this time.
    pub disagreement: f32,
    /// Fraction of samples that emit a token here while the reference emits nothing nearby.
    pub extra_emission: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// The reference has tokens here, but most samples hear something else.
    Unstable,
    /// The reference has no tokens here, but many samples do.
    Dropped,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub text: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UncertainSpan {
    pub kind: SpanKind,
    pub start: f32,
    pub end: f32,
    /// Reference token range `[token_start, token_end)`. A dropped span covers no token, so both
    /// values are its insert position.
    pub token_start: usize,
    pub token_end: usize,
    /// What the samples heard in this span, most frequent first.
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Analysis {
    pub tokens: Vec<TokenAgreement>,
    pub timeline: Vec<TimelineBin>,
    pub spans: Vec<UncertainSpan>,
}

#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    /// Two tokens with the same text and times closer than this are the same token.
    pub tolerance_seconds: f32,
    /// Two tokens that the alignment matches are the same token up to this distance.
    pub aligned_tolerance_seconds: f32,
    pub bin_seconds: f32,
    /// A reference token with agreement below this fraction is part of an unstable span.
    pub agreement_threshold: f32,
    /// A bin where at least this fraction of samples emits extra tokens is part of a dropped span.
    pub extra_emission_threshold: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            tolerance_seconds: 0.2,
            aligned_tolerance_seconds: 0.6,
            bin_seconds: 0.05,
            agreement_threshold: 0.6,
            extra_emission_threshold: 0.4,
        }
    }
}

/// Comparison key of a token. Case, punctuation, and the BPE word marker do not change what was heard.
pub fn normalize_token(text: &str) -> String {
    text.chars()
        .filter(|c| *c != '▁' && !c.is_ascii_punctuation() && !is_cjk_punctuation(*c))
        .collect::<String>()
        .trim()
        .to_lowercase()
}

fn is_cjk_punctuation(c: char) -> bool {
    matches!(
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
            | '“'
            | '”'
            | '‘'
            | '’'
    )
}

fn has_text(token: &TimedToken) -> bool {
    !normalize_token(&token.text).is_empty()
}

#[derive(Debug, Clone, Copy)]
struct Pair {
    reference: Option<usize>,
    sample: Option<usize>,
    matched: bool,
}

/// Levenshtein alignment over non-empty keys. Ties prefer the diagonal, so a substitution stays one pair.
fn align(reference: &Hypothesis, sample: &Hypothesis) -> Vec<Pair> {
    let keyed = |h: &Hypothesis| -> Vec<(String, usize)> {
        h.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let key = normalize_token(&t.text);
                (!key.is_empty()).then_some((key, i))
            })
            .collect()
    };
    let (r, s) = (keyed(reference), keyed(sample));
    let columns = s.len() + 1;
    let mut cost = vec![0u32; (r.len() + 1) * columns];
    for row in 0..=r.len() {
        cost[row * columns] = row as u32;
    }
    for (column, value) in cost.iter_mut().take(columns).enumerate() {
        *value = column as u32;
    }
    for row in 1..=r.len() {
        for column in 1..=s.len() {
            let substitution =
                cost[(row - 1) * columns + column - 1] + u32::from(r[row - 1].0 != s[column - 1].0);
            let deletion = cost[(row - 1) * columns + column] + 1;
            let insertion = cost[row * columns + column - 1] + 1;
            cost[row * columns + column] = substitution.min(deletion).min(insertion);
        }
    }

    let mut pairs = Vec::new();
    let (mut row, mut column) = (r.len(), s.len());
    while row > 0 || column > 0 {
        let current = cost[row * columns + column];
        if row > 0 && column > 0 {
            let matched = r[row - 1].0 == s[column - 1].0;
            if current == cost[(row - 1) * columns + column - 1] + u32::from(!matched) {
                pairs.push(Pair {
                    reference: Some(r[row - 1].1),
                    sample: Some(s[column - 1].1),
                    matched,
                });
                row -= 1;
                column -= 1;
                continue;
            }
        }
        if row > 0 && current == cost[(row - 1) * columns + column] + 1 {
            pairs.push(Pair {
                reference: Some(r[row - 1].1),
                sample: None,
                matched: false,
            });
            row -= 1;
            continue;
        }
        pairs.push(Pair {
            reference: None,
            sample: Some(s[column - 1].1),
            matched: false,
        });
        column -= 1;
    }
    pairs.reverse();
    pairs
}

/// Pairs of reference and sample token indices that are the same token: the same text within
/// `tolerance_seconds`, wherever the alignment puts it, or an alignment match within `aligned_tolerance_seconds`.
fn correspondences(
    reference: &Hypothesis,
    sample: &Hypothesis,
    pairs: &[Pair],
    options: &AnalysisOptions,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for (ri, rt) in reference.tokens.iter().enumerate() {
        let key = normalize_token(&rt.text);
        if key.is_empty() {
            continue;
        }
        for (si, st) in sample.tokens.iter().enumerate() {
            if (st.time - rt.time).abs() <= options.tolerance_seconds
                && normalize_token(&st.text) == key
            {
                result.push((ri, si));
            }
        }
    }
    for pair in pairs {
        if let (true, Some(ri), Some(si)) = (pair.matched, pair.reference, pair.sample) {
            if (reference.tokens[ri].time - sample.tokens[si].time).abs()
                <= options.aligned_tolerance_seconds
            {
                result.push((ri, si));
            }
        }
    }
    result
}

/// Sample tokens that match no reference token and have no reference token of any text near them.
fn extra_tokens<'a>(
    reference: &Hypothesis,
    sample: &'a Hypothesis,
    matched: &BTreeSet<usize>,
    tolerance: f32,
) -> Vec<&'a TimedToken> {
    let reference_times: Vec<f32> = reference
        .tokens
        .iter()
        .filter(|t| has_text(t))
        .map(|t| t.time)
        .collect();
    sample
        .tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            has_text(t)
                && !matched.contains(i)
                && reference_times
                    .iter()
                    .all(|time| (time - t.time).abs() > tolerance)
        })
        .map(|(_, t)| t)
        .collect()
}

fn count_variants(texts: impl IntoIterator<Item = String>) -> Vec<Variant> {
    let mut counts: Vec<Variant> = Vec::new();
    for text in texts {
        match counts.iter_mut().find(|v| v.text == text) {
            Some(v) => v.count += 1,
            None => counts.push(Variant { text, count: 1 }),
        }
    }
    // Stable, so ties keep first-seen order.
    counts.sort_by_key(|v| std::cmp::Reverse(v.count));
    counts
}

/// Number of reference tokens that start before `seconds`. A span at that time is inserted there.
pub fn insert_position(reference: &Hypothesis, seconds: f32) -> usize {
    reference
        .tokens
        .iter()
        .position(|t| t.time >= seconds)
        .unwrap_or(reference.tokens.len())
}

pub fn analyze(
    reference: &Hypothesis,
    samples: &[Hypothesis],
    duration_seconds: f32,
    options: &AnalysisOptions,
) -> Analysis {
    let alignments: Vec<Vec<Pair>> = samples.iter().map(|s| align(reference, s)).collect();
    let matches: Vec<Vec<(usize, usize)>> = samples
        .iter()
        .zip(&alignments)
        .map(|(s, a)| correspondences(reference, s, a, options))
        .collect();

    let mut text_matches = vec![0usize; reference.tokens.len()];
    let mut time_matches = vec![0usize; reference.tokens.len()];
    for (pairs, found) in alignments.iter().zip(&matches) {
        for pair in pairs {
            if let (true, Some(ri)) = (pair.matched, pair.reference) {
                text_matches[ri] += 1;
            }
        }
        for ri in found.iter().map(|m| m.0).collect::<BTreeSet<_>>() {
            time_matches[ri] += 1;
        }
    }

    // With no samples nothing disagrees, so every token is fully stable.
    let fraction = |count: usize| {
        if samples.is_empty() {
            1.0
        } else {
            count as f32 / samples.len() as f32
        }
    };
    let tokens: Vec<TokenAgreement> = (0..reference.tokens.len())
        .map(|i| TokenAgreement {
            by_time: fraction(time_matches[i]),
            by_text: fraction(text_matches[i]),
        })
        .collect();

    let extras: Vec<Vec<&TimedToken>> = samples
        .iter()
        .zip(&matches)
        .map(|(s, found)| {
            extra_tokens(
                reference,
                s,
                &found.iter().map(|m| m.1).collect(),
                options.tolerance_seconds,
            )
        })
        .collect();
    let timeline = measure_timeline(reference, samples, &extras, duration_seconds, options);

    let mut spans = unstable_spans(reference, &tokens, samples, &alignments, options);
    spans.extend(dropped_spans(reference, &timeline, &extras, options));
    Analysis {
        tokens,
        timeline,
        spans,
    }
}

fn measure_timeline(
    reference: &Hypothesis,
    samples: &[Hypothesis],
    extras: &[Vec<&TimedToken>],
    duration: f32,
    options: &AnalysisOptions,
) -> Vec<TimelineBin> {
    let tolerance = options.tolerance_seconds;
    let keys_near = |h: &Hypothesis, center: f32| -> Vec<String> {
        h.tokens
            .iter()
            .filter(|t| (t.time - center).abs() <= tolerance && has_text(t))
            .map(|t| normalize_token(&t.text))
            .collect()
    };
    let mut bins = Vec::new();
    let mut index = 0;
    loop {
        let start = index as f32 * options.bin_seconds;
        if start >= duration {
            break;
        }
        let center = start + options.bin_seconds / 2.0;
        let reference_keys = keys_near(reference, center);
        let distance: f32 = samples
            .iter()
            .map(|s| jaccard_distance(&reference_keys, &keys_near(s, center)))
            .sum();
        // Half the tolerance keeps an extra token from widening the run by a full tolerance on both sides.
        let emitting = extras
            .iter()
            .filter(|tokens| {
                tokens
                    .iter()
                    .any(|t| (t.time - center).abs() <= tolerance / 2.0)
            })
            .count();
        let n = samples.len().max(1) as f32;
        bins.push(TimelineBin {
            start,
            disagreement: distance / n,
            extra_emission: emitting as f32 / n,
        });
        index += 1;
    }
    bins
}

/// Multiset Jaccard distance. Two empty sets agree.
fn jaccard_distance(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut remaining: HashMap<&str, usize> = HashMap::new();
    for key in a {
        *remaining.entry(key).or_default() += 1;
    }
    let mut intersection = 0;
    for key in b {
        if let Some(count) = remaining.get_mut(key.as_str()).filter(|c| **c > 0) {
            *count -= 1;
            intersection += 1;
        }
    }
    1.0 - intersection as f32 / (a.len() + b.len() - intersection) as f32
}

fn unstable_spans(
    reference: &Hypothesis,
    tokens: &[TokenAgreement],
    samples: &[Hypothesis],
    alignments: &[Vec<Pair>],
    options: &AnalysisOptions,
) -> Vec<UncertainSpan> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut previous_with_text: Option<usize> = None;
    for (index, agreement) in tokens.iter().enumerate() {
        // Tokens with no text, such as a lone word marker, neither start nor break a run.
        if !has_text(&reference.tokens[index]) {
            continue;
        }
        if agreement.by_time < options.agreement_threshold {
            match runs.last_mut() {
                Some(last) if previous_with_text == Some(last.1 - 1) => last.1 = index + 1,
                _ => runs.push((index, index + 1)),
            }
        }
        previous_with_text = Some(index);
    }

    widen_to_words(reference, runs)
        .into_iter()
        .map(|(token_start, token_end)| {
            // Inside a short span the stable tokens on both sides anchor the alignment, so its
            // pairs say what each sample heard there. Insertions inside the span belong to it.
            let heard = alignments.iter().zip(samples).map(|(pairs, sample)| {
                let first = pairs
                    .iter()
                    .position(|p| p.reference.is_some_and(|r| r >= token_start))
                    .unwrap_or(pairs.len());
                let after = pairs
                    .iter()
                    .position(|p| p.reference.is_some_and(|r| r >= token_end))
                    .unwrap_or(pairs.len());
                join_tokens(
                    pairs[first..after]
                        .iter()
                        .filter_map(|p| p.sample)
                        .map(|i| sample.tokens[i].text.as_str()),
                )
            });
            let times: Vec<f32> = reference.tokens[token_start..token_end]
                .iter()
                .map(|t| t.time)
                .collect();
            let min = times.iter().copied().fold(f32::INFINITY, f32::min);
            let max = times.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            UncertainSpan {
                kind: SpanKind::Unstable,
                start: (min - options.tolerance_seconds / 2.0).max(0.0),
                end: max + options.tolerance_seconds / 2.0,
                token_start,
                token_end,
                variants: count_variants(heard),
            }
        })
        .collect()
}

/// True when the token starts a word: it carries the BPE word marker, or it or the token before it
/// is CJK, where every character is its own unit.
fn starts_word(tokens: &[TimedToken], index: usize) -> bool {
    let cjk = |t: &TimedToken| {
        t.text.chars().any(|c| matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF))
    };
    index == 0
        || tokens[index].text.starts_with('▁')
        || cjk(&tokens[index])
        || cjk(&tokens[index - 1])
}

/// Grows token runs to whole words and merges the runs that then touch.
///
/// A BPE model can be unsure of one piece of a word, such as `THEL` in `BROTHELS`. The rewrite model
/// needs the whole word and what the other decodes heard for it, not a fragment.
fn widen_to_words(reference: &Hypothesis, runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let tokens = &reference.tokens;
    let mut widened: Vec<(usize, usize)> = Vec::new();
    for (mut start, mut end) in runs {
        while !starts_word(tokens, start) {
            start -= 1;
        }
        while end < tokens.len() && !starts_word(tokens, end) {
            end += 1;
        }
        match widened.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => widened.push((start, end)),
        }
    }
    widened
}

fn dropped_spans(
    reference: &Hypothesis,
    timeline: &[TimelineBin],
    extras: &[Vec<&TimedToken>],
    options: &AnalysisOptions,
) -> Vec<UncertainSpan> {
    let mut runs: Vec<(f32, f32)> = Vec::new();
    for bin in timeline
        .iter()
        .filter(|b| b.extra_emission >= options.extra_emission_threshold)
    {
        match runs.last_mut() {
            Some(last) if (last.1 - bin.start).abs() < options.bin_seconds / 2.0 => {
                last.1 = bin.start + options.bin_seconds
            }
            _ => runs.push((bin.start, bin.start + options.bin_seconds)),
        }
    }
    let half = options.tolerance_seconds / 2.0;
    runs.into_iter()
        .map(|(start, end)| {
            let position = insert_position(reference, start);
            let heard = extras
                .iter()
                .map(|tokens| {
                    join_tokens(
                        tokens
                            .iter()
                            .filter(|t| t.time >= start - half && t.time <= end + half)
                            .map(|t| t.text.as_str()),
                    )
                })
                .filter(|text| !text.is_empty());
            UncertainSpan {
                kind: SpanKind::Dropped,
                start,
                end,
                token_start: position,
                token_end: position,
                variants: count_variants(heard),
            }
        })
        .collect()
}

/// Renders the reference transcript with span markers, as the rewrite model reads it.
///
/// `我 [unclear: "流氓", also heard "硫磺"]`
pub fn format_for_llm(reference: &Hypothesis, spans: &[UncertainSpan]) -> String {
    let texts: Vec<&str> = reference.tokens.iter().map(|t| t.text.as_str()).collect();
    let mut inserts: Vec<&UncertainSpan> = spans
        .iter()
        .filter(|s| s.kind == SpanKind::Dropped)
        .collect();
    inserts.sort_by_key(|s| s.token_start);
    let mut parts: Vec<String> = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    let mut insert_index = 0;
    let mut index = 0;

    let flush = |pending: &mut Vec<&str>, parts: &mut Vec<String>| {
        if !pending.is_empty() {
            parts.push(join_tokens(pending.drain(..)));
        }
    };

    while index <= texts.len() {
        while insert_index < inserts.len() && inserts[insert_index].token_start <= index {
            flush(&mut pending, &mut parts);
            parts.push(describe(inserts[insert_index], ""));
            insert_index += 1;
        }
        if index == texts.len() {
            break;
        }
        match spans
            .iter()
            .find(|s| s.kind == SpanKind::Unstable && s.token_start == index)
        {
            Some(span) => {
                flush(&mut pending, &mut parts);
                parts.push(describe(
                    span,
                    &join_tokens(texts[span.token_start..span.token_end].iter().copied()),
                ));
                index = span.token_end;
            }
            None => {
                pending.push(texts[index]);
                index += 1;
            }
        }
    }
    flush(&mut pending, &mut parts);
    parts.join(" ")
}

fn describe(span: &UncertainSpan, reference_text: &str) -> String {
    let others: Vec<String> = span
        .variants
        .iter()
        .map(|v| v.text.as_str())
        .filter(|t| !t.is_empty() && *t != reference_text)
        .take(3)
        .map(|t| format!("\"{t}\""))
        .collect();
    match span.kind {
        SpanKind::Unstable if others.is_empty() => format!("[unclear: \"{reference_text}\"]"),
        SpanKind::Unstable => format!(
            "[unclear: \"{reference_text}\", also heard {}]",
            others.join(", ")
        ),
        SpanKind::Dropped if others.is_empty() => {
            format!("[possibly missed speech ~{:.1}s]", span.end - span.start)
        }
        SpanKind::Dropped => format!("[possibly missed speech: {}]", others.join(" or ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyp(tokens: &[(&str, f32)]) -> Hypothesis {
        Hypothesis {
            tokens: tokens
                .iter()
                .map(|(text, time)| TimedToken {
                    id: 0,
                    text: (*text).to_owned(),
                    frame: (*time / 0.04) as usize,
                    time: *time,
                    log_prob: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn unanimous_samples_leave_no_spans() {
        let reference = hyp(&[("▁HELLO", 0.2), ("▁WORLD", 0.6)]);
        let analysis = analyze(
            &reference,
            &[reference.clone(), reference.clone()],
            1.0,
            &AnalysisOptions::default(),
        );
        assert!(analysis.spans.is_empty());
        assert!(analysis.tokens.iter().all(|t| t.by_time == 1.0));
    }

    #[test]
    fn marks_a_token_that_most_samples_hear_differently() {
        let reference = hyp(&[("我", 0.1), ("流", 0.3), ("氓", 0.5), ("了", 0.7)]);
        let other = hyp(&[("我", 0.1), ("硫", 0.3), ("磺", 0.5), ("了", 0.7)]);
        let samples = vec![other.clone(), other.clone(), other, reference.clone()];
        let analysis = analyze(&reference, &samples, 1.0, &AnalysisOptions::default());
        assert_eq!(analysis.spans.len(), 1);
        let span = &analysis.spans[0];
        assert_eq!(
            (span.kind, span.token_start, span.token_end),
            (SpanKind::Unstable, 1, 3)
        );
        assert_eq!(
            span.variants[0],
            Variant {
                text: "硫磺".into(),
                count: 3
            }
        );
        assert_eq!(
            format_for_llm(&reference, &analysis.spans),
            "我 [unclear: \"流氓\", also heard \"硫磺\"] 了"
        );
    }

    #[test]
    fn early_insertion_does_not_shift_later_agreement() {
        let reference = hyp(&[("▁A", 0.2), ("▁B", 0.6), ("▁C", 1.0)]);
        let inserted = hyp(&[("▁X", 0.05), ("▁A", 0.2), ("▁B", 0.6), ("▁C", 1.0)]);
        let analysis = analyze(
            &reference,
            &[inserted.clone(), inserted],
            1.2,
            &AnalysisOptions::default(),
        );
        assert!(analysis.tokens.iter().all(|t| t.by_time == 1.0));
    }

    #[test]
    fn widens_an_unstable_piece_to_its_word() {
        let reference = hyp(&[
            ("▁THE", 0.1),
            ("▁B", 0.4),
            ("RO", 0.5),
            ("THEL", 0.6),
            ("S", 0.7),
            ("▁END", 1.0),
        ]);
        let other = hyp(&[
            ("▁THE", 0.1),
            ("▁B", 0.4),
            ("RO", 0.5),
            ("CKET", 0.6),
            ("S", 0.7),
            ("▁END", 1.0),
        ]);
        let analysis = analyze(
            &reference,
            &[other.clone(), other],
            1.2,
            &AnalysisOptions::default(),
        );
        assert_eq!(
            (analysis.spans[0].token_start, analysis.spans[0].token_end),
            (1, 5)
        );
        assert_eq!(
            format_for_llm(&reference, &analysis.spans),
            "THE [unclear: \"BROTHELS\", also heard \"BROCKETS\"] END"
        );
    }

    #[test]
    fn marks_speech_that_the_reference_dropped() {
        let reference = hyp(&[("▁I", 0.1), ("▁AGREE", 1.2)]);
        let heard = hyp(&[("▁I", 0.1), ("▁DO", 0.6), ("▁AGREE", 1.2)]);
        let analysis = analyze(
            &reference,
            &[heard.clone(), heard, reference.clone()],
            1.5,
            &AnalysisOptions::default(),
        );
        let dropped: Vec<_> = analysis
            .spans
            .iter()
            .filter(|s| s.kind == SpanKind::Dropped)
            .collect();
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].token_start, 1);
        assert_eq!(
            format_for_llm(&reference, &analysis.spans),
            "I [possibly missed speech: \"DO\"] AGREE"
        );
    }
}
