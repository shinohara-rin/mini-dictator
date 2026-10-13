//! Picks hotwords from screen text.
//!
//! Adapted from AIRI's conversation term extraction (`stage-ui/libs/voice/transcription-bias.ts`).
//! A term qualifies when its written form marks it as specific: a quoted or bracketed span,
//! CamelCase, an acronym, a word with digits or joiners such as `Node.js`, or a run of capitalized
//! words inside a sentence. Plain lowercase words do not qualify, because nothing in their form
//! separates a name from ordinary speech. Screen text adds one form that chat text lacks: a short
//! CJK element on its own, such as a contact, a channel, or a file name.

use std::collections::{HashMap, HashSet};

use super::{ScreenContext, TextBlock};

const PHRASE_CHARS: usize = 32;
const PHRASE_WORDS: usize = 4;
/// A CJK label longer than this is a sentence, not a name.
const LABEL_CJK_CHARS: usize = 12;

const COMMON_CAPITALIZED: &[&str] = &[
    "I", "I'm", "I'll", "I've", "I'd", "OK", "Okay", "Yes", "No", "Oh", "Hi", "Hello", "Hey",
    "Thanks", "Sure", "The", "A", "An",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermForm {
    Quoted,
    Identifier,
    Name,
    Label,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub phrase: String,
    /// Sum of the region weights of its occurrences.
    pub score: f32,
    pub form: TermForm,
}

/// Ranks the terms of a screen snapshot, highest score first, at most `limit`.
pub fn rank_terms(context: &ScreenContext, limit: usize) -> Vec<Term> {
    // Capitalized words that also appear inside a sentence are names, and only these count at a sentence start.
    let names: HashSet<String> = context
        .blocks
        .iter()
        .flat_map(|block| terms_of(&block.text, &HashSet::new()))
        .flat_map(|(phrase, _)| {
            phrase
                .to_lowercase()
                .split(' ')
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();

    let mut scores: HashMap<String, (Term, usize)> = HashMap::new();
    let mut order = 0;
    for TextBlock { region, text } in &context.blocks {
        let mut found = terms_of(text, &names);
        if let Some(label) = cjk_label(text) {
            found.push((label, TermForm::Label));
        }
        for (phrase, form) in found {
            let key = phrase.to_lowercase();
            let entry = scores.entry(key).or_insert_with(|| {
                order += 1;
                (
                    Term {
                        phrase,
                        score: 0.0,
                        form,
                    },
                    order,
                )
            });
            entry.0.score += region.weight();
        }
    }
    let mut ranked: Vec<(Term, usize)> = scores.into_values().collect();
    ranked.sort_by(|a, b| b.0.score.total_cmp(&a.0.score).then(a.1.cmp(&b.1)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(term, _)| term)
        .collect()
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF)
}

/// A whole element text that is a short CJK name.
fn cjk_label(text: &str) -> Option<String> {
    let text = text.trim();
    let count = text.chars().count();
    let cjk = text.chars().filter(|c| is_cjk(*c)).count();
    (cjk >= 2
        && count <= LABEL_CJK_CHARS
        && !text
            .chars()
            .any(|c| c.is_whitespace() || "，。！？；：,.!?;:".contains(c)))
    .then(|| text.to_owned())
}

fn is_phrase(value: &str) -> bool {
    let chars = value.chars().count();
    (2..=PHRASE_CHARS).contains(&chars)
        && value.split_whitespace().count() <= PHRASE_WORDS
        && !value.ends_with(|c: char| ",.;:!?，。；：！？".contains(c))
}

const OPENERS: &str = "“「『《【\"`";
const CLOSERS: &str = "”」』》】\"`";

/// Quoted spans and the text with each span replaced by a word joiner, so the Latin scan does not
/// count its words again. The joiner is not white space, so it also breaks a run of names.
fn split_quoted(text: &str) -> (Vec<String>, String) {
    let mut spans = Vec::new();
    let mut rest = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if OPENERS.contains(chars[i]) {
            let end = (i + 1..chars.len().min(i + 2 + PHRASE_CHARS))
                .find(|j| CLOSERS.contains(chars[*j]) || chars[*j] == '\n');
            if let Some(j) = end.filter(|j| chars[*j] != '\n' && j - i > 2) {
                spans.push(chars[i + 1..j].iter().collect::<String>().trim().to_owned());
                rest.push('\u{2060}');
                i = j + 1;
                continue;
            }
        }
        rest.push(chars[i]);
        i += 1;
    }
    (spans, rest)
}

/// Splits after `.!?` followed by white space, and after `。！？` or a newline.
fn sentences(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let next_is_space = iter.peek().is_some_and(|(_, n)| n.is_whitespace());
        if "。！？\n".contains(c) || (".!?".contains(c) && next_is_space) {
            let end = i + c.len_utf8();
            result.push(&text[start..end]);
            start = end;
        }
    }
    result.push(&text[start..]);
    result
}

/// Latin words with inner joiners, as `(start, word)`. A word starts with a letter.
fn latin_words(sentence: &str) -> Vec<(usize, usize, &str)> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in sentence
        .char_indices()
        .chain(std::iter::once((sentence.len(), ' ')))
    {
        let joiner = c.is_ascii_alphanumeric() || "_'+#.-".contains(c);
        match (start, joiner) {
            (None, true) if c.is_ascii_alphabetic() => start = Some(i),
            (Some(s), false) => {
                let word = sentence[s..i].trim_end_matches(['.', '\'', '-']);
                words.push((s, s + word.len(), word));
                start = None;
            }
            _ => {}
        }
    }
    words
}

/// A written form that ordinary words lack, such as `useMemo`, `ASR`, `x86`, `Node.js`, or `C++`.
fn is_specific_form(word: &str) -> bool {
    if word.len() < 2 || COMMON_CAPITALIZED.contains(&word) {
        return false;
    }
    let bytes = word.as_bytes();
    let camel = bytes
        .windows(2)
        .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
    let acronym = {
        let core = word.strip_suffix('s').unwrap_or(word);
        core.len() >= 2 && core.bytes().all(|b| b.is_ascii_uppercase())
    };
    let digits =
        word.bytes().any(|b| b.is_ascii_digit()) && word.bytes().any(|b| b.is_ascii_alphabetic());
    let joined = bytes
        .windows(2)
        .any(|w| w[0].is_ascii_alphabetic() && b".+#".contains(&w[1]));
    camel || acronym || digits || joined
}

fn terms_of(text: &str, names: &HashSet<String>) -> Vec<(String, TermForm)> {
    let (quoted, unquoted) = split_quoted(text);
    let mut terms: Vec<(String, TermForm)> = quoted
        .into_iter()
        .filter(|s| is_phrase(s))
        .map(|s| (s, TermForm::Quoted))
        .collect();

    for sentence in sentences(&unquoted) {
        let mut run: Vec<&str> = Vec::new();
        let flush = |run: &mut Vec<&str>, terms: &mut Vec<(String, TermForm)>| {
            let phrase = run.join(" ");
            if !run.is_empty() && is_phrase(&phrase) {
                terms.push((phrase, TermForm::Name));
            }
            run.clear();
        };
        let mut previous_end = 0;
        for (start, end, word) in latin_words(sentence) {
            // A run of capitalized words breaks on anything but one space.
            if &sentence[previous_end..start] != " " {
                flush(&mut run, &mut terms);
            }
            previous_end = end;
            if is_specific_form(word) {
                flush(&mut run, &mut terms);
                terms.push((word.to_owned(), TermForm::Identifier));
                continue;
            }
            let capitalized = word.len() >= 2
                && word.as_bytes()[0].is_ascii_uppercase()
                && word.as_bytes()[1].is_ascii_lowercase();
            let sentence_start = sentence[..start].trim().is_empty();
            if capitalized
                && !COMMON_CAPITALIZED.contains(&word)
                && (!sentence_start || names.contains(&word.to_lowercase()))
            {
                run.push(word);
                continue;
            }
            flush(&mut run, &mut terms);
        }
        flush(&mut run, &mut terms);
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{ContextSource, Region};

    fn screen(blocks: &[(Region, &str)]) -> ScreenContext {
        ScreenContext {
            app: None,
            blocks: blocks
                .iter()
                .map(|(region, text)| TextBlock {
                    region: *region,
                    text: (*text).into(),
                })
                .collect(),
            source: ContextSource::Accessibility,
        }
    }

    fn phrases(context: &ScreenContext) -> Vec<String> {
        rank_terms(context, 100)
            .into_iter()
            .map(|t| t.phrase)
            .collect()
    }

    #[test]
    fn finds_identifiers_names_and_quotes() {
        let context = screen(&[(Region::Window, "I built a store with Zustand and React.js. Ask the Sulfur Cube fans about 「硫方怪」 and gpt-4o.")]);
        let found = phrases(&context);
        for expected in ["Zustand", "React.js", "Sulfur Cube", "硫方怪", "gpt-4o"] {
            assert!(
                found.contains(&expected.to_owned()),
                "{expected} missing from {found:?}"
            );
        }
        assert!(!found.iter().any(|p| p == "I" || p == "Ask"), "{found:?}");
    }

    #[test]
    fn a_sentence_start_counts_only_for_known_names() {
        let context = screen(&[(Region::Window, "Zustand is small. We like Zustand.")]);
        assert_eq!(phrases(&context), vec!["Zustand"]);
        assert!(phrases(&screen(&[(Region::Window, "Today we ship.")])).is_empty());
    }

    #[test]
    fn short_cjk_elements_are_labels() {
        let context = screen(&[
            (Region::Window, "王小明"),
            (Region::Window, "这是一个很长的句子，不是名字。"),
        ]);
        assert_eq!(phrases(&context), vec!["王小明"]);
    }

    #[test]
    fn focused_text_outranks_the_rest_of_the_window() {
        let context = screen(&[
            (Region::Window, "runs on Kubernetes"),
            (Region::Focused, "deploy with ArgoCD"),
        ]);
        assert_eq!(phrases(&context), vec!["ArgoCD", "Kubernetes"]);
    }
}
