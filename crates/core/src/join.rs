//! Joins the text of pause-split segments into one passage.
//!
//! The recognizer hears each segment alone, so it writes every one as a full sentence: "If we can.
//! Use speculative decoding. Or other hacks." A segment's final period is therefore held back
//! until the next segment shows whether the sentence went on. It is typed when the next segment
//! starts a new sentence or the dictation ends, and dropped when the next segment continues it.
//!
//! The rewrite model sees the text typed so far and the pause, and says which it is by how it
//! starts its reply: lowercase continues the sentence, a capital starts a new one. Without a
//! rewrite, the pause decides: a sentence goes on across a pause shorter than
//! [`JoinConfig::sentence_pause_seconds`], and its next word loses the capital the recognizer gave it.

use crate::audio::SAMPLE_RATE;
use crate::vad::{SegmenterConfig, SpeechSegment};

#[derive(Debug, Clone)]
pub struct JoinConfig {
    /// Without a rewrite, a pause at least this long ends a sentence and a shorter one does not.
    pub sentence_pause_seconds: f32,
}

impl Default for JoinConfig {
    fn default() -> Self {
        Self {
            sentence_pause_seconds: 1.5,
        }
    }
}

/// One segment's text, ready to join.
pub struct Piece<'a> {
    pub text: &'a str,
    /// Silence between the previous typed piece and this one, in seconds. None for the first.
    pub pause_seconds: Option<f32>,
    /// The rewrite model wrote this text, so its first letter's case was chosen with context.
    pub rewritten: bool,
    /// Words whose capitals are kept, such as names from the screen.
    pub names: &'a [String],
}

/// What one dictation has typed, for joining the next piece.
#[derive(Debug, Clone, Default)]
pub struct Joiner {
    /// The final period of the last piece, not typed yet.
    held: &'static str,
    /// Closing quotes or brackets after that period, typed whether or not the sentence goes on.
    closing: String,
    /// The sample where the speech of the last typed piece ended.
    speech_end: Option<u64>,
}

impl Joiner {
    /// The silence before `segment`'s speech, not counting the audio the segmenter keeps around it.
    pub fn pause_before(&self, segment: &SpeechSegment, config: &SegmenterConfig) -> Option<f32> {
        let (start, _) = speech_span(segment, config);
        self.speech_end
            .map(|end| start.saturating_sub(end) as f32 / SAMPLE_RATE as f32)
    }

    /// Records that `segment`'s text was typed.
    pub fn typed(&mut self, segment: &SpeechSegment, config: &SegmenterConfig) {
        self.speech_end = Some(speech_span(segment, config).1);
    }

    /// The text to type for `piece` after `typed`: a held period when the piece starts a new
    /// sentence, a separating space, and the piece without its own final period.
    pub fn join(&mut self, typed: &str, piece: &Piece, config: &JoinConfig) -> String {
        let (body, period, closing) = split_final_period(piece.text.trim());
        let continues = piece.pause_seconds.is_some_and(|pause| {
            // An opening quote or bracket says nothing; the letter after it does.
            let first = body.chars().find(|&c| !is_opening_punctuation(c));
            if body.chars().next().is_some_and(is_joining_punctuation)
                || first.is_some_and(char::is_lowercase)
            {
                true
            } else if piece.rewritten
                && first.is_some_and(char::is_uppercase)
                && !keeps_capital(first_word(body), piece.names)
            {
                // "I", a name, or an acronym keeps its capital mid-sentence, so only the pause
                // can tell.
                false
            } else {
                // A question or exclamation already ended its sentence.
                let ended = typed
                    .trim_end()
                    .trim_end_matches(is_closing_punctuation)
                    .ends_with(['?', '!', '？', '！']);
                !ended && pause < config.sentence_pause_seconds
            }
        });
        let mut text = if continues {
            std::mem::take(&mut self.closing)
        } else {
            self.finish()
        };
        self.held = period;
        self.closing = closing.to_owned();
        let body = if continues && !piece.rewritten {
            lowercase_first_word(body, piece.names)
        } else {
            body.to_owned()
        };
        // Pieces join with a space unless the text is CJK, which has none between sentences, or
        // the piece opens with punctuation that attaches to the word before.
        let last = text.chars().last().or_else(|| typed.chars().last());
        if let (Some(last), Some(first)) = (last, body.chars().next()) {
            if !last.is_whitespace()
                && !is_cjk(last)
                && !is_cjk(first)
                && !is_joining_punctuation(first)
            {
                text.push(' ');
            }
        }
        text.push_str(&body);
        text
    }

    /// The held period and the closing quotes after it, typed when the dictation ends.
    pub fn finish(&mut self) -> String {
        let closing = std::mem::take(&mut self.closing);
        format!("{}{closing}", std::mem::take(&mut self.held))
    }
}

/// Where a segment's speech starts and ends, without its pre-roll and post-roll. A segment cut at
/// the length limit has no post-roll, and the part after it no pre-roll; counting them anyway
/// makes the pause between them a quarter second, still short.
fn speech_span(segment: &SpeechSegment, config: &SegmenterConfig) -> (u64, u64) {
    let ms = |ms: u32| ms as u64 * SAMPLE_RATE as u64 / 1000;
    let end = segment.start_sample + segment.samples.len() as u64;
    let start = (segment.start_sample + ms(config.pre_roll_ms)).min(end);
    let end = if segment.truncated {
        end
    } else {
        end.saturating_sub(ms(config.post_roll_ms)).max(start)
    };
    (start, end)
}

/// Splits off a final period, Latin or CJK, with any closing quotes or brackets after it:
/// `He said “fast.”` gives `He said “fast` and `.`, `”`. An ellipsis stays, and so does the period
/// of an abbreviation like "Dr.", and `?` and `!`, which the recognizer rarely adds where a
/// sentence goes on.
fn split_final_period(text: &str) -> (&str, &'static str, &str) {
    let inner = text.trim_end_matches(is_closing_punctuation);
    let closing = &text[inner.len()..];
    for period in [".", "。", "．"] {
        if let Some(body) = inner.strip_suffix(period) {
            let last_word = body.rsplit(char::is_whitespace).next().unwrap_or("");
            if !body.ends_with(['.', '。', '．', '…'])
                && !body.trim().is_empty()
                && !is_abbreviation(last_word)
            {
                return (body.trim_end(), period, closing);
            }
        }
    }
    (text, "", "")
}

/// Words whose period belongs to them, not to the sentence.
fn is_abbreviation(word: &str) -> bool {
    const ABBREVIATIONS: [&str; 16] = [
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "inc", "ltd",
        "co", "no",
    ];
    ABBREVIATIONS
        .iter()
        .any(|a| a.eq_ignore_ascii_case(word.trim_start_matches(is_opening_punctuation)))
}

/// Punctuation that closes a quote or an aside.
fn is_closing_punctuation(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '”' | '’' | ')' | ']' | '»' | '」' | '』' | '）'
    )
}

fn first_word(text: &str) -> &str {
    text.trim_start_matches(is_opening_punctuation)
        .split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
        .next()
        .unwrap_or("")
}

/// "I", "I'm", "I'll" and the like keep their capital wherever they are.
fn is_i(word: &str) -> bool {
    word == "I" || word.starts_with("I'") || word.starts_with("I’")
}

/// True when a capitalized word keeps its capital mid-sentence: "I", one of `names`, or a word
/// with other capitals (an acronym or a name like "McDonald").
fn keeps_capital(word: &str, names: &[String]) -> bool {
    is_i(word)
        || word.chars().skip(1).any(char::is_uppercase)
        || names.iter().any(|name| {
            name.split_whitespace()
                .any(|n| n.eq_ignore_ascii_case(word))
        })
}

/// Lowercases the first letter unless the word [keeps its capital](keeps_capital).
fn lowercase_first_word(text: &str, names: &[String]) -> String {
    let word = first_word(text);
    let Some(first) = word.chars().next() else {
        return text.to_owned();
    };
    if !first.is_uppercase() || keeps_capital(word, names) || !text.starts_with(word) {
        return text.to_owned();
    }
    first
        .to_lowercase()
        .chain(text[first.len_utf8()..].chars())
        .collect()
}

/// Punctuation that opens a quote, an aside, or a list item.
fn is_opening_punctuation(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '“' | '‘' | '(' | '[' | '«' | '「' | '『' | '（' | '-' | '*' | '•'
    )
}

/// Punctuation that attaches to the word before it.
fn is_joining_punctuation(c: char) -> bool {
    matches!(
        c,
        ',' | ';' | ':' | '.' | '!' | '?' | ')' | '…' | '，' | '、' | '；' | '：' | '）'
    )
}

pub(crate) fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xFF00..=0xFFEF)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Joins pieces typed after the given pauses, as the recognizer or rewrite wrote them.
    fn dictate(pieces: &[(f32, &str)], rewritten: bool) -> String {
        let mut joiner = Joiner::default();
        let mut typed = String::new();
        let names = ["Rin".to_owned(), "Sulfur Cube".to_owned()];
        for (i, (pause, text)) in pieces.iter().enumerate() {
            let piece = Piece {
                text,
                pause_seconds: (i > 0).then_some(*pause),
                rewritten,
                names: &names,
            };
            let text = joiner.join(&typed, &piece, &JoinConfig::default());
            typed.push_str(&text);
        }
        typed.push_str(&joiner.finish());
        typed
    }

    #[test]
    fn a_short_pause_continues_the_sentence() {
        assert_eq!(
            dictate(
                &[
                    (0.0, "If we can."),
                    (0.8, "Use speculative decoding."),
                    (0.7, "Or other hacks to speed up.")
                ],
                false
            ),
            "If we can use speculative decoding or other hacks to speed up."
        );
    }

    #[test]
    fn closing_quotes_stay_after_a_held_period() {
        assert_eq!(
            dictate(
                &[(0.0, "He called it “fast.”"), (0.7, "and reliable.")],
                false
            ),
            "He called it “fast” and reliable."
        );
        assert_eq!(
            dictate(&[(0.0, "He called it “fast.”"), (2.0, "Then left.")], false),
            "He called it “fast.” Then left."
        );
        assert_eq!(dictate(&[(0.0, "「速い。」")], false), "「速い。」");
    }

    #[test]
    fn a_long_pause_ends_the_sentence() {
        assert_eq!(
            dictate(&[(0.0, "That works."), (2.0, "Next, the icon.")], false),
            "That works. Next, the icon."
        );
        // So does a question mark, however short the pause.
        assert_eq!(
            dictate(&[(0.0, "Are you ready?"), (0.7, "Yes.")], false),
            "Are you ready? Yes."
        );
    }

    #[test]
    fn names_acronyms_and_i_keep_their_capitals() {
        assert_eq!(
            dictate(
                &[
                    (0.0, "Ask."),
                    (0.7, "Rin about it."),
                    (0.7, "I think."),
                    (0.7, "ONNX is fine.")
                ],
                false
            ),
            "Ask Rin about it I think ONNX is fine."
        );
    }

    #[test]
    fn the_rewrite_model_decides_by_its_first_letter() {
        // A capital starts a sentence even after a short pause, lowercase goes on after a long one.
        assert_eq!(
            dictate(
                &[
                    (0.0, "That works."),
                    (0.7, "Next, the icon."),
                    (2.0, "and the menu.")
                ],
                true
            ),
            "That works. Next, the icon and the menu."
        );
        // "I", a name or an acronym says nothing, so the pause decides.
        assert_eq!(
            dictate(&[(0.0, "Sure."), (0.7, "I can.")], true),
            "Sure I can."
        );
        assert_eq!(
            dictate(
                &[
                    (0.0, "I think."),
                    (0.7, "Rin can help."),
                    (2.0, "ONNX works.")
                ],
                true
            ),
            "I think Rin can help. ONNX works."
        );
        assert_eq!(
            dictate(&[(0.0, "Ask."), (0.7, "Cube can help.")], true),
            "Ask Cube can help."
        );
        // An opening quote does not hide the capital after it.
        assert_eq!(
            dictate(
                &[(0.0, "That works."), (0.7, "“Next step,” he said.")],
                true
            ),
            "That works. “Next step,” he said."
        );
        // A reply that opens with punctuation attaches to the text before it.
        assert_eq!(
            dictate(&[(0.0, "Apples."), (0.7, ", pears, and plums.")], true),
            "Apples, pears, and plums."
        );
    }

    #[test]
    fn questions_and_ellipses_are_typed_at_once() {
        let mut joiner = Joiner::default();
        let piece = |text| Piece {
            text,
            pause_seconds: None,
            rewritten: false,
            names: &[],
        };
        let config = JoinConfig::default();
        assert_eq!(joiner.join("", &piece("Ready?"), &config), "Ready?");
        assert_eq!(joiner.join("", &piece("Well..."), &config), "Well...");
        assert_eq!(joiner.join("", &piece("Done."), &config), "Done");
        assert_eq!(joiner.finish(), ".");
        assert_eq!(
            joiner.join("", &piece("He called it “fast.”"), &config),
            "He called it “fast"
        );
        assert_eq!(joiner.finish(), ".”");
        assert_eq!(joiner.join("", &piece("Ask Dr."), &config), "Ask Dr.");
        assert_eq!(joiner.finish(), "");
    }

    #[test]
    fn cjk_pieces_join_without_spaces() {
        assert_eq!(
            dictate(
                &[
                    (0.0, "我们可以。"),
                    (0.7, "用推测解码。"),
                    (2.0, "然后呢。")
                ],
                false
            ),
            "我们可以用推测解码。然后呢。"
        );
    }

    #[test]
    fn pauses_leave_out_the_pre_roll_and_post_roll() {
        let config = SegmenterConfig::default();
        let segment = |start_sample: u64, seconds: f32, truncated| SpeechSegment {
            samples: vec![0.0; (seconds * SAMPLE_RATE as f32) as usize],
            start_sample,
            truncated,
        };
        let mut joiner = Joiner::default();
        let first = segment(0, 2.0, false);
        assert_eq!(joiner.pause_before(&first, &config), None);
        joiner.typed(&first, &config);
        // Speech ended at 1.85 s and the next starts at 2.5 + 0.25 s.
        let pause = joiner
            .pause_before(&segment(40_000, 1.0, false), &config)
            .unwrap();
        assert!((pause - 0.9).abs() < 1e-3, "{pause}");

        // The part after a cut at the length limit follows without a pause worth a period.
        let cut = segment(0, 20.0, true);
        joiner.typed(&cut, &config);
        let pause = joiner
            .pause_before(&segment(320_000, 1.0, false), &config)
            .unwrap();
        assert!(
            pause < JoinConfig::default().sentence_pause_seconds,
            "{pause}"
        );
    }
}
