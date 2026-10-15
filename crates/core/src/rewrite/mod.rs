//! The LLM rewrite step: turns a marked transcript into the text the user meant to type.

#[cfg(all(feature = "http", unix))]
mod llama;
#[cfg(feature = "http")]
mod openai;
#[cfg(feature = "http")]
mod wav;

#[cfg(all(feature = "http", unix))]
pub use llama::{Builtin, LlamaServer};
#[cfg(feature = "http")]
pub use openai::OpenAiCompatible;

use crate::Result;

/// Everything the rewrite model reads for one utterance.
#[derive(Debug, Clone, Default)]
pub struct RewriteRequest {
    /// The clean decode with `[unclear: ...]` and `[possibly missed speech ...]` markers.
    pub marked_transcript: String,
    /// The clean decode without markers, the fallback when the rewrite fails.
    pub plain_transcript: String,
    pub app: Option<String>,
    /// Terms from the screen, highest ranked first.
    pub screen_terms: Vec<String>,
    /// Text of the focused field before the cursor. Empty when screen text is not shared.
    pub preceding_text: String,
    /// Text already inserted in this dictation, so a sentence split by a pause continues.
    pub session_text: String,
    /// Seconds of silence between `session_text` and this transcript.
    pub pause_seconds: Option<f32>,
    /// The utterance at 16 kHz, for models that listen as well as read. Empty when not shared.
    pub audio: Vec<f32>,
}

/// Rewrites a transcript. The HTTP implementation is [`OpenAiCompatible`]; shells can add others.
pub trait Rewriter: Send + Sync {
    /// The rewritten text, or None when no model ran and the plain transcript stands.
    fn rewrite(&self, request: &RewriteRequest) -> Result<Option<String>>;
}

/// Inserts the plain transcript. Used when no rewrite model is configured.
pub struct PassThrough;

impl Rewriter for PassThrough {
    fn rewrite(&self, _: &RewriteRequest) -> Result<Option<String>> {
        Ok(None)
    }
}

pub const SYSTEM_PROMPT: &str = "\
You turn raw speech recognition output into the text the user meant to type.

- Fix recognition errors, punctuation, capitalization, and spacing. Remove fillers, false starts, and repeated words.
- Keep the user's wording, language, and meaning. Reply in the language of the transcript and never translate it. Do not answer, summarize, or add content.
- Spans marked [unclear: \"X\", also heard \"Y\"] are where the recognizer was unsure. Pick the reading that fits the context and the screen terms, or keep X.
- [possibly missed speech] marks audio the recognizer may have skipped. Insert a word there only when the context makes it obvious.
- Screen terms show how names are spelled where the user is typing. Use their spelling when the speech sounds like them.
- The user pauses while speaking, and each pause starts a new transcript. When text was already dictated, the transcript continues it. If it continues the same sentence, begin with a lowercase letter unless the first word is a name or \"I\", or with the punctuation that joins it, such as a comma. If it starts a new sentence, begin with a capital letter. A sentence often goes on after a short pause, and a long pause usually ends one. Never repeat the dictated text.
- When the user dictates formatting, such as \"new line\" or \"bullet point\", apply it.
- Reply with the final text only, with no quotes or explanation.";

/// Added to the system prompt when the recording goes with the transcript.
pub const AUDIO_NOTE: &str = "
- The recording of the utterance follows the transcript. Use it to check names, unclear spans, and words the recognizer may have missed or misheard. The transcript is still the text you edit: remove fillers, false starts, and repeated words even when you hear them, and write the utterance once.";

/// Worked examples sent as earlier turns before the request. Small models follow them much more
/// closely than the rules alone: they keep the language, stop echoing the prompt, and continue a
/// sentence after a pause.
pub fn examples() -> Vec<(RewriteRequest, &'static str)> {
    let example = |app: &str, terms: &[&str], session: &str, transcript: &str| RewriteRequest {
        marked_transcript: transcript.into(),
        app: Some(app.into()),
        screen_terms: terms.iter().map(|t| t.to_string()).collect(),
        session_text: session.into(),
        pause_seconds: (!session.is_empty()).then_some(0.7),
        ..RewriteRequest::default()
    };
    // The last example is the closest to the request, so it is English: a Chinese one there
    // pulled some replies into Chinese.
    vec![
        (
            example("WeChat", &[], "", "嗯那个周末我们去哪里吃饭"),
            "周末我们去哪里吃饭？",
        ),
        (
            example("Slack", &["TestFlight"], "", "えっとテストフライトのビルドはもう出しましたか"),
            "TestFlightのビルドはもう出しましたか？",
        ),
        (
            example(
                "Slack",
                &["Grafana", "Prometheus"],
                "",
                "uh can you check the [unclear: \"graph fauna\", also heard \"grafana\"] dashboard i think i think the alert is broken",
            ),
            "Can you check the Grafana dashboard? I think the alert is broken.",
        ),
        (
            example("Notes", &[], "", "shopping list new line apples new line coffee"),
            "Shopping list\nApples\nCoffee",
        ),
        (
            example(
                "Mail",
                &[],
                "I'll send the draft tonight,",
                "So you can review it tomorrow morning.",
            ),
            "so you can review it tomorrow morning.",
        ),
    ]
}

/// Builds the user message of a rewrite request.
pub fn user_prompt(request: &RewriteRequest) -> String {
    let mut prompt = String::new();
    if let Some(app) = &request.app {
        prompt.push_str(&format!("App: {app}\n"));
    }
    if !request.screen_terms.is_empty() {
        prompt.push_str(&format!(
            "Screen terms: {}\n",
            request.screen_terms.join(", ")
        ));
    }
    if !request.preceding_text.is_empty() {
        prompt.push_str(&format!(
            "Text before the cursor:\n<before>\n{}\n</before>\n",
            request.preceding_text
        ));
    }
    if !request.session_text.is_empty() {
        prompt.push_str(&format!(
            "Already dictated in this session:\n<dictated>\n{}\n</dictated>\n",
            request.session_text
        ));
        if let Some(pause) = request.pause_seconds {
            prompt.push_str(&format!("Pause before the transcript: {pause:.1} s\n"));
        }
    }
    prompt.push_str(&format!(
        "Transcript:\n<transcript>\n{}\n</transcript>",
        request.marked_transcript
    ));
    prompt
}

/// Removes what some models and servers add around the answer: chat template tokens that leak
/// through (`<|im_start|>assistant`, `<|im_end|>`), a reasoning block (`<think>...</think>`), a
/// `<transcript>` wrapping, or quotes. A reasoning block that never closes leaves nothing. A reply
/// that says the same text twice in a row keeps one copy: with the recording attached, Gemma 4
/// sometimes writes what it heard and then the edited transcript, word for word the same.
pub fn clean_reply(reply: &str) -> String {
    let mut text = reply.trim();
    // A role line and a reasoning block can come in either order.
    loop {
        if let Some(rest) = text.strip_prefix("<|im_start|>") {
            text = rest.split_once('\n').map_or("", |(_, answer)| answer);
        } else if let Some(rest) = text.strip_prefix("<think>") {
            text = rest.split_once("</think>").map_or("", |(_, answer)| answer);
        } else {
            break;
        }
        text = text.trim_start();
    }
    // The end of the turn, or a new turn the model started on its own.
    for end in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
        if let Some((answer, _)) = text.split_once(end) {
            text = answer;
        }
    }
    let trimmed = text.trim();
    let unwrapped = trimmed
        .strip_prefix("<transcript>")
        .and_then(|s| s.strip_suffix("</transcript>"))
        .unwrap_or(trimmed)
        .trim();
    let unwrapped = said_once(unwrapped);
    let quoted = unwrapped.len() >= 2
        && ((unwrapped.starts_with('"') && unwrapped.ends_with('"'))
            || (unwrapped.starts_with('“') && unwrapped.ends_with('”')));
    if quoted {
        let mut chars = unwrapped.chars();
        chars.next();
        chars.next_back();
        chars.as_str().to_owned()
    } else {
        unwrapped.to_owned()
    }
}

/// The first half of a text that is one copy repeated twice, with or without whitespace between.
fn said_once(text: &str) -> &str {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    // Try each split near the middle: the copies can be joined by whitespace.
    let half = chars.len() / 2;
    for mid in half.saturating_sub(2)..=(half + 2).min(chars.len()) {
        let Some(&(at, _)) = chars.get(mid) else {
            continue;
        };
        let (first, second) = (text[..at].trim_end(), text[at..].trim_start());
        if first.chars().count() >= 4 && first == second {
            return first;
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_carries_context_and_markers() {
        let prompt = user_prompt(&RewriteRequest {
            marked_transcript: "use [unclear: \"zoo stand\", also heard \"zustand\"] for state"
                .into(),
            app: Some("Code".into()),
            screen_terms: vec!["Zustand".into(), "React".into()],
            ..RewriteRequest::default()
        });
        assert!(prompt.starts_with("App: Code\nScreen terms: Zustand, React\n"));
        assert!(prompt.ends_with("<transcript>\nuse [unclear: \"zoo stand\", also heard \"zustand\"] for state\n</transcript>"));
    }

    #[test]
    fn strips_quotes_and_tags_from_replies() {
        assert_eq!(clean_reply("\"Hello there.\"\n"), "Hello there.");
        assert_eq!(clean_reply("<transcript>\nHi\n</transcript>"), "Hi");
        assert_eq!(clean_reply("He said \"hi\""), "He said \"hi\"");
    }

    #[test]
    fn keeps_one_copy_of_a_doubled_reply() {
        assert_eq!(
            clean_reply("抜群の運動神経を持ち合わせた。抜群の運動神経を持ち合わせた。"),
            "抜群の運動神経を持ち合わせた。"
        );
        assert_eq!(clean_reply("I agree. I agree."), "I agree.");
        assert_eq!(clean_reply("Hello there.\nHello there."), "Hello there.");
        // Short repeats and partial repeats are left alone.
        assert_eq!(clean_reply("no no"), "no no");
        assert_eq!(
            clean_reply("Go now. Go now, please."),
            "Go now. Go now, please."
        );
    }

    #[test]
    fn strips_chat_tokens_and_reasoning_from_replies() {
        assert_eq!(
            clean_reply("<|im_start|>assistant\nHello.<|im_end|>"),
            "Hello."
        );
        assert_eq!(
            clean_reply("Hello.<|im_end|>\n<|im_start|>user\nmore"),
            "Hello."
        );
        assert_eq!(clean_reply("Hello.<|im_start|>assistant\nagain"), "Hello.");
        assert_eq!(
            clean_reply("<think>\nfix the name\n</think>\n\nHello."),
            "Hello."
        );
        assert_eq!(clean_reply("<think>\n\n</think>\n\n\"Hello.\""), "Hello.");
        assert_eq!(clean_reply("<think>still thinking"), "");
        assert_eq!(
            clean_reply("<think>fix it</think>\n<|im_start|>assistant\nFixed.<|im_end|>"),
            "Fixed."
        );
        assert_eq!(
            clean_reply("<|im_start|>assistant\n<think>fix it</think>\nFixed.<|im_end|>"),
            "Fixed."
        );
    }
}
