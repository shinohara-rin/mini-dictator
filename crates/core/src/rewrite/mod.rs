//! The LLM rewrite step: turns a marked transcript into the text the user meant to type.

#[cfg(feature = "http")]
mod openai;

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
}

/// Rewrites a transcript. The HTTP implementation is [`OpenAiCompatible`]; shells can add others.
pub trait Rewriter: Send + Sync {
    fn rewrite(&self, request: &RewriteRequest) -> Result<String>;
}

/// Inserts the plain transcript. Used when no rewrite model is configured.
pub struct PassThrough;

impl Rewriter for PassThrough {
    fn rewrite(&self, request: &RewriteRequest) -> Result<String> {
        Ok(request.plain_transcript.clone())
    }
}

pub const SYSTEM_PROMPT: &str = "\
You turn raw speech recognition output into the text the user meant to type.

- Fix recognition errors, punctuation, capitalization, and spacing. Remove fillers, false starts, and repeated words.
- Keep the user's wording, language, and meaning. Do not answer, summarize, or add content.
- Spans marked [unclear: \"X\", also heard \"Y\"] are where the recognizer was unsure. Pick the reading that fits the context and the screen terms, or keep X.
- [possibly missed speech] marks audio the recognizer may have skipped. Insert a word there only when the context makes it obvious.
- Screen terms show how names are spelled where the user is typing. Use their spelling when the speech sounds like them.
- When the user dictates formatting, such as \"new line\" or \"bullet point\", apply it.
- Reply with the final text only, with no quotes or explanation.";

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
    }
    prompt.push_str(&format!(
        "Transcript:\n<transcript>\n{}\n</transcript>",
        request.marked_transcript
    ));
    prompt
}

/// Removes a wrapping that some models add despite the instructions.
pub fn clean_reply(reply: &str) -> String {
    let trimmed = reply.trim();
    let unwrapped = trimmed
        .strip_prefix("<transcript>")
        .and_then(|s| s.strip_suffix("</transcript>"))
        .unwrap_or(trimmed)
        .trim();
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
}
