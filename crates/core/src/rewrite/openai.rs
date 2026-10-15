use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use super::{
    clean_reply, parse_yes_no, user_prompt, RewriteRequest, Rewriter, SENTENCE_PROMPT,
    SYSTEM_PROMPT,
};
use crate::{Error, Result};

/// Any endpoint that speaks the OpenAI chat completions API: hosted providers, OpenRouter, or a
/// local server such as llama.cpp or Ollama.
pub struct OpenAiCompatible {
    /// Base URL, for example `https://api.openai.com/v1` or `http://localhost:11434/v1`.
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub timeout: Duration,
}

#[derive(Deserialize)]
struct Completion {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}

impl OpenAiCompatible {
    /// The reply's text for these messages.
    fn complete(&self, messages: serde_json::Value, max_tokens: Option<u32>) -> Result<String> {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .build()
            .new_agent();
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut body = json!({
            "model": self.model,
            "temperature": 0,
            "messages": messages,
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        let mut call = agent.post(&url);
        if let Some(key) = &self.api_key {
            call = call.header("Authorization", &format!("Bearer {key}"));
        }
        let completion: Completion = call
            .send_json(&body)
            .map_err(|e| Error::Rewrite(e.to_string()))?
            .body_mut()
            .read_json()
            .map_err(|e| Error::Rewrite(e.to_string()))?;
        completion
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| Error::Rewrite("empty completion".into()))
    }
}

impl Rewriter for OpenAiCompatible {
    fn rewrite(&self, request: &RewriteRequest) -> Result<Option<String>> {
        let content = self.complete(
            json!([
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": user_prompt(request) },
            ]),
            None,
        )?;
        let text = clean_reply(&content);
        // An answer that was all reasoning or template would delete the dictation.
        if text.is_empty() && !request.marked_transcript.trim().is_empty() {
            return Err(Error::Rewrite("the reply has no text".into()));
        }
        Ok(Some(text))
    }

    fn ends_sentence(&self, text: &str) -> Result<Option<bool>> {
        let content = self.complete(
            json!([
                { "role": "system", "content": SENTENCE_PROMPT },
                { "role": "user", "content": format!("<text>\n{text}\n</text>") },
            ]),
            Some(1),
        )?;
        Ok(parse_yes_no(&content))
    }
}
