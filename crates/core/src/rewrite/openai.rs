use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use super::{clean_reply, user_prompt, RewriteRequest, Rewriter, SYSTEM_PROMPT};
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

impl Rewriter for OpenAiCompatible {
    fn rewrite(&self, request: &RewriteRequest) -> Result<Option<String>> {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .build()
            .new_agent();
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": self.model,
            "temperature": 0,
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": user_prompt(request) },
            ],
        });
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
        let content = completion
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| Error::Rewrite("empty completion".into()))?;
        let text = clean_reply(&content);
        // An answer that was all reasoning or template would delete the dictation.
        if text.is_empty() && !request.marked_transcript.trim().is_empty() {
            return Err(Error::Rewrite("the reply has no text".into()));
        }
        Ok(Some(text))
    }
}
