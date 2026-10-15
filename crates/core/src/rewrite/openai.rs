use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use super::wav::base64_wav;
use super::{
    clean_reply, examples, user_prompt, RewriteRequest, Rewriter, AUDIO_NOTE, SYSTEM_PROMPT,
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
    /// Attach the recording to the request, for models that take audio input (Gemma 4 E2B and E4B
    /// through llama.cpp, for example).
    pub send_audio: bool,
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
    fn messages(&self, request: &RewriteRequest) -> Vec<serde_json::Value> {
        let audio = self.send_audio && !request.audio.is_empty();
        let system = if audio {
            format!("{SYSTEM_PROMPT}{AUDIO_NOTE}")
        } else {
            SYSTEM_PROMPT.to_owned()
        };
        let mut messages = vec![json!({ "role": "system", "content": system })];
        for (example, answer) in examples() {
            messages.push(json!({ "role": "user", "content": user_prompt(&example) }));
            messages.push(json!({ "role": "assistant", "content": answer }));
        }
        messages.push(if audio {
            json!({ "role": "user", "content": [
                { "type": "text", "text": user_prompt(request) },
                { "type": "input_audio", "input_audio": {
                    "data": base64_wav(&request.audio),
                    "format": "wav",
                } },
            ] })
        } else {
            json!({ "role": "user", "content": user_prompt(request) })
        });
        messages
    }
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
            "messages": self.messages(request),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn client(send_audio: bool) -> OpenAiCompatible {
        OpenAiCompatible {
            base_url: String::new(),
            api_key: None,
            model: "m".into(),
            timeout: Duration::from_secs(1),
            send_audio,
        }
    }

    #[test]
    fn sends_examples_before_the_transcript() {
        let request = RewriteRequest {
            marked_transcript: "hello".into(),
            audio: vec![0.0; 160],
            ..RewriteRequest::default()
        };
        let messages = client(false).messages(&request);
        assert_eq!(messages.len(), 2 + 2 * examples().len());
        assert_eq!(messages[0]["content"], SYSTEM_PROMPT);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        let last = messages.last().unwrap();
        assert_eq!(last["content"], user_prompt(&request));
    }

    #[test]
    fn attaches_the_recording_when_asked_and_present() {
        let mut request = RewriteRequest {
            marked_transcript: "hello".into(),
            audio: vec![0.0; 160],
            ..RewriteRequest::default()
        };
        let messages = client(true).messages(&request);
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .ends_with(AUDIO_NOTE));
        let parts = messages.last().unwrap()["content"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(parts[0]["text"], user_prompt(&request));
        assert_eq!(parts[1]["input_audio"]["format"], "wav");
        assert!(parts[1]["input_audio"]["data"]
            .as_str()
            .unwrap()
            .starts_with("UklGR"));

        // No audio, no attachment.
        request.audio.clear();
        let messages = client(true).messages(&request);
        assert_eq!(messages[0]["content"], SYSTEM_PROMPT);
        assert!(messages.last().unwrap()["content"].is_string());
    }
}
