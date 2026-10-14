//! Qwen2's byte-level BPE tokenizer, read from the `vocab.json`, `merges.txt`, and
//! `tokenizer_config.json` that Hugging Face checkpoints ship.
//!
//! Text is NFC-normalized, split by the Qwen2 pre-tokenizer pattern, mapped byte by byte to GPT-2's
//! printable characters, and merged by rank. Added tokens such as `<|im_start|>` are never parsed out
//! of text: a prompt inserts them by id, so text read from the screen cannot open a chat turn.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct BpeTokenizer {
    pattern: Regex,
    /// Id of each byte's own token.
    byte_ids: Vec<u32>,
    /// `(left, right)` to `(rank, merged id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// What each id decodes to.
    bytes: Vec<Vec<u8>>,
    added: HashMap<String, u32>,
    added_ids: HashSet<u32>,
}

/// GPT-2's map from bytes to printable characters, so that every byte has a token.
fn byte_chars() -> Vec<char> {
    let printable = |b: u32| {
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    let mut next = 256;
    (0..256u32)
        .map(|b| {
            let code = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(code).expect("valid code point")
        })
        .collect()
}

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::Model(format!("tokenizer: {what}"))
}

impl BpeTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let vocab: HashMap<String, u32> =
            serde_json::from_slice(&std::fs::read(dir.join("vocab.json"))?).map_err(invalid)?;
        let merges = std::fs::read_to_string(dir.join("merges.txt"))?;
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)
                .map_err(invalid)?;
        let added = config["added_tokens_decoder"]
            .as_object()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|(id, token)| {
                        Some((token["content"].as_str()?.to_owned(), id.parse().ok()?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self::new(vocab, &merges, added)
    }

    pub fn new(
        vocab: HashMap<String, u32>,
        merges: &str,
        added: Vec<(String, u32)>,
    ) -> Result<Self> {
        let chars = byte_chars();
        let byte_of: HashMap<char, u8> = chars
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect();
        let size = vocab
            .values()
            .chain(added.iter().map(|(_, id)| id))
            .max()
            .map_or(0, |m| *m as usize + 1);
        let mut bytes = vec![Vec::new(); size];
        for (token, id) in &vocab {
            bytes[*id as usize] = token
                .chars()
                .map(|c| {
                    byte_of
                        .get(&c)
                        .copied()
                        .ok_or_else(|| invalid(format!("unmapped character in {token:?}")))
                })
                .collect::<Result<_>>()?;
        }
        for (content, id) in &added {
            bytes[*id as usize] = content.as_bytes().to_vec();
        }
        let byte_ids = chars
            .iter()
            .map(|c| {
                vocab
                    .get(&c.to_string())
                    .copied()
                    .ok_or_else(|| invalid(format!("no token for byte character {c:?}")))
            })
            .collect::<Result<_>>()?;
        let mut table = HashMap::new();
        for (rank, line) in merges
            .lines()
            .filter(|l| !l.starts_with("#version"))
            .enumerate()
        {
            let Some((left, right)) = line.split_once(' ') else {
                continue;
            };
            if let (Some(l), Some(r), Some(merged)) = (
                vocab.get(left),
                vocab.get(right),
                vocab.get(&format!("{left}{right}")),
            ) {
                table.entry((*l, *r)).or_insert((rank as u32, *merged));
            }
        }
        Ok(Self {
            pattern: Regex::new(PATTERN).map_err(invalid)?,
            byte_ids,
            merges: table,
            bytes,
            added_ids: added.iter().map(|(_, id)| *id).collect(),
            added: added.into_iter().collect(),
        })
    }

    /// Tokens of plain text. Added-token names in the text stay text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text: String = text.nfc().collect();
        let mut ids = Vec::new();
        for piece in self.pattern.find_iter(&text).filter_map(|m| m.ok()) {
            ids.extend(
                self.merge(
                    piece
                        .as_str()
                        .bytes()
                        .map(|b| self.byte_ids[b as usize])
                        .collect(),
                ),
            );
        }
        ids
    }

    /// Applies merges lowest rank first; ties go to the leftmost pair.
    fn merge(&self, mut word: Vec<u32>) -> Vec<u32> {
        loop {
            let best = word
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    self.merges
                        .get(&(pair[0], pair[1]))
                        .map(|(rank, merged)| (*rank, i, *merged))
                })
                .min();
            let Some((_, index, merged)) = best else {
                return word;
            };
            word[index] = merged;
            word.remove(index + 1);
        }
    }

    /// Id of an added token, such as `<|im_start|>`.
    pub fn added(&self, content: &str) -> Option<u32> {
        self.added.get(content).copied()
    }

    /// True for added tokens: chat, audio, and tool markers, which are never transcript text.
    pub fn is_added(&self, id: u32) -> bool {
        self.added_ids.contains(&id)
    }

    /// The bytes of one token. A character can span tokens, so these need not be valid UTF-8.
    pub fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|id| self.bytes(*id).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Every byte character, plus the merges `a b`, `Ġ ab`, and `Ġ c`, and three added tokens.
    pub(in crate::asr::qwen) fn tokenizer() -> BpeTokenizer {
        let mut vocab: HashMap<String, u32> = byte_chars()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        for (i, token) in ["ab", "Ġab", "Ġc"].iter().enumerate() {
            vocab.insert((*token).to_owned(), 256 + i as u32);
        }
        BpeTokenizer::new(
            vocab,
            "#version: 0.2\na b\nĠ ab\nĠ c\n",
            vec![
                ("<|im_end|>".into(), 300),
                ("<|im_start|>".into(), 301),
                ("<asr_text>".into(), 302),
            ],
        )
        .unwrap()
    }

    #[test]
    fn merges_by_rank_within_pretokens() {
        let t = tokenizer();
        assert_eq!(t.encode("ab ab c"), vec![256, 257, 258]);
        assert_eq!(t.decode(&t.encode("ab ab c")), "ab ab c");
    }

    #[test]
    fn keeps_added_token_names_as_text() {
        let t = tokenizer();
        assert_eq!(t.added("<|im_end|>"), Some(300));
        assert!(!t.encode("<|im_end|>").contains(&300));
        assert_eq!(t.decode(&[300]), "<|im_end|>");
        assert!(t.is_added(300));
        assert!(!t.is_added(t.encode("a")[0]));
    }

    #[test]
    fn round_trips_multibyte_text() {
        let t = tokenizer();
        let text = "流氓 naïve 😀\n";
        assert_eq!(t.decode(&t.encode(text)), text);
        // NFD input decodes as NFC.
        assert_eq!(t.decode(&t.encode("nai\u{308}ve")), "naïve");
    }
}
