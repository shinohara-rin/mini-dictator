use std::collections::HashMap;
use std::path::Path;

use crate::{Error, Result};

/// The `tokens.txt` of a sherpa-onnx model: one `symbol id` pair per line.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    texts: Vec<String>,
    ids: HashMap<String, u32>,
}

impl SymbolTable {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let table = Self::from_lines(content.lines().map(str::to_owned));
        if table.texts.is_empty() {
            return Err(Error::Model(format!("no symbols in {}", path.display())));
        }
        Ok(table)
    }

    pub fn from_lines(lines: impl IntoIterator<Item = String>) -> Self {
        let mut table = Self::default();
        for line in lines {
            // The symbol itself can be a space in some tables, so split at the last space.
            let Some((symbol, id)) = line.trim_end().rsplit_once(' ') else {
                continue;
            };
            let Ok(id) = id.parse::<u32>() else { continue };
            if table.texts.len() <= id as usize {
                table.texts.resize(id as usize + 1, String::new());
            }
            table.texts[id as usize] = symbol.to_owned();
            table.ids.insert(symbol.to_owned(), id);
        }
        table
    }

    pub fn text(&self, id: u32) -> &str {
        self.texts.get(id as usize).map_or("", String::as_str)
    }

    pub fn id(&self, symbol: &str) -> Option<u32> {
        self.ids.get(symbol).copied()
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

/// Spells hotwords in model tokens with greedy longest match.
///
/// For CJK character models this is exact. For BPE models it approximates SentencePiece, which
/// is good enough for biasing: a hotword spelled with other pieces than the model would emit gets
/// no boost, and nothing worse. Text is upper-cased first when the table has no lower-case pieces,
/// because many English zipformer models use upper-case BPE.
pub struct Tokenizer<'a> {
    symbols: &'a SymbolTable,
    max_len: usize,
    upper: bool,
}

impl<'a> Tokenizer<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        let max_len = symbols
            .texts
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(1);
        // Special symbols such as `<blk>` and `<unk>` say nothing about the case of the pieces.
        let upper = !symbols
            .texts
            .iter()
            .filter(|s| !s.starts_with('<'))
            .any(|s| s.chars().any(char::is_lowercase));
        Self {
            symbols,
            max_len,
            upper,
        }
    }

    /// Token ids of `phrase`, or `None` when part of it has no token.
    pub fn encode(&self, phrase: &str) -> Option<Vec<u32>> {
        let phrase = if self.upper {
            phrase.to_uppercase()
        } else {
            phrase.to_owned()
        };
        let mut ids = Vec::new();
        for word in phrase.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            while start < chars.len() {
                let found = (1..=self.max_len.min(chars.len() - start))
                    .rev()
                    .find_map(|len| {
                        let piece: String = chars[start..start + len].iter().collect();
                        let marked = if start == 0 {
                            self.symbols.id(&format!("▁{piece}"))
                        } else {
                            None
                        };
                        marked
                            .or_else(|| self.symbols.id(&piece))
                            .map(|id| (id, len))
                    });
                let (id, len) = found?;
                ids.push(id);
                start += len;
            }
        }
        (!ids.is_empty()).then_some(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spells_bpe_words_with_the_word_marker() {
        let table = SymbolTable::from_lines(
            ["<blk> 0", "▁ZU 1", "STAND 2", "S 3", "▁ 4", "T 5"].map(String::from),
        );
        assert_eq!(Tokenizer::new(&table).encode("zustand"), Some(vec![1, 2]));
    }

    #[test]
    fn spells_cjk_characters() {
        let table = SymbolTable::from_lines(["<blk> 0", "硫 1", "方 2", "怪 3"].map(String::from));
        assert_eq!(Tokenizer::new(&table).encode("硫方怪"), Some(vec![1, 2, 3]));
        assert_eq!(Tokenizer::new(&table).encode("硫磺"), None);
    }
}
