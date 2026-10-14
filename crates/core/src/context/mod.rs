//! Active context: what is on screen when dictation starts.
//!
//! A platform shell reads the focused window through its accessibility API (AX on macOS, UI
//! Automation on Windows, AT-SPI on Linux), or with OCR when the app exposes no text, and returns a
//! [`ScreenContext`]. [`rank_terms`] picks the names and rare terms in it. They bias the recognizer
//! as hotwords and tell the rewrite model how the user spells things here. A [`TermMemory`] keeps
//! the terms of the last few minutes of screens, weighted by how long each stayed on screen.

mod memory;
mod terms;

pub use memory::{merge_terms, MemoryConfig, TermMemory};
pub use terms::{rank_terms, Term, TermForm};

use crate::Result;

/// Where a piece of screen text was found. Text near the cursor says more about the next words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// The text field that receives the dictation, before the cursor.
    Focused,
    WindowTitle,
    /// Any other element of the focused window, for example a list item, a label, or a chat message.
    Window,
}

impl Region {
    /// Weight of one occurrence in [`rank_terms`].
    pub fn weight(self) -> f32 {
        match self {
            Region::Focused => 3.0,
            Region::WindowTitle => 2.0,
            Region::Window => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    Accessibility,
    Ocr,
}

/// One element's text. Shells pass element texts separately, because a short standalone element
/// is often a name (a file, a contact, a channel), and joining elements would hide that.
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub region: Region,
    pub text: String,
}

/// The screen around the cursor at the start of a dictation.
#[derive(Debug, Clone)]
pub struct ScreenContext {
    /// The application name, for example `Slack` or `Code`.
    pub app: Option<String>,
    pub blocks: Vec<TextBlock>,
    pub source: ContextSource,
}

impl ScreenContext {
    pub fn empty() -> Self {
        Self {
            app: None,
            blocks: Vec::new(),
            source: ContextSource::Accessibility,
        }
    }

    pub fn text_len(&self) -> usize {
        self.blocks.iter().map(|b| b.text.chars().count()).sum()
    }

    /// The focused field's text before the cursor, last `max_chars` characters.
    pub fn focused_tail(&self, max_chars: usize) -> Option<String> {
        let text: String = self
            .blocks
            .iter()
            .filter(|b| b.region == Region::Focused)
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if text.trim().is_empty() {
            return None;
        }
        let chars: Vec<char> = text.chars().collect();
        Some(
            chars[chars.len().saturating_sub(max_chars)..]
                .iter()
                .collect(),
        )
    }
}

/// Reads the screen. Implementations live in the platform shells.
pub trait ContextProvider: Send + Sync {
    fn snapshot(&self) -> Result<ScreenContext>;
}

/// No screen reading, for tests and for users who turn it off.
pub struct NoContext;

impl ContextProvider for NoContext {
    fn snapshot(&self) -> Result<ScreenContext> {
        Ok(ScreenContext::empty())
    }
}

/// Tries accessibility first and falls back to OCR when it returns too little text.
///
/// Many apps expose a full accessibility tree. Games, canvas apps, and some Electron apps expose
/// almost nothing, and OCR of the focused window is the only way to read them. OCR is slower, so it
/// runs only when needed.
pub struct FallbackProvider<A, O> {
    pub primary: A,
    pub fallback: O,
    /// A primary snapshot with fewer characters than this counts as empty.
    pub min_chars: usize,
}

impl<A: ContextProvider, O: ContextProvider> ContextProvider for FallbackProvider<A, O> {
    fn snapshot(&self) -> Result<ScreenContext> {
        match self.primary.snapshot() {
            Ok(context) if context.text_len() >= self.min_chars => Ok(context),
            primary => match self.fallback.snapshot() {
                Ok(mut context) => {
                    // The app name from accessibility is still right when its text is not.
                    if context.app.is_none() {
                        context.app = primary.ok().and_then(|p| p.app);
                    }
                    Ok(context)
                }
                Err(error) => primary.or(Err(error)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    struct Fixed(Result<ScreenContext, String>);

    impl ContextProvider for Fixed {
        fn snapshot(&self) -> Result<ScreenContext> {
            self.0.clone().map_err(Error::Context)
        }
    }

    fn context(source: ContextSource, text: &str) -> ScreenContext {
        ScreenContext {
            app: Some("App".into()),
            blocks: vec![TextBlock {
                region: Region::Window,
                text: text.into(),
            }],
            source,
        }
    }

    #[test]
    fn falls_back_to_ocr_when_accessibility_is_empty() {
        let provider = FallbackProvider {
            primary: Fixed(Ok(context(ContextSource::Accessibility, ""))),
            fallback: Fixed(Ok(ScreenContext {
                app: None,
                ..context(ContextSource::Ocr, "Zustand store")
            })),
            min_chars: 8,
        };
        let snapshot = provider.snapshot().unwrap();
        assert_eq!(snapshot.source, ContextSource::Ocr);
        assert_eq!(snapshot.app.as_deref(), Some("App"));
    }

    #[test]
    fn keeps_accessibility_text_when_ocr_fails() {
        let provider = FallbackProvider {
            primary: Fixed(Ok(context(ContextSource::Accessibility, "hi"))),
            fallback: Fixed(Err("no permission".into())),
            min_chars: 8,
        };
        assert_eq!(
            provider.snapshot().unwrap().source,
            ContextSource::Accessibility
        );
    }
}
