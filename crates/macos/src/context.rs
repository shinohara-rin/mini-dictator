//! Reads the focused window through the macOS accessibility API.
//!
//! The reader collects the focused field's text before the cursor, the window title, and the text
//! of other elements within an element, character, and time budget. Web content comes first, in
//! document order, since that is where a page's names are; the rest of the window follows breadth
//! first. Each element costs one request for all the attributes it needs, because every request is
//! a round trip to the app.
//!
//! Chromium browsers and Electron apps build their accessibility tree only when asked, so the reader
//! sets `AXManualAccessibility` on the focused app first. When a web area still has no content, the
//! app either ignores that attribute or is building its tree just now: the reader sets
//! `AXEnhancedUserInterface`, which is what VoiceOver sets, and waits a moment for the content.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use mini_dictator_core::context::{
    ContextProvider, ContextSource, Region, ScreenContext, TextBlock,
};
use mini_dictator_core::{Error, Result};
use objc2_application_services::{
    kAXTrustedCheckOptionPrompt, AXCopyMultipleAttributeOptions, AXError, AXIsProcessTrusted,
    AXIsProcessTrustedWithOptions, AXUIElement, AXValue, AXValueType,
};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFRange, CFRetained, CFString, CFType,
};

pub struct AccessibilityReader {
    pub max_elements: usize,
    pub max_chars: usize,
    pub budget: Duration,
    /// Longest wait for a web area that has no content yet.
    pub web_wait: Duration,
}

impl Default for AccessibilityReader {
    fn default() -> Self {
        Self {
            max_elements: 1500,
            max_chars: 20_000,
            budget: Duration::from_millis(150),
            web_wait: Duration::from_millis(300),
        }
    }
}

/// The attributes read from each element, in one request. The first text attribute present is the
/// element's text.
const NODE_ATTRIBUTES: [&str; 5] = [
    "AXRole",
    "AXChildren",
    "AXValue",
    "AXTitle",
    "AXDescription",
];

struct Node {
    role: Option<String>,
    text: Option<String>,
    children: Vec<CFRetained<AXUIElement>>,
}

fn non_empty(value: &CFType) -> Option<String> {
    let text = value.downcast_ref::<CFString>()?.to_string();
    (!text.trim().is_empty()).then_some(text)
}

fn element_array(value: CFRetained<CFType>) -> Vec<CFRetained<AXUIElement>> {
    let Ok(array) = value.downcast::<CFArray>() else {
        return Vec::new();
    };
    let array: &CFArray<CFType> = unsafe { array.cast_unchecked() };
    array
        .iter()
        .filter_map(|child| child.downcast::<AXUIElement>().ok())
        .collect()
}

/// Reads an element's role, children, and text. A missing attribute comes back as an error value,
/// which is neither a string nor an array, so it is skipped.
fn node(element: &AXUIElement, names: &CFArray<CFString>) -> Node {
    let mut values: *const CFArray = std::ptr::null();
    let error = unsafe {
        element.copy_multiple_attribute_values(
            names.as_opaque(),
            AXCopyMultipleAttributeOptions(0),
            NonNull::from(&mut values),
        )
    };
    let values = (error == AXError::Success)
        .then(|| NonNull::new(values as *mut CFArray))
        .flatten()
        .map(|ptr| unsafe { CFRetained::from_raw(ptr) });
    let Some(values) = values else {
        return Node {
            role: None,
            text: None,
            children: Vec::new(),
        };
    };
    let values: &CFArray<CFType> = unsafe { values.cast_unchecked() };
    Node {
        role: values.get(0).as_deref().and_then(non_empty),
        children: values.get(1).map(element_array).unwrap_or_default(),
        text: (2..NODE_ATTRIBUTES.len()).find_map(|i| values.get(i).as_deref().and_then(non_empty)),
    }
}

/// True when the app may use the accessibility API. With `prompt`, macOS shows its permission dialog.
pub fn is_trusted(prompt: bool) -> bool {
    if !prompt {
        return unsafe { AXIsProcessTrusted() };
    }
    let key: &CFString = unsafe { kAXTrustedCheckOptionPrompt };
    let options = CFDictionary::<CFString, CFBoolean>::from_slices(&[key], &[CFBoolean::new(true)]);
    unsafe { AXIsProcessTrustedWithOptions(Some(options.as_opaque())) }
}

/// The focused window and its title: a cheap check for whether the screen changed, two requests
/// instead of a full read. Two windows with the same title still differ, and a tab switch that
/// keeps the window changes its title.
#[derive(PartialEq)]
pub struct FocusedWindow {
    /// The window, or the app when it has no focused window. Equal when it is the same element.
    element: CFRetained<AXUIElement>,
    title: String,
}

pub fn focused_window() -> Option<FocusedWindow> {
    let system = unsafe { AXUIElement::new_system_wide() };
    unsafe { system.set_messaging_timeout(0.1) };
    let app = element(&system, "AXFocusedApplication")?;
    let element = element(&app, "AXFocusedWindow").unwrap_or(app);
    let title = string(&element, "AXTitle").unwrap_or_default();
    Some(FocusedWindow { element, title })
}

fn attribute(element: &AXUIElement, name: &str) -> Option<CFRetained<CFType>> {
    let name = CFString::from_str(name);
    let mut value: *const CFType = std::ptr::null();
    let error = unsafe { element.copy_attribute_value(&name, NonNull::from(&mut value)) };
    if error != AXError::Success {
        return None;
    }
    NonNull::new(value as *mut CFType).map(|ptr| unsafe { CFRetained::from_raw(ptr) })
}

fn string(element: &AXUIElement, name: &str) -> Option<String> {
    non_empty(&*attribute(element, name)?)
}

fn element(element: &AXUIElement, name: &str) -> Option<CFRetained<AXUIElement>> {
    attribute(element, name)?.downcast::<AXUIElement>().ok()
}

fn children(element: &AXUIElement) -> Vec<CFRetained<AXUIElement>> {
    attribute(element, "AXChildren")
        .map(element_array)
        .unwrap_or_default()
}

fn set_true(element: &AXUIElement, name: &str) -> AXError {
    unsafe { element.set_attribute_value(&CFString::from_str(name), CFBoolean::new(true).as_ref()) }
}

/// The focused field's text before the cursor. `AXSelectedTextRange` counts UTF-16 units.
fn text_before_cursor(focused: &AXUIElement) -> Option<String> {
    let value = string(focused, "AXValue")?;
    let range = attribute(focused, "AXSelectedTextRange")
        .and_then(|v| v.downcast::<AXValue>().ok())
        .and_then(|v| {
            let mut range = CFRange {
                location: 0,
                length: 0,
            };
            unsafe { v.value(AXValueType::CFRange, NonNull::from(&mut range).cast()) }
                .then_some(range)
        });
    let Some(range) = range else {
        return Some(value);
    };
    let units: Vec<u16> = value
        .encode_utf16()
        .take(range.location.max(0) as usize)
        .collect();
    Some(String::from_utf16_lossy(&units))
}

impl ContextProvider for AccessibilityReader {
    fn snapshot(&self) -> Result<ScreenContext> {
        if !is_trusted(false) {
            return Err(Error::Context(
                "accessibility permission is not granted".into(),
            ));
        }
        let started = Instant::now();
        let system = unsafe { AXUIElement::new_system_wide() };
        unsafe { system.set_messaging_timeout(0.1) };
        let app = element(&system, "AXFocusedApplication")
            .ok_or_else(|| Error::Context("no focused application".into()))?;
        // Electron and Chromium build their tree on request. Other apps ignore the attribute.
        set_true(&app, "AXManualAccessibility");
        let names: Vec<CFRetained<CFString>> = NODE_ATTRIBUTES
            .iter()
            .map(|n| CFString::from_str(n))
            .collect();
        let names = CFArray::from_retained_objects(&names);

        let mut blocks = Vec::new();
        let focused = element(&system, "AXFocusedUIElement");
        if let Some(text) = focused.as_deref().and_then(text_before_cursor) {
            blocks.push(TextBlock {
                region: Region::Focused,
                text,
            });
        }
        let window = element(&app, "AXFocusedWindow");
        if let Some(title) = window.as_deref().and_then(|w| string(w, "AXTitle")) {
            blocks.push(TextBlock {
                region: Region::WindowTitle,
                text: title,
            });
        }

        let mut chars = blocks.iter().map(|b| b.text.chars().count()).sum::<usize>();
        // The window breadth first, until a web area turns up; its content then goes first, depth
        // first in document order.
        let mut web: Vec<CFRetained<AXUIElement>> = Vec::new();
        let mut queue: VecDeque<CFRetained<AXUIElement>> = window.into_iter().collect();
        let mut visited = 0;
        let mut waited = Duration::ZERO;
        loop {
            let (current, in_web) = match web.pop() {
                Some(current) => (current, true),
                None => match queue.pop_front() {
                    Some(current) => (current, false),
                    None => break,
                },
            };
            if visited >= self.max_elements
                || chars >= self.max_chars
                || started.elapsed() >= self.budget + waited
            {
                break;
            }
            visited += 1;
            // The focused field is already in, with only its text before the cursor.
            if focused.as_deref().is_some_and(|f| f == &*current) {
                continue;
            }
            let mut node = node(&current, &names);
            let web_area = node.role.as_deref() == Some("AXWebArea");
            // A page's own web area, not an empty frame inside it, with no content: still being
            // built, or the app waits for the switch VoiceOver uses. The wait does not count
            // against the budget, and only a page whose tree was off pays it.
            if web_area && !in_web && node.children.is_empty() && waited.is_zero() {
                let waiting = Instant::now();
                set_true(&app, "AXEnhancedUserInterface");
                while node.children.is_empty() && waiting.elapsed() < self.web_wait {
                    std::thread::sleep(Duration::from_millis(30));
                    node.children = children(&current);
                }
                waited = waiting.elapsed();
            }
            if let Some(text) = node.text {
                chars += text.chars().count();
                blocks.push(TextBlock {
                    region: Region::Window,
                    text,
                });
            }
            if in_web || web_area {
                web.extend(node.children.into_iter().rev());
            } else {
                queue.extend(node.children);
            }
        }

        Ok(ScreenContext {
            app: string(&app, "AXTitle"),
            blocks,
            source: ContextSource::Accessibility,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_dictator_core::context::rank_terms;

    fn read() -> ScreenContext {
        let started = Instant::now();
        let screen = AccessibilityReader::default().snapshot().unwrap();
        println!(
            "read {} blocks from {:?} in {:?}",
            screen.blocks.len(),
            screen.app,
            started.elapsed()
        );
        for block in &screen.blocks {
            println!("  {:?}: {:?}", block.region, block.text);
        }
        screen
    }

    /// Reads `tests/data/page.html`, which `scripts/accessibility-e2e.sh` opens in Chrome, as the
    /// first dictation after Chrome starts would: Chrome builds its tree only once asked.
    #[test]
    #[ignore = "needs the page open in Chrome and accessibility permission; run scripts/accessibility-e2e.sh"]
    fn reads_names_from_a_chrome_page() {
        assert!(is_trusted(false), "accessibility permission is not granted");
        let first = read();
        let second = read();
        let hotwords: Vec<String> = rank_terms(&first, 100)
            .into_iter()
            .map(|t| t.phrase)
            .collect();
        println!("hotwords: {}", hotwords.join(", "));

        assert_eq!(first.app.as_deref(), Some("Google Chrome"));
        let title = first
            .blocks
            .iter()
            .find(|b| b.region == Region::WindowTitle);
        assert!(
            title.is_some_and(|b| b.text.contains("Kestrelholm release notes")),
            "{title:?}"
        );
        for name in ["Kestrelholm", "Airicraft", "Thessaly Varga", "Quillon"] {
            assert!(
                hotwords.iter().any(|h| h == name),
                "{name} is not among the hotwords"
            );
        }
        // The focused field may need the tree the first read asked for.
        let focused = |screen: &ScreenContext| screen.focused_tail(400);
        assert_eq!(
            focused(&second).as_deref(),
            Some("Dear Quillon, the next build is"),
            "first read: {:?}",
            focused(&first)
        );
    }
}
