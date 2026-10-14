//! Reads the focused window through the macOS accessibility API.
//!
//! The reader collects the focused field's text before the cursor, the window title, and the text
//! of other elements, breadth first, within an element, character, and time budget. Electron apps
//! build their accessibility tree only when asked, so the reader sets `AXManualAccessibility` on
//! the focused app first.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use mini_dictator_core::context::{
    ContextProvider, ContextSource, Region, ScreenContext, TextBlock,
};
use mini_dictator_core::{Error, Result};
use objc2_application_services::{
    kAXTrustedCheckOptionPrompt, AXError, AXIsProcessTrusted, AXIsProcessTrustedWithOptions,
    AXUIElement, AXValue, AXValueType,
};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFRange, CFRetained, CFString, CFType,
};

pub struct AccessibilityReader {
    pub max_elements: usize,
    pub max_chars: usize,
    pub budget: Duration,
}

impl Default for AccessibilityReader {
    fn default() -> Self {
        Self {
            max_elements: 800,
            max_chars: 20_000,
            budget: Duration::from_millis(150),
        }
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
    let value = attribute(element, name)?;
    let text = value.downcast_ref::<CFString>()?.to_string();
    (!text.trim().is_empty()).then_some(text)
}

fn element(element: &AXUIElement, name: &str) -> Option<CFRetained<AXUIElement>> {
    attribute(element, name)?.downcast::<AXUIElement>().ok()
}

fn children(element: &AXUIElement) -> Vec<CFRetained<AXUIElement>> {
    let Some(value) = attribute(element, "AXChildren") else {
        return Vec::new();
    };
    let Ok(array) = value.downcast::<CFArray>() else {
        return Vec::new();
    };
    let array: &CFArray<CFType> = unsafe { array.cast_unchecked() };
    array
        .iter()
        .filter_map(|child| child.downcast::<AXUIElement>().ok())
        .collect()
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
        unsafe {
            app.set_attribute_value(
                &CFString::from_str("AXManualAccessibility"),
                CFBoolean::new(true).as_ref(),
            )
        };

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
        let mut queue: VecDeque<CFRetained<AXUIElement>> = window.into_iter().collect();
        let mut visited = 0;
        while let Some(current) = queue.pop_front() {
            if visited >= self.max_elements
                || chars >= self.max_chars
                || started.elapsed() >= self.budget
            {
                break;
            }
            visited += 1;
            // The focused field is already in, with only its text before the cursor.
            if focused.as_deref().is_some_and(|f| f == &*current) {
                continue;
            }
            for name in ["AXValue", "AXTitle", "AXDescription"] {
                if let Some(text) = string(&current, name) {
                    chars += text.chars().count();
                    blocks.push(TextBlock {
                        region: Region::Window,
                        text,
                    });
                    break;
                }
            }
            queue.extend(children(&current));
        }

        Ok(ScreenContext {
            app: string(&app, "AXTitle"),
            blocks,
            source: ContextSource::Accessibility,
        })
    }
}
