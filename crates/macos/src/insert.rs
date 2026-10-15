//! Inserts text into the focused field without the clipboard.
//!
//! Native fields take the text through the accessibility API, at the cursor in one step. Others,
//! web pages and Electron apps among them, get it typed as Unicode key events, a few characters per
//! event, with Return pressed for each line break.
//!
//! Text is inserted only into what had focus when the dictation started. Once the focus moves, the
//! rest of the dictation is held, and the menu offers it to copy.

use std::thread::sleep;
use std::time::Duration;

use mini_dictator_core::pipeline::{Delivery, TextSink};
use mini_dictator_core::{Error, Result};
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};
use objc2_foundation::NSString;

use crate::context::{insert_at_cursor, Focus};
use crate::keyboard;

/// UTF-16 units per key event. Some apps read no more than 20 from one event.
const CHUNK: usize = 20;

/// Pause between key events, so an app that handles them one per run-loop pass keeps their order.
const EVENT_GAP: Duration = Duration::from_millis(2);

/// The virtual key code of Return.
const KEY_RETURN: u16 = 36;

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    /// On while a password field or a terminal's secure keyboard entry has focus. macOS then
    /// keeps synthetic key events from every app, so typed text goes nowhere.
    fn IsSecureEventInputEnabled() -> bool;
}

pub struct InsertSink {
    /// What had focus when the dictation started. None when it could not be read.
    focus: Option<Focus>,
    /// Why the dictation stopped typing, kept for its remaining text.
    held: Option<String>,
}

impl InsertSink {
    /// `focus` is what had focus at the hotkey press: what the dictation is typed into.
    pub fn new(focus: Option<Focus>) -> InsertSink {
        InsertSink { focus, held: None }
    }
}

/// Posts one key press with `text` as the characters it types, or a plain press of `key` when
/// `text` is empty.
fn press(source: Option<&CGEventSource>, key: u16, text: &[u16]) -> Result<()> {
    for down in [true, false] {
        let event = CGEvent::new_keyboard_event(source, key, down)
            .ok_or_else(|| Error::Sink("cannot create a key event".into()))?;
        // No modifiers, whatever the user still holds: Cmd with a letter would be a shortcut.
        CGEvent::set_flags(Some(&event), CGEventFlags(0));
        if !text.is_empty() {
            unsafe {
                CGEvent::keyboard_set_unicode_string(Some(&event), text.len() as _, text.as_ptr())
            };
        }
        // So the hotkey taps let it through.
        CGEvent::set_integer_value_field(
            Some(&event),
            CGEventField::EventSourceUserData,
            keyboard::OWN_EVENT,
        );
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
    }
    sleep(EVENT_GAP);
    Ok(())
}

/// Types `text` as Unicode key events.
fn type_text(text: &str) -> Result<()> {
    // A private source, so the modifiers the user holds, such as a held hotkey, do not join in.
    let source = CGEventSource::new(CGEventSourceStateID::Private);
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            press(source.as_deref(), KEY_RETURN, &[])?;
        }
        let units: Vec<u16> = line.trim_end_matches('\r').encode_utf16().collect();
        let mut start = 0;
        while start < units.len() {
            let mut end = (start + CHUNK).min(units.len());
            // Never split a surrogate pair across two events.
            if end < units.len() && (0xD800..0xDC00).contains(&units[end - 1]) {
                end -= 1;
            }
            // The key code only matters to apps that ignore the Unicode string.
            press(source.as_deref(), 0, &units[start..end])?;
            start = end;
        }
    }
    Ok(())
}

impl TextSink for InsertSink {
    fn insert(&mut self, text: &str) -> Result<Delivery> {
        if self.held.is_none() {
            let moved = match (&self.focus, Focus::now()) {
                (Some(before), Some(now)) => now.moved_from(before),
                _ => false,
            };
            if moved {
                self.held = Some("the focus moved".into());
            }
        }
        if self.held.is_none() && unsafe { IsSecureEventInputEnabled() } {
            // Kept for the rest too: later pieces would continue text the target never got.
            self.held = Some("secure input is on".into());
        }
        if let Some(reason) = &self.held {
            return Ok(Delivery::Held(reason.clone()));
        }
        if !insert_at_cursor(text) {
            type_text(text)?;
        }
        Ok(Delivery::Typed)
    }
}

/// Puts `text` on the clipboard for the user to paste themselves.
pub fn copy(text: &str) {
    let pasteboard = NSPasteboard::generalPasteboard();
    pasteboard.clearContents();
    pasteboard.setString_forType(&NSString::from_str(text), unsafe { NSPasteboardTypeString });
}
