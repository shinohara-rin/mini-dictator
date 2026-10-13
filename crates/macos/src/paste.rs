//! Inserts text by pasting it, then puts the clipboard back.
//!
//! Pasting works in nearly every app and inserts long text at once. Typed Unicode events are the
//! alternative, but some apps drop or reorder them. The clipboard is restored only if nothing else
//! changed it during the paste.

use std::thread::sleep;
use std::time::Duration;

use mini_dictator_core::pipeline::TextSink;
use mini_dictator_core::{Error, Result};
use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_core_graphics::{CGEvent, CGEventFlags, CGEventTapLocation};
use objc2_foundation::NSString;

/// The virtual key code of V on ANSI keyboards. macOS maps Cmd+V by key position.
const KEY_V: u16 = 9;

pub struct PasteSink;

fn press_command_v() -> Result<()> {
    for down in [true, false] {
        let event = CGEvent::new_keyboard_event(None, KEY_V, down)
            .ok_or_else(|| Error::Sink("cannot create a key event".into()))?;
        CGEvent::set_flags(Some(&event), CGEventFlags::MaskCommand);
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
    }
    Ok(())
}

impl TextSink for PasteSink {
    fn insert(&mut self, text: &str) -> Result<()> {
        let pasteboard = NSPasteboard::generalPasteboard();
        let kind = unsafe { NSPasteboardTypeString };
        let previous = pasteboard.stringForType(kind);

        pasteboard.clearContents();
        if !pasteboard.setString_forType(&NSString::from_str(text), kind) {
            return Err(Error::Sink("cannot write the clipboard".into()));
        }
        let ours = pasteboard.changeCount();
        press_command_v()?;

        // The target app reads the clipboard when it handles the key event, which can take a moment.
        sleep(Duration::from_millis(250));
        if pasteboard.changeCount() == ours {
            pasteboard.clearContents();
            if let Some(previous) = previous {
                pasteboard.setString_forType(&previous, kind);
            }
        }
        Ok(())
    }
}
