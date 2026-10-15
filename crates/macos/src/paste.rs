//! Inserts text by pasting it, then puts the clipboard back.
//!
//! Pasting works in nearly every app and inserts long text at once. Typed Unicode events are the
//! alternative, but some apps drop or reorder them. The clipboard is restored with every item and
//! type it held (an image, files, rich text), and only if nothing else changed it meanwhile.
//!
//! Text is pasted only into what had focus when the dictation started. Once the focus moves, the
//! rest of the dictation is held, and the menu offers it to copy.

use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;

use mini_dictator_core::pipeline::{Delivery, TextSink};
use mini_dictator_core::{Error, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{
    NSPasteboard, NSPasteboardItem, NSPasteboardType, NSPasteboardTypeString, NSPasteboardWriting,
};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};
use objc2_foundation::{NSArray, NSData, NSString};

use crate::context::Focus;
use crate::keyboard;

/// The virtual key code of V on ANSI keyboards. macOS maps Cmd+V by key position.
const KEY_V: u16 = 9;

/// How long after the last paste the clipboard is put back. The target app reads the clipboard
/// when it handles the key event, and a busy app can take a while; restoring before that pastes
/// the old clipboard instead.
const RESTORE_AFTER: Duration = Duration::from_millis(1500);

/// Time between the key going down and up, for apps that miss a press with no length.
const KEY_HOLD: Duration = Duration::from_millis(15);

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    /// On while a password field or a terminal's secure keyboard entry has focus. macOS then
    /// keeps synthetic key events from every app, so a paste does nothing.
    fn IsSecureEventInputEnabled() -> bool;
}

pub struct PasteSink {
    /// What had focus when the dictation started. None when it could not be read.
    focus: Option<Focus>,
    /// Why the dictation stopped typing, kept for its remaining text.
    held: Option<String>,
}

impl PasteSink {
    /// Call when the dictation starts, so the focus it records is the one the user dictates into.
    pub fn new() -> PasteSink {
        PasteSink {
            focus: Focus::now(),
            held: None,
        }
    }
}

/// The clipboard's items, each with the data of every type it offers, as plain bytes so the
/// restore can wait on another thread.
type Snapshot = Vec<Vec<(String, Vec<u8>)>>;

/// The user's clipboard, saved before a paste and not put back yet.
struct Saved {
    snapshot: Snapshot,
    /// The clipboard's change count right after the paste wrote it.
    ours: isize,
}

/// Shared across dictations: a paste that comes before the last one was restored keeps the
/// user's clipboard instead of saving the dictated text as theirs.
static SAVED: Mutex<Option<Saved>> = Mutex::new(None);

fn snapshot(pasteboard: &NSPasteboard) -> Snapshot {
    let Some(items) = pasteboard.pasteboardItems() else {
        return Vec::new();
    };
    items
        .iter()
        .map(|item| {
            item.types()
                .iter()
                .filter_map(|kind| {
                    let data = item.dataForType(&kind)?;
                    Some((kind.to_string(), data.to_vec()))
                })
                .collect()
        })
        .collect()
}

fn restore(pasteboard: &NSPasteboard, snapshot: &Snapshot) {
    pasteboard.clearContents();
    let items: Vec<Retained<ProtocolObject<dyn NSPasteboardWriting>>> = snapshot
        .iter()
        .map(|types| {
            let item = NSPasteboardItem::new();
            for (kind, data) in types {
                let kind: Retained<NSPasteboardType> = NSString::from_str(kind);
                item.setData_forType(&NSData::with_bytes(data), &kind);
            }
            ProtocolObject::from_retained(item)
        })
        .collect();
    if !items.is_empty() {
        pasteboard.writeObjects(&NSArray::from_retained_slice(&items));
    }
}

/// Puts the user's clipboard back once the paste that wrote `ours` had time to land, unless a later
/// paste took over or something else changed the clipboard.
fn restore_later(ours: isize) {
    std::thread::spawn(move || {
        sleep(RESTORE_AFTER);
        let mut saved = SAVED.lock().unwrap_or_else(|e| e.into_inner());
        if saved.as_ref().is_none_or(|s| s.ours != ours) {
            return;
        }
        let pasteboard = NSPasteboard::generalPasteboard();
        if let Some(saved) = saved.take() {
            if pasteboard.changeCount() == ours {
                restore(&pasteboard, &saved.snapshot);
            }
        }
    });
}

fn press_command_v() -> Result<()> {
    // A private source, so the modifiers the user still holds, such as a held hotkey, do not join
    // the event: Cmd+Option+V or Cmd+Ctrl+V does something else or nothing in most apps.
    let source = CGEventSource::new(CGEventSourceStateID::Private);
    for down in [true, false] {
        let event = CGEvent::new_keyboard_event(source.as_deref(), KEY_V, down)
            .ok_or_else(|| Error::Sink("cannot create a key event".into()))?;
        CGEvent::set_flags(Some(&event), CGEventFlags::MaskCommand);
        // So a hotkey recording in progress lets the paste through.
        CGEvent::set_integer_value_field(
            Some(&event),
            CGEventField::EventSourceUserData,
            keyboard::OWN_EVENT,
        );
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
        if down {
            sleep(KEY_HOLD);
        }
    }
    Ok(())
}

impl TextSink for PasteSink {
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
        if let Some(reason) = &self.held {
            return Ok(Delivery::Held(reason.clone()));
        }
        if unsafe { IsSecureEventInputEnabled() } {
            // Only this piece: secure input can go off again, and the focus did not move.
            return Ok(Delivery::Held("secure input is on".into()));
        }

        let pasteboard = NSPasteboard::generalPasteboard();
        let kind = unsafe { NSPasteboardTypeString };
        let mut saved = SAVED.lock().unwrap_or_else(|e| e.into_inner());
        // The clipboard still holds an earlier paste: what to restore is what was saved then.
        let previous = match saved.take() {
            Some(s) if pasteboard.changeCount() == s.ours => s.snapshot,
            _ => snapshot(&pasteboard),
        };

        pasteboard.clearContents();
        if !pasteboard.setString_forType(&NSString::from_str(text), kind) {
            restore(&pasteboard, &previous);
            return Err(Error::Sink("cannot write the clipboard".into()));
        }
        let ours = pasteboard.changeCount();
        if let Err(error) = press_command_v() {
            restore(&pasteboard, &previous);
            return Err(error);
        }
        *saved = Some(Saved {
            snapshot: previous,
            ours,
        });
        drop(saved);
        restore_later(ours);
        Ok(Delivery::Typed)
    }
}

/// Puts `text` on the clipboard for the user to paste themselves.
pub fn copy(text: &str) {
    let pasteboard = NSPasteboard::generalPasteboard();
    pasteboard.clearContents();
    pasteboard.setString_forType(&NSString::from_str(text), unsafe { NSPasteboardTypeString });
}
