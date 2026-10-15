//! Inserts text by pasting it, then puts the clipboard back.
//!
//! Pasting works in nearly every app and inserts long text at once. Typed Unicode events are the
//! alternative, but some apps drop or reorder them. The clipboard is restored with every item and
//! type it held (an image, files, rich text), and only if nothing else changed it during the paste.

use std::thread::sleep;
use std::time::Duration;

use mini_dictator_core::pipeline::TextSink;
use mini_dictator_core::{Error, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{
    NSPasteboard, NSPasteboardItem, NSPasteboardType, NSPasteboardTypeString, NSPasteboardWriting,
};
use objc2_core_graphics::{CGEvent, CGEventField, CGEventFlags, CGEventTapLocation};
use objc2_foundation::{NSArray, NSData, NSString};

use crate::keyboard;

/// The virtual key code of V on ANSI keyboards. macOS maps Cmd+V by key position.
const KEY_V: u16 = 9;

pub struct PasteSink;

/// The clipboard's items, each with the data of every type it offers.
type Snapshot = Vec<Vec<(Retained<NSPasteboardType>, Retained<NSData>)>>;

fn snapshot(pasteboard: &NSPasteboard) -> Snapshot {
    let Some(items) = pasteboard.pasteboardItems() else {
        return Vec::new();
    };
    items
        .iter()
        .map(|item| {
            item.types()
                .iter()
                .filter_map(|kind| item.dataForType(&kind).map(|data| (kind, data)))
                .collect()
        })
        .collect()
}

fn restore(pasteboard: &NSPasteboard, snapshot: Snapshot) {
    pasteboard.clearContents();
    let items: Vec<Retained<ProtocolObject<dyn NSPasteboardWriting>>> = snapshot
        .into_iter()
        .map(|types| {
            let item = NSPasteboardItem::new();
            for (kind, data) in &types {
                item.setData_forType(data, kind);
            }
            ProtocolObject::from_retained(item)
        })
        .collect();
    if !items.is_empty() {
        pasteboard.writeObjects(&NSArray::from_retained_slice(&items));
    }
}

fn press_command_v() -> Result<()> {
    for down in [true, false] {
        let event = CGEvent::new_keyboard_event(None, KEY_V, down)
            .ok_or_else(|| Error::Sink("cannot create a key event".into()))?;
        CGEvent::set_flags(Some(&event), CGEventFlags::MaskCommand);
        // So a hotkey recording in progress lets the paste through.
        CGEvent::set_integer_value_field(
            Some(&event),
            CGEventField::EventSourceUserData,
            keyboard::OWN_EVENT,
        );
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
    }
    Ok(())
}

impl TextSink for PasteSink {
    fn insert(&mut self, text: &str) -> Result<()> {
        let pasteboard = NSPasteboard::generalPasteboard();
        let kind = unsafe { NSPasteboardTypeString };
        let previous = snapshot(&pasteboard);

        pasteboard.clearContents();
        if !pasteboard.setString_forType(&NSString::from_str(text), kind) {
            restore(&pasteboard, previous);
            return Err(Error::Sink("cannot write the clipboard".into()));
        }
        let ours = pasteboard.changeCount();
        if let Err(error) = press_command_v() {
            restore(&pasteboard, previous);
            return Err(error);
        }

        // The target app reads the clipboard when it handles the key event, which can take a moment.
        sleep(Duration::from_millis(250));
        if pasteboard.changeCount() == ours {
            restore(&pasteboard, previous);
        }
        Ok(())
    }
}
