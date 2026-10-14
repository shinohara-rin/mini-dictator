//! Records a new hotkey from the keyboard, for the menu's Change Hotkey item.
//!
//! While recording, an event tap sees every key and keeps it from reaching other apps. The
//! combination is the last key pressed with the modifiers held at that moment, and it is taken
//! only once every key is up, so the release of the new hotkey cannot trigger it. Escape on its
//! own cancels. Creating the tap needs the Accessibility permission the app already asks for.

use std::collections::HashSet;
use std::ffi::c_void;
use std::ptr::{null_mut, NonNull};

use objc2_core_foundation::{
    kCFRunLoopCommonModes, CFMachPort, CFRetained, CFRunLoop, CFRunLoopSource,
};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventMask, CGEventTapLocation, CGEventTapOptions,
    CGEventTapPlacement, CGEventTapProxy, CGEventType,
};

/// The virtual key code of Escape.
const KEY_ESCAPE: u16 = 0x35;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A hotkey in the form the config file and `HotKey::from_str` take, such as `Ctrl+Shift+D`.
    Recorded(String),
    Cancelled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub control: bool,
    pub option: bool,
    pub shift: bool,
    pub command: bool,
}

impl Modifiers {
    fn from_flags(flags: CGEventFlags) -> Self {
        Self {
            control: flags.contains(CGEventFlags::MaskControl),
            option: flags.contains(CGEventFlags::MaskAlternate),
            shift: flags.contains(CGEventFlags::MaskShift),
            command: flags.contains(CGEventFlags::MaskCommand),
        }
    }

    fn any(self) -> bool {
        self.control || self.option || self.shift || self.command
    }
}

/// The keys seen so far, and the combination they make.
#[derive(Debug, Default)]
pub struct Recorder {
    down: HashSet<u16>,
    modifiers: Modifiers,
    combination: Option<(Modifiers, &'static str)>,
}

impl Recorder {
    pub fn key_down(&mut self, code: u16, modifiers: Modifiers) -> Option<Outcome> {
        self.modifiers = modifiers;
        if code == KEY_ESCAPE && !modifiers.any() {
            return Some(Outcome::Cancelled);
        }
        self.down.insert(code);
        // A key global-hotkey cannot register, such as Fn, leaves the combination as it was.
        if let Some(name) = key_name(code) {
            self.combination = Some((modifiers, name));
        }
        None
    }

    pub fn key_up(&mut self, code: u16, modifiers: Modifiers) -> Option<Outcome> {
        self.down.remove(&code);
        self.modifiers = modifiers;
        self.finish()
    }

    pub fn modifiers_changed(&mut self, modifiers: Modifiers) -> Option<Outcome> {
        self.modifiers = modifiers;
        self.finish()
    }

    /// Modifiers pressed and released on their own keep recording: a hotkey needs a key.
    fn finish(&self) -> Option<Outcome> {
        if !self.down.is_empty() || self.modifiers.any() {
            return None;
        }
        let (modifiers, key) = self.combination?;
        Some(Outcome::Recorded(format(modifiers, key)))
    }
}

/// Modifiers in the order macOS menus show them.
fn format(modifiers: Modifiers, key: &str) -> String {
    let mut parts = Vec::new();
    for (held, name) in [
        (modifiers.control, "Ctrl"),
        (modifiers.option, "Alt"),
        (modifiers.shift, "Shift"),
        (modifiers.command, "Cmd"),
    ] {
        if held {
            parts.push(name);
        }
    }
    parts.push(key);
    parts.join("+")
}

/// The name global-hotkey parses for a virtual key code, for every key it can register on macOS.
fn key_name(code: u16) -> Option<&'static str> {
    Some(match code {
        0x00 => "A",
        0x01 => "S",
        0x02 => "D",
        0x03 => "F",
        0x04 => "H",
        0x05 => "G",
        0x06 => "Z",
        0x07 => "X",
        0x08 => "C",
        0x09 => "V",
        0x0b => "B",
        0x0c => "Q",
        0x0d => "W",
        0x0e => "E",
        0x0f => "R",
        0x10 => "Y",
        0x11 => "T",
        0x12 => "1",
        0x13 => "2",
        0x14 => "3",
        0x15 => "4",
        0x16 => "6",
        0x17 => "5",
        0x18 => "Equal",
        0x19 => "9",
        0x1a => "7",
        0x1b => "Minus",
        0x1c => "8",
        0x1d => "0",
        0x1e => "BracketRight",
        0x1f => "O",
        0x20 => "U",
        0x21 => "BracketLeft",
        0x22 => "I",
        0x23 => "P",
        0x24 => "Enter",
        0x25 => "L",
        0x26 => "J",
        0x27 => "Quote",
        0x28 => "K",
        0x29 => "Semicolon",
        0x2a => "Backslash",
        0x2b => "Comma",
        0x2c => "Slash",
        0x2d => "N",
        0x2e => "M",
        0x2f => "Period",
        0x30 => "Tab",
        0x31 => "Space",
        0x32 => "Backquote",
        0x33 => "Backspace",
        0x35 => "Escape",
        0x40 => "F17",
        0x41 => "NumpadDecimal",
        0x43 => "NumpadMultiply",
        0x45 => "NumpadAdd",
        0x47 => "NumLock",
        0x4b => "NumpadDivide",
        0x4c => "NumpadEnter",
        0x4e => "NumpadSubtract",
        0x4f => "F18",
        0x50 => "F19",
        0x51 => "NumpadEqual",
        0x52 => "Numpad0",
        0x53 => "Numpad1",
        0x54 => "Numpad2",
        0x55 => "Numpad3",
        0x56 => "Numpad4",
        0x57 => "Numpad5",
        0x58 => "Numpad6",
        0x59 => "Numpad7",
        0x5a => "F20",
        0x5b => "Numpad8",
        0x5c => "Numpad9",
        0x60 => "F5",
        0x61 => "F6",
        0x62 => "F7",
        0x63 => "F3",
        0x64 => "F8",
        0x65 => "F9",
        0x67 => "F11",
        0x69 => "F13",
        0x6a => "F16",
        0x6b => "F14",
        0x6d => "F10",
        0x6f => "F12",
        0x71 => "F15",
        0x72 => "Insert",
        0x73 => "Home",
        0x74 => "PageUp",
        0x75 => "Delete",
        0x76 => "F4",
        0x77 => "End",
        0x78 => "F2",
        0x79 => "PageDown",
        0x7a => "F1",
        0x7b => "Left",
        0x7c => "Right",
        0x7d => "Down",
        0x7e => "Up",
        _ => return None,
    })
}

struct TapState {
    recorder: Recorder,
    /// The tap itself, to turn it back on when macOS disables it for being slow.
    port: Option<CFRetained<CFMachPort>>,
    done: bool,
    on_done: Box<dyn FnMut(Outcome)>,
}

/// A recording in progress. Dropping it removes the tap, so keys reach other apps again.
pub struct Capture {
    port: CFRetained<CFMachPort>,
    source: CFRetained<CFRunLoopSource>,
    state: *mut TapState,
}

/// Starts recording on the main run loop, or returns `None` when the tap cannot be created, which
/// means the app lacks the Accessibility permission. `on_done` runs once, on the main thread.
pub fn start(on_done: impl FnMut(Outcome) + 'static) -> Option<Capture> {
    let mask: CGEventMask = (1 << CGEventType::KeyDown.0)
        | (1 << CGEventType::KeyUp.0)
        | (1 << CGEventType::FlagsChanged.0);
    let state = Box::into_raw(Box::new(TapState {
        recorder: Recorder::default(),
        port: None,
        done: false,
        on_done: Box::new(on_done),
    }));
    // SAFETY: the callback matches the signature CoreGraphics calls, and `state` stays valid until
    // `Capture` drops, which removes the tap first.
    let port = unsafe {
        CGEvent::tap_create(
            CGEventTapLocation::SessionEventTap,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::Default,
            mask,
            Some(on_event),
            state.cast(),
        )
    };
    let source = port
        .as_ref()
        .and_then(|port| CFMachPort::new_run_loop_source(None, Some(port), 0));
    let run_loop = CFRunLoop::main();
    let (Some(port), Some(source), Some(run_loop)) = (port, source, run_loop) else {
        // SAFETY: no tap is running, so nothing else holds the pointer.
        drop(unsafe { Box::from_raw(state) });
        return None;
    };
    // SAFETY: the tap only calls back on the main run loop, which is this thread.
    unsafe { (*state).port = Some(port.clone()) };
    run_loop.add_source(Some(&source), unsafe { kCFRunLoopCommonModes });
    CGEvent::tap_enable(&port, true);
    Some(Capture {
        port,
        source,
        state,
    })
}

impl Drop for Capture {
    fn drop(&mut self) {
        CGEvent::tap_enable(&self.port, false);
        if let Some(run_loop) = CFRunLoop::main() {
            run_loop.remove_source(Some(&self.source), unsafe { kCFRunLoopCommonModes });
        }
        self.port.invalidate();
        // SAFETY: the tap is invalidated, so the callback no longer runs with this pointer.
        drop(unsafe { Box::from_raw(self.state) });
    }
}

unsafe extern "C-unwind" fn on_event(
    _proxy: CGEventTapProxy,
    kind: CGEventType,
    event: NonNull<CGEvent>,
    user_info: *mut c_void,
) -> *mut CGEvent {
    // SAFETY: `user_info` is the `TapState` that `start` made, alive while the tap exists.
    let state = unsafe { &mut *user_info.cast::<TapState>() };
    if kind == CGEventType::TapDisabledByTimeout || kind == CGEventType::TapDisabledByUserInput {
        if let Some(port) = &state.port {
            CGEvent::tap_enable(port, true);
        }
        return event.as_ptr();
    }
    // SAFETY: CoreGraphics passes a valid event for the duration of the call.
    let event_ref = unsafe { event.as_ref() };
    if !state.done {
        let modifiers = Modifiers::from_flags(CGEvent::flags(Some(event_ref)));
        let code = CGEvent::integer_value_field(Some(event_ref), CGEventField::KeyboardEventKeycode)
            as u16;
        let outcome = match kind {
            CGEventType::KeyDown => state.recorder.key_down(code, modifiers),
            CGEventType::KeyUp => state.recorder.key_up(code, modifiers),
            CGEventType::FlagsChanged => state.recorder.modifiers_changed(modifiers),
            _ => None,
        };
        if let Some(outcome) = outcome {
            state.done = true;
            (state.on_done)(outcome);
        }
    }
    // Modifier changes pass through so other apps do not see a modifier stuck down.
    if kind == CGEventType::FlagsChanged {
        event.as_ptr()
    } else {
        null_mut()
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use global_hotkey::hotkey::HotKey;

    use super::*;

    const NONE: Modifiers = Modifiers {
        control: false,
        option: false,
        shift: false,
        command: false,
    };
    const CTRL: Modifiers = Modifiers {
        control: true,
        ..NONE
    };
    const CTRL_SHIFT: Modifiers = Modifiers {
        control: true,
        shift: true,
        ..NONE
    };
    const KEY_D: u16 = 0x02;
    const KEY_SPACE: u16 = 0x31;

    #[test]
    fn commits_once_every_key_is_up() {
        let mut recorder = Recorder::default();
        assert_eq!(recorder.modifiers_changed(CTRL), None);
        assert_eq!(recorder.modifiers_changed(CTRL_SHIFT), None);
        assert_eq!(recorder.key_down(KEY_D, CTRL_SHIFT), None);
        assert_eq!(recorder.key_up(KEY_D, CTRL_SHIFT), None);
        assert_eq!(recorder.modifiers_changed(CTRL), None);
        assert_eq!(
            recorder.modifiers_changed(NONE),
            Some(Outcome::Recorded("Ctrl+Shift+D".into()))
        );
    }

    #[test]
    fn modifiers_alone_keep_recording() {
        let mut recorder = Recorder::default();
        recorder.modifiers_changed(CTRL);
        assert_eq!(recorder.modifiers_changed(NONE), None);
        recorder.key_down(KEY_SPACE, NONE);
        assert_eq!(
            recorder.key_up(KEY_SPACE, NONE),
            Some(Outcome::Recorded("Space".into()))
        );
    }

    #[test]
    fn the_last_key_wins() {
        let mut recorder = Recorder::default();
        recorder.key_down(KEY_D, CTRL);
        recorder.key_down(KEY_SPACE, CTRL);
        assert_eq!(recorder.key_up(KEY_D, CTRL), None);
        assert_eq!(recorder.key_up(KEY_SPACE, CTRL), None);
        assert_eq!(
            recorder.modifiers_changed(NONE),
            Some(Outcome::Recorded("Ctrl+Space".into()))
        );
    }

    #[test]
    fn escape_cancels_but_a_modified_escape_records() {
        let mut recorder = Recorder::default();
        assert_eq!(
            recorder.key_down(KEY_ESCAPE, NONE),
            Some(Outcome::Cancelled)
        );
        let mut recorder = Recorder::default();
        assert_eq!(recorder.key_down(KEY_ESCAPE, CTRL), None);
        assert_eq!(recorder.key_up(KEY_ESCAPE, CTRL), None);
        assert_eq!(
            recorder.modifiers_changed(NONE),
            Some(Outcome::Recorded("Ctrl+Escape".into()))
        );
    }

    #[test]
    fn every_recorded_hotkey_parses() {
        let all = Modifiers {
            control: true,
            option: true,
            shift: true,
            command: true,
        };
        for code in 0..=u8::MAX as u16 {
            if let Some(key) = key_name(code) {
                let text = format(all, key);
                assert!(HotKey::from_str(&text).is_ok(), "{text}");
            }
        }
    }
}
