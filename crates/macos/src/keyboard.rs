//! The keyboard through CoreGraphics event taps: the hotkey listener that runs all the time, and
//! the recorder behind the menu's Change Hotkey item. Both taps run on the main run loop and need
//! the Accessibility permission the app already asks for. Each keeps the keys it uses from
//! reaching other apps, and lets modifier changes through so no modifier looks stuck elsewhere.

use std::ffi::c_void;
use std::ptr::{null_mut, NonNull};

use objc2_core_foundation::{
    kCFRunLoopCommonModes, CFMachPort, CFRetained, CFRunLoop, CFRunLoopSource,
};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventMask, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
    CGEventTapOptions, CGEventTapPlacement, CGEventTapProxy, CGEventType,
};

use crate::hotkey::{Held, Hotkey, Matcher, Outcome, Recorder, Signal};

/// A key event as the state machines take it.
enum Key {
    Down {
        code: u16,
        repeat: bool,
    },
    Up {
        code: u16,
    },
    Modifiers(Held),
    /// A mouse button went down or the wheel scrolled. Never kept from other apps.
    Click,
}

const KEYS: CGEventMask = (1 << CGEventType::KeyDown.0)
    | (1 << CGEventType::KeyUp.0)
    | (1 << CGEventType::FlagsChanged.0);
const CLICKS: CGEventMask = (1 << CGEventType::LeftMouseDown.0)
    | (1 << CGEventType::RightMouseDown.0)
    | (1 << CGEventType::OtherMouseDown.0)
    | (1 << CGEventType::ScrollWheel.0);

/// Returns whether to keep the event from other apps.
type Handler = Box<dyn FnMut(Key) -> bool>;

struct TapState {
    handler: Handler,
    /// The tap itself, to turn it back on when macOS disables it for being slow.
    port: Option<CFRetained<CFMachPort>>,
}

/// An event tap on the main run loop. Dropping it removes the tap.
struct Tap {
    port: CFRetained<CFMachPort>,
    source: CFRetained<CFRunLoopSource>,
    state: *mut TapState,
}

impl Tap {
    /// `None` when the tap cannot be created, which means the app lacks Accessibility.
    fn new(mask: CGEventMask, handler: Handler) -> Option<Tap> {
        let state = Box::into_raw(Box::new(TapState {
            handler,
            port: None,
        }));
        // SAFETY: the callback matches the signature CoreGraphics calls, and `state` stays valid
        // until `Tap` drops, which removes the tap first.
        let port = unsafe {
            CGEvent::tap_create(
                // Where keys enter the window server, ahead of system shortcuts such as the
                // dictation key's.
                CGEventTapLocation::HIDEventTap,
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
        Some(Tap {
            port,
            source,
            state,
        })
    }
}

impl Drop for Tap {
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
    // SAFETY: `user_info` is the `TapState` that `Tap::new` made, alive while the tap exists.
    let state = unsafe { &mut *user_info.cast::<TapState>() };
    if kind == CGEventType::TapDisabledByTimeout || kind == CGEventType::TapDisabledByUserInput {
        if let Some(port) = &state.port {
            CGEvent::tap_enable(port, true);
        }
        return event.as_ptr();
    }
    // SAFETY: CoreGraphics passes a valid event for the duration of the call.
    let event_ref = Some(unsafe { event.as_ref() });
    let code = CGEvent::integer_value_field(event_ref, CGEventField::KeyboardEventKeycode) as u16;
    let key = match kind {
        CGEventType::KeyDown => Key::Down {
            code,
            repeat: CGEvent::integer_value_field(event_ref, CGEventField::KeyboardEventAutorepeat)
                != 0,
        },
        CGEventType::KeyUp => Key::Up { code },
        CGEventType::FlagsChanged => {
            Key::Modifiers(Held::from_flags(CGEvent::flags(event_ref).bits()))
        }
        CGEventType::LeftMouseDown
        | CGEventType::RightMouseDown
        | CGEventType::OtherMouseDown
        | CGEventType::ScrollWheel => Key::Click,
        _ => return event.as_ptr(),
    };
    let passes = matches!(key, Key::Modifiers(_) | Key::Click);
    if (state.handler)(key) && !passes {
        null_mut()
    } else {
        event.as_ptr()
    }
}

/// Watches for the hotkey until dropped.
pub struct Listener {
    _tap: Tap,
}

/// `on_signal` runs on the main thread.
pub fn listen(hotkey: Hotkey, mut on_signal: impl FnMut(Signal) + 'static) -> Option<Listener> {
    let held = Held::from_flags(
        CGEventSource::flags_state(CGEventSourceStateID::CombinedSessionState).bits(),
    );
    let mut matcher = Matcher::new(hotkey, held);
    let tap = Tap::new(
        KEYS | CLICKS,
        Box::new(move |key| {
            let (signal, swallow) = match key {
                Key::Down { code, repeat } => matcher.key_down(code, repeat),
                Key::Up { code } => matcher.key_up(code),
                Key::Modifiers(held) => (matcher.modifiers_changed(held), false),
                Key::Click => (matcher.mouse_down(), false),
            };
            if let Some(signal) = signal {
                on_signal(signal);
            }
            swallow
        }),
    )?;
    Some(Listener { _tap: tap })
}

/// A recording in progress. Dropping it stops recording.
pub struct Capture {
    _tap: Tap,
}

/// Records the next hotkey, keeping every key from other apps meanwhile. `on_outcome` runs on
/// the main thread, until it is given a recorded hotkey or a cancel.
pub fn record(mut on_outcome: impl FnMut(Outcome) + 'static) -> Option<Capture> {
    let mut recorder = Recorder::default();
    let mut done = false;
    let tap = Tap::new(
        KEYS,
        Box::new(move |key| {
            if !done {
                let outcome = match key {
                    Key::Down { code, repeat } => recorder.key_down(code, repeat),
                    Key::Up { code } => recorder.key_up(code),
                    Key::Modifiers(held) => recorder.modifiers_changed(held),
                    Key::Click => None,
                };
                if let Some(outcome) = outcome {
                    done = !matches!(outcome, Outcome::NeedsModifier);
                    on_outcome(outcome);
                }
            }
            true
        }),
    )?;
    Some(Capture { _tap: tap })
}
