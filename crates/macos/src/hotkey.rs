//! Hotkeys as the config file writes them, and the two keyboard state machines that use them: one
//! that matches the hotkey while the app runs, and one that records a new hotkey.
//!
//! A hotkey is modifiers and one key, such as `Ctrl+Shift+D`, or modifiers alone, such as
//! `RightCmd`. Each modifier may name a side, as in `LeftAlt` or `RightCmd`; without one, either
//! side matches. Both machines take virtual key codes and the modifier keys held, so they run the
//! same in tests as behind the event tap.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

/// The virtual key code of Escape.
pub const KEY_ESCAPE: u16 = 0x35;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Either,
    Left,
    Right,
    /// Left and right together, as a chord of both Command keys.
    Both,
}

/// Control, Option, Shift and Command, in the order macOS menus show them.
const MODIFIERS: [(&str, &[&str]); 4] = [
    ("Ctrl", &["ctrl", "control"]),
    ("Alt", &["alt", "option", "opt"]),
    ("Shift", &["shift"]),
    // global-hotkey's CmdOrCtrl means Command on macOS.
    (
        "Cmd",
        &[
            "cmd",
            "command",
            "super",
            "meta",
            "cmdorctrl",
            "cmdorcontrol",
            "commandorctrl",
            "commandorcontrol",
        ],
    ),
];
const CONTROL: usize = 0;
#[cfg(test)]
const OPTION: usize = 1;
const SHIFT: usize = 2;
#[cfg(test)]
const COMMAND: usize = 3;

/// The left and right device bits of each modifier in an event's flags, then its side-blind bit.
const DEVICE_BITS: [(u64, u64, u64); 4] = [
    (0x0000_0001, 0x0000_2000, 0x0004_0000),
    (0x0000_0020, 0x0000_0040, 0x0008_0000),
    (0x0000_0002, 0x0000_0004, 0x0002_0000),
    (0x0000_0008, 0x0000_0010, 0x0010_0000),
];

/// The modifier keys held down, by side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Held {
    left: [bool; 4],
    right: [bool; 4],
}

impl Held {
    /// From an event's flags. An event without device bits, as some synthetic ones are, counts
    /// a set modifier as its left key.
    pub fn from_flags(flags: u64) -> Self {
        let mut held = Self::default();
        for (i, (left, right, any)) in DEVICE_BITS.into_iter().enumerate() {
            held.left[i] = flags & left != 0;
            held.right[i] = flags & right != 0;
            if flags & any != 0 && !held.left[i] && !held.right[i] {
                held.left[i] = true;
            }
        }
        held
    }

    pub fn is_empty(self) -> bool {
        self == Self::default()
    }

    fn union(self, other: Self) -> Self {
        let mut held = self;
        for i in 0..4 {
            held.left[i] |= other.left[i];
            held.right[i] |= other.right[i];
        }
        held
    }

    fn side(self, i: usize) -> Option<Side> {
        match (self.left[i], self.right[i]) {
            (false, false) => None,
            (true, false) => Some(Side::Left),
            (false, true) => Some(Side::Right),
            (true, true) => Some(Side::Both),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hotkey {
    modifiers: [Option<Side>; 4],
    key: Option<u16>,
}

impl Hotkey {
    /// Whether the hotkey is modifiers alone, so a press may turn out to be part of a shortcut.
    pub fn is_modifier_only(&self) -> bool {
        self.key.is_none()
    }

    /// What the recorder saw: Option and Command keep their side, Control and Shift match
    /// either, since few keyboards tell those apart by feel.
    fn recorded(held: Held, key: Option<u16>) -> Self {
        let mut modifiers = [None; 4];
        for (i, modifier) in modifiers.iter_mut().enumerate() {
            *modifier = match held.side(i) {
                Some(_) if i == CONTROL || i == SHIFT => Some(Side::Either),
                side => side,
            };
        }
        Self { modifiers, key }
    }

    /// Whether exactly these modifiers are down: each one on its side, and no others.
    fn matches(&self, held: Held) -> bool {
        (0..4).all(|i| {
            let (left, right) = (held.left[i], held.right[i]);
            match self.modifiers[i] {
                None => !left && !right,
                Some(Side::Either) => left || right,
                Some(Side::Left) => left && !right,
                Some(Side::Right) => right && !left,
                Some(Side::Both) => left && right,
            }
        })
    }

    /// Whether a modifier outside the hotkey is down, counting the other side of a sided one.
    fn has_extra(&self, held: Held) -> bool {
        (0..4).any(|i| match self.modifiers[i] {
            None => held.side(i).is_some(),
            Some(Side::Either | Side::Both) => false,
            Some(Side::Left) => held.right[i],
            Some(Side::Right) => held.left[i],
        })
    }
}

impl fmt::Display for Hotkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        for (i, (name, _)) in MODIFIERS.iter().enumerate() {
            match self.modifiers[i] {
                None => {}
                Some(Side::Either) => parts.push((*name).to_owned()),
                Some(Side::Left) => parts.push(format!("Left{name}")),
                Some(Side::Right) => parts.push(format!("Right{name}")),
                Some(Side::Both) => parts.push(format!("Left{name}+Right{name}")),
            }
        }
        if let Some(name) = self.key.and_then(key_name) {
            parts.push(name.to_owned());
        }
        write!(f, "{}", parts.join("+"))
    }
}

impl FromStr for Hotkey {
    type Err = String;

    /// Also takes the names global-hotkey used, such as `KeyD`, `Digit1` and `ArrowUp`, so older
    /// config files keep working.
    fn from_str(text: &str) -> Result<Self, String> {
        let mut hotkey = Self {
            modifiers: [None; 4],
            key: None,
        };
        for part in text.split('+') {
            let lower: String = part
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .to_ascii_lowercase();
            let (side, bare) = if let Some(rest) = lower.strip_prefix("left") {
                (Side::Left, rest)
            } else if let Some(rest) = lower.strip_prefix("right") {
                (Side::Right, rest)
            } else {
                (Side::Either, lower.as_str())
            };
            if let Some(i) = MODIFIERS
                .iter()
                .position(|(_, names)| names.contains(&bare))
            {
                // `LeftCmd+RightCmd` is both Command keys together.
                hotkey.modifiers[i] = match (hotkey.modifiers[i], side) {
                    (None, side) => Some(side),
                    (Some(Side::Left), Side::Right) | (Some(Side::Right), Side::Left) => {
                        Some(Side::Both)
                    }
                    _ => return Err(format!("{text}: {part} appears twice")),
                };
                continue;
            }
            let code = key_code(&lower).ok_or_else(|| format!("{text}: unknown key {part}"))?;
            if hotkey.key.replace(code).is_some() {
                return Err(format!(
                    "{text}: a hotkey has at most one key besides modifiers"
                ));
            }
        }
        if hotkey.modifiers == [None; 4] {
            match hotkey.key {
                None => return Err("an empty hotkey".into()),
                Some(code) if !stands_alone(code) => {
                    return Err(format!(
                        "{text}: only F1 to F20 and Dictation work without a modifier"
                    ))
                }
                Some(_) => {}
            }
        }
        Ok(hotkey)
    }
}

/// What the hotkey did, for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Pressed,
    Released,
    /// A modifier-only hotkey became part of another shortcut, such as Command in Command+C,
    /// after it was already reported as pressed.
    Interrupted,
}

/// Follows the keyboard and reports when the hotkey goes down and up.
#[derive(Debug)]
pub struct Matcher {
    hotkey: Hotkey,
    held: Held,
    /// Keys other than modifiers that are down, which keep a modifier-only hotkey from firing.
    keys: HashSet<u16>,
    active: bool,
    /// Whether a modifier-only hotkey may fire: not after another modifier joined, until every
    /// modifier is up again.
    armed: bool,
    /// Whether a mouse button is down, as when Option turns a drag into a copy.
    button_down: bool,
}

impl Matcher {
    pub fn new(hotkey: Hotkey, held: Held) -> Self {
        Self {
            hotkey,
            held,
            keys: HashSet::new(),
            active: false,
            armed: held.is_empty(),
            button_down: false,
        }
    }

    /// The signal, if any, and whether to keep the key from other apps.
    pub fn key_down(&mut self, code: u16, repeat: bool) -> (Option<Signal>, bool) {
        match self.hotkey.key {
            Some(key) if code == key => {
                if self.active {
                    (None, true)
                } else if !repeat && self.hotkey.matches(self.held) {
                    self.active = true;
                    (Some(Signal::Pressed), true)
                } else {
                    (None, false)
                }
            }
            Some(_) => (None, false),
            None => {
                self.keys.insert(code);
                // A key typed with a modifier down makes that press part of a shortcut.
                if !self.held.is_empty() {
                    self.armed = false;
                }
                if std::mem::take(&mut self.active) {
                    (Some(Signal::Interrupted), false)
                } else {
                    (None, false)
                }
            }
        }
    }

    /// A click or scroll with a modifier-only hotkey down, as in Command-click or Control-scroll,
    /// uses it for something else.
    pub fn mouse_down(&mut self) -> Option<Signal> {
        if self.hotkey.key.is_some() {
            return None;
        }
        if !self.held.is_empty() {
            self.armed = false;
        }
        std::mem::take(&mut self.active).then_some(Signal::Interrupted)
    }

    pub fn key_up(&mut self, code: u16) -> (Option<Signal>, bool) {
        if self.hotkey.key == Some(code) && self.active {
            self.active = false;
            return (Some(Signal::Released), true);
        }
        self.keys.remove(&code);
        (None, false)
    }

    /// A modifier-only hotkey does not fire while a mouse button is down.
    pub fn set_button_down(&mut self, down: bool) {
        self.button_down = down;
    }

    pub fn modifiers_changed(&mut self, held: Held) -> Option<Signal> {
        self.held = held;
        if self.hotkey.key.is_some() {
            return None;
        }
        if held.is_empty() {
            self.armed = true;
            return std::mem::take(&mut self.active).then_some(Signal::Released);
        }
        if self.hotkey.has_extra(held) {
            self.armed = false;
            return std::mem::take(&mut self.active).then_some(Signal::Interrupted);
        }
        if self.active && !self.hotkey.matches(held) {
            // One of the hotkey's modifiers went up while the others are still down.
            self.active = false;
            return Some(Signal::Released);
        }
        if !self.active
            && self.armed
            && self.keys.is_empty()
            && !self.button_down
            && self.hotkey.matches(held)
        {
            self.active = true;
            return Some(Signal::Pressed);
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Recorded(Hotkey),
    /// A letter or similar key was pressed without a modifier. Recording goes on.
    NeedsModifier,
    Cancelled,
}

/// Records the next hotkey. A key counts with the modifiers held when it went down, the last key
/// wins, and the result is taken once every key is up, so letting go of the new hotkey cannot
/// trigger it. Modifiers pressed and let go with no key between become a modifier-only hotkey.
/// Escape on its own cancels.
#[derive(Debug, Default)]
pub struct Recorder {
    held: Held,
    /// Every modifier held since the keyboard was last empty.
    peak: Held,
    keys: HashSet<u16>,
    /// The last named key pressed, while it is still down.
    last_key: Option<u16>,
    /// Every modifier held since `last_key` went down.
    key_held: Held,
    pressed_key: bool,
    combination: Option<Hotkey>,
}

impl Recorder {
    /// Starts with the modifiers already down, as when Command-clicking the menu item, so a key
    /// pressed with them counts. They alone never become the hotkey unless pressed again.
    pub fn new(held: Held) -> Self {
        Self {
            held,
            ..Self::default()
        }
    }

    pub fn key_down(&mut self, code: u16, repeat: bool) -> Option<Outcome> {
        if repeat {
            return None;
        }
        if code == KEY_ESCAPE && self.held.is_empty() {
            return Some(Outcome::Cancelled);
        }
        self.keys.insert(code);
        self.pressed_key = true;
        // A key with no name here, such as Fn, leaves the combination as it was.
        key_name(code)?;
        self.last_key = Some(code);
        self.key_held = self.held;
        if self.held.is_empty() && !stands_alone(code) {
            self.combination = None;
            return Some(Outcome::NeedsModifier);
        }
        self.combination = Some(Hotkey::recorded(self.held, Some(code)));
        None
    }

    pub fn key_up(&mut self, code: u16) -> Option<Outcome> {
        self.keys.remove(&code);
        if self.last_key == Some(code) {
            self.last_key = None;
        }
        self.finish()
    }

    pub fn modifiers_changed(&mut self, held: Held) -> Option<Outcome> {
        self.held = held;
        self.peak = self.peak.union(held);
        // Modifiers added to a key already held, as the menu asks after a bare letter, join it.
        // Letting go of some before the key keeps them.
        if let Some(code) = self.last_key {
            self.key_held = self.key_held.union(held);
            if !self.key_held.is_empty() {
                self.combination = Some(Hotkey::recorded(self.key_held, Some(code)));
            }
        }
        self.finish()
    }

    fn finish(&mut self) -> Option<Outcome> {
        if !self.keys.is_empty() || !self.held.is_empty() {
            return None;
        }
        let outcome = match self.combination.take() {
            Some(hotkey) => Some(Outcome::Recorded(hotkey)),
            None if !self.pressed_key && !self.peak.is_empty() => {
                Some(Outcome::Recorded(Hotkey::recorded(self.peak, None)))
            }
            None => None,
        };
        self.peak = Held::default();
        self.pressed_key = false;
        outcome
    }
}

/// F1 to F20 and the dictation key work as a hotkey alone; any other key needs a modifier, or it
/// could not be typed.
fn stands_alone(code: u16) -> bool {
    code == KEY_DICTATION
        || key_name(code).is_some_and(|name| name.len() > 1 && name.starts_with('F'))
}

/// The key code for a name, ignoring case.
fn key_code(lower: &str) -> Option<u16> {
    let name = match lower {
        "esc" => "escape",
        "return" => "enter",
        "arrowleft" => "left",
        "arrowright" => "right",
        "arrowup" => "up",
        "arrowdown" => "down",
        _ => lower
            .strip_prefix("key")
            .filter(|rest| rest.len() == 1)
            .or_else(|| lower.strip_prefix("digit").filter(|rest| rest.len() == 1))
            .unwrap_or(lower),
    };
    KEYS.iter()
        .find(|(_, key)| key.eq_ignore_ascii_case(name))
        .map(|(code, _)| *code)
}

fn key_name(code: u16) -> Option<&'static str> {
    KEYS.iter().find(|(c, _)| *c == code).map(|(_, name)| *name)
}

/// Virtual key codes on ANSI keyboards and their names.
const KEYS: &[(u16, &str)] = &[
    (0x00, "A"),
    (0x01, "S"),
    (0x02, "D"),
    (0x03, "F"),
    (0x04, "H"),
    (0x05, "G"),
    (0x06, "Z"),
    (0x07, "X"),
    (0x08, "C"),
    (0x09, "V"),
    // The extra key beside left Shift on ISO keyboards.
    (0x0a, "IntlBackslash"),
    (0x0b, "B"),
    (0x0c, "Q"),
    (0x0d, "W"),
    (0x0e, "E"),
    (0x0f, "R"),
    (0x10, "Y"),
    (0x11, "T"),
    (0x12, "1"),
    (0x13, "2"),
    (0x14, "3"),
    (0x15, "4"),
    (0x16, "6"),
    (0x17, "5"),
    (0x18, "Equal"),
    (0x19, "9"),
    (0x1a, "7"),
    (0x1b, "Minus"),
    (0x1c, "8"),
    (0x1d, "0"),
    (0x1e, "BracketRight"),
    (0x1f, "O"),
    (0x20, "U"),
    (0x21, "BracketLeft"),
    (0x22, "I"),
    (0x23, "P"),
    (0x24, "Enter"),
    (0x25, "L"),
    (0x26, "J"),
    (0x27, "Quote"),
    (0x28, "K"),
    (0x29, "Semicolon"),
    (0x2a, "Backslash"),
    (0x2b, "Comma"),
    (0x2c, "Slash"),
    (0x2d, "N"),
    (0x2e, "M"),
    (0x2f, "Period"),
    (0x30, "Tab"),
    (0x31, "Space"),
    (0x32, "Backquote"),
    (0x33, "Backspace"),
    (0x35, "Escape"),
    (0x40, "F17"),
    (0x41, "NumpadDecimal"),
    (0x43, "NumpadMultiply"),
    (0x45, "NumpadAdd"),
    (0x47, "NumLock"),
    (0x4b, "NumpadDivide"),
    (0x4c, "NumpadEnter"),
    (0x4e, "NumpadSubtract"),
    (0x4f, "F18"),
    (0x50, "F19"),
    (0x51, "NumpadEqual"),
    (0x52, "Numpad0"),
    (0x53, "Numpad1"),
    (0x54, "Numpad2"),
    (0x55, "Numpad3"),
    (0x56, "Numpad4"),
    (0x57, "Numpad5"),
    (0x58, "Numpad6"),
    (0x59, "Numpad7"),
    (0x5a, "F20"),
    (0x5b, "Numpad8"),
    (0x5c, "Numpad9"),
    // Japanese keyboards.
    (0x5d, "IntlYen"),
    (0x5e, "IntlRo"),
    (0x5f, "NumpadComma"),
    (0x60, "F5"),
    (0x61, "F6"),
    (0x62, "F7"),
    (0x63, "F3"),
    (0x64, "F8"),
    (0x65, "F9"),
    (0x66, "Lang2"),
    (0x67, "F11"),
    (0x68, "Lang1"),
    (0x69, "F13"),
    (0x6a, "F16"),
    (0x6b, "F14"),
    (0x6d, "F10"),
    (0x6f, "F12"),
    (0x71, "F15"),
    (0x72, "Insert"),
    (0x73, "Home"),
    (0x74, "PageUp"),
    (0x75, "Delete"),
    (0x76, "F4"),
    (0x77, "End"),
    (0x78, "F2"),
    (0x79, "PageDown"),
    (0x7a, "F1"),
    (0x7b, "Left"),
    (0x7c, "Right"),
    (0x7d, "Down"),
    (0x7e, "Up"),
    (KEY_DICTATION, "Dictation"),
];

/// The microphone key on F5 of recent Mac keyboards, when F keys are not set to act as standard
/// function keys.
const KEY_DICTATION: u16 = 0xb0;

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_D: u16 = 0x02;
    const KEY_SPACE: u16 = 0x31;
    const KEY_C: u16 = 0x08;
    const KEY_F5: u16 = 0x60;

    fn held(left: &[usize], right: &[usize]) -> Held {
        let mut held = Held::default();
        for &i in left {
            held.left[i] = true;
        }
        for &i in right {
            held.right[i] = true;
        }
        held
    }

    fn hotkey(text: &str) -> Hotkey {
        text.parse().unwrap()
    }

    #[test]
    fn parses_and_writes_hotkeys() {
        for text in [
            "Alt+Space",
            "Ctrl+Shift+D",
            "RightCmd",
            "LeftAlt+RightCmd",
            "F5",
            "Cmd+Up",
        ] {
            assert_eq!(hotkey(text).to_string(), text);
        }
        assert_eq!(hotkey("option+command+KeyD").to_string(), "Alt+Cmd+D");
        assert_eq!(hotkey("right option").to_string(), "RightAlt");
        assert_eq!(hotkey("Cmd+Left").to_string(), "Cmd+Left");
        assert_eq!(hotkey("super+Digit1").to_string(), "Cmd+1");
        assert_eq!(hotkey("CommandOrControl+Space").to_string(), "Cmd+Space");
        assert!("Ctrl+Ctrl+D".parse::<Hotkey>().is_err());
        assert!("Ctrl+D+E".parse::<Hotkey>().is_err());
        assert!("Ctrl+Nope".parse::<Hotkey>().is_err());
        assert!("".parse::<Hotkey>().is_err());
        assert!(
            "D".parse::<Hotkey>().is_err(),
            "a letter alone would block typing it"
        );
    }

    #[test]
    fn every_key_name_parses_back() {
        for (code, name) in KEYS {
            assert_eq!(hotkey(&format!("Ctrl+{name}")).key, Some(*code), "{name}");
        }
    }

    #[test]
    fn reads_sides_from_event_flags() {
        // Right Command: its device bit and the side-blind Command bit.
        assert_eq!(Held::from_flags(0x0010_0010), held(&[], &[COMMAND]));
        assert_eq!(Held::from_flags(0x0008_0020), held(&[OPTION], &[]));
        assert_eq!(Held::from_flags(0x0010_0000), held(&[COMMAND], &[]));
        assert!(
            Held::from_flags(0x0080_0000).is_empty(),
            "Fn is not a modifier here"
        );
    }

    #[test]
    fn a_key_combination_fires_with_exactly_its_modifiers() {
        let mut matcher = Matcher::new(hotkey("Alt+Space"), Held::default());
        matcher.modifiers_changed(held(&[], &[OPTION]));
        assert_eq!(
            matcher.key_down(KEY_SPACE, false),
            (Some(Signal::Pressed), true)
        );
        assert_eq!(matcher.key_down(KEY_SPACE, true), (None, true));
        assert_eq!(matcher.key_up(KEY_SPACE), (Some(Signal::Released), true));

        matcher.modifiers_changed(held(&[OPTION, SHIFT], &[]));
        assert_eq!(matcher.key_down(KEY_SPACE, false), (None, false));
    }

    #[test]
    fn sides_must_match() {
        let mut matcher = Matcher::new(hotkey("RightAlt+Space"), Held::default());
        matcher.modifiers_changed(held(&[OPTION], &[]));
        assert_eq!(matcher.key_down(KEY_SPACE, false), (None, false));
        matcher.key_up(KEY_SPACE);
        matcher.modifiers_changed(held(&[], &[OPTION]));
        assert_eq!(
            matcher.key_down(KEY_SPACE, false),
            (Some(Signal::Pressed), true)
        );
    }

    #[test]
    fn a_lone_modifier_fires_on_its_own() {
        let mut matcher = Matcher::new(hotkey("RightCmd"), Held::default());
        assert_eq!(matcher.modifiers_changed(held(&[COMMAND], &[])), None);
        assert_eq!(matcher.modifiers_changed(Held::default()), None);
        assert_eq!(
            matcher.modifiers_changed(held(&[], &[COMMAND])),
            Some(Signal::Pressed)
        );
        assert_eq!(
            matcher.modifiers_changed(Held::default()),
            Some(Signal::Released)
        );
    }

    #[test]
    fn a_lone_modifier_used_in_a_shortcut_is_interrupted() {
        let mut matcher = Matcher::new(hotkey("RightCmd"), Held::default());
        matcher.modifiers_changed(held(&[], &[COMMAND]));
        assert_eq!(
            matcher.key_down(KEY_C, false),
            (Some(Signal::Interrupted), false)
        );
        matcher.key_up(KEY_C);
        assert_eq!(matcher.modifiers_changed(Held::default()), None);

        matcher.modifiers_changed(held(&[], &[COMMAND]));
        assert_eq!(
            matcher.modifiers_changed(held(&[SHIFT], &[COMMAND])),
            Some(Signal::Interrupted)
        );
        // Letting go of Shift first does not make it fire again.
        assert_eq!(matcher.modifiers_changed(held(&[], &[COMMAND])), None);
        assert_eq!(matcher.modifiers_changed(Held::default()), None);
    }

    #[test]
    fn typing_and_clicking_alone_leave_a_lone_modifier_ready() {
        let mut matcher = Matcher::new(hotkey("RightCmd"), Held::default());
        matcher.key_down(KEY_D, false);
        matcher.key_up(KEY_D);
        assert_eq!(matcher.mouse_down(), None);
        assert_eq!(
            matcher.modifiers_changed(held(&[], &[COMMAND])),
            Some(Signal::Pressed)
        );
    }

    #[test]
    fn a_lone_modifier_does_not_fire_during_a_drag() {
        let mut matcher = Matcher::new(hotkey("RightAlt"), Held::default());
        matcher.set_button_down(true);
        assert_eq!(matcher.modifiers_changed(held(&[], &[OPTION])), None);
        matcher.modifiers_changed(Held::default());
        matcher.set_button_down(false);
        assert_eq!(
            matcher.modifiers_changed(held(&[], &[OPTION])),
            Some(Signal::Pressed)
        );
    }

    #[test]
    fn a_lone_modifier_does_not_fire_while_a_key_is_down() {
        let mut matcher = Matcher::new(hotkey("RightAlt"), Held::default());
        matcher.key_down(KEY_D, false);
        assert_eq!(matcher.modifiers_changed(held(&[], &[OPTION])), None);
    }

    #[test]
    fn records_a_combination_once_every_key_is_up() {
        let mut recorder = Recorder::default();
        assert_eq!(recorder.modifiers_changed(held(&[CONTROL], &[])), None);
        assert_eq!(recorder.modifiers_changed(held(&[CONTROL], &[SHIFT])), None);
        assert_eq!(recorder.key_down(KEY_D, false), None);
        assert_eq!(recorder.key_up(KEY_D), None);
        assert_eq!(recorder.modifiers_changed(held(&[CONTROL], &[])), None);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("Ctrl+Shift+D")))
        );
    }

    #[test]
    fn records_the_side_of_option_and_command() {
        let mut recorder = Recorder::default();
        recorder.modifiers_changed(held(&[], &[COMMAND]));
        recorder.modifiers_changed(held(&[OPTION], &[COMMAND]));
        recorder.key_down(KEY_SPACE, false);
        recorder.key_up(KEY_SPACE);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("LeftAlt+RightCmd+Space")))
        );
    }

    #[test]
    fn records_a_lone_modifier() {
        let mut recorder = Recorder::default();
        recorder.modifiers_changed(held(&[], &[OPTION]));
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("RightAlt")))
        );
    }

    #[test]
    fn a_letter_alone_needs_a_modifier() {
        let mut recorder = Recorder::default();
        assert_eq!(
            recorder.key_down(KEY_D, false),
            Some(Outcome::NeedsModifier)
        );
        assert_eq!(recorder.key_up(KEY_D), None);
        // Recording goes on.
        recorder.key_down(KEY_F5, false);
        assert_eq!(
            recorder.key_up(KEY_F5),
            Some(Outcome::Recorded(hotkey("F5")))
        );
        recorder.key_down(KEY_DICTATION, false);
        assert_eq!(
            recorder.key_up(KEY_DICTATION),
            Some(Outcome::Recorded(hotkey("Dictation")))
        );
    }

    #[test]
    fn a_modifier_added_to_a_held_letter_records_both() {
        let mut recorder = Recorder::default();
        assert_eq!(
            recorder.key_down(KEY_D, false),
            Some(Outcome::NeedsModifier)
        );
        assert_eq!(recorder.modifiers_changed(held(&[], &[COMMAND])), None);
        assert_eq!(recorder.key_up(KEY_D), None);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("RightCmd+D")))
        );
    }

    #[test]
    fn both_sides_of_a_modifier_make_a_chord() {
        let mut recorder = Recorder::default();
        recorder.modifiers_changed(held(&[COMMAND], &[]));
        recorder.modifiers_changed(held(&[COMMAND], &[COMMAND]));
        let recorded = recorder.modifiers_changed(Held::default());
        assert_eq!(
            recorded,
            Some(Outcome::Recorded(hotkey("LeftCmd+RightCmd")))
        );
        assert_eq!(hotkey("RightCmd+LeftCmd").to_string(), "LeftCmd+RightCmd");
        assert!("LeftCmd+LeftCmd".parse::<Hotkey>().is_err());

        let mut matcher = Matcher::new(hotkey("LeftCmd+RightCmd"), Held::default());
        assert_eq!(matcher.modifiers_changed(held(&[COMMAND], &[])), None);
        assert_eq!(
            matcher.modifiers_changed(held(&[COMMAND], &[COMMAND])),
            Some(Signal::Pressed)
        );
    }

    #[test]
    fn modifiers_held_when_recording_starts_count() {
        let mut recorder = Recorder::new(held(&[COMMAND], &[]));
        assert_eq!(recorder.key_down(KEY_D, false), None);
        recorder.key_up(KEY_D);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("LeftCmd+D")))
        );
        // Only letting go of them records nothing.
        let mut recorder = Recorder::new(held(&[COMMAND], &[]));
        assert_eq!(recorder.modifiers_changed(Held::default()), None);
    }

    #[test]
    fn modifiers_added_to_a_held_letter_all_count() {
        let mut recorder = Recorder::default();
        recorder.key_down(KEY_D, false);
        recorder.modifiers_changed(held(&[CONTROL], &[]));
        recorder.modifiers_changed(held(&[CONTROL, SHIFT], &[]));
        // Letting go of Shift before D keeps it.
        recorder.modifiers_changed(held(&[CONTROL], &[]));
        recorder.key_up(KEY_D);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("Ctrl+Shift+D")))
        );
    }

    #[test]
    fn the_other_side_interrupts_a_sided_lone_modifier() {
        let mut matcher = Matcher::new(hotkey("RightCmd"), Held::default());
        matcher.modifiers_changed(held(&[], &[COMMAND]));
        assert_eq!(
            matcher.modifiers_changed(held(&[COMMAND], &[COMMAND])),
            Some(Signal::Interrupted)
        );
        assert_eq!(matcher.modifiers_changed(held(&[], &[COMMAND])), None);
        assert_eq!(matcher.modifiers_changed(Held::default()), None);
    }

    #[test]
    fn a_click_interrupts_a_lone_modifier() {
        let mut matcher = Matcher::new(hotkey("RightCmd"), Held::default());
        matcher.modifiers_changed(held(&[], &[COMMAND]));
        assert_eq!(matcher.mouse_down(), Some(Signal::Interrupted));
        assert_eq!(matcher.modifiers_changed(Held::default()), None);

        let mut matcher = Matcher::new(hotkey("Alt+Space"), Held::default());
        assert_eq!(matcher.mouse_down(), None);
    }

    #[test]
    fn escape_cancels_but_a_modified_escape_records() {
        let mut recorder = Recorder::default();
        assert_eq!(
            recorder.key_down(KEY_ESCAPE, false),
            Some(Outcome::Cancelled)
        );
        let mut recorder = Recorder::default();
        recorder.modifiers_changed(held(&[CONTROL], &[]));
        assert_eq!(recorder.key_down(KEY_ESCAPE, false), None);
        recorder.key_up(KEY_ESCAPE);
        assert_eq!(
            recorder.modifiers_changed(Held::default()),
            Some(Outcome::Recorded(hotkey("Ctrl+Escape")))
        );
    }
}
