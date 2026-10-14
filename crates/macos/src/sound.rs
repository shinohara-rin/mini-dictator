//! Sounds for what the dictation is doing. `minimal` marks when the microphone opens and closes;
//! `verbose` adds a cue for each step of the pipeline, drawn by [`crate::synth`].

use std::cell::{Cell, RefCell};
use std::time::Instant;

use objc2::rc::Retained;
use objc2::AllocAnyThread;
use objc2_app_kit::NSSound;
use objc2_foundation::{NSData, NSString};
use serde::{Deserialize, Serialize};

pub use crate::synth::Jingle;
use crate::synth::Limiter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Off,
    /// A sound when the microphone opens and closes.
    Minimal,
    /// Also a sound for each segment heard, unclear words, nothing heard, catching up, text typed,
    /// and errors.
    Verbose,
}

/// The microphone cues, system sounds from `/System/Library/Sounds`.
pub enum Cue {
    Start,
    Stop,
}

const VOLUME: f32 = 0.5;
/// The verbose cues are frequent, so they sit a little below the microphone cues.
const JINGLE_VOLUME: f32 = 0.35;

/// Lives on the main thread, which plays every sound. Playing never waits for a sound to finish.
pub struct Sounds {
    level: Cell<Level>,
    jingles: Vec<(Jingle, Retained<NSSound>)>,
    limiter: RefCell<Limiter>,
}

impl Sounds {
    pub fn new(level: Level) -> Self {
        let jingles = Jingle::ALL
            .into_iter()
            .filter_map(|jingle| {
                let data = NSData::with_bytes(&jingle.wav());
                let sound = NSSound::initWithData(NSSound::alloc(), &data)?;
                sound.setVolume(JINGLE_VOLUME);
                Some((jingle, sound))
            })
            .collect();
        Sounds {
            level: Cell::new(level),
            jingles,
            limiter: RefCell::new(Limiter::new()),
        }
    }

    pub fn set_level(&self, level: Level) {
        self.level.set(level);
    }

    pub fn cue(&self, cue: Cue) {
        if self.level.get() == Level::Off {
            return;
        }
        let name = match cue {
            Cue::Start => "Tink",
            Cue::Stop => "Pop",
        };
        if let Some(sound) = NSSound::soundNamed(&NSString::from_str(name)) {
            sound.setVolume(VOLUME);
            sound.play();
        }
    }

    /// Plays in the verbose set only, unless the same cue just played.
    pub fn jingle(&self, jingle: Jingle) {
        if self.level.get() != Level::Verbose
            || !self.limiter.borrow_mut().allow(jingle, Instant::now())
        {
            return;
        }
        if let Some((_, sound)) = self.jingles.iter().find(|(j, _)| *j == jingle) {
            // A sound still playing ignores `play`; start it over instead.
            sound.stop();
            sound.play();
        }
    }
}
