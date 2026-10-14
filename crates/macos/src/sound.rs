//! Short sounds that mark when the microphone opens and closes.

use objc2_app_kit::NSSound;
use objc2_foundation::NSString;

/// A system sound, from `/System/Library/Sounds`.
pub enum Cue {
    Start,
    Stop,
}

/// Plays the cue at half the system volume, without waiting for it to finish.
pub fn play(cue: Cue) {
    let name = match cue {
        Cue::Start => "Tink",
        Cue::Stop => "Pop",
    };
    if let Some(sound) = NSSound::soundNamed(&NSString::from_str(name)) {
        sound.setVolume(0.5);
        sound.play();
    }
}
