//! `~/Library/Application Support/Mini Dictator/config.toml`. The first launch writes the defaults.

use std::path::{Path, PathBuf};

use mini_dictator_core::join::JoinConfig;
use serde::{Deserialize, Serialize};

use crate::sound::Level;

pub const APP_NAME: &str = "Mini Dictator";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// A sherpa-onnx Qwen3-ASR export or offline transducer directory. Relative paths start at the
    /// support directory.
    pub model_dir: PathBuf,
    /// Silero VAD. Without it, a loudness detector gates the audio.
    pub vad_model: Option<PathBuf>,
    /// ONNX Runtime. Empty means `ORT_DYLIB_PATH`, then `Contents/Frameworks` of the app bundle.
    pub onnxruntime: Option<PathBuf>,
    /// For example `Alt+Space`, `Ctrl+Shift+D`, `F13`, `Dictation` (the microphone key), or a
    /// modifier alone such as `RightCmd`.
    /// `LeftAlt` and `RightAlt` (or `Cmd`, `Ctrl`, `Shift`) name a side. Tap to toggle, hold to talk.
    pub hotkey: String,
    /// Read the focused window for hotwords.
    pub read_screen: bool,
    /// Remember names seen on screen for about this long (the half-life of their weight), sampling
    /// the focused window every few seconds between dictations. 0 turns the memory off.
    pub screen_memory_seconds: u64,
    /// Send the text before the cursor to the rewrite model.
    pub share_screen_text: bool,
    /// `"off"`, `"minimal"` (a sound when the microphone opens and closes), or `"verbose"` (also a
    /// cue for each segment heard, unclear words, nothing heard, catching up, text typed, and errors). The menu
    /// changes it until the app quits.
    // A config written before this field keeps its own setting, not the new default.
    #[serde(default)]
    pub sounds: Option<Level>,
    /// The older switch for the microphone sounds, read when `sounds` is not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beep: Option<bool>,
    /// Write `debug.log` next to this file: the screen text and hotwords, each decode and its
    /// uncertainty check, the rewrite prompt and reply, and timings. The menu turns it on and off
    /// until the app quits.
    pub debug: bool,
    /// Without a rewrite, a pause this long ends a sentence, and after a shorter one the next
    /// words continue it. With a rewrite, the model decides and is told the pause.
    pub sentence_pause_seconds: f32,
    pub uncertainty: Uncertainty,
    pub rewrite: Rewrite,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Uncertainty {
    /// Noisy decodes per segment. 0 turns the check off. Empty means the model's default.
    pub samples: Option<usize>,
    /// Noise as a fraction of each encoder dimension's spread. Empty means the model's default.
    pub sigma: Option<f32>,
    /// A segment that reaches the recognizer this many seconds after it was spoken skips the
    /// check, so a slow Mac catches up. Empty means 2.
    pub max_lag_seconds: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Rewrite {
    pub enabled: bool,
    /// An OpenAI-compatible base URL, such as `https://api.openai.com/v1` or `http://localhost:11434/v1`.
    pub base_url: String,
    pub model: String,
    /// Stored here because apps started from Finder do not see shell environment variables.
    /// The file is created readable by the user only.
    pub api_key: String,
    pub timeout_seconds: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"),
            vad_model: Some(PathBuf::from("models/silero_vad.onnx")),
            onnxruntime: None,
            hotkey: "Alt+Space".into(),
            read_screen: true,
            screen_memory_seconds: 60,
            share_screen_text: true,
            sounds: Some(Level::Verbose),
            beep: None,
            debug: false,
            sentence_pause_seconds: JoinConfig::default().sentence_pause_seconds,
            uncertainty: Uncertainty::default(),
            rewrite: Rewrite::default(),
        }
    }
}

impl Default for Rewrite {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:11434/v1".into(),
            model: "qwen3:4b".into(),
            api_key: String::new(),
            timeout_seconds: 15,
        }
    }
}

pub fn support_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("Library/Application Support").join(APP_NAME)
}

pub fn config_path() -> PathBuf {
    support_dir().join("config.toml")
}

/// Where a write to `path` should land: a symlinked config, such as one kept in a dotfiles
/// repository, is replaced at its target, even when that target does not exist yet.
fn link_target(path: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = std::fs::canonicalize(path) {
        return Ok(path);
    }
    let mut path = path.to_path_buf();
    // A bound on the hops, in case the links form a loop.
    for _ in 0..32 {
        let Ok(target) = std::fs::read_link(&path) else {
            return Ok(path);
        };
        path = match path.parent() {
            Some(parent) => parent.join(target),
            None => target,
        };
    }
    Err(format!("{}: too many symbolic links", path.display()))
}

/// Writes the config file, readable by the user only since it can hold an API key. The text goes
/// to a temporary file that then replaces the config, so a failed write leaves the old one whole.
fn write(text: &str) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    std::fs::create_dir_all(support_dir().join("models")).map_err(|e| e.to_string())?;
    let path = link_target(&config_path())?;
    let temporary = path.with_extension("toml.tmp");
    let written = (|| {
        // A leftover temporary could be open elsewhere with looser permissions, so start fresh:
        // `mode` applies only to a file this call creates.
        match std::fs::remove_file(&temporary) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&temporary, &path)
    })();
    written.map_err(|e| {
        let _ = std::fs::remove_file(&temporary);
        format!("{}: {e}", path.display())
    })
}

/// The config text with `hotkey` set, its comments and layout untouched. A file that does not
/// parse is an error rather than something to overwrite.
fn with_hotkey(text: &str, hotkey: &str) -> Result<String, String> {
    let mut document: toml_edit::DocumentMut = text.parse().map_err(|e| format!("{e}"))?;
    match document
        .get_mut("hotkey")
        .and_then(|item| item.as_value_mut())
    {
        Some(value) => {
            let decor = value.decor().clone();
            *value = hotkey.into();
            *value.decor_mut() = decor;
        }
        None => document["hotkey"] = toml_edit::value(hotkey),
    }
    Ok(document.to_string())
}

pub fn debug_log_path() -> PathBuf {
    support_dir().join("debug.log")
}

impl Config {
    /// Loads the config, writing the defaults first if there is none.
    pub fn load_or_create() -> Result<Self, String> {
        let path = config_path();
        if !path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self) -> Result<(), String> {
        write(&toml::to_string_pretty(self).map_err(|e| e.to_string())?)
    }

    /// Writes only the hotkey into the config file, keeping the rest as the user wrote it.
    pub fn save_hotkey(hotkey: &str) -> Result<(), String> {
        let path = config_path();
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(format!("{}: {error}", path.display())),
        };
        write(&with_hotkey(&text, hotkey).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    /// Expands `~` and resolves relative paths against the support directory.
    pub fn resolve(path: &Path) -> PathBuf {
        if let Ok(rest) = path.strip_prefix("~") {
            return std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(rest);
        }
        if path.is_relative() {
            return support_dir().join(path);
        }
        path.to_path_buf()
    }

    /// `sounds`, else `beep` from a config written before it, else the minimal sounds that config had.
    pub fn sound_level(&self) -> Level {
        self.sounds.unwrap_or(match self.beep {
            Some(false) => Level::Off,
            _ => Level::Minimal,
        })
    }

    /// ONNX Runtime: the config, then `ORT_DYLIB_PATH`, then the app bundle's Frameworks folder.
    pub fn onnxruntime_path(&self) -> Option<PathBuf> {
        if let Some(path) = self
            .onnxruntime
            .as_deref()
            .filter(|p| !p.as_os_str().is_empty())
        {
            return Some(Self::resolve(path));
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            return None;
        }
        let exe = std::env::current_exe().ok()?;
        Some(
            exe.parent()?
                .parent()?
                .join("Frameworks/libonnxruntime.dylib"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dangling_config_link_is_written_at_its_target() {
        let dir = std::env::temp_dir().join(format!("mini-dictator-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("dotfiles")).unwrap();
        let link = dir.join("config.toml");
        std::os::unix::fs::symlink("dotfiles/dictator.toml", &link).unwrap();
        assert_eq!(link_target(&link), Ok(dir.join("dotfiles/dictator.toml")));
        // A loop is refused rather than overwritten.
        let looped = dir.join("loop.toml");
        std::os::unix::fs::symlink("loop.toml", &looped).unwrap();
        assert!(link_target(&looped).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn setting_the_hotkey_keeps_the_rest_of_the_file() {
        let text = "# mine\nhotkey = \"Alt+Space\" # tap or hold\nbeep = false\n\n[rewrite]\nmodel = \"x\"\n";
        assert_eq!(
            with_hotkey(text, "Ctrl+Shift+D").unwrap(),
            "# mine\nhotkey = \"Ctrl+Shift+D\" # tap or hold\nbeep = false\n\n[rewrite]\nmodel = \"x\"\n"
        );
    }

    #[test]
    fn a_missing_hotkey_goes_before_the_tables() {
        let text = "beep = false\n\n[rewrite]\nmodel = \"x\"\n";
        let updated = with_hotkey(text, "F5").unwrap();
        let config: Config = toml::from_str(&updated).unwrap();
        assert_eq!(config.hotkey, "F5");
        assert_eq!(config.rewrite.model, "x");
        assert_eq!(config.sound_level(), Level::Off);
    }

    #[test]
    fn sounds_win_over_the_older_beep_switch() {
        let level = |text| toml::from_str::<Config>(text).unwrap().sound_level();
        assert_eq!(
            level("sounds = \"verbose\"\nbeep = false\n"),
            Level::Verbose
        );
        assert_eq!(level("beep = true\n"), Level::Minimal);
        assert_eq!(level("beep = false\n"), Level::Off);
        // A config from before either switch had the microphone sounds.
        assert_eq!(level(""), Level::Minimal);
        assert_eq!(Config::default().sound_level(), Level::Verbose);
    }

    #[test]
    fn a_broken_file_is_left_alone() {
        assert!(with_hotkey("hotkey = ", "F5").is_err());
    }
}
