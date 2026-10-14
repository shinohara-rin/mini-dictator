//! `~/Library/Application Support/Mini Dictator/config.toml`. The first launch writes the defaults.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
    /// For example `Alt+Space` or `Ctrl+Shift+D`. Tap to toggle, hold to talk.
    pub hotkey: String,
    /// Read the focused window for hotwords.
    pub read_screen: bool,
    /// Send the text before the cursor to the rewrite model.
    pub share_screen_text: bool,
    /// Play a short sound when the microphone opens and closes.
    pub beep: bool,
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
            share_screen_text: true,
            beep: true,
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

/// Writes the config file, readable by the user only since it can hold an API key.
fn write(text: &str) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let path = config_path();
    std::fs::create_dir_all(support_dir().join("models")).map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(text.as_bytes()).map_err(|e| e.to_string())
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
        assert!(!config.beep);
    }

    #[test]
    fn a_broken_file_is_left_alone() {
        assert!(with_hotkey("hotkey = ", "F5").is_err());
    }
}
