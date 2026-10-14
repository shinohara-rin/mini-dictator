//! The debug log, off by default: `debug.log` in the support directory, next to `config.toml`.
//!
//! It records what each dictation read from the screen, the hotwords, every decode with its noisy
//! decodes, the rewrite prompt and reply, and how long each step took. That includes text from the
//! screen, so the file is readable by the user only. When it passes [`MAX_BYTES`] it moves to
//! `debug.log.1`, replacing the previous one, so the two files never hold much more than twice that.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub const MAX_BYTES: u64 = 5 * 1024 * 1024;

pub struct DebugLog {
    path: PathBuf,
    max_bytes: u64,
    /// Shared with the engine, which builds debug events only while it is set.
    enabled: Arc<AtomicBool>,
    /// Open while logging is on.
    file: Mutex<Option<File>>,
}

impl DebugLog {
    pub fn new(path: PathBuf, max_bytes: u64, enabled: Arc<AtomicBool>) -> Self {
        Self {
            path,
            max_bytes,
            enabled,
            file: Mutex::new(None),
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        // Under the lock, so no write that started before this call lands after it.
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        self.enabled.store(enabled, Ordering::Relaxed);
        if !enabled {
            file.take();
        }
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Appends one timestamped entry when logging is on. An entry longer than a quarter of the
    /// limit is cut, so neither file grows past it. A failed write is dropped: the log must never
    /// stop a dictation.
    pub fn write(&self, text: &str) {
        if !self.enabled() {
            return;
        }
        let limit = (self.max_bytes / 4) as usize;
        let entry = if text.len() > limit {
            let mut end = limit;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            format!(
                "[{}] {}… ({} bytes cut)\n",
                timestamp(),
                &text[..end],
                text.len() - end
            )
        } else {
            format!("[{}] {text}\n", timestamp())
        };
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if self.enabled() {
            let _ = self.append(&mut file, entry.as_bytes());
        }
    }

    fn append(&self, file: &mut Option<File>, entry: &[u8]) -> std::io::Result<()> {
        let size = match file {
            Some(open) => open.metadata()?.len(),
            None => std::fs::metadata(&self.path).map_or(0, |m| m.len()),
        };
        if size > 0 && size + entry.len() as u64 > self.max_bytes {
            file.take();
            let mut rotated = self.path.clone().into_os_string();
            rotated.push(".1");
            std::fs::rename(&self.path, rotated)?;
        }
        if file.is_none() {
            if let Some(dir) = self.path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let opened = OpenOptions::new()
                .append(true)
                .create(true)
                .mode(0o600)
                .open(&self.path)?;
            // The mode applies only to a new file; an older one may be readable by others.
            opened.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            *file = Some(opened);
        }
        file.as_mut().expect("opened above").write_all(entry)
    }
}

static LOG: OnceLock<DebugLog> = OnceLock::new();

/// Sets up the app's log, on while `enabled` is set. Later calls change nothing.
pub fn init(path: PathBuf, enabled: Arc<AtomicBool>) {
    let _ = LOG.set(DebugLog::new(path, MAX_BYTES, enabled));
}

pub fn set_enabled(enabled: bool) {
    if let Some(log) = LOG.get() {
        log.set_enabled(enabled);
    }
}

/// Writes to the app's log when debug logging is on.
pub fn write(text: &str) {
    if let Some(log) = LOG.get() {
        log.write(text);
    }
}

/// Local time with milliseconds, such as `2026-10-14 13:46:58.123`.
fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() as libc::time_t;
    // SAFETY: both pointers are valid for the call, and localtime_r is thread safe.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&seconds, &mut tm) }.is_null() {
        return format!("{}.{:03}", now.as_secs(), now.subsec_millis());
    }
    format!(
        "{}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        now.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("md-debug-log-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("debug.log")
    }

    #[test]
    fn writes_nothing_while_off() {
        let path = scratch("off");
        let log = DebugLog::new(path.clone(), 1024, Arc::new(AtomicBool::new(false)));
        log.write("hidden");
        assert!(!path.exists());
        log.set_enabled(true);
        log.write("shown");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("] shown\n"), "{text}");
        assert!(!text.contains("hidden"));
    }

    #[test]
    fn moves_a_full_log_aside() {
        let path = scratch("rotate");
        let log = DebugLog::new(path.clone(), 100, Arc::new(AtomicBool::new(true)));
        for i in 0..6 {
            log.write(&format!("entry {i} {}", "x".repeat(20)));
        }
        let current = std::fs::read_to_string(&path).unwrap();
        let rotated = std::fs::read_to_string(path.with_extension("log.1")).unwrap();
        assert!(current.len() <= 100 && rotated.len() <= 100);
        assert!(current.contains("entry 5"), "{current}");
        assert!(rotated.contains("entry 4"), "{rotated}");
    }

    #[test]
    fn cuts_an_entry_too_long_for_the_limit() {
        let path = scratch("long");
        let log = DebugLog::new(path.clone(), 400, Arc::new(AtomicBool::new(true)));
        log.write(&"é".repeat(300));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.len() < 200, "{}", text.len());
        assert!(text.ends_with("bytes cut)\n"), "{text}");
    }

    #[test]
    fn makes_an_existing_log_private() {
        let path = scratch("mode");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "old\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        DebugLog::new(path.clone(), 1024, Arc::new(AtomicBool::new(true))).write("new");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
