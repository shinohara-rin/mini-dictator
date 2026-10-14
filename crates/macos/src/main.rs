//! The macOS shell: a menu bar icon, a global hotkey, the microphone, the accessibility tree, and
//! paste. Everything else is `mini-dictator-core`.

#[cfg(target_os = "macos")]
mod app;
#[cfg(target_os = "macos")]
mod audio;
#[cfg(target_os = "macos")]
mod config;
#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod debug_log;
#[cfg(target_os = "macos")]
mod icons;
#[cfg(target_os = "macos")]
mod paste;
#[cfg(target_os = "macos")]
mod permissions;
#[cfg(target_os = "macos")]
mod sound;
#[cfg(target_os = "macos")]
mod switches;
// Pure Rust, so its tests run on every platform.
#[cfg(any(target_os = "macos", test))]
mod synth;

#[cfg(target_os = "macos")]
fn main() {
    app::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mini-dictator-macos runs on macOS only");
    std::process::exit(1);
}
