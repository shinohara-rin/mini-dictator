//! The event loop: tray menu, hotkey, and one dictation at a time.
//!
//! Tap the hotkey to start and tap again to stop, or hold it while speaking. Change Hotkey records
//! a new one from the keyboard and writes it to the config file. Each dictation runs
//! `pipeline::run_dictation` on a worker thread. The microphone stream stays on the main thread,
//! and dropping it closes the audio channel, which ends the dictation.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, SearchConfig};
use mini_dictator_core::context::{rank_terms, MemoryConfig, TermMemory};
use mini_dictator_core::join::JoinConfig;
use mini_dictator_core::pipeline::{run_dictation, Engine, EngineConfig, Event as CoreEvent};
use mini_dictator_core::rewrite::{OpenAiCompatible, Rewriter};
use mini_dictator_core::vad::{EnergyVad, SileroVad};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

use crate::config::{debug_log_path, support_dir, Config, APP_NAME};
use crate::context::{focused_window, is_trusted, AccessibilityReader, FocusedWindow};
use crate::debug_log;
use crate::hotkey::{Hotkey, Outcome, Signal};
use crate::icons::{icon, State};
use crate::keyboard::{self, Capture, Listener};
use crate::paste::PasteSink;
use crate::permissions;
use crate::sound::{self, Cue};
use crate::switches::Switchable;

/// A press longer than this is push-to-talk: releasing the key stops the dictation.
const HOLD_THRESHOLD: Duration = Duration::from_millis(350);

/// How often the menu checks permissions, which the user can grant at any time in System Settings.
const PERMISSION_POLL: Duration = Duration::from_secs(2);
/// How often the term memory checks the focused window between dictations.
const MEMORY_SAMPLE: Duration = Duration::from_secs(5);
/// A full read of an unchanged window, for content that scrolls or updates. Each read briefly
/// holds up the app being read, so the memory reads again only when the window changes or this
/// much time has passed.
const MEMORY_REREAD: Duration = Duration::from_secs(30);

enum AppEvent {
    Menu(MenuEvent),
    HotKey(Signal),
    Loaded(Result<(Arc<Engine>, Switches), String>),
    Core(CoreEvent),
    /// The worker finished, with an error if the dictation failed.
    Done(Option<String>),
    CheckPermissions,
    /// The result of the recording with this number. A recording cancelled from the menu may
    /// still deliver one, which is then ignored.
    HotKeyRecorded(u64, Outcome),
}

struct Switches {
    rewrite: Arc<AtomicBool>,
    screen: Arc<AtomicBool>,
}

fn load_engine(config: &Config, debug: Arc<AtomicBool>) -> Result<(Engine, Switches), String> {
    init_runtime(config.onnxruntime_path().as_deref()).map_err(|e| format!("ONNX Runtime: {e}"))?;
    let recognizer = asr::load(
        &Config::resolve(&config.model_dir),
        performance_cores(),
        SearchConfig {
            language: Some(config.language.clone()).filter(|l| !l.is_empty()),
            ..SearchConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let mut monte_carlo = recognizer.monte_carlo();
    monte_carlo.samples = config.uncertainty.samples.unwrap_or(monte_carlo.samples);
    monte_carlo.relative_sigma = config
        .uncertainty
        .sigma
        .unwrap_or(monte_carlo.relative_sigma);

    let rewriter: Option<Box<dyn Rewriter>> = (!config.rewrite.model.is_empty()).then(|| {
        Box::new(OpenAiCompatible {
            base_url: config.rewrite.base_url.clone(),
            api_key: Some(config.rewrite.api_key.clone()).filter(|k| !k.is_empty()),
            model: config.rewrite.model.clone(),
            timeout: Duration::from_secs(config.rewrite.timeout_seconds),
        }) as Box<dyn Rewriter>
    });
    let rewriter = Switchable::new(rewriter, config.rewrite.enabled);
    let screen = Switchable::new(AccessibilityReader::default(), config.read_screen);
    let switches = Switches {
        rewrite: rewriter.enabled.clone(),
        screen: screen.enabled.clone(),
    };

    let memory = TermMemory::new(if config.screen_memory_seconds > 0 {
        MemoryConfig {
            half_life: Duration::from_secs(config.screen_memory_seconds),
            max_gap: MEMORY_SAMPLE * 2,
            ..MemoryConfig::default()
        }
    } else {
        MemoryConfig {
            max_terms: 0,
            ..MemoryConfig::default()
        }
    });
    memory.set_enabled(config.read_screen);

    let engine = Engine {
        asr: recognizer,
        context: Box::new(screen),
        memory: Arc::new(memory),
        rewriter: Box::new(rewriter),
        config: EngineConfig {
            monte_carlo,
            max_lag_seconds: config
                .uncertainty
                .max_lag_seconds
                .unwrap_or(EngineConfig::default().max_lag_seconds),
            share_screen_text: config.share_screen_text,
            join: JoinConfig {
                sentence_pause_seconds: config.sentence_pause_seconds,
            },
            debug,
            ..EngineConfig::default()
        },
    };
    Ok((engine, switches))
}

/// Performance cores on Apple silicon. ONNX Runtime splits work evenly across its threads, so a
/// thread on an efficiency core holds up the others.
fn performance_cores() -> usize {
    let mut cores: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    // SAFETY: the name is NUL-terminated and the output buffer is a c_int of the given size.
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&mut cores as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 && cores > 0 {
        cores as usize
    } else {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    }
}

struct Dictation {
    /// Dropping it stops the microphone and ends the dictation.
    stream: Option<cpal::Stream>,
    pressed_at: Instant,
}

struct Ui {
    tray: TrayIcon,
    status: MenuItem,
    toggle: MenuItem,
    last: MenuItem,
    permissions: MenuItem,
    rewrite: CheckMenuItem,
    screen: CheckMenuItem,
    change_hotkey: MenuItem,
    debug: CheckMenuItem,
    open_config: MenuItem,
    open_folder: MenuItem,
    quit: MenuItem,
    beep: bool,
    hotkey: String,
}

impl Ui {
    fn new(config: &Config) -> Ui {
        let status = MenuItem::new("Loading the model…", false, None);
        let toggle = MenuItem::new(start_label(&config.hotkey), false, None);
        let last = MenuItem::new("Nothing dictated yet", false, None);
        let permissions = MenuItem::new(permissions::summary(&permissions::missing()), false, None);
        let rewrite = CheckMenuItem::new(
            "Rewrite with LLM",
            !config.rewrite.model.is_empty(),
            config.rewrite.enabled,
            None,
        );
        let screen = CheckMenuItem::new("Read Screen for Names", true, config.read_screen, None);
        let change_hotkey = MenuItem::new(change_label(&config.hotkey), true, None);
        let debug = CheckMenuItem::new("Write Debug Log", true, config.debug, None);
        let open_config = MenuItem::new("Open Settings Folder…", true, None);
        let open_folder = MenuItem::new("Open Models Folder…", true, None);
        let quit = MenuItem::new(format!("Quit {APP_NAME}"), true, None);
        let menu = Menu::new();
        let separator = PredefinedMenuItem::separator();
        menu.append_items(&[
            &status,
            &toggle,
            &last,
            &permissions,
            &separator,
            &rewrite,
            &screen,
            &change_hotkey,
            &debug,
            &PredefinedMenuItem::separator(),
            &open_config,
            &open_folder,
            &PredefinedMenuItem::separator(),
            &quit,
        ])
        .expect("menu items");
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_icon_templated(icon(State::Working))
            .with_tooltip(APP_NAME)
            .build()
            .expect("tray icon");
        Ui {
            tray,
            status,
            toggle,
            last,
            permissions,
            rewrite,
            screen,
            change_hotkey,
            debug,
            open_config,
            open_folder,
            quit,
            beep: config.beep,
            hotkey: config.hotkey.clone(),
        }
    }

    fn show(&self, state: State, status: &str) {
        let _ = self.tray.set_icon_templated(Some(icon(state)));
        self.status.set_text(status);
    }

    fn set_hotkey(&mut self, hotkey: &str) {
        hotkey.clone_into(&mut self.hotkey);
        self.change_hotkey.set_text(change_label(hotkey));
        if self.toggle.text().starts_with("Start") {
            self.toggle.set_text(start_label(hotkey));
        }
    }

    fn cue(&self, cue: Cue) {
        if self.beep {
            sound::play(cue);
        }
    }
}

const RECORDING_LABEL: &str = "Press the New Hotkey (Esc to Cancel)…";
const NEEDS_MODIFIER_LABEL: &str = "Add ⌃, ⌥, ⇧ or ⌘ to That Key (Esc to Cancel)…";

fn start_label(hotkey: &str) -> String {
    format!("Start Dictation ({hotkey})")
}

fn change_label(hotkey: &str) -> String {
    format!("Change Hotkey ({hotkey})…")
}

fn shorten(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        text.to_owned()
    } else {
        format!("…{}", chars[chars.len() - max..].iter().collect::<String>())
    }
}

fn open(path: &std::path::Path) {
    let _ = std::process::Command::new("open").arg(path).spawn();
}

pub fn run() {
    let mut config = match Config::load_or_create() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("config: {error}; using defaults");
            Config::default()
        }
    };
    let debug = Arc::new(AtomicBool::new(config.debug));
    debug_log::init(debug_log_path(), debug.clone());

    let mut event_loop = EventLoopBuilder::<AppEvent>::with_user_event().build();
    // A menu bar app: no Dock icon, no app menu.
    event_loop.set_activation_policy(ActivationPolicy::Accessory);
    let proxy: EventLoopProxy<AppEvent> = event_loop.create_proxy();
    {
        let proxy = proxy.clone();
        MenuEvent::set_event_handler(Some(move |event| {
            let _ = proxy.send_event(AppEvent::Menu(event));
        }));
    }
    let mut hotkey = Hotkey::from_str(&config.hotkey).unwrap_or_else(|error| {
        eprintln!("hotkey: {error}; using Alt+Space");
        config.hotkey = "Alt+Space".into();
        Hotkey::from_str(&config.hotkey).expect("default hotkey")
    });
    // The listener needs Accessibility, so without it the permission poll tries again.
    let listen = {
        let proxy = proxy.clone();
        move |hotkey: Hotkey| -> Option<Listener> {
            let proxy = proxy.clone();
            keyboard::listen(hotkey, move |signal| {
                let _ = proxy.send_event(AppEvent::HotKey(signal));
            })
        }
    };
    let mut listener: Option<Listener> = None;
    // Whether the hotkey press still held down started the dictation.
    let mut started_by_press = false;
    // Whether the lone-modifier press still held down stops the dictation once it is let go.
    let mut stop_on_release = false;

    let vad_path = config
        .vad_model
        .as_deref()
        .map(Config::resolve)
        .filter(|p| p.exists());
    let mut ui: Option<Ui> = None;
    let mut engine: Option<Arc<Engine>> = None;
    let mut switches: Option<Switches> = None;
    let mut dictation: Option<Dictation> = None;
    let mut busy = false;
    let mut capture: Option<Capture> = None;
    let mut recording: u64 = 0;
    // Shared with the memory sampler, which pauses while a dictation reads the screen itself.
    let dictating = Arc::new(AtomicBool::new(false));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            // The status item must be created after the app finishes launching.
            Event::NewEvents(StartCause::Init) => {
                ui = Some(Ui::new(&config));
                listener = listen(hotkey);
                // Prompts for the permission that both reading the screen and pasting need.
                is_trusted(true);
                {
                    let proxy = proxy.clone();
                    std::thread::spawn(move || loop {
                        std::thread::sleep(PERMISSION_POLL);
                        if proxy.send_event(AppEvent::CheckPermissions).is_err() {
                            break;
                        }
                    });
                }
                let proxy = proxy.clone();
                let config = config.clone();
                let debug = debug.clone();
                std::thread::spawn(move || {
                    let loaded = load_engine(&config, debug)
                        .map(|(engine, found)| (Arc::new(engine), found));
                    let _ = proxy.send_event(AppEvent::Loaded(loaded));
                });
            }
            Event::UserEvent(AppEvent::Loaded(result)) => {
                let Some(ui) = &ui else { return };
                match result {
                    Ok((loaded, found)) => {
                        if config.screen_memory_seconds > 0 {
                            sample_screen(loaded.clone(), found.screen.clone(), dictating.clone());
                        }
                        debug_log::write(&format!(
                            "loaded {}",
                            Config::resolve(&config.model_dir).display()
                        ));
                        engine = Some(loaded);
                        switches = Some(found);
                        ui.toggle.set_enabled(true);
                        ui.show(State::Idle, "Ready");
                    }
                    Err(error) => {
                        eprintln!("model: {error}");
                        debug_log::write(&format!("model: {error}"));
                        ui.show(State::Error, &shorten(&format!("Model: {error}"), 60));
                    }
                }
            }
            // While recording a new hotkey, the current one does nothing.
            Event::UserEvent(AppEvent::HotKey(signal)) if capture.is_none() => {
                let Some(ui) = &ui else { return };
                let by_press = std::mem::take(&mut started_by_press);
                let pending_stop = std::mem::take(&mut stop_on_release);
                match (signal, &dictation) {
                    (Signal::Pressed, None) if !busy => {
                        if let Some(engine) = &engine {
                            dictation = start(engine.clone(), vad_path.clone(), proxy.clone(), ui);
                            busy = dictation.is_some();
                            started_by_press = busy;
                            dictating.store(busy, Ordering::Relaxed);
                        }
                    }
                    // A lone modifier may be the start of a shortcut such as Command-C, so it
                    // stops only when let go without one.
                    (Signal::Pressed, Some(_)) if hotkey.is_modifier_only() => {
                        stop_on_release = true;
                    }
                    (Signal::Pressed, Some(_)) => stop(&mut dictation, ui),
                    (Signal::Released, Some(_)) if pending_stop => stop(&mut dictation, ui),
                    (Signal::Released, Some(current))
                        if current.pressed_at.elapsed() >= HOLD_THRESHOLD =>
                    {
                        stop(&mut dictation, ui)
                    }
                    // A lone-modifier hotkey turned out to be part of another shortcut.
                    (Signal::Interrupted, Some(_)) if by_press => stop(&mut dictation, ui),
                    _ => {}
                }
            }
            Event::UserEvent(AppEvent::Menu(event)) => {
                let Some(ui) = &ui else { return };
                if event.id == ui.toggle.id() {
                    match (&dictation, &engine) {
                        (Some(_), _) => stop(&mut dictation, ui),
                        (None, Some(engine)) if !busy => {
                            dictation = start(engine.clone(), vad_path.clone(), proxy.clone(), ui);
                            busy = dictation.is_some();
                            dictating.store(busy, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                } else if event.id == ui.rewrite.id() {
                    if let Some(s) = &switches {
                        s.rewrite.store(ui.rewrite.is_checked(), Ordering::Relaxed);
                    }
                } else if event.id == ui.screen.id() {
                    if let Some(s) = &switches {
                        s.screen.store(ui.screen.is_checked(), Ordering::Relaxed);
                    }
                    if let Some(engine) = &engine {
                        engine.memory.set_enabled(ui.screen.is_checked());
                    }
                } else if event.id == ui.change_hotkey.id() {
                    if capture.take().is_some() {
                        // Clicked again while recording: keep the current hotkey.
                        ui.change_hotkey.set_text(change_label(&ui.hotkey));
                        listener = listen(hotkey);
                        return;
                    }
                    let proxy = proxy.clone();
                    recording += 1;
                    let current = recording;
                    capture = keyboard::record(move |outcome| {
                        let _ = proxy.send_event(AppEvent::HotKeyRecorded(current, outcome));
                    });
                    if capture.is_some() {
                        ui.change_hotkey.set_text(RECORDING_LABEL);
                        // Keys pressed while recording are not the current hotkey's. Without a
                        // listener, none of them can queue a signal that lands after recording.
                        listener = None;
                        started_by_press = false;
                        stop_on_release = false;
                    } else {
                        ui.status.set_text("Allow Accessibility to record a hotkey");
                        is_trusted(true);
                    }
                } else if event.id == ui.debug.id() {
                    debug_log::set_enabled(ui.debug.is_checked());
                } else if event.id == ui.open_config.id() {
                    open(&support_dir());
                } else if event.id == ui.open_folder.id() {
                    open(&support_dir().join("models"));
                } else if event.id == ui.quit.id() {
                    *control_flow = ControlFlow::Exit;
                }
            }
            Event::UserEvent(AppEvent::Core(event)) => {
                let Some(ui) = &ui else { return };
                match event {
                    CoreEvent::Listening { hotwords } => ui.show(
                        State::Listening,
                        &format!("Listening ({hotwords} names on screen)"),
                    ),
                    CoreEvent::Transcribed {
                        unclear_spans,
                        behind_seconds,
                        ..
                    } => {
                        let state = if dictation.is_some() {
                            State::Listening
                        } else {
                            State::Working
                        };
                        let status = match behind_seconds {
                            Some(seconds) => format!("Catching up, {seconds:.0} s behind"),
                            None => format!("Transcribed, {unclear_spans} unclear spans"),
                        };
                        ui.show(state, &status);
                    }
                    // The period typed when a dictation ends keeps the preview of its text.
                    CoreEvent::Inserted { text } if text.chars().any(char::is_alphanumeric) => ui
                        .last
                        .set_text(format!("Last: {}", shorten(text.trim(), 40))),
                    CoreEvent::Inserted { .. } => {}
                    CoreEvent::Warning(warning) => {
                        eprintln!("warning: {warning}");
                        debug_log::write(&format!("warning: {warning}"));
                        ui.last
                            .set_text(shorten(&format!("Warning: {warning}"), 60));
                    }
                    // The worker writes these to the log itself.
                    CoreEvent::Debug(_) | CoreEvent::Finished => {}
                }
            }
            Event::UserEvent(AppEvent::CheckPermissions) => {
                let Some(ui) = &ui else { return };
                if listener.is_none() && capture.is_none() {
                    listener = listen(hotkey);
                }
                let summary = permissions::summary(&permissions::missing());
                if ui.permissions.text() != summary {
                    ui.permissions.set_text(summary);
                }
            }
            Event::UserEvent(AppEvent::HotKeyRecorded(number, outcome)) => {
                let Some(ui) = &mut ui else { return };
                if number != recording || capture.is_none() {
                    return;
                }
                let new = match outcome {
                    Outcome::Recorded(new) => new,
                    Outcome::NeedsModifier => {
                        ui.change_hotkey.set_text(NEEDS_MODIFIER_LABEL);
                        return;
                    }
                    Outcome::Cancelled => {
                        capture = None;
                        ui.change_hotkey.set_text(change_label(&ui.hotkey));
                        listener = listen(hotkey);
                        return;
                    }
                };
                capture = None;
                hotkey = new;
                listener = listen(hotkey);
                let recorded = new.to_string();
                ui.set_hotkey(&recorded);
                if let Err(error) = Config::save_hotkey(&recorded) {
                    eprintln!("config: {error}");
                    ui.status
                        .set_text(shorten(&format!("Hotkey not saved: {error}"), 60));
                }
            }
            Event::UserEvent(AppEvent::Done(error)) => {
                busy = false;
                dictating.store(false, Ordering::Relaxed);
                let Some(ui) = &ui else { return };
                // The dictation failed while the microphone was on: stop recording.
                if dictation.take().is_some() {
                    ui.toggle.set_text(start_label(&ui.hotkey));
                    ui.cue(Cue::Stop);
                }
                match error {
                    Some(error) => {
                        eprintln!("dictation: {error}");
                        debug_log::write(&format!("dictation failed: {error}"));
                        ui.show(State::Error, &shorten(&format!("Error: {error}"), 60));
                    }
                    None => ui.show(State::Idle, "Ready"),
                }
            }
            _ => {}
        }
    });
}

/// Feeds the term memory every [`MEMORY_SAMPLE`] while no dictation runs and screen reading is on.
/// An unchanged window counts its last terms again without a read.
fn sample_screen(engine: Arc<Engine>, screen_on: Arc<AtomicBool>, dictating: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut last: Option<(FocusedWindow, Instant)> = None;
        let mut terms = Vec::new();
        loop {
            std::thread::sleep(MEMORY_SAMPLE);
            if !screen_on.load(Ordering::Relaxed) {
                // The menu cleared the memory; nothing seen before stays.
                last = None;
                terms.clear();
                continue;
            }
            if dictating.load(Ordering::Relaxed) {
                // The dictation reads the screen itself; read afresh after it.
                last = None;
                continue;
            }
            let Some(window) = focused_window() else {
                continue;
            };
            let fresh = last
                .as_ref()
                .is_some_and(|(seen, read)| *seen == window && read.elapsed() < MEMORY_REREAD);
            if !fresh {
                // Every term; the memory applies its own cap. A failed read is retried next time.
                (terms, last) = match engine.context.snapshot() {
                    Ok(screen) => (
                        rank_terms(&screen, usize::MAX),
                        Some((window, Instant::now())),
                    ),
                    Err(_) => (Vec::new(), None),
                };
            }
            // Ignored if the switch went off during the read.
            engine.memory.observe(&terms, Instant::now());
        }
    });
}

fn start(
    engine: Arc<Engine>,
    vad_path: Option<PathBuf>,
    proxy: EventLoopProxy<AppEvent>,
    ui: &Ui,
) -> Option<Dictation> {
    let (sender, receiver) = mpsc::channel();
    let stream = match crate::audio::start(sender) {
        Ok(stream) => stream,
        Err(error) => {
            ui.show(State::Error, &shorten(&format!("Microphone: {error}"), 60));
            return None;
        }
    };
    ui.toggle.set_text("Stop Dictation");
    ui.show(State::Listening, "Listening");
    ui.cue(Cue::Start);

    std::thread::spawn(move || {
        let mut sink = PasteSink;
        let events = proxy.clone();
        let mut forward = move |event: CoreEvent| match event {
            // Written here, so long screen text never passes through the event loop.
            CoreEvent::Debug(text) => debug_log::write(&text),
            event => {
                let _ = events.send_event(AppEvent::Core(event));
            }
        };
        debug_log::write("dictation started");
        let result = match vad_path.map(|p| SileroVad::load(&p)) {
            Some(Ok(vad)) => run_dictation(&engine, vad, receiver, &mut sink, &mut forward),
            Some(Err(error)) => {
                forward(CoreEvent::Warning(format!(
                    "Silero VAD: {error}; using the loudness detector"
                )));
                run_dictation(
                    &engine,
                    EnergyVad::default(),
                    receiver,
                    &mut sink,
                    &mut forward,
                )
            }
            None => run_dictation(
                &engine,
                EnergyVad::default(),
                receiver,
                &mut sink,
                &mut forward,
            ),
        };
        if let Ok(text) = &result {
            debug_log::write(&format!("dictation finished, inserted {text:?}"));
        }
        let _ = proxy.send_event(AppEvent::Done(result.err().map(|e| e.to_string())));
    });
    Some(Dictation {
        stream: Some(stream),
        pressed_at: Instant::now(),
    })
}

fn stop(dictation: &mut Option<Dictation>, ui: &Ui) {
    if let Some(mut current) = dictation.take() {
        current.stream.take();
        ui.cue(Cue::Stop);
        ui.toggle.set_text(start_label(&ui.hotkey));
        ui.show(State::Working, "Finishing…");
    }
}
