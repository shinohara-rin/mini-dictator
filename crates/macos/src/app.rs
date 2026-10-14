//! The event loop: tray menu, hotkey, and one dictation at a time.
//!
//! Tap the hotkey to start and tap again to stop, or hold it while speaking. Each dictation runs
//! `pipeline::run_dictation` on a worker thread. The microphone stream stays on the main thread,
//! and dropping it closes the audio channel, which ends the dictation.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use mini_dictator_core::asr::onnx::init_runtime;
use mini_dictator_core::asr::{self, SearchConfig};
use mini_dictator_core::context::{rank_terms, MemoryConfig, TermMemory};
use mini_dictator_core::pipeline::{run_dictation, Engine, EngineConfig, Event as CoreEvent};
use mini_dictator_core::rewrite::{OpenAiCompatible, Rewriter};
use mini_dictator_core::vad::{EnergyVad, SileroVad};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

use crate::config::{config_path, debug_log_path, support_dir, Config, APP_NAME};
use crate::context::{focused_window, is_trusted, AccessibilityReader};
use crate::debug_log;
use crate::icons::{icon, State};
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
    HotKey(GlobalHotKeyEvent),
    Loaded(Result<(Arc<Engine>, Switches), String>),
    Core(CoreEvent),
    /// The worker finished, with an error if the dictation failed.
    Done(Option<String>),
    CheckPermissions,
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
        SearchConfig::default(),
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

    let engine = Engine {
        asr: recognizer,
        context: Box::new(screen),
        memory: Arc::new(TermMemory::new(if config.screen_memory_seconds > 0 {
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
        })),
        rewriter: Box::new(rewriter),
        config: EngineConfig {
            monte_carlo,
            max_lag_seconds: config
                .uncertainty
                .max_lag_seconds
                .unwrap_or(EngineConfig::default().max_lag_seconds),
            share_screen_text: config.share_screen_text,
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
    debug: CheckMenuItem,
    open_config: MenuItem,
    open_folder: MenuItem,
    quit: MenuItem,
    beep: bool,
}

impl Ui {
    fn new(config: &Config) -> Ui {
        let status = MenuItem::new("Loading the model…", false, None);
        let toggle = MenuItem::new(format!("Start Dictation ({})", config.hotkey), false, None);
        let last = MenuItem::new("Nothing dictated yet", false, None);
        let permissions = MenuItem::new(permissions::summary(&permissions::missing()), false, None);
        let rewrite = CheckMenuItem::new(
            "Rewrite with LLM",
            !config.rewrite.model.is_empty(),
            config.rewrite.enabled,
            None,
        );
        let screen = CheckMenuItem::new("Read Screen for Names", true, config.read_screen, None);
        let debug = CheckMenuItem::new("Write Debug Log", true, config.debug, None);
        let open_config = MenuItem::new("Open Settings File…", true, None);
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
            debug,
            open_config,
            open_folder,
            quit,
            beep: config.beep,
        }
    }

    fn show(&self, state: State, status: &str) {
        let _ = self.tray.set_icon_templated(Some(icon(state)));
        self.status.set_text(status);
    }

    fn cue(&self, cue: Cue) {
        if self.beep {
            sound::play(cue);
        }
    }
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
    let config = match Config::load_or_create() {
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
    {
        let proxy = proxy.clone();
        GlobalHotKeyEvent::set_event_handler(Some(move |event| {
            let _ = proxy.send_event(AppEvent::HotKey(event));
        }));
    }

    let hotkeys = GlobalHotKeyManager::new().expect("hotkey manager");
    let hotkey = HotKey::from_str(&config.hotkey).unwrap_or_else(|error| {
        eprintln!("hotkey {}: {error}; using Alt+Space", config.hotkey);
        HotKey::from_str("Alt+Space").expect("default hotkey")
    });

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
    // Shared with the memory sampler, which pauses while a dictation reads the screen itself.
    let dictating = Arc::new(AtomicBool::new(false));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            // The status item must be created after the app finishes launching.
            Event::NewEvents(StartCause::Init) => {
                ui = Some(Ui::new(&config));
                if let Err(error) = hotkeys.register(hotkey) {
                    eprintln!("hotkey: {error}");
                }
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
            Event::UserEvent(AppEvent::HotKey(event)) if event.id == hotkey.id() => {
                let Some(ui) = &ui else { return };
                match (event.state, &dictation) {
                    (HotKeyState::Pressed, None) if !busy => {
                        if let Some(engine) = &engine {
                            dictation = start(engine.clone(), vad_path.clone(), proxy.clone(), ui);
                            busy = dictation.is_some();
                            dictating.store(busy, Ordering::Relaxed);
                        }
                    }
                    (HotKeyState::Pressed, Some(_)) => stop(&mut dictation, ui),
                    (HotKeyState::Released, Some(current))
                        if current.pressed_at.elapsed() >= HOLD_THRESHOLD =>
                    {
                        stop(&mut dictation, ui)
                    }
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
                        s.screen.store(ui.screen.is_checked(), Ordering::SeqCst);
                    }
                    if let Some(engine) = engine.as_ref().filter(|_| !ui.screen.is_checked()) {
                        engine.memory.clear();
                    }
                } else if event.id == ui.debug.id() {
                    debug_log::set_enabled(ui.debug.is_checked());
                } else if event.id == ui.open_config.id() {
                    open(&config_path());
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
                    CoreEvent::Inserted { text } => ui
                        .last
                        .set_text(format!("Last: {}", shorten(text.trim(), 40))),
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
                let summary = permissions::summary(&permissions::missing());
                if ui.permissions.text() != summary {
                    ui.permissions.set_text(summary);
                }
            }
            Event::UserEvent(AppEvent::Done(error)) => {
                busy = false;
                dictating.store(false, Ordering::Relaxed);
                let Some(ui) = &ui else { return };
                // The dictation failed while the microphone was on: stop recording.
                if dictation.take().is_some() {
                    ui.toggle.set_text("Start Dictation");
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
        let mut last: Option<((String, String), Instant)> = None;
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
                terms = match engine.context.snapshot() {
                    Ok(screen) => rank_terms(&screen, engine.config.max_hotwords),
                    Err(_) => Vec::new(),
                };
                last = Some((window, Instant::now()));
            }
            engine.memory.observe(&terms, Instant::now());
            // The menu stores the switch, then clears the memory. If it went off around this
            // write, the clear may have come first, so clear again.
            if !screen_on.load(Ordering::SeqCst) {
                engine.memory.clear();
            }
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
        ui.toggle.set_text("Start Dictation");
        ui.show(State::Working, "Finishing…");
    }
}
