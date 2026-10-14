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
use mini_dictator_core::pipeline::{run_dictation, Engine, EngineConfig, Event as CoreEvent};
use mini_dictator_core::rewrite::{OpenAiCompatible, Rewriter};
use mini_dictator_core::vad::{EnergyVad, SileroVad};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

use crate::config::{config_path, support_dir, Config, APP_NAME};
use crate::context::{is_trusted, AccessibilityReader};
use crate::icons::{icon, State};
use crate::paste::PasteSink;
use crate::sound::{self, Cue};
use crate::switches::Switchable;

/// A press longer than this is push-to-talk: releasing the key stops the dictation.
const HOLD_THRESHOLD: Duration = Duration::from_millis(350);

enum AppEvent {
    Menu(MenuEvent),
    HotKey(GlobalHotKeyEvent),
    Loaded(Result<(Arc<Engine>, Switches), String>),
    Core(CoreEvent),
    /// The worker finished, with an error if the dictation failed.
    Done(Option<String>),
}

struct Switches {
    rewrite: Arc<AtomicBool>,
    screen: Arc<AtomicBool>,
}

fn load_engine(config: &Config) -> Result<(Engine, Switches), String> {
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
        rewriter: Box::new(rewriter),
        config: EngineConfig {
            monte_carlo,
            share_screen_text: config.share_screen_text,
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
    rewrite: CheckMenuItem,
    screen: CheckMenuItem,
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
        let rewrite = CheckMenuItem::new(
            "Rewrite with LLM",
            !config.rewrite.model.is_empty(),
            config.rewrite.enabled,
            None,
        );
        let screen = CheckMenuItem::new("Read Screen for Names", true, config.read_screen, None);
        let open_config = MenuItem::new("Open Settings File…", true, None);
        let open_folder = MenuItem::new("Open Models Folder…", true, None);
        let quit = MenuItem::new(format!("Quit {APP_NAME}"), true, None);
        let menu = Menu::new();
        let separator = PredefinedMenuItem::separator();
        menu.append_items(&[
            &status,
            &toggle,
            &last,
            &separator,
            &rewrite,
            &screen,
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
            rewrite,
            screen,
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
                let proxy = proxy.clone();
                let config = config.clone();
                std::thread::spawn(move || {
                    let loaded =
                        load_engine(&config).map(|(engine, found)| (Arc::new(engine), found));
                    let _ = proxy.send_event(AppEvent::Loaded(loaded));
                });
            }
            Event::UserEvent(AppEvent::Loaded(result)) => {
                let Some(ui) = &ui else { return };
                match result {
                    Ok((loaded, found)) => {
                        engine = Some(loaded);
                        switches = Some(found);
                        ui.toggle.set_enabled(true);
                        ui.show(State::Idle, "Ready");
                    }
                    Err(error) => {
                        eprintln!("model: {error}");
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
                    CoreEvent::Transcribed { unclear_spans, .. } => {
                        let state = if dictation.is_some() {
                            State::Listening
                        } else {
                            State::Working
                        };
                        ui.show(
                            state,
                            &format!("Transcribed, {unclear_spans} unclear spans"),
                        );
                    }
                    CoreEvent::Inserted { text } => ui
                        .last
                        .set_text(format!("Last: {}", shorten(text.trim(), 40))),
                    CoreEvent::Warning(warning) => {
                        eprintln!("warning: {warning}");
                        ui.last
                            .set_text(shorten(&format!("Warning: {warning}"), 60));
                    }
                    CoreEvent::Finished => {}
                }
            }
            Event::UserEvent(AppEvent::Done(error)) => {
                busy = false;
                let Some(ui) = &ui else { return };
                // The dictation failed while the microphone was on: stop recording.
                if dictation.take().is_some() {
                    ui.toggle.set_text("Start Dictation");
                    ui.cue(Cue::Stop);
                }
                match error {
                    Some(error) => {
                        eprintln!("dictation: {error}");
                        ui.show(State::Error, &shorten(&format!("Error: {error}"), 60));
                    }
                    None => ui.show(State::Idle, "Ready"),
                }
            }
            _ => {}
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
        let mut forward = move |event: CoreEvent| {
            let _ = events.send_event(AppEvent::Core(event));
        };
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
