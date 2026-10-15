//! Microphone capture. With echo cancellation, an `AVAudioEngine` with macOS voice processing
//! records the microphone and plays the cues, so the cues it plays are taken out of what it
//! records. Without it, or if the engine fails to start, cpal records the microphone as is.
//! Either lives on the main thread, and dropping it stops capture and ends the dictation.

use std::cell::Cell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use block2::RcBlock;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use mini_dictator_core::audio::AudioChunk;
use objc2::rc::Retained;
use objc2::AllocAnyThread;
use objc2_avf_audio::{
    AVAudioCommonFormat, AVAudioEngine, AVAudioFormat, AVAudioInputNode, AVAudioPCMBuffer,
    AVAudioPlayerNode, AVAudioTime,
};

use crate::sound::{Jingle, Playback};

/// The microphone while a dictation records.
pub enum Capture {
    /// The raw microphone.
    Plain(#[allow(dead_code)] Stream),
    Echo(EchoCapture),
}

impl Capture {
    /// Where the cues play while this capture runs, when they are cancelled from it.
    pub fn echo_output(&self) -> Option<EchoOutput> {
        match self {
            Capture::Plain(_) => None,
            Capture::Echo(capture) => Some(capture.output.clone()),
        }
    }
}

/// Opens the microphone, with echo cancellation if asked and available, and sends its blocks
/// until the capture is dropped. Blocks recorded while the app plays a sound it cannot cancel
/// are marked as such. Returns why echo cancellation was not used, if it was asked for and failed.
pub fn start(
    sender: Sender<AudioChunk>,
    playback: Playback,
    cancel_echo: bool,
) -> Result<(Capture, Option<String>), String> {
    let mut fallback = None;
    if cancel_echo {
        match EchoCapture::start(sender.clone(), playback.clone(), Jingle::ALL.len()) {
            Ok(capture) => return Ok((Capture::Echo(capture), None)),
            Err(error) => fallback = Some(error),
        }
    }
    Ok((Capture::Plain(start_plain(sender, playback)?), fallback))
}

/// The cue players of a running [`EchoCapture`], one per cue so that different cues overlap in
/// the mixer instead of queueing behind each other. Main thread only.
#[derive(Clone)]
pub struct EchoOutput {
    players: Rc<Vec<Retained<AVAudioPlayerNode>>>,
    /// False once the capture stopped, after which the player must not be used.
    running: Rc<Cell<bool>>,
}

impl EchoOutput {
    /// Plays mono samples at `rate` on player `slot`, starting over whatever that player was
    /// still playing. False if the capture stopped or there is no such player.
    pub fn play(&self, slot: usize, samples: &[f32], rate: f64) -> bool {
        let Some(player) = self.players.get(slot).filter(|_| self.running.get()) else {
            return false;
        };
        // SAFETY: the engine is running, the buffer matches the format the player was connected
        // with, and its single channel is filled within its capacity before it is scheduled.
        unsafe {
            let Some(format) = cue_format(rate) else {
                return false;
            };
            let Some(buffer) = AVAudioPCMBuffer::initWithPCMFormat_frameCapacity(
                AVAudioPCMBuffer::alloc(),
                &format,
                samples.len() as u32,
            ) else {
                return false;
            };
            buffer.setFrameLength(samples.len() as u32);
            let channel = *buffer.floatChannelData();
            std::ptr::copy_nonoverlapping(samples.as_ptr(), channel.as_ptr(), samples.len());
            // A buffer scheduled without a time waits for the ones before it, so drop those.
            player.stop();
            player.scheduleBuffer_completionHandler(&buffer, std::ptr::null_mut());
            player.play();
        }
        true
    }
}

/// The rate the cues are drawn at; the engine's mixer converts them for the speakers.
pub const CUE_RATE: f64 = crate::synth::RATE as f64;

fn cue_format(rate: f64) -> Option<Retained<AVAudioFormat>> {
    // SAFETY: a standard mono float format.
    unsafe {
        AVAudioFormat::initWithCommonFormat_sampleRate_channels_interleaved(
            AVAudioFormat::alloc(),
            AVAudioCommonFormat::PCMFormatFloat32,
            rate,
            1,
            false,
        )
    }
}

/// The microphone through macOS voice processing, which removes from it what the same engine
/// plays: the cues. Voice processing also turns down other apps' audio while it runs.
pub struct EchoCapture {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    output: EchoOutput,
    /// Taken on drop, which closes the audio channel even if the tap block outlives the engine.
    sender: Arc<Mutex<Option<Sender<AudioChunk>>>>,
}

impl EchoCapture {
    fn start(
        sender: Sender<AudioChunk>,
        playback: Playback,
        players: usize,
    ) -> Result<Self, String> {
        let describe =
            |e: Retained<objc2_foundation::NSError>| e.localizedDescription().to_string();
        // SAFETY: the engine is set up while stopped, the player is attached before it is
        // connected, and the tap block only reads the buffer it is handed, within its length.
        unsafe {
            let engine = AVAudioEngine::new();
            let input = engine.inputNode();
            input
                .setVoiceProcessingEnabled_error(true)
                .map_err(describe)?;
            let format = cue_format(CUE_RATE).ok_or("no cue format")?;
            let players: Vec<_> = (0..players)
                .map(|_| {
                    let player = AVAudioPlayerNode::new();
                    engine.attachNode(&player);
                    engine.connect_to_format(&player, &engine.mainMixerNode(), Some(&format));
                    player
                })
                .collect();

            let input_format = input.outputFormatForBus(0);
            let sample_rate = input_format.sampleRate() as u32;
            if sample_rate == 0 || input_format.channelCount() == 0 {
                return Err("the microphone has no input format".into());
            }
            let sender = Arc::new(Mutex::new(Some(sender)));
            let tap_sender = sender.clone();
            let tap = RcBlock::new(
                move |buffer: NonNull<AVAudioPCMBuffer>, _: NonNull<AVAudioTime>| {
                    let buffer = buffer.as_ref();
                    let channels = buffer.floatChannelData();
                    if channels.is_null() {
                        return;
                    }
                    // Voice processing can report several channels; the first carries the voice.
                    let first = *channels;
                    let samples =
                        std::slice::from_raw_parts(first.as_ptr(), buffer.frameLength() as usize);
                    if let Some(sender) = tap_sender.lock().ok().and_then(|s| s.clone()) {
                        let _ = sender.send(AudioChunk {
                            samples: samples.to_vec(),
                            sample_rate,
                            channels: 1,
                            captured: Instant::now(),
                            sound_playing: playback.is_playing(),
                        });
                    }
                },
            );
            input.installTapOnBus_bufferSize_format_block(0, 1024, None, RcBlock::as_ptr(&tap));
            engine.prepare();
            if let Err(error) = engine.startAndReturnError() {
                input.removeTapOnBus(0);
                return Err(describe(error));
            }
            Ok(EchoCapture {
                engine,
                input,
                output: EchoOutput {
                    players: Rc::new(players),
                    running: Rc::new(Cell::new(true)),
                },
                sender,
            })
        }
    }
}

impl Drop for EchoCapture {
    fn drop(&mut self) {
        self.output.running.set(false);
        // SAFETY: stops what `start` started, on the thread that started it.
        unsafe {
            self.input.removeTapOnBus(0);
            for player in self.output.players.iter() {
                player.stop();
            }
            self.engine.stop();
        }
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
    }
}

/// Opens the default input device with cpal.
fn start_plain(sender: Sender<AudioChunk>, playback: Playback) -> Result<Stream, String> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("no microphone")?;
    let supported = device.default_input_config().map_err(|e| e.to_string())?;
    let format = supported.sample_format();
    let config = supported.config();
    let (sample_rate, channels) = (config.sample_rate, config.channels);
    let error = |e: cpal::Error| eprintln!("microphone: {e}");

    let stream = match format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                let _ = sender.send(AudioChunk {
                    samples: data.to_vec(),
                    sample_rate,
                    channels,
                    captured: Instant::now(),
                    sound_playing: playback.is_playing(),
                });
            },
            error,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let samples = data.iter().map(|s| *s as f32 / 32768.0).collect();
                let _ = sender.send(AudioChunk {
                    samples,
                    sample_rate,
                    channels,
                    captured: Instant::now(),
                    sound_playing: playback.is_playing(),
                });
            },
            error,
            None,
        ),
        other => return Err(format!("unsupported microphone format {other}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}
