//! Microphone capture with cpal. The stream lives on the main thread, because a cpal stream is
//! not `Send` on every backend, and dropping it stops capture.

use std::sync::mpsc::Sender;
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use mini_dictator_core::audio::AudioChunk;

use crate::sound::Playback;

/// Opens the default input device and sends its blocks until the returned stream is dropped.
/// Blocks recorded while the app plays a sound are marked as such.
pub fn start(sender: Sender<AudioChunk>, playback: Playback) -> Result<Stream, String> {
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
