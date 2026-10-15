//! Qwen3-ASR through ONNX Runtime, in the export that sherpa-onnx ships
//! (`sherpa-onnx-qwen3-asr-0.6B-int8-*`).
//!
//! - `conv_frontend.onnx` downsamples Whisper mel features by 8, to 12.5 audio tokens per second.
//! - `encoder.onnx` turns them into audio features in the LLM's embedding space.
//! - `decoder.onnx` runs the LLM on a chat prompt whose `<|audio_pad|>` tokens stand for the audio
//!   features. It returns the key/value cache entries of the tokens it read, and the caller keeps
//!   the cache.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorRef};

use super::{audio_token_count, pick, Pick, QwenModel};
use crate::asr::onnx::load_session;
use crate::asr::EncoderOutput;
use crate::features::Features;
use crate::{Error, Result};

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

/// The files of one Qwen3-ASR export.
#[derive(Debug, Clone)]
pub struct QwenFiles {
    pub conv_frontend: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    /// Directory with `vocab.json`, `merges.txt`, and `tokenizer_config.json`.
    pub tokenizer: PathBuf,
}

impl QwenFiles {
    /// True when `dir` looks like a Qwen3-ASR export rather than a transducer.
    pub fn is_qwen(dir: &Path) -> bool {
        dir.join("conv_frontend.onnx").is_file()
    }

    /// Finds the files in a model directory, preferring the int8 encoder and decoder.
    pub fn discover(dir: &Path) -> Result<Self> {
        let pick = |stem: &str| -> Result<PathBuf> {
            [format!("{stem}.int8.onnx"), format!("{stem}.onnx")]
                .into_iter()
                .map(|name| dir.join(name))
                .find(|path| path.is_file())
                .ok_or_else(|| Error::Model(format!("no {stem}.onnx in {}", dir.display())))
        };
        Ok(Self {
            conv_frontend: pick("conv_frontend")?,
            encoder: pick("encoder")?,
            decoder: pick("decoder")?,
            tokenizer: dir.join("tokenizer"),
        })
    }
}

/// Shapes the decoder's inputs reveal.
struct DecoderShape {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
}

pub struct OrtQwen {
    frontend: Mutex<Session>,
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    shape: DecoderShape,
}

/// Key and value buffers of every layer, `[batch, total, kv_heads, head_dim]` each, on the host.
pub struct OrtCache {
    buffers: Vec<Vec<f32>>,
    batch: usize,
    total: usize,
}

impl OrtQwen {
    pub fn load(files: &QwenFiles, threads: usize) -> Result<Self> {
        let decoder = load_session(&files.decoder, threads)?;
        let layers = decoder
            .inputs()
            .iter()
            .filter(|input| input.name().starts_with("cache_key_"))
            .count();
        let dims: Vec<i64> = decoder
            .inputs()
            .iter()
            .find(|input| input.name() == "cache_key_0")
            .and_then(|input| input.dtype().tensor_shape())
            .map(|shape| shape.iter().copied().collect())
            .unwrap_or_default();
        let (kv_heads, head_dim) = match dims.as_slice() {
            [_, _, heads, dim] if *heads > 0 && *dim > 0 => (*heads as usize, *dim as usize),
            _ => {
                return Err(Error::Model(format!(
                    "unexpected decoder cache shape {dims:?} in {}",
                    files.decoder.display()
                )))
            }
        };
        Ok(Self {
            frontend: Mutex::new(load_session(&files.conv_frontend, threads)?),
            encoder: Mutex::new(load_session(&files.encoder, threads)?),
            decoder: Mutex::new(decoder),
            shape: DecoderShape {
                layers,
                kv_heads,
                head_dim,
            },
        })
    }

    fn width(&self) -> usize {
        self.shape.kv_heads * self.shape.head_dim
    }
}

impl QwenModel for OrtQwen {
    type Cache = OrtCache;

    fn encode(&self, features: Features) -> Result<EncoderOutput> {
        let frames = features.frames;
        let input = Tensor::from_array(([1, features.frames, features.dim], features.data))
            .map_err(model_error)?;
        let (conv_shape, conv) = {
            let mut session = self.frontend.lock().expect("frontend lock");
            let outputs = session
                .run(ort::inputs!["input_features" => input])
                .map_err(model_error)?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(model_error)?;
            (shape.iter().copied().collect::<Vec<i64>>(), data.to_vec())
        };
        let tokens = conv_shape[1] as usize;
        let valid = audio_token_count(frames).min(tokens);
        let mask: Vec<bool> = (0..tokens).map(|i| i < valid).collect();
        let conv = Tensor::from_array((conv_shape, conv)).map_err(model_error)?;
        let mask = Tensor::from_array(([1, tokens], mask)).map_err(model_error)?;
        let mut session = self.encoder.lock().expect("encoder lock");
        let outputs = session
            .run(ort::inputs!["input_features" => conv, "feature_attention_mask" => mask])
            .map_err(model_error)?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let dim = shape[2] as usize;
        Ok(EncoderOutput {
            frames: valid,
            dim,
            data: data[..valid * dim].to_vec(),
        })
    }

    fn cache(&self, batch: usize, total: usize) -> OrtCache {
        OrtCache {
            buffers: (0..2 * self.shape.layers)
                .map(|_| vec![0.0; batch * total * self.width()])
                .collect(),
            batch,
            total,
        }
    }

    /// Copies the rows whole: the decoder reads a row's cache only up to the position it is given.
    fn fork(&self, cache: &OrtCache, rows: &[usize], _length: usize) -> OrtCache {
        let slab = cache.total * self.width();
        OrtCache {
            buffers: cache
                .buffers
                .iter()
                .map(|buffer| {
                    rows.iter()
                        .flat_map(|b| buffer[b * slab..(b + 1) * slab].iter().copied())
                        .collect()
                })
                .collect(),
            batch: rows.len(),
            total: cache.total,
        }
    }

    fn cache_bytes(&self, tokens: usize) -> usize {
        2 * self.shape.layers * tokens * self.width() * 4
    }

    fn step(
        &self,
        input: &[i64],
        length: usize,
        start: usize,
        audio: &[f32],
        (frames, dim): (usize, usize),
        cache: &mut OrtCache,
        keep_from: usize,
    ) -> Result<Vec<Vec<Pick>>> {
        let batch = cache.batch;
        let total = cache.total;
        let shape = &self.shape;
        let width = self.width();
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            (
                "input_ids".into(),
                Tensor::from_array(([batch, length], input.to_vec()))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "audio_features".into(),
                TensorRef::from_array_view(([batch, frames, dim], audio))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array(([batch, length], vec![1i64; batch * length]))
                    .map_err(model_error)?
                    .into(),
            ),
            (
                "cache_position".into(),
                Tensor::from_array((
                    [length],
                    (start as i64..(start + length) as i64).collect::<Vec<_>>(),
                ))
                .map_err(model_error)?
                .into(),
            ),
        ];
        let dims = [batch, total, shape.kv_heads, shape.head_dim];
        for (layer, pair) in cache.buffers.chunks(2).enumerate() {
            inputs.push((
                format!("cache_key_{layer}"),
                TensorRef::from_array_view((dims, pair[0].as_slice()))
                    .map_err(model_error)?
                    .into(),
            ));
            inputs.push((
                format!("cache_value_{layer}"),
                TensorRef::from_array_view((dims, pair[1].as_slice()))
                    .map_err(model_error)?
                    .into(),
            ));
        }

        let mut session = self.decoder.lock().expect("decoder lock");
        let outputs = session.run(inputs).map_err(model_error)?;
        let (logits_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let vocab = logits_shape[2] as usize;
        let steps = logits_shape[1] as usize;
        let picks = (0..batch)
            .map(|b| {
                (keep_from..steps)
                    .map(|p| {
                        let at = (b * steps + p) * vocab;
                        pick(&logits[at..at + vocab])
                    })
                    .collect()
            })
            .collect();

        for (layer, pair) in cache.buffers.chunks_mut(2).enumerate() {
            for (kind, buffer) in ["key", "value"].iter().zip(pair.iter_mut()) {
                let (delta_shape, delta) = outputs[format!("{kind}_delta_{layer}").as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(model_error)?;
                let rows = (delta_shape[1] as usize).min(total - start);
                for b in 0..batch {
                    let from = b * delta_shape[1] as usize * width;
                    let to = (b * total + start) * width;
                    buffer[to..to + rows * width]
                        .copy_from_slice(&delta[from..from + rows * width]);
                }
            }
        }
        Ok(picks)
    }
}
