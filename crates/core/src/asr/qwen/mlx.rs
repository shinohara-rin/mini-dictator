//! Qwen3-ASR on MLX, from an MLX checkpoint such as `mlx-community/Qwen3-ASR-0.6B-4bit`: the GPU of
//! Apple silicon runs the encoder and the decoder, with the decoder's weights quantized as the
//! checkpoint stores them.
//!
//! This follows the native MLX runtime of sglang-omni (`sglang_omni_mlx/native`, Apache-2.0), which
//! Voxt runs: the same weights, layers and audio windows, in Rust on mlx-rs. Its greedy loop and
//! prompt are [`super::QwenAsr`]'s, shared with the ONNX export, so hotwords and the noisy check work
//! the same on both.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mlx_rs::error::Exception;
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{argmax_axis, take_along_axis, TryIndexOp};
use mlx_rs::{nn, ops, Array, Dtype};
use serde_json::Value;

use super::{audio_token_count, Pick, QwenModel, CHUNK_FRAMES, MEL_BINS};
use crate::asr::EncoderOutput;
use crate::features::Features;
use crate::{Error, Result};

type Mx<T> = std::result::Result<T, Exception>;

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(format!("MLX: {error}"))
}

const LAYER_NORM_EPS: f32 = 1e-5;

struct AudioConfig {
    layers: usize,
    heads: i32,
    width: i32,
    /// Mel frames per chunk of the convolutions, and per attention window.
    chunk_frames: usize,
    window_frames: usize,
}

struct TextConfig {
    layers: usize,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    rope_theta: f32,
}

/// The checkpoint's weights, with the layers built on them. `Array` is not `Sync`, so
/// [`MlxQwen`] keeps this behind a lock.
struct Weights {
    arrays: HashMap<String, Array>,
    /// Group size and bits of the quantized layers.
    quantization: Option<(i32, i32)>,
    audio: AudioConfig,
    text: TextConfig,
}

pub struct MlxQwen {
    weights: Mutex<Weights>,
    audio_pad: u32,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// Keys and values of every decoder layer, `[batch, kv_heads, tokens, head_dim]` each.
pub struct MlxCache {
    layers: Vec<Option<(Array, Array)>>,
    batch: usize,
    offset: usize,
}

fn config_int(config: &Value, path: &[&str]) -> Result<i64> {
    path.iter()
        .try_fold(config, |value, key| value.get(key))
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::Model(format!("config.json has no {}", path.join("."))))
}

fn config_float(config: &Value, path: &[&str]) -> Result<f64> {
    path.iter()
        .try_fold(config, |value, key| value.get(key))
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Model(format!("config.json has no {}", path.join("."))))
}

impl MlxQwen {
    /// True when `dir` holds an MLX checkpoint of Qwen3-ASR: a `config.json` of model type
    /// `qwen3_asr` with `.safetensors` weights.
    pub fn is_checkpoint(dir: &Path) -> bool {
        let qwen = std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|config| config["model_type"] == "qwen3_asr");
        qwen && !weight_files(dir).is_empty()
    }

    /// Loads the checkpoint in `dir` and evaluates its weights.
    pub fn load(dir: &Path) -> Result<Self> {
        #[cfg(target_os = "macos")]
        use_bundled_kernels();
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let config: Value = serde_json::from_str(&text).map_err(model_error)?;
        let audio_pad = config_int(&config, &["thinker_config", "audio_token_id"])? as u32;
        let audio = ["thinker_config", "audio_config"];
        let at = |key: &'static str| [audio[0], audio[1], key];
        let n_window = config_int(&config, &at("n_window"))? as usize;
        let audio = AudioConfig {
            layers: config_int(&config, &at("encoder_layers"))? as usize,
            heads: config_int(&config, &at("encoder_attention_heads"))? as i32,
            width: config_int(&config, &at("d_model"))? as i32,
            chunk_frames: n_window * 2,
            window_frames: config_int(&config, &at("n_window_infer"))? as usize,
        };
        if audio.chunk_frames != CHUNK_FRAMES || config_int(&config, &at("num_mel_bins"))? != 128 {
            return Err(Error::Model(
                "the checkpoint's audio encoder does not read 100-frame chunks of 128 mel bins"
                    .into(),
            ));
        }
        let text_at = |key: &'static str| ["thinker_config", "text_config", key];
        let text = TextConfig {
            layers: config_int(&config, &text_at("num_hidden_layers"))? as usize,
            heads: config_int(&config, &text_at("num_attention_heads"))? as i32,
            kv_heads: config_int(&config, &text_at("num_key_value_heads"))? as i32,
            head_dim: config_int(&config, &text_at("head_dim"))? as i32,
            rms_norm_eps: config_float(&config, &text_at("rms_norm_eps"))? as f32,
            rope_theta: config_float(&config, &text_at("rope_theta"))? as f32,
        };
        let quantization = match config.get("quantization") {
            Some(q) => Some((
                config_int(q, &["group_size"])? as i32,
                config_int(q, &["bits"])? as i32,
            )),
            None => None,
        };

        let files = weight_files(dir);
        if files.is_empty() {
            return Err(Error::Model(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut arrays = HashMap::new();
        for file in files {
            arrays.extend(Array::load_safetensors(&file).map_err(model_error)?);
        }
        mlx_rs::transforms::eval(arrays.values()).map_err(model_error)?;
        Ok(Self {
            audio_pad,
            layers: text.layers,
            kv_heads: text.kv_heads as usize,
            head_dim: text.head_dim as usize,
            weights: Mutex::new(Weights {
                arrays,
                quantization,
                audio,
                text,
            }),
        })
    }
}

/// Points MLX at the Metal kernels in the app bundle's `Contents/Resources`. Outside a bundle,
/// MLX finds `mlx.metallib` beside the binary, or where the build put it.
#[cfg(target_os = "macos")]
fn use_bundled_kernels() {
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.parent()?.join("Resources/mlx.metallib")))
        .filter(|path| path.is_file());
    if let Some(path) = bundled.and_then(|path| path.to_str().map(str::to_owned)) {
        let _ = mlx_rs::metal::set_metallib_path(path);
    }
}

fn weight_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

impl Weights {
    fn weight(&self, name: &str) -> Mx<&Array> {
        self.arrays
            .get(name)
            .ok_or_else(|| Exception::custom(format!("the checkpoint is missing {name}")))
    }

    fn get(&self, name: &str) -> Option<&Array> {
        self.arrays.get(name)
    }

    fn quantized(&self, x: &Array, prefix: &str, scales: &Array) -> Mx<Array> {
        let (group_size, bits) = self.quantization.unwrap_or((64, 4));
        ops::quantized_matmul(
            x,
            self.weight(&format!("{prefix}.weight"))?,
            scales,
            self.get(&format!("{prefix}.biases")),
            true,
            group_size,
            bits,
        )
    }

    /// `x W^T + b`, quantized where the checkpoint is.
    fn linear(&self, x: &Array, prefix: &str) -> Mx<Array> {
        let y = match self.get(&format!("{prefix}.scales")) {
            Some(scales) => self.quantized(x, prefix, scales)?,
            None => x.matmul(self.weight(&format!("{prefix}.weight"))?.transpose()?)?,
        };
        match self.get(&format!("{prefix}.bias")) {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    fn layer_norm(&self, x: &Array, prefix: &str) -> Mx<Array> {
        fast::layer_norm(
            x,
            self.weight(&format!("{prefix}.weight"))?,
            self.weight(&format!("{prefix}.bias"))?,
            LAYER_NORM_EPS,
        )
    }

    fn rms_norm(&self, x: &Array, prefix: &str) -> Mx<Array> {
        fast::rms_norm(
            x,
            Some(self.weight(&format!("{prefix}.weight"))?),
            self.text.rms_norm_eps,
        )
    }

    /// Rows of the token embedding table.
    fn embed(&self, ids: &Array) -> Mx<Array> {
        let prefix = "model.embed_tokens";
        let weight = self
            .weight(&format!("{prefix}.weight"))?
            .take_axis(ids, 0)?;
        match self.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let (group_size, bits) = self.quantization.unwrap_or((64, 4));
                let biases = self
                    .get(&format!("{prefix}.biases"))
                    .map(|b| b.take_axis(ids, 0))
                    .transpose()?;
                ops::dequantize(
                    &weight,
                    scales.take_axis(ids, 0)?,
                    biases.as_ref(),
                    group_size,
                    bits,
                )
            }
            None => Ok(weight),
        }
    }

    /// Logits from the tied embedding table.
    fn logits(&self, hidden: &Array) -> Mx<Array> {
        let prefix = "model.embed_tokens";
        match self.get(&format!("{prefix}.scales")) {
            Some(scales) => self.quantized(hidden, prefix, scales),
            None => hidden.matmul(self.weight(&format!("{prefix}.weight"))?.transpose()?),
        }
    }

    fn conv(&self, x: &Array, prefix: &str) -> Mx<Array> {
        let y = ops::conv2d(
            x,
            self.weight(&format!("{prefix}.weight"))?,
            (2, 2),
            (1, 1),
            None,
            None,
        )?;
        nn::gelu(y.add(self.weight(&format!("{prefix}.bias"))?)?)
    }

    fn audio_layer(&self, x: &Array, layer: usize) -> Mx<Array> {
        let prefix = format!("audio_tower.layers.{layer}");
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2));
        let heads = self.audio.heads;
        let head_dim = self.audio.width / heads;
        let normed = self.layer_norm(x, &format!("{prefix}.self_attn_layer_norm"))?;
        let project = |name: &str| -> Mx<Array> {
            self.linear(&normed, &format!("{prefix}.self_attn.{name}"))?
                .reshape(&[batch, length, heads, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])
        };
        let attended = fast::scaled_dot_product_attention(
            project("q_proj")?,
            project("k_proj")?,
            project("v_proj")?,
            (head_dim as f32).powf(-0.5),
            None::<ScaledDotProductAttentionMask>,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, length, width])?;
        let hidden = x.add(self.linear(&attended, &format!("{prefix}.self_attn.out_proj"))?)?;
        let normed = self.layer_norm(&hidden, &format!("{prefix}.final_layer_norm"))?;
        let inner = nn::gelu(self.linear(&normed, &format!("{prefix}.fc1"))?)?;
        hidden.add(self.linear(&inner, &format!("{prefix}.fc2"))?)
    }

    /// Audio features `[tokens, output_dim]` of mel features `[frames, mel_bins]`. Each chunk of
    /// 100 frames goes through the convolutions on its own, and attention stays within windows of
    /// chunks.
    fn encode(&self, mel: &Array) -> Mx<Array> {
        let frames = mel.dim(0) as usize;
        let chunk_frames = self.audio.chunk_frames;
        let chunks: Vec<(usize, usize)> = (0..frames)
            .step_by(chunk_frames)
            .map(|start| (start, chunk_frames.min(frames - start)))
            .collect();
        let longest = chunks.iter().map(|(_, length)| *length).max().unwrap_or(0) as i32;
        let padded = chunks
            .iter()
            .map(|(start, length)| {
                let chunk = mel.try_index((*start as i32..(start + length) as i32, ..))?;
                ops::pad(&chunk, &[(0, longest - *length as i32), (0, 0)], None, None)
            })
            .collect::<Mx<Vec<Array>>>()?;
        // [chunks, mel_bins, frames, 1]: the mel bins are the image's height.
        let x = ops::stack(&padded, 0)?
            .transpose_axes(&[0, 2, 1])?
            .expand_dims(-1)?;
        let x = self.conv(&x, "audio_tower.conv2d1")?;
        let x = self.conv(&x, "audio_tower.conv2d2")?;
        let x = self.conv(&x, "audio_tower.conv2d3")?;
        let (count, bins, conv_frames, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        let x = x
            .transpose_axes(&[0, 2, 3, 1])?
            .reshape(&[count, conv_frames, channels * bins])?;
        let x = self.linear(&x, "audio_tower.conv_out")?;
        let x = x.add(sinusoidal_positions(conv_frames, self.audio.width)?.as_dtype(x.dtype())?)?;

        // Each chunk keeps the rows of its own length.
        let kept: Vec<usize> = chunks
            .iter()
            .map(|(_, length)| audio_token_count(*length).min(conv_frames as usize))
            .collect();
        let rows = kept
            .iter()
            .enumerate()
            .map(|(i, kept)| x.try_index((i as i32, ..*kept as i32, ..)))
            .collect::<Mx<Vec<Array>>>()?;
        let hidden = ops::concatenate(&rows, 0)?;

        // Windows of whole chunks; windows of one length run as one batch.
        let per_window = (self.audio.window_frames / chunk_frames).max(1);
        let mut windows: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for group in kept.chunks(per_window) {
            let length: usize = group.iter().sum();
            if length > 0 {
                windows.push((start, start + length));
            }
            start += length;
        }
        let mut encoded: Vec<Option<Array>> = vec![None; windows.len()];
        let mut lengths: Vec<usize> = windows.iter().map(|(s, e)| e - s).collect();
        lengths.sort_unstable();
        lengths.dedup();
        for length in lengths {
            let same: Vec<usize> = (0..windows.len())
                .filter(|w| windows[*w].1 - windows[*w].0 == length)
                .collect();
            let batch = same
                .iter()
                .map(|w| {
                    let (s, e) = windows[*w];
                    hidden.try_index((s as i32..e as i32, ..))
                })
                .collect::<Mx<Vec<Array>>>()?;
            let mut batch = ops::stack(&batch, 0)?;
            for layer in 0..self.audio.layers {
                batch = self.audio_layer(&batch, layer)?;
            }
            for (row, w) in same.iter().enumerate() {
                encoded[*w] = Some(batch.try_index(row as i32)?);
            }
        }
        let encoded: Vec<Array> = encoded.into_iter().flatten().collect();
        let hidden = self.layer_norm(&ops::concatenate(&encoded, 0)?, "audio_tower.ln_post")?;
        let hidden = nn::gelu(self.linear(&hidden, "audio_tower.proj1")?)?;
        self.linear(&hidden, "audio_tower.proj2")
    }

    fn decoder_layer(
        &self,
        x: &Array,
        layer: usize,
        cache: &mut Option<(Array, Array)>,
        offset: i32,
    ) -> Mx<Array> {
        let prefix = format!("model.layers.{layer}");
        let t = &self.text;
        let (batch, length) = (x.dim(0), x.dim(1));
        let normed = self.rms_norm(x, &format!("{prefix}.input_layernorm"))?;
        let project = |name: &str, heads: i32| -> Mx<Array> {
            self.linear(&normed, &format!("{prefix}.self_attn.{name}"))?
                .reshape(&[batch, length, heads, t.head_dim])
        };
        let queries = self.rms_norm(
            &project("q_proj", t.heads)?,
            &format!("{prefix}.self_attn.q_norm"),
        )?;
        let keys = self.rms_norm(
            &project("k_proj", t.kv_heads)?,
            &format!("{prefix}.self_attn.k_norm"),
        )?;
        let values = project("v_proj", t.kv_heads)?.transpose_axes(&[0, 2, 1, 3])?;
        let rope = |x: Array| -> Mx<Array> {
            fast::rope(
                x.transpose_axes(&[0, 2, 1, 3])?,
                t.head_dim,
                false,
                t.rope_theta,
                1.0,
                offset,
                None,
            )
        };
        let queries = rope(queries)?;
        let keys = rope(keys)?;
        let (keys, values) = match cache.take() {
            Some((old_keys, old_values)) => (
                ops::concatenate(&[old_keys, keys], 2)?,
                ops::concatenate(&[old_values, values], 2)?,
            ),
            None => (keys, values),
        };
        let mask = (length > 1).then_some(ScaledDotProductAttentionMask::Causal);
        let attended = fast::scaled_dot_product_attention(
            queries,
            &keys,
            &values,
            (t.head_dim as f32).powf(-0.5),
            mask,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, length, -1])?;
        *cache = Some((keys, values));
        let hidden = x.add(self.linear(&attended, &format!("{prefix}.self_attn.o_proj"))?)?;
        let normed = self.rms_norm(&hidden, &format!("{prefix}.post_attention_layernorm"))?;
        let gate = nn::silu(self.linear(&normed, &format!("{prefix}.mlp.gate_proj"))?)?;
        let up = self.linear(&normed, &format!("{prefix}.mlp.up_proj"))?;
        hidden.add(self.linear(&gate.multiply(&up)?, &format!("{prefix}.mlp.down_proj"))?)
    }
}

/// Whisper's sinusoidal positions, `[length, channels]`.
fn sinusoidal_positions(length: i32, channels: i32) -> Mx<Array> {
    let half = channels / 2;
    let step = (10000f64.ln() / (half - 1) as f64) as f32;
    let inverse = Array::arange::<_, f32>(None, half, None)?
        .multiply(Array::from_f32(-step))?
        .exp()?;
    let scaled = Array::arange::<_, f32>(None, length, None)?
        .expand_dims(1)?
        .multiply(inverse.expand_dims(0)?)?;
    ops::concatenate(&[scaled.sin()?, scaled.cos()?], 1)
}

impl MlxQwen {
    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &self,
        input: &[i64],
        length: usize,
        start: usize,
        audio: &[f32],
        (frames, dim): (usize, usize),
        cache: &mut MlxCache,
        keep_from: usize,
    ) -> Mx<Vec<Vec<Pick>>> {
        let weights = self.weights.lock().expect("MLX weights lock");
        let batch = cache.batch;
        if cache.offset != start {
            return Err(Exception::custom(format!(
                "decoder cache at {}, step at {start}",
                cache.offset
            )));
        }
        let ids: Vec<i32> = input.iter().map(|id| *id as i32).collect();
        let ids = Array::from_slice(&ids, &[batch as i32, length as i32]);
        let mut embeddings = weights.embed(&ids)?;
        // The audio features take the places of the `<|audio_pad|>` run, as one block per row.
        let row = &input[..length];
        if let Some(at) = row.iter().position(|id| *id == i64::from(self.audio_pad)) {
            let pads = row[at..]
                .iter()
                .take_while(|id| **id == i64::from(self.audio_pad))
                .count();
            if pads != frames {
                return Err(Exception::custom(format!(
                    "{pads} audio tokens in the prompt for {frames} audio features"
                )));
            }
            let features = Array::from_slice(
                &audio[..batch * frames * dim],
                &[batch as i32, frames as i32, dim as i32],
            )
            .as_dtype(embeddings.dtype())?;
            embeddings = ops::concatenate(
                &[
                    embeddings.try_index((.., ..at as i32, ..))?,
                    features,
                    embeddings.try_index((.., (at + frames) as i32.., ..))?,
                ],
                1,
            )?;
        }
        let mut hidden = embeddings;
        for (layer, entry) in cache.layers.iter_mut().enumerate() {
            hidden = weights.decoder_layer(&hidden, layer, entry, start as i32)?;
        }
        cache.offset += length;
        let hidden = hidden.try_index((.., keep_from as i32.., ..))?;
        let logits = weights
            .logits(&weights.rms_norm(&hidden, "model.norm")?)?
            .as_dtype(Dtype::Float32)?;
        let best = argmax_axis(&logits, -1, true)?;
        let log_probs =
            take_along_axis(&logits, &best, -1)?.subtract(logits.logsumexp_axis(-1, true)?)?;
        let best = best.as_dtype(Dtype::Uint32)?;
        mlx_rs::transforms::eval([&best, &log_probs])?;
        let kept = length - keep_from;
        let (ids, log_probs) = (best.as_slice::<u32>(), log_probs.as_slice::<f32>());
        Ok((0..batch)
            .map(|b| {
                (0..kept)
                    .map(|p| (ids[b * kept + p], log_probs[b * kept + p]))
                    .collect()
            })
            .collect())
    }
}

impl QwenModel for MlxQwen {
    type Cache = MlxCache;

    fn encode(&self, features: Features) -> Result<EncoderOutput> {
        debug_assert_eq!(features.dim, MEL_BINS);
        let weights = self.weights.lock().expect("MLX weights lock");
        let mel = Array::from_slice(
            &features.data,
            &[features.frames as i32, features.dim as i32],
        );
        let encoded = weights
            .encode(&mel)
            .and_then(|a| a.as_dtype(Dtype::Float32))
            .map_err(model_error)?;
        encoded.eval().map_err(model_error)?;
        Ok(EncoderOutput {
            frames: encoded.dim(0) as usize,
            dim: encoded.dim(1) as usize,
            data: encoded.as_slice::<f32>().to_vec(),
        })
    }

    fn cache(&self, batch: usize, _total: usize) -> MlxCache {
        MlxCache {
            layers: vec![None; self.layers],
            batch,
            offset: 0,
        }
    }

    fn fork(&self, cache: &MlxCache, rows: &[usize], length: usize) -> MlxCache {
        let _weights = self.weights.lock().expect("MLX weights lock");
        let index: Vec<u32> = rows.iter().map(|b| *b as u32).collect();
        let index = Array::from_slice(&index, &[rows.len() as i32]);
        let pick = |array: &Array| -> Array {
            array
                .take_axis(&index, 0)
                .and_then(|rows| rows.try_index((.., .., ..length as i32, ..)))
                .expect("MLX cache rows")
        };
        MlxCache {
            layers: cache
                .layers
                .iter()
                .map(|entry| entry.as_ref().map(|(k, v)| (pick(k), pick(v))))
                .collect(),
            batch: rows.len(),
            offset: length.min(cache.offset),
        }
    }

    /// Keys and values in bfloat16.
    fn cache_bytes(&self, tokens: usize) -> usize {
        2 * self.layers * tokens * self.kv_heads * self.head_dim * 2
    }

    fn step(
        &self,
        input: &[i64],
        length: usize,
        start: usize,
        audio: &[f32],
        shape: (usize, usize),
        cache: &mut MlxCache,
        keep_from: usize,
    ) -> Result<Vec<Vec<Pick>>> {
        self.run_step(input, length, start, audio, shape, cache, keep_from)
            .map_err(model_error)
    }
}
