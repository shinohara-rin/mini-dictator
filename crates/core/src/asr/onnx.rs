//! sherpa-onnx transducer models through ONNX Runtime.
//!
//! The model directory holds `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, and `tokens.txt`, as
//! the sherpa-onnx releases ship them. ONNX Runtime is loaded at run time, so the app can bundle the
//! shared library next to its binary on every platform.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use super::{EncoderOutput, SymbolTable, TransducerModel};
use crate::features::Features;
use crate::{Error, Result};

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::Model(error.to_string())
}

static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();

/// Loads ONNX Runtime from `library`, or from `ORT_DYLIB_PATH` when it is `None`. Call once before
/// loading any model; later calls keep the first runtime.
pub fn init_runtime(library: Option<&Path>) -> Result<()> {
    if ENVIRONMENT.get().is_some() {
        return Ok(());
    }
    let builder = match library {
        Some(path) => ort::init_from(path).map_err(model_error)?,
        None => ort::init(),
    };
    let environment = builder
        .with_name("mini-dictator")
        .build()
        .map_err(model_error)?;
    let _ = ENVIRONMENT.set(environment);
    Ok(())
}

pub(crate) fn load_session(path: &Path, threads: usize) -> Result<Session> {
    let environment = ENVIRONMENT.get().ok_or_else(|| {
        Error::Model("ONNX Runtime is not initialized; call init_runtime first".into())
    })?;
    Session::builder(environment)
        .map_err(model_error)?
        .with_intra_threads(threads)
        .map_err(model_error)?
        .commit_from_file(path)
        .map_err(model_error)
}

/// The files of one transducer model.
#[derive(Debug, Clone)]
pub struct TransducerFiles {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

impl TransducerFiles {
    /// Finds the files in a sherpa-onnx model directory. `int8` prefers the quantized encoder and
    /// joiner, which are about four times smaller and barely less accurate.
    pub fn discover(dir: &Path, int8: bool) -> Result<Self> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        let find = |prefix: &str, quantized: bool| -> Result<PathBuf> {
            let matches = |p: &&PathBuf| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with(prefix) && name.ends_with(".onnx")
            };
            let is_int8 = |p: &&PathBuf| p.to_string_lossy().contains(".int8.");
            entries
                .iter()
                .filter(matches)
                .find(|p| is_int8(p) == quantized)
                .or_else(|| entries.iter().find(matches))
                .cloned()
                .ok_or_else(|| Error::Model(format!("no {prefix}*.onnx in {}", dir.display())))
        };
        Ok(Self {
            encoder: find("encoder", int8)?,
            // The decoder is small, and its int8 version loses more accuracy than it saves.
            decoder: find("decoder", false)?,
            joiner: find("joiner", int8)?,
            tokens: dir.join("tokens.txt"),
        })
    }
}

/// An offline (non-streaming) zipformer transducer.
pub struct OnnxTransducer {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joiner: Mutex<Session>,
    context_size: usize,
    vocab_size: usize,
}

impl OnnxTransducer {
    pub fn load(files: &TransducerFiles, threads: usize) -> Result<(Self, SymbolTable)> {
        let decoder = load_session(&files.decoder, 1)?;
        let metadata = decoder.metadata().map_err(model_error)?;
        let read = |key: &str| -> Result<usize> {
            metadata
                .custom(key)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| Error::Model(format!("decoder metadata has no {key}")))
        };
        let (context_size, vocab_size) = (read("context_size")?, read("vocab_size")?);
        drop(metadata);
        let symbols = SymbolTable::load(&files.tokens)?;
        let model = Self {
            encoder: Mutex::new(load_session(&files.encoder, threads)?),
            decoder: Mutex::new(decoder),
            joiner: Mutex::new(load_session(&files.joiner, 1)?),
            context_size,
            vocab_size,
        };
        Ok((model, symbols))
    }
}

/// Flattens equal-length rows for a batched input.
fn stack(rows: &[&[f32]]) -> (Vec<i64>, Vec<f32>) {
    let width = rows.first().map_or(0, |r| r.len());
    (
        vec![rows.len() as i64, width as i64],
        rows.iter().flat_map(|r| r.iter().copied()).collect(),
    )
}

fn unstack(shape: &[i64], data: &[f32]) -> Vec<Vec<f32>> {
    let width = *shape.last().unwrap_or(&0) as usize;
    data.chunks(width.max(1)).map(<[f32]>::to_vec).collect()
}

impl TransducerModel for OnnxTransducer {
    fn context_size(&self) -> usize {
        self.context_size
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn encode(&self, features: &Features) -> Result<EncoderOutput> {
        let x = Tensor::from_array((
            vec![1i64, features.frames as i64, features.dim as i64],
            features.data.clone(),
        ))
        .map_err(model_error)?;
        let x_lens =
            Tensor::from_array((vec![1i64], vec![features.frames as i64])).map_err(model_error)?;
        let mut session = self.encoder.lock().expect("encoder lock");
        let outputs = session
            .run(ort::inputs!["x" => x, "x_lens" => x_lens])
            .map_err(model_error)?;
        let (_, lens) = outputs["encoder_out_lens"]
            .try_extract_tensor::<i64>()
            .map_err(model_error)?;
        let (shape, data) = outputs["encoder_out"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let (frames, dim) = ((lens[0] as usize).min(shape[1] as usize), shape[2] as usize);
        Ok(EncoderOutput {
            frames,
            dim,
            data: data[..frames * dim].to_vec(),
        })
    }

    fn decode(&self, contexts: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let y: Vec<i64> = contexts
            .iter()
            .flat_map(|c| c.iter().map(|t| *t as i64))
            .collect();
        let y = Tensor::from_array((vec![contexts.len() as i64, self.context_size as i64], y))
            .map_err(model_error)?;
        let mut session = self.decoder.lock().expect("decoder lock");
        let outputs = session.run(ort::inputs!["y" => y]).map_err(model_error)?;
        let (shape, data) = outputs["decoder_out"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        Ok(unstack(shape, data))
    }

    fn join(&self, encoder: &[&[f32]], decoder: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let (encoder_shape, encoder_data) = stack(encoder);
        let (decoder_shape, decoder_data) = stack(decoder);
        let encoder_out = Tensor::from_array((encoder_shape, encoder_data)).map_err(model_error)?;
        let decoder_out = Tensor::from_array((decoder_shape, decoder_data)).map_err(model_error)?;
        let mut session = self.joiner.lock().expect("joiner lock");
        let outputs = session
            .run(ort::inputs!["encoder_out" => encoder_out, "decoder_out" => decoder_out])
            .map_err(model_error)?;
        let (shape, data) = outputs["logit"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        Ok(unstack(shape, data))
    }
}
