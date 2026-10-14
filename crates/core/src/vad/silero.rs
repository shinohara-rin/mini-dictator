use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use super::VoiceActivityDetector;
use crate::asr::onnx::load_session;
use crate::{Error, Result};

enum Version {
    /// Inputs `x`, `h`, `c` with LSTM states of `[2, 1, 64]`, as sherpa-onnx ships it.
    V4 { h: Vec<f32>, c: Vec<f32> },
    /// Inputs `input`, `state` of `[2, 1, 128]`, and `sr`. The frame is prefixed with the last 64
    /// samples of the previous frame.
    V5 { state: Vec<f32>, context: Vec<f32> },
}

/// Silero VAD on 32 ms frames.
pub struct SileroVad {
    session: Session,
    version: Version,
}

const FRAME: usize = 512;
const V5_CONTEXT: usize = 64;

impl SileroVad {
    pub fn load(path: &Path) -> Result<Self> {
        let session = load_session(path, 1)?;
        let v5 = session.inputs().iter().any(|input| input.name() == "state");
        let mut vad = Self {
            session,
            version: Version::V4 {
                h: Vec::new(),
                c: Vec::new(),
            },
        };
        vad.version = if v5 {
            Version::V5 {
                state: Vec::new(),
                context: Vec::new(),
            }
        } else {
            Version::V4 {
                h: Vec::new(),
                c: Vec::new(),
            }
        };
        vad.reset();
        Ok(vad)
    }

    fn run(&mut self, frame: &[f32]) -> Result<f32> {
        let error = |e: ort::Error| Error::Model(e.to_string());
        match &mut self.version {
            Version::V4 { h, c } => {
                let inputs = ort::inputs![
                    "x" => Tensor::from_array((vec![1i64, FRAME as i64], frame.to_vec())).map_err(error)?,
                    "h" => Tensor::from_array((vec![2i64, 1, 64], h.clone())).map_err(error)?,
                    "c" => Tensor::from_array((vec![2i64, 1, 64], c.clone())).map_err(error)?,
                ];
                let outputs = self.session.run(inputs).map_err(error)?;
                *h = outputs["new_h"]
                    .try_extract_tensor::<f32>()
                    .map_err(error)?
                    .1
                    .to_vec();
                *c = outputs["new_c"]
                    .try_extract_tensor::<f32>()
                    .map_err(error)?
                    .1
                    .to_vec();
                Ok(outputs["prob"]
                    .try_extract_tensor::<f32>()
                    .map_err(error)?
                    .1[0])
            }
            Version::V5 { state, context } => {
                let mut input = context.clone();
                input.extend_from_slice(frame);
                *context = frame[FRAME - V5_CONTEXT..].to_vec();
                let inputs = ort::inputs![
                    "input" => Tensor::from_array((vec![1i64, input.len() as i64], input)).map_err(error)?,
                    "state" => Tensor::from_array((vec![2i64, 1, 128], state.clone())).map_err(error)?,
                    "sr" => Tensor::from_array((Vec::<i64>::new(), vec![16_000i64])).map_err(error)?,
                ];
                let outputs = self.session.run(inputs).map_err(error)?;
                *state = outputs["stateN"]
                    .try_extract_tensor::<f32>()
                    .map_err(error)?
                    .1
                    .to_vec();
                Ok(outputs["output"]
                    .try_extract_tensor::<f32>()
                    .map_err(error)?
                    .1[0])
            }
        }
    }
}

impl VoiceActivityDetector for SileroVad {
    fn frame_size(&self) -> usize {
        FRAME
    }

    fn probability(&mut self, frame: &[f32]) -> f32 {
        // A failed frame counts as silence; the segmenter then ends the utterance as at a pause.
        self.run(frame).unwrap_or(0.0)
    }

    fn reset(&mut self) {
        match &mut self.version {
            Version::V4 { h, c } => {
                *h = vec![0.0; 128];
                *c = vec![0.0; 128];
            }
            Version::V5 { state, context } => {
                *state = vec![0.0; 256];
                *context = vec![0.0; V5_CONTEXT];
            }
        }
    }
}
