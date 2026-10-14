//! The platform-independent core of Mini Dictator.
//!
//! One dictation runs this path:
//!
//! 1. [`context`] reads the screen around the cursor before speech starts and turns it into hotwords.
//! 2. [`vad`] gates microphone audio into speech segments.
//! 3. [`features`] and [`asr`] decode each segment with Qwen3-ASR or a transducer, biased toward the
//!    hotwords.
//! 4. [`uncertainty`] decodes the same encoder output again with Gaussian noise and marks the spans
//!    where the decodes disagree.
//! 5. [`rewrite`] asks an LLM for the final text, with the marked transcript and the screen context.
//!
//! [`pipeline`] runs these steps. Everything that touches the operating system (audio capture,
//! accessibility, OCR, text insertion, the tray) is a trait here and lives in a platform shell.

pub mod asr;
pub mod audio;
pub mod context;
pub mod error;
pub mod features;
pub mod pipeline;
pub mod rewrite;
pub mod rng;
pub mod uncertainty;
pub mod vad;

pub use error::{Error, Result};
