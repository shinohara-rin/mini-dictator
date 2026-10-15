//! Audio plumbing shared by the shells and the core. The core works on mono `f32` samples in
//! `[-1, 1]` at [`SAMPLE_RATE`].

use std::time::Instant;

/// The sample rate of every model in the core.
pub const SAMPLE_RATE: u32 = 16_000;

/// A block of microphone audio as a shell delivers it.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// When the block's last sample was recorded. The core measures how far behind it is from
    /// this.
    pub captured: Instant,
    /// The shell was playing a sound of its own while the block was recorded, which the microphone
    /// may have picked up. Speech cannot start in such a block; speech already going on continues.
    pub sound_playing: bool,
}

impl AudioChunk {
    /// Mono samples at [`SAMPLE_RATE`]. Channels are averaged.
    pub fn to_model_input(&self, resampler: &mut Resampler) -> Vec<f32> {
        let channels = self.channels.max(1) as usize;
        let mono: Vec<f32> = if channels == 1 {
            self.samples.clone()
        } else {
            self.samples
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                .collect()
        };
        resampler.process(&mono, self.sample_rate)
    }
}

/// Streaming linear resampler to [`SAMPLE_RATE`].
///
/// Linear interpolation is enough for speech recognition at 16 kHz from the usual 44.1 or 48 kHz
/// capture rates. It keeps the fractional position and the last sample between blocks, so block
/// boundaries add no clicks.
#[derive(Debug, Default, Clone)]
pub struct Resampler {
    position: f64,
    previous: Option<f32>,
    source_rate: u32,
}

impl Resampler {
    pub fn process(&mut self, input: &[f32], source_rate: u32) -> Vec<f32> {
        if source_rate == SAMPLE_RATE {
            return input.to_vec();
        }
        if source_rate != self.source_rate {
            *self = Self {
                source_rate,
                ..Self::default()
            };
        }
        // Positions count source samples from the last sample of the previous block, so position 1.0
        // is `input[0]`. The first block has no previous sample and starts there.
        let previous = match self.previous {
            Some(sample) => sample,
            None => {
                self.position = 1.0;
                input.first().copied().unwrap_or(0.0)
            }
        };
        let step = source_rate as f64 / SAMPLE_RATE as f64;
        let sample = |index: usize| {
            if index == 0 {
                previous
            } else {
                input[index - 1]
            }
        };

        let mut output = Vec::with_capacity((input.len() as f64 / step) as usize + 1);
        let mut position = self.position;
        while (position.floor() as usize) < input.len() + usize::from(position.fract() == 0.0) {
            let left = position.floor() as usize;
            let fraction = position.fract() as f32;
            let right = if fraction == 0.0 { left } else { left + 1 };
            output.push(sample(left) * (1.0 - fraction) + sample(right) * fraction);
            position += step;
        }
        self.position = position - input.len() as f64;
        if let Some(last) = input.last() {
            self.previous = Some(*last);
        }
        output
    }
}

/// Root mean square amplitude.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_48k_to_16k_across_blocks() {
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut whole = Resampler::default();
        let expected = whole.process(&tone, 48_000);

        let mut blocks = Resampler::default();
        let mut streamed = Vec::new();
        for block in tone.chunks(1234) {
            streamed.extend(blocks.process(block, 48_000));
        }
        assert!(
            (expected.len() as i64 - 16_000).abs() <= 1,
            "{}",
            expected.len()
        );
        assert_eq!(expected.len(), streamed.len());
        for (a, b) in expected.iter().zip(&streamed) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
