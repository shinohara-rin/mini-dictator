//! Log mel features for the recognizers.
//!
//! [`Fbank`] computes Kaldi-compatible filterbank features, as icefall and sherpa-onnx compute them for
//! zipformer transducers: 25 ms Povey windows every 10 ms, 80 mel bins, no dither, no snipped edges.
//! [`WhisperMel`] computes the log mel spectrogram of Whisper, which Qwen3-ASR reads too.

mod whisper;

use std::f32::consts::PI;

use crate::audio::SAMPLE_RATE;

pub use whisper::WhisperMel;

#[derive(Debug, Clone)]
pub struct FbankConfig {
    pub num_bins: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub low_freq: f32,
    /// Negative values count down from the Nyquist frequency, as in Kaldi.
    pub high_freq: f32,
    pub preemphasis: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            num_bins: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            low_freq: 20.0,
            high_freq: -400.0,
            preemphasis: 0.97,
        }
    }
}

/// Feature frames in row-major order.
#[derive(Debug, Clone, Default)]
pub struct Features {
    pub frames: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Features {
    pub fn frame(&self, index: usize) -> &[f32] {
        &self.data[index * self.dim..(index + 1) * self.dim]
    }
}

/// Computes [`Features`] from 16 kHz samples in `[-1, 1]`.
pub struct Fbank {
    config: FbankConfig,
    frame_length: usize,
    frame_shift: usize,
    fft_size: usize,
    window: Vec<f32>,
    /// For each mel bin, the first FFT bin it covers and its weights from there.
    mel_banks: Vec<(usize, Vec<f32>)>,
    fft: Fft,
}

impl Fbank {
    pub fn new(config: FbankConfig) -> Self {
        let frame_length = (SAMPLE_RATE as f32 * config.frame_length_ms / 1000.0) as usize;
        let frame_shift = (SAMPLE_RATE as f32 * config.frame_shift_ms / 1000.0) as usize;
        let fft_size = frame_length.next_power_of_two();
        let window = (0..frame_length)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * PI * i as f32 / (frame_length - 1) as f32).cos()).powf(0.85)
            })
            .collect();
        let mel_banks = mel_banks(&config, fft_size);
        Self {
            config,
            frame_length,
            frame_shift,
            fft_size,
            window,
            mel_banks,
            fft: Fft::new(fft_size),
        }
    }

    /// Frame count without snipped edges: one frame per shift, centered on it.
    pub fn frame_count(&self, samples: usize) -> usize {
        (samples + self.frame_shift / 2) / self.frame_shift
    }

    pub fn compute(&self, samples: &[f32]) -> Features {
        let frames = self.frame_count(samples.len());
        let dim = self.config.num_bins;
        let mut data = Vec::with_capacity(frames * dim);
        let mut frame = vec![0.0f32; self.frame_length];
        let mut real = vec![0.0f32; self.fft_size];
        let mut imag = vec![0.0f32; self.fft_size];

        for index in 0..frames {
            let start = (index * self.frame_shift + self.frame_shift / 2) as isize
                - (self.frame_length / 2) as isize;
            for (offset, value) in frame.iter_mut().enumerate() {
                *value = samples
                    .get(reflect(start + offset as isize, samples.len()))
                    .copied()
                    .unwrap_or(0.0);
            }

            let mean = frame.iter().sum::<f32>() / frame.len() as f32;
            frame.iter_mut().for_each(|v| *v -= mean);
            for i in (1..frame.len()).rev() {
                frame[i] -= self.config.preemphasis * frame[i - 1];
            }
            frame[0] -= self.config.preemphasis * frame[0];

            real.fill(0.0);
            imag.fill(0.0);
            for (i, value) in frame.iter().enumerate() {
                real[i] = value * self.window[i];
            }
            self.fft.transform(&mut real, &mut imag);

            for (first, weights) in &self.mel_banks {
                let energy: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * (real[first + k].powi(2) + imag[first + k].powi(2)))
                    .sum();
                data.push(energy.max(f32::EPSILON).ln());
            }
        }
        Features { frames, dim, data }
    }
}

/// Kaldi's reflection at the edges for frames that start before the first sample or end after the last.
fn reflect(index: isize, len: usize) -> usize {
    let len = len as isize;
    if len == 0 {
        return 0;
    }
    let mut index = index;
    while index < 0 || index >= len {
        index = if index < 0 {
            -index - 1
        } else {
            2 * len - 1 - index
        };
    }
    index as usize
}

fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn mel_banks(config: &FbankConfig, fft_size: usize) -> Vec<(usize, Vec<f32>)> {
    let nyquist = SAMPLE_RATE as f32 / 2.0;
    let high_freq = if config.high_freq <= 0.0 {
        nyquist + config.high_freq
    } else {
        config.high_freq
    };
    let fft_bins = fft_size / 2;
    let bin_width = SAMPLE_RATE as f32 / fft_size as f32;
    let (low_mel, high_mel) = (mel(config.low_freq), mel(high_freq));
    let delta = (high_mel - low_mel) / (config.num_bins + 1) as f32;

    (0..config.num_bins)
        .map(|bin| {
            let left = low_mel + bin as f32 * delta;
            let center = left + delta;
            let right = center + delta;
            let mut first = None;
            let mut weights = Vec::new();
            for k in 0..fft_bins {
                let m = mel(bin_width * k as f32);
                if m > left && m < right {
                    let weight = if m <= center {
                        (m - left) / (center - left)
                    } else {
                        (right - m) / (right - center)
                    };
                    first.get_or_insert(k);
                    weights.push(weight);
                } else if first.is_some() {
                    break;
                }
            }
            (first.unwrap_or(0), weights)
        })
        .collect()
}

/// In-place iterative radix-2 FFT. The frame size is fixed, so the twiddles are computed once.
struct Fft {
    size: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(size: usize) -> Self {
        assert!(size.is_power_of_two());
        let cos = (0..size / 2)
            .map(|i| (2.0 * PI * i as f32 / size as f32).cos())
            .collect();
        let sin = (0..size / 2)
            .map(|i| -(2.0 * PI * i as f32 / size as f32).sin())
            .collect();
        Self { size, cos, sin }
    }

    fn transform(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }
        let mut length = 2;
        while length <= n {
            let stride = n / length;
            for start in (0..n).step_by(length) {
                for k in 0..length / 2 {
                    let (wr, wi) = (self.cos[k * stride], self.sin[k * stride]);
                    let (a, b) = (start + k, start + k + length / 2);
                    let tr = real[b] * wr - imag[b] * wi;
                    let ti = real[b] * wi + imag[b] * wr;
                    real[b] = real[a] - tr;
                    imag[b] = imag[a] - ti;
                    real[a] += tr;
                    imag[a] += ti;
                }
            }
            length <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let fft = Fft::new(n);
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 % 5) as f32) - 2.0).collect();
        let (mut real, mut imag) = (input.clone(), vec![0.0; n]);
        fft.transform(&mut real, &mut imag);
        for k in 0..n {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (t, x) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k * t) as f32 / n as f32;
                re += x * angle.cos();
                im += x * angle.sin();
            }
            assert!(
                (re - real[k]).abs() < 1e-3 && (im - imag[k]).abs() < 1e-3,
                "bin {k}"
            );
        }
    }

    #[test]
    fn frame_count_matches_kaldi_without_snip_edges() {
        let fbank = Fbank::new(FbankConfig::default());
        assert_eq!(fbank.frame_count(16_000), 100);
        assert_eq!(fbank.frame_count(16_079), 100);
        assert_eq!(fbank.frame_count(16_080), 101);
    }

    #[test]
    fn tone_peaks_in_the_matching_mel_bin() {
        let fbank = Fbank::new(FbankConfig::default());
        let samples: Vec<f32> = (0..16_000)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / 16_000.0).sin())
            .collect();
        let features = fbank.compute(&samples);
        let frame = features.frame(50);
        let peak = (0..frame.len())
            .max_by(|a, b| frame[*a].total_cmp(&frame[*b]))
            .unwrap();
        let (first, weights) = &fbank.mel_banks[peak];
        let bin_1k = 1000 * 512 / 16_000;
        assert!(
            *first <= bin_1k && bin_1k < first + weights.len(),
            "peak bin {peak}"
        );
    }
}
