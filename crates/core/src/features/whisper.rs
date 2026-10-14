//! The log mel spectrogram of Whisper's feature extractor, as Hugging Face computes it.

use super::Features;

const FFT_SIZE: usize = 400;
const HOP: usize = 160;
const BINS: usize = FFT_SIZE / 2 + 1;

/// Whisper features: a centered 400-point STFT with a periodic Hann window every 10 ms, Slaney mel
/// filters up to 8 kHz, log10, a floor 8 (80 dB) below the loudest value, and a shift and scale to
/// about `[-1, 1]`.
pub struct WhisperMel {
    mels: usize,
    window: Vec<f32>,
    /// `cos(2 pi k n / N)` and `sin(2 pi k n / N)` by frequency bin `k`, then sample `n`.
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// For each mel bin, the first FFT bin it covers and its weights from there.
    filters: Vec<(usize, Vec<f32>)>,
}

impl WhisperMel {
    pub fn new(mels: usize) -> Self {
        let tau = 2.0 * std::f64::consts::PI;
        let window = (0..FFT_SIZE)
            .map(|n| (0.5 - 0.5 * (tau * n as f64 / FFT_SIZE as f64).cos()) as f32)
            .collect();
        let mut cos = Vec::with_capacity(BINS * FFT_SIZE);
        let mut sin = Vec::with_capacity(BINS * FFT_SIZE);
        for k in 0..BINS {
            for n in 0..FFT_SIZE {
                // The product modulo N keeps the angle small, so f64 rounding never grows with k * n.
                let angle = tau * ((k * n) % FFT_SIZE) as f64 / FFT_SIZE as f64;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            mels,
            window,
            cos,
            sin,
            filters: slaney_filters(mels, 16_000.0, 8_000.0),
        }
    }

    /// Frames of 16 kHz samples: one every 10 ms, `samples.len() / 160` in all.
    pub fn compute(&self, samples: &[f32]) -> Features {
        let frames = samples.len() / HOP;
        let mut data = vec![0.0f32; frames * self.mels];
        let mut frame = [0.0f32; FFT_SIZE];
        let mut power = [0.0f32; BINS];
        let half = (FFT_SIZE / 2) as isize;
        for t in 0..frames {
            for (n, value) in frame.iter_mut().enumerate() {
                let index = mirror((t * HOP + n) as isize - half, samples.len());
                *value = samples[index] * self.window[n];
            }
            for (k, p) in power.iter_mut().enumerate() {
                let re = dot(&frame, &self.cos[k * FFT_SIZE..(k + 1) * FFT_SIZE]);
                let im = dot(&frame, &self.sin[k * FFT_SIZE..(k + 1) * FFT_SIZE]);
                *p = re * re + im * im;
            }
            for (m, (first, weights)) in self.filters.iter().enumerate() {
                let energy: f32 = weights
                    .iter()
                    .zip(&power[*first..])
                    .map(|(w, p)| w * p)
                    .sum();
                data[t * self.mels + m] = energy.max(1e-10).log10();
            }
        }
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut data {
            *value = (value.max(max - 8.0) + 4.0) / 4.0;
        }
        Features {
            frames,
            dim: self.mels,
            data,
        }
    }
}

/// Dot product with eight running sums, which the compiler can keep in vector registers.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let (a_chunks, b_chunks) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in a_chunks.zip(b_chunks) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    sums.iter().sum::<f32>() + tail
}

/// NumPy's `reflect` padding: the edge sample is not repeated.
fn mirror(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let folded = index.rem_euclid(period);
    (if folded < len as isize {
        folded
    } else {
        period - folded
    }) as usize
}

fn hz_to_mel(hz: f64) -> f64 {
    let step = 200.0 / 3.0;
    if hz < 1000.0 {
        hz / step
    } else {
        1000.0 / step + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let step = 200.0 / 3.0;
    let knee = 1000.0 / step;
    if mel < knee {
        mel * step
    } else {
        1000.0 * ((mel - knee) * (6.4f64.ln() / 27.0)).exp()
    }
}

/// Triangular filters on the Slaney mel scale with Slaney area normalization, as librosa builds them.
fn slaney_filters(mels: usize, sample_rate: f64, max_hz: f64) -> Vec<(usize, Vec<f32>)> {
    let fft_hz: Vec<f64> = (0..BINS)
        .map(|k| k as f64 * sample_rate / FFT_SIZE as f64)
        .collect();
    let top = hz_to_mel(max_hz);
    let edges: Vec<f64> = (0..mels + 2)
        .map(|i| mel_to_hz(top * i as f64 / (mels + 1) as f64))
        .collect();
    (0..mels)
        .map(|m| {
            let (low, center, high) = (edges[m], edges[m + 1], edges[m + 2]);
            let scale = 2.0 / (high - low);
            let weights: Vec<f64> = fft_hz
                .iter()
                .map(|&hz| {
                    let rising = (hz - low) / (center - low);
                    let falling = (high - hz) / (high - center);
                    rising.min(falling).max(0.0) * scale
                })
                .collect();
            let first = weights.iter().position(|w| *w > 0.0).unwrap_or(0);
            let last = weights
                .iter()
                .rposition(|w| *w > 0.0)
                .map_or(first, |l| l + 1);
            (
                first,
                weights[first..last].iter().map(|w| *w as f32).collect(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_matches_numpy_reflect() {
        let indices: Vec<usize> = (-3..7).map(|i| mirror(i, 4)).collect();
        assert_eq!(indices, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn slaney_scale_round_trips() {
        for hz in [0.0, 440.0, 999.0, 1000.0, 4321.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6);
        }
    }

    #[test]
    fn tone_peaks_in_the_matching_mel_bin() {
        let mel = WhisperMel::new(128);
        let samples: Vec<f32> = (0..16_000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin())
            .collect();
        let features = mel.compute(&samples);
        assert_eq!((features.frames, features.dim), (100, 128));
        let frame = features.frame(50);
        let peak = (0..frame.len())
            .max_by(|a, b| frame[*a].total_cmp(&frame[*b]))
            .unwrap();
        let (first, weights) = &mel.filters[peak];
        // 1 kHz is FFT bin 25.
        assert!(
            *first <= 25 && 25 < first + weights.len(),
            "peak bin {peak}"
        );
    }
}
