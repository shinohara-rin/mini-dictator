//! A small seeded generator. Monte Carlo runs must repeat for the same seed, so a threshold change
//! can be compared on the same noise.

/// SplitMix64. Enough for noise, and it needs no dependency.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fills `out` with independent standard normal values (Box-Muller, two values per pair of draws).
    pub fn fill_gaussian(&mut self, out: &mut [f32]) {
        let mut chunks = out.chunks_mut(2);
        for pair in &mut chunks {
            let radius = (-2.0 * (1.0 - self.next_f64()).ln()).sqrt();
            let angle = std::f64::consts::TAU * self.next_f64();
            pair[0] = (radius * angle.cos()) as f32;
            if pair.len() > 1 {
                pair[1] = (radius * angle.sin()) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_has_unit_variance() {
        let mut rng = SplitMix64::new(7);
        let mut values = vec![0.0f32; 100_000];
        rng.fill_gaussian(&mut values);
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
    }

    #[test]
    fn same_seed_same_values() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
