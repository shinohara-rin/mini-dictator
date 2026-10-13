use super::VoiceActivityDetector;

/// A detector from loudness alone, relative to a tracked noise floor.
///
/// It needs no model, so tests and first runs work without downloads. Silero VAD is the default in
/// the app, because loudness cannot tell speech from a fan or typing.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    noise_floor_db: f32,
    /// Loudness above the floor that scores 0.5.
    pub margin_db: f32,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self {
            noise_floor_db: -60.0,
            margin_db: 12.0,
        }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn frame_size(&self) -> usize {
        512
    }

    fn probability(&mut self, frame: &[f32]) -> f32 {
        let db = 20.0 * (crate::audio::rms(frame) + 1e-9).log10();
        // The floor falls at once and rises slowly, so it tracks the quietest recent frames.
        self.noise_floor_db = if db < self.noise_floor_db {
            db
        } else {
            self.noise_floor_db + 0.05 * (db - self.noise_floor_db).min(1.0)
        };
        let above = db - self.noise_floor_db - self.margin_db;
        1.0 / (1.0 + (-above / 3.0).exp())
    }

    fn reset(&mut self) {
        *self = Self {
            margin_db: self.margin_db,
            ..Self::default()
        };
    }
}
