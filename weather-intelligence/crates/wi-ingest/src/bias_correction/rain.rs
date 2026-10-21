//! Koreksi bias curah hujan — BUKAN aditif murni.
//!
//! Hujan tidak negatif, miring ke kanan, dan banyak bernilai nol.
//! Digunakan rasio (korreksi multiplikatif) atau pemetaan kuantil,
//! lalu hasil dibatasi agar tidak negatif.

use chrono::{DateTime, Utc};
use wi_core::error::WIError;
use wi_core::types::WeatherVariable;

use super::{BiasKey, BiasStore, BiasUpdateConfig};

/// Metode koreksi hujan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainCorrectionMethod {
    /// Rasio: x_koreksi = x_prakiraan * ratio, ratio = observation / forecast (jika forecast > 0)
    Ratio,
    /// Pemetaan kuantil sederhana (placeholder — implementasi lengkap butuh data historis)
    QuantileMapping,
}

/// Hasil koreksi hujan.
#[derive(Debug, Clone)]
pub struct RainCorrection {
    pub raw_value: f32,
    pub bias_ratio: f32,
    pub corrected_value: f32,
    pub station_id: String,
    pub lead_seconds: u32,
    pub cycle_hour: u8,
    pub method: RainCorrectionMethod,
}

/// Pemroses koreksi bias hujan.
#[derive(Debug, Clone)]
pub struct RainBiasCorrector {
    store: BiasStore,
    config: BiasUpdateConfig,
    method: RainCorrectionMethod,
}

impl RainBiasCorrector {
    pub fn new(config: BiasUpdateConfig, method: RainCorrectionMethod) -> Self {
        Self {
            store: BiasStore::new(),
            config,
            method,
        }
    }

    /// Koreksi curah hujan prakiraan.
    /// Metode Rasio: corrected = raw * ratio, di mana ratio = b (disimpan sebagai ratio)
    /// Jika raw = 0, corrected = 0 (tidak menimbulkan hujan dari kosong).
    pub fn correct(
        &self,
        raw_value: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: DateTime<Utc>,
    ) -> Result<RainCorrection, WIError> {
        let _ = now; // diizinkan untuk masa depan (logging, audit)
        let key = BiasKey::without_season(
            station_id,
            WeatherVariable::Precipitation,
            lead_seconds,
            cycle_hour,
        );
        let ratio = self.store.get(&key).map(|s| s.b).unwrap_or(1.0);

        let corrected = match self.method {
            RainCorrectionMethod::Ratio => {
                if raw_value <= 0.0 {
                    0.0
                } else {
                    let c = raw_value * ratio;
                    c.max(0.0)
                }
            }
            RainCorrectionMethod::QuantileMapping => {
                // Placeholder: tanpa data historis kuantil, gunakan ratio
                if raw_value <= 0.0 {
                    0.0
                } else {
                    let c = raw_value * ratio;
                    c.max(0.0)
                }
            }
        };

        Ok(RainCorrection {
            raw_value,
            bias_ratio: ratio,
            corrected_value: corrected,
            station_id: station_id.to_string(),
            lead_seconds,
            cycle_hour,
            method: self.method,
        })
    }

    /// Perbarui ratio bias hujan.
    /// ratio_baru = (1 - α) * ratio_lama + α * (observation / forecast) (jika forecast > 0)
    pub fn update_bias(
        &mut self,
        forecast: f32,
        observation: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: DateTime<Utc>,
    ) -> Result<(), WIError> {
        let alpha = self.config.require_alpha()?;

        let ratio_observed = if forecast > 0.0 {
            observation / forecast
        } else if observation == 0.0 {
            1.0 // kedua-duanya nol, tidak ada koreksi
        } else {
            // Forecast nol tapi observation ada → underforecast, set ratio tinggi
            2.0
        };

        let key = BiasKey::without_season(
            station_id,
            WeatherVariable::Precipitation,
            lead_seconds,
            cycle_hour,
        );
        let state = self.store.get_or_default(key);
        state.update(ratio_observed, alpha, now);
        Ok(())
    }

    pub fn reset_all(&mut self) {
        self.store.reset_all();
    }

    pub fn config(&self) -> &BiasUpdateConfig {
        &self.config
    }
}

impl Default for RainBiasCorrector {
    fn default() -> Self {
        Self::new(BiasUpdateConfig::new(), RainCorrectionMethod::Ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_rain_correction_ratio() {
        let corrector =
            RainBiasCorrector::new(BiasUpdateConfig::new(), RainCorrectionMethod::Ratio);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // Tanpa state, ratio = 1.0, corrected = raw
        let corr = corrector.correct(10.0, "stn-001", 3600, 12, now).unwrap();
        assert_eq!(corr.corrected_value, 10.0);
        assert_eq!(corr.bias_ratio, 1.0);
    }

    #[test]
    fn test_rain_raw_zero_stays_zero() {
        let corrector =
            RainBiasCorrector::new(BiasUpdateConfig::new(), RainCorrectionMethod::Ratio);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let corr = corrector.correct(0.0, "stn-001", 3600, 12, now).unwrap();
        assert_eq!(corr.corrected_value, 0.0);
    }

    #[test]
    fn test_rain_bias_update_requires_alpha() {
        let mut corrector =
            RainBiasCorrector::new(BiasUpdateConfig::new(), RainCorrectionMethod::Ratio);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let err = corrector.update_bias(10.0, 8.0, "stn-001", 3600, 12, now);
        let err = err.unwrap_err();
        assert!(err.to_string().contains("belum diputuskan"));
    }

    #[test]
    fn test_rain_bias_update_with_alpha() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.5).unwrap();
        let mut corrector = RainBiasCorrector::new(config, RainCorrectionMethod::Ratio);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // forecast 10, observation 8 → ratio = 0.8, b_baru = (1-0.5)*0 + 0.5*0.8 = 0.4
        corrector
            .update_bias(10.0, 8.0, "stn-001", 3600, 12, now)
            .unwrap();

        // corrected = 10 * 0.4 = 4.0
        let corr = corrector.correct(10.0, "stn-001", 3600, 12, now).unwrap();
        assert!((corr.corrected_value - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rain_clamps_non_negative() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.9).unwrap();
        let mut corrector = RainBiasCorrector::new(config, RainCorrectionMethod::Ratio);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // forecast 10, observation 0 → ratio = 0, b_baru = 0.1*1 + 0.9*0 = 0.1
        corrector
            .update_bias(10.0, 0.0, "stn-001", 3600, 12, now)
            .unwrap();

        // corrected = 10 * 0.1 = 1.0 (non-negative)
        let corr = corrector.correct(10.0, "stn-001", 3600, 12, now).unwrap();
        assert!(corr.corrected_value >= 0.0);
    }
}
