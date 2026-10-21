//! # Bias Correction
//!
//! Koreksi bias lokal terhadap pengamatan Indonesia.
//! - Aditif: suhu 2m, tekanan permukaan laut, angin (u,v), kelembapan relatif (clamp 0-100%).
//!   Rumus: x_koreksi = x_prakiraan - b; b_baru = (1-α)·b_lama + α·(prakiraan − pengamatan)
//! - Hujan: BUKAN aditif. Pakai rasio / pemetaan kuantil, clamp ≥ 0.
//!
//! α dan panjang jendela BELUM DIPUTUSKAN → mengembalikan WIError::BelumDiputuskan,
//! tidak boleh diisi tebakan.
//!
//! Acuan pengamatan: obs.bmkg_stn, obs.isd (setelah QC). Bukan ref.era5.

mod additive;
mod bias_state;
mod rain;

pub use additive::{AdditiveBiasCorrector, AdditiveCorrection, AdditiveVariable};
pub use bias_state::{BiasKey, BiasState, BiasStore, BiasUpdateConfig};
pub use rain::{RainBiasCorrector, RainCorrection, RainCorrectionMethod};

use wi_core::error::WIError;
use wi_core::types::WeatherVariable;

/// Kelompok variabel untuk koreksi bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasVariableGroup {
    Additive(AdditiveVariable),
    Rain,
    NotCorrectable,
}

impl BiasVariableGroup {
    pub fn classify(var: WeatherVariable) -> Self {
        match var {
            WeatherVariable::Temperature
            | WeatherVariable::Pressure
            | WeatherVariable::SurfacePressure
            | WeatherVariable::WindU
            | WeatherVariable::WindV
            | WeatherVariable::RelativeHumidity => {
                let add = AdditiveVariable::from_weather_variable(var)
                    .expect("klasifikasi variabel aditif konsisten");
                Self::Additive(add)
            }
            WeatherVariable::Precipitation => Self::Rain,
            _ => Self::NotCorrectable,
        }
    }
}

/// Engine koreksi bias utama: menggabungkan aditif dan hujan.
#[derive(Debug, Clone)]
pub struct BiasCorrectionEngine {
    additive: AdditiveBiasCorrector,
    rain: RainBiasCorrector,
}

impl BiasCorrectionEngine {
    pub fn new(config: BiasUpdateConfig) -> Self {
        Self {
            additive: AdditiveBiasCorrector::new(config),
            rain: RainBiasCorrector::new(config, RainCorrectionMethod::Ratio),
        }
    }

    pub fn with_rain_method(config: BiasUpdateConfig, method: RainCorrectionMethod) -> Self {
        Self {
            additive: AdditiveBiasCorrector::new(config),
            rain: RainBiasCorrector::new(config, method),
        }
    }

    /// Koreksi nilai prakiraan untuk variabel apa pun.
    /// Mengembalikan error jika variabel tidak bisa dikoreksi atau konfigurasi belum lengkap.
    pub fn correct(
        &self,
        variable: WeatherVariable,
        raw_value: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<CorrectedValue, WIError> {
        match BiasVariableGroup::classify(variable) {
            BiasVariableGroup::Additive(_) => {
                let corr = self.additive.correct(
                    variable,
                    raw_value,
                    station_id,
                    lead_seconds,
                    cycle_hour,
                    now,
                )?;
                Ok(CorrectedValue::Additive(corr))
            }
            BiasVariableGroup::Rain => {
                let corr =
                    self.rain
                        .correct(raw_value, station_id, lead_seconds, cycle_hour, now)?;
                Ok(CorrectedValue::Rain(corr))
            }
            BiasVariableGroup::NotCorrectable => Err(WIError::validation(format!(
                "Variabel {:?} tidak bisa dikoreksi bias",
                variable
            ))),
        }
    }

    /// Perbarui state bias dengan pasangan prakiraan dan pengamatan.
    /// Memerlukan alpha yang sudah disetel (BelumDiputuskan jika belum).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        variable: WeatherVariable,
        forecast: f32,
        observation: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), WIError> {
        match BiasVariableGroup::classify(variable) {
            BiasVariableGroup::Additive(_) => self.additive.update_bias(
                variable,
                forecast,
                observation,
                station_id,
                lead_seconds,
                cycle_hour,
                now,
            ),
            BiasVariableGroup::Rain => self.rain.update_bias(
                forecast,
                observation,
                station_id,
                lead_seconds,
                cycle_hour,
                now,
            ),
            BiasVariableGroup::NotCorrectable => Err(WIError::validation(format!(
                "Variabel {:?} tidak bisa dikoreksi bias",
                variable
            ))),
        }
    }

    /// Reset semua state bias (dipanggil saat versi model berubah).
    pub fn reset_all(&mut self) {
        self.additive.reset_all();
        self.rain.reset_all();
    }

    pub fn config(&self) -> &BiasUpdateConfig {
        self.additive.config()
    }
}

impl Default for BiasCorrectionEngine {
    fn default() -> Self {
        Self::new(BiasUpdateConfig::new())
    }
}

/// Hasil koreksi bias terdefinisi tipe.
#[derive(Debug, Clone)]
pub enum CorrectedValue {
    Additive(additive::AdditiveCorrection),
    Rain(rain::RainCorrection),
}

impl CorrectedValue {
    pub fn corrected_value(&self) -> f32 {
        match self {
            Self::Additive(c) => c.corrected_value,
            Self::Rain(c) => c.corrected_value,
        }
    }

    pub fn raw_value(&self) -> f32 {
        match self {
            Self::Additive(c) => c.raw_value,
            Self::Rain(c) => c.raw_value,
        }
    }

    pub fn variable(&self) -> WeatherVariable {
        match self {
            Self::Additive(c) => c.variable,
            Self::Rain(_) => WeatherVariable::Precipitation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use chrono::Utc;

    #[test]
    fn test_engine_additive() {
        let engine = BiasCorrectionEngine::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let result = engine
            .correct(
                WeatherVariable::Temperature,
                300.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(result.corrected_value(), 300.0);
    }

    #[test]
    fn test_engine_rain() {
        let engine = BiasCorrectionEngine::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let result = engine
            .correct(
                WeatherVariable::Precipitation,
                10.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(result.corrected_value(), 10.0);
    }

    #[test]
    fn test_engine_not_correctable() {
        let engine = BiasCorrectionEngine::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let err = engine.correct(WeatherVariable::WindSpeed, 5.0, "stn-001", 3600, 12, now);
        assert!(err.is_err());
    }

    #[test]
    fn test_engine_update_requires_alpha() {
        let mut engine = BiasCorrectionEngine::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        let err = engine.update(
            WeatherVariable::Temperature,
            300.0,
            299.0,
            "stn-001",
            3600,
            12,
            now,
        );
        let err = err.unwrap_err();
        assert!(err.to_string().contains("belum diputuskan"));
    }

    #[test]
    fn test_engine_reset_all() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.5).unwrap();
        let mut engine = BiasCorrectionEngine::new(config);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        engine
            .update(
                WeatherVariable::Temperature,
                300.0,
                299.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();

        engine.reset_all();
        // Setelah reset, corrected = raw (b = 0)
        let result = engine
            .correct(
                WeatherVariable::Temperature,
                300.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(result.corrected_value(), 300.0);
    }
}
