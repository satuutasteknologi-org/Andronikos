//! Koreksi bias aditif untuk variabel kontinu (suhu, tekanan, angin, kelembapan).
//!
//! Rumus: x_koreksi = x_prakiraan - b
//! Pembaruan: b_baru = (1 - α) * b_lama + α * (prakiraan - pengamatan)

use chrono::{DateTime, Utc};
use wi_core::error::WIError;
use wi_core::types::WeatherVariable;

use super::{BiasKey, BiasStore, BiasUpdateConfig};

/// Variabel yang boleh dikoreksi aditif langsung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdditiveVariable {
    Temperature2m,
    MeanSeaLevelPressure,
    WindU10,
    WindV10,
    RelativeHumidity,
}

impl AdditiveVariable {
    pub fn from_weather_variable(var: WeatherVariable) -> Option<Self> {
        match var {
            WeatherVariable::Temperature => Some(Self::Temperature2m),
            WeatherVariable::Pressure | WeatherVariable::SurfacePressure => {
                Some(Self::MeanSeaLevelPressure)
            }
            WeatherVariable::WindU => Some(Self::WindU10),
            WeatherVariable::WindV => Some(Self::WindV10),
            WeatherVariable::RelativeHumidity => Some(Self::RelativeHumidity),
            _ => None,
        }
    }

    pub fn weather_variable(self) -> WeatherVariable {
        match self {
            Self::Temperature2m => WeatherVariable::Temperature,
            Self::MeanSeaLevelPressure => WeatherVariable::Pressure,
            Self::WindU10 => WeatherVariable::WindU,
            Self::WindV10 => WeatherVariable::WindV,
            Self::RelativeHumidity => WeatherVariable::RelativeHumidity,
        }
    }
}

/// Hasil koreksi aditif.
#[derive(Debug, Clone)]
pub struct AdditiveCorrection {
    pub variable: WeatherVariable,
    pub raw_value: f32,
    pub bias_b: f32,
    pub corrected_value: f32,
    pub station_id: String,
    pub lead_seconds: u32,
    pub cycle_hour: u8,
}

/// Pemroses koreksi bias aditif.
#[derive(Debug, Clone)]
pub struct AdditiveBiasCorrector {
    store: BiasStore,
    config: BiasUpdateConfig,
}

impl AdditiveBiasCorrector {
    pub fn new(config: BiasUpdateConfig) -> Self {
        Self {
            store: BiasStore::new(),
            config,
        }
    }

    /// Koreksi nilai prakiraan dengan bias b yang tersimpan.
    /// Jika tidak ada state untuk kunci ini, b = 0 (tidak ada koreksi).
    pub fn correct(
        &self,
        variable: WeatherVariable,
        raw_value: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: DateTime<Utc>,
    ) -> Result<AdditiveCorrection, WIError> {
        let _ = now;
        let add_var = AdditiveVariable::from_weather_variable(variable).ok_or_else(|| {
            WIError::validation(format!(
                "Variabel {:?} tidak mendukung koreksi aditif",
                variable
            ))
        })?;

        let key = BiasKey::without_season(station_id, variable, lead_seconds, cycle_hour);
        let b = self.store.get(&key).map(|s| s.b).unwrap_or(0.0);

        let corrected = match add_var {
            AdditiveVariable::RelativeHumidity => {
                let c = raw_value - b;
                // Clamp ke rentang fisik 0-100%
                c.clamp(0.0, 100.0)
            }
            AdditiveVariable::WindU10 | AdditiveVariable::WindV10 => raw_value - b,
            AdditiveVariable::Temperature2m => raw_value - b,
            AdditiveVariable::MeanSeaLevelPressure => raw_value - b,
        };

        Ok(AdditiveCorrection {
            variable,
            raw_value,
            bias_b: b,
            corrected_value: corrected,
            station_id: station_id.to_string(),
            lead_seconds,
            cycle_hour,
        })
    }

    /// Perbarui state bias dengan pasangan prakiraan dan pengamatan.
    /// Memerlukan alpha yang sudah disetel (BelumDiputuskan jika belum).
    #[allow(clippy::too_many_arguments)]
    pub fn update_bias(
        &mut self,
        variable: WeatherVariable,
        forecast: f32,
        observation: f32,
        station_id: &str,
        lead_seconds: u32,
        cycle_hour: u8,
        now: DateTime<Utc>,
    ) -> Result<(), WIError> {
        let alpha = self.config.require_alpha()?;

        let _ = AdditiveVariable::from_weather_variable(variable).ok_or_else(|| {
            WIError::validation(format!(
                "Variabel {:?} tidak mendukung koreksi aditif",
                variable
            ))
        })?;

        let key = BiasKey::without_season(station_id, variable, lead_seconds, cycle_hour);
        let state = self.store.get_or_default(key);
        let diff = forecast - observation;
        state.update(diff, alpha, now);
        Ok(())
    }

    /// Reset semua state bias (dipanggil saat versi model berubah).
    pub fn reset_all(&mut self) {
        self.store.reset_all();
    }

    pub fn config(&self) -> &BiasUpdateConfig {
        &self.config
    }

    pub fn store(&self) -> &BiasStore {
        &self.store
    }
}

impl Default for AdditiveBiasCorrector {
    fn default() -> Self {
        Self::new(BiasUpdateConfig::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_additive_correction_temperature() {
        let corrector = AdditiveBiasCorrector::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // Tanpa state sebelumnya, b = 0, corrected = raw
        let corr = corrector
            .correct(
                WeatherVariable::Temperature,
                300.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(corr.corrected_value, 300.0);
        assert_eq!(corr.bias_b, 0.0);
    }

    #[test]
    fn test_additive_bias_update_requires_alpha() {
        let mut corrector = AdditiveBiasCorrector::new(BiasUpdateConfig::new());
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // Alpha belum diset → BelumDiputuskan
        let err = corrector.update_bias(
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
    fn test_additive_bias_update_with_alpha() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.5).unwrap();
        let mut corrector = AdditiveBiasCorrector::new(config);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // Forecast 300, observation 299 → diff = 1, b_baru = (1-0.5)*0 + 0.5*1 = 0.5
        corrector
            .update_bias(
                WeatherVariable::Temperature,
                300.0,
                299.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();

        // Sekarang corrected = 300 - 0.5 = 299.5
        let corr = corrector
            .correct(
                WeatherVariable::Temperature,
                300.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert!((corr.corrected_value - 299.5).abs() < 1e-6);
    }

    #[test]
    fn test_humidity_clamp() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.5).unwrap();
        let mut corrector = AdditiveBiasCorrector::new(config);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        // Set bias besar agar corrected melampaui batas
        corrector
            .update_bias(
                WeatherVariable::RelativeHumidity,
                50.0,
                10.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();

        // raw 120, b = 0.5*(50-10) = 20, corrected = 120-20 = 100 (clamp)
        let corr = corrector
            .correct(
                WeatherVariable::RelativeHumidity,
                120.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(corr.corrected_value, 100.0);

        // raw 0, corrected = 0-20 = -20 → clamp ke 0
        let corr = corrector
            .correct(
                WeatherVariable::RelativeHumidity,
                0.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();
        assert_eq!(corr.corrected_value, 0.0);
    }

    #[test]
    fn test_reset_all() {
        let mut config = BiasUpdateConfig::new();
        config.set_alpha(0.5).unwrap();
        let mut corrector = AdditiveBiasCorrector::new(config);
        let now = Utc.with_ymd_and_hms(2026, 10, 15, 12, 0, 0).unwrap();

        corrector
            .update_bias(
                WeatherVariable::Temperature,
                300.0,
                299.0,
                "stn-001",
                3600,
                12,
                now,
            )
            .unwrap();

        assert!(!corrector.store().is_empty());
        corrector.reset_all();
        assert!(corrector.store().is_empty());
    }
}
