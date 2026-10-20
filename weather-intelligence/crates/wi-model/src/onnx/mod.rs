//! # ONNX Downscaler
//!
//! ONNX Runtime inference untuk regional downscaling 0.5km Indonesia.
//!
//! Input (14 features per cell):
//!   lat, lon, lead_hr, T_coarse, hum, precip, wind_u, wind_v, pressure,
//!   elevation, land_mask, month, hour, sin_month
//!
//! Output (24 features = 6 vars × 4 quantiles):
//!   [temp, humidity, precip, wind_u, wind_v, pressure] × [q10, q25, q75, q90]
//!
//! CATATAN: Memerlukan file model .onnx yang sudah di-train.
//! Gunakan `OnnxDownscaler::placeholder()` untuk development tanpa model.

use async_trait::async_trait;
use ndarray::Array1;

use crate::traits::{Downscaler, DownscalerInput, DownscalerOutput, ModelInfo, QuantilePrediction};
use wi_core::{Result, WIError, WeatherVariable};

// ─────────────────────────────────────────────────────────────────────────────
// Feature engineering
// ─────────────────────────────────────────────────────────────────────────────

/// Buat feature vector [14] dari DownscalerInput
pub fn build_features(input: &DownscalerInput) -> Array1<f32> {
    let sin_month = (input.month as f32 * 2.0 * std::f32::consts::PI / 12.0).sin();
    Array1::from(vec![
        input.lat as f32,
        input.lon as f32,
        input.lead_seconds as f32 / 3600.0,
        input.coarse_temp,
        input.coarse_humidity,
        input.coarse_precip,
        input.coarse_wind_u,
        input.coarse_wind_v,
        input.coarse_pressure,
        input.elevation,
        input.land_mask,
        input.month as f32,
        input.hour as f32,
        sin_month,
    ])
}

/// Konversi output flat [24] → DownscalerOutput (6 vars × 4 quantiles)
pub fn extract_output(flat: &[f32]) -> Result<DownscalerOutput> {
    if flat.len() < 24 {
        return Err(WIError::model(format!(
            "Output tensor terlalu pendek: {} < 24",
            flat.len()
        )));
    }
    let q = |idx: usize| -> QuantilePrediction {
        QuantilePrediction {
            p10: flat[idx * 4],
            p50: flat[idx * 4 + 1],
            p90: flat[idx * 4 + 2],
            prob_rain: flat[idx * 4 + 3],
        }
    };
    Ok(DownscalerOutput {
        temp: q(0),
        humidity: q(1),
        precip: q(2),
        wind_u: q(3),
        wind_v: q(4),
        pressure: q(5),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// OnnxDownscaler
// ─────────────────────────────────────────────────────────────────────────────

/// ONNX Downscaler — regional 0.5km dengan quantile output
///
/// Untuk development tanpa model .onnx, gunakan `OnnxDownscaler::placeholder()`.
/// Untuk production, model .onnx di-export dari PyTorch training pipeline.
///
/// TODO (Phase 2): Implementasi penuh dengan ort::Session setelah model tersedia.
/// ```rust,ignore
/// use ort::{GraphOptimizationLevel, Session};
/// let session = Session::builder()?
///     .with_optimization_level(GraphOptimizationLevel::Level3)?
///     .with_intra_threads(4)?
///     .commit_from_file(model_path)?;
/// ```
#[derive(Debug)]
pub struct OnnxDownscaler {
    is_placeholder: bool,
    model_version: String,
}

impl OnnxDownscaler {
    /// Buat downscaler placeholder (deterministik, output zeros).
    /// Digunakan untuk development sebelum model .onnx tersedia.
    pub fn placeholder() -> Self {
        Self {
            is_placeholder: true,
            model_version: "0.0.0-placeholder".to_string(),
        }
    }

    /// Buat downscaler dari file .onnx (production).
    /// Saat ini mengembalikan placeholder — akan menggunakan ort::Session setelah
    /// model .onnx tersedia dari Python training pipeline.
    pub fn from_file(model_path: &str, version: &str) -> Result<Self> {
        // Validasi path setidaknya ada (belum buka file)
        if model_path.is_empty() {
            return Err(WIError::validation("model_path tidak boleh kosong"));
        }
        // TODO Phase 2: buka ort::Session dari file
        Ok(Self {
            is_placeholder: true, // sampai model .onnx tersedia
            model_version: version.to_string(),
        })
    }

    /// Validasi input sebelum inferensi
    fn validate(&self, input: &DownscalerInput) -> Result<()> {
        if !input.lat.is_finite() || !input.lon.is_finite() {
            return Err(WIError::validation("lat/lon tidak finite"));
        }
        if input.month == 0 || input.month > 12 {
            return Err(WIError::validation(format!(
                "month tidak valid: {} (harus 1-12)",
                input.month
            )));
        }
        if input.hour > 23 {
            return Err(WIError::validation(format!(
                "hour tidak valid: {} (harus 0-23)",
                input.hour
            )));
        }
        Ok(())
    }

    /// Jalankan inferensi — placeholder deterministik
    fn run_placeholder(&self, _input: &DownscalerInput) -> DownscalerOutput {
        let zero_q = || QuantilePrediction {
            p10: 0.0,
            p50: 0.0,
            p90: 0.0,
            prob_rain: 0.0,
        };
        DownscalerOutput {
            temp: zero_q(),
            humidity: zero_q(),
            precip: zero_q(),
            wind_u: zero_q(),
            wind_v: zero_q(),
            pressure: zero_q(),
        }
    }
}

#[async_trait]
impl Downscaler for OnnxDownscaler {
    async fn downscale(&self, input: DownscalerInput) -> Result<DownscalerOutput> {
        self.validate(&input)?;

        if self.is_placeholder {
            return Ok(self.run_placeholder(&input));
        }

        // Real ONNX inference — TODO Phase 2
        Err(WIError::model(
            "ONNX session belum diinisialisasi. Gunakan placeholder() untuk development.",
        ))
    }

    fn info(&self) -> ModelInfo {
        let arch = if self.is_placeholder {
            "Placeholder — 14 features → 24 quantile outputs (zeros, belum trained)"
        } else {
            "ONNX — 14 features → 24 quantile outputs (6 vars × 4 quantiles)"
        };
        ModelInfo {
            name: "ONNX Downscaler (wi-model)".to_string(),
            version: self.model_version.clone(),
            architecture: arch.to_string(),
            num_params: 0,
            input_vars: WeatherVariable::surface_variables(),
            output_vars: WeatherVariable::surface_variables(),
            supported_lead_times: vec![3600, 10800, 21600, 43200, 86400, 172800],
            supported_resolutions: vec![500, 1000],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input() -> DownscalerInput {
        DownscalerInput {
            lat: -6.2,
            lon: 106.8,
            lead_seconds: 3600,
            coarse_temp: 300.0,
            coarse_humidity: 0.7,
            coarse_precip: 1.5,
            coarse_wind_u: 2.0,
            coarse_wind_v: -1.0,
            coarse_pressure: 1013.0,
            elevation: 8.0,
            land_mask: 1.0,
            month: 10,
            hour: 12,
        }
    }

    #[test]
    fn test_build_features_length() {
        let input = make_input();
        let feat = build_features(&input);
        assert_eq!(feat.len(), 14);
    }

    #[test]
    fn test_build_features_lead_conversion() {
        let input = make_input();
        let feat = build_features(&input);
        // lead_seconds=3600 → lead_hr=1.0
        assert!((feat[2] - 1.0).abs() < 1e-5, "lead_hr: {}", feat[2]);
    }

    #[test]
    fn test_build_features_sin_month() {
        let input = make_input(); // month=10
        let feat = build_features(&input);
        let expected = (10.0_f32 * 2.0 * std::f32::consts::PI / 12.0).sin();
        assert!(
            (feat[13] - expected).abs() < 1e-5,
            "sin_month: {}",
            feat[13]
        );
    }

    #[test]
    fn test_extract_output_ok() {
        let flat: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = extract_output(&flat);
        assert!(out.is_ok(), "{:?}", out.err());
        let o = out.unwrap();
        assert!((o.temp.p10 - 0.0).abs() < 1e-5);
        assert!((o.temp.p50 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_output_too_short() {
        let flat: Vec<f32> = vec![0.0; 10];
        let out = extract_output(&flat);
        assert!(out.is_err(), "harus error untuk output pendek");
    }

    #[tokio::test]
    async fn test_placeholder_downscale_ok() {
        let ds = OnnxDownscaler::placeholder();
        let input = make_input();
        let result = ds.downscale(input).await;
        assert!(result.is_ok(), "{:?}", result.err());
        let o = result.unwrap();
        // Placeholder returns zeros
        assert_eq!(o.temp.p50, 0.0);
    }

    #[tokio::test]
    async fn test_placeholder_invalid_month() {
        let ds = OnnxDownscaler::placeholder();
        let mut input = make_input();
        input.month = 13;
        let result = ds.downscale(input).await;
        assert!(result.is_err(), "month=13 harus error");
    }

    #[tokio::test]
    async fn test_placeholder_invalid_hour() {
        let ds = OnnxDownscaler::placeholder();
        let mut input = make_input();
        input.hour = 24;
        let result = ds.downscale(input).await;
        assert!(result.is_err(), "hour=24 harus error");
    }

    #[tokio::test]
    async fn test_placeholder_invalid_lat() {
        let ds = OnnxDownscaler::placeholder();
        let mut input = make_input();
        input.lat = f64::NAN;
        let result = ds.downscale(input).await;
        assert!(result.is_err(), "lat=NaN harus error");
    }

    #[test]
    fn test_from_file_empty_path_error() {
        let result = OnnxDownscaler::from_file("", "1.0");
        assert!(result.is_err(), "path kosong harus error");
    }

    #[test]
    fn test_from_file_valid_path_ok() {
        // Non-existent path OK karena belum buka file (Phase 2 TODO)
        let result = OnnxDownscaler::from_file("/path/to/model.onnx", "1.0.0");
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_info_placeholder() {
        let ds = OnnxDownscaler::placeholder();
        let info = ds.info();
        assert!(info.name.contains("ONNX"));
        assert!(info.architecture.contains("Placeholder"));
        assert!(!info.supported_lead_times.is_empty());
    }
}
