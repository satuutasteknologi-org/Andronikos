//! # Model Traits
//!
//! Common traits for ML models.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::fusion::FusionOutput;
use crate::tiling::CoordinatedInferenceResult;
use wi_core::types::{BatchGridPredictOutput, BatchGridPredictRequest, Bbox, GridPredictRequest};
use wi_core::{DeviceType, Result, WeatherObservation, WeatherVariable};

/// Maksimum lead time: 168 jam (7 hari) per K-19. Menit hanya untuk nowcasting 0-3 jam.
pub const MAX_LEAD_SECONDS: u32 = 168 * 3600;

/// Target waktu maksimum untuk inferensi 38.9M cell Indonesia 0.5km: 1 menit (60 detik).
pub const TARGET_INFERENCE_SECONDS: u64 = 60;

/// Trait for weather forecasters
#[async_trait]
pub trait Forecaster: Send + Sync + std::fmt::Debug {
    async fn predict_point(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
        variables: &[WeatherVariable],
    ) -> Result<FusionOutput>;

    /// Prediksi grid untuk satu area (legacy, akan diarahkan ke predict_grid_batch)
    async fn predict_grid(&self, request: GridPredictRequest) -> Result<Vec<GridPrediction>>;

    /// Prediksi grid batch untuk banyak area dengan tiling dan GPU support.
    /// Target: < 1 menit untuk 0.5km multi-area.
    async fn predict_grid_batch(
        &self,
        request: BatchGridPredictRequest,
    ) -> Result<BatchGridPredictOutput>;

    /// Inferensi distributed untuk satu area besar (bisa 38.9M cell).
    /// Otomatis split ke multiple tiles dan process secara parallel.
    /// Guard: gagal jika total waktu > TARGET_INFERENCE_SECONDS.
    async fn infer_area_distributed(
        &self,
        area_id: &str,
        bbox: &Bbox,
        resolution: u8,
        lead_seconds: u32,
        variables: &[WeatherVariable],
    ) -> Result<CoordinatedInferenceResult>;

    /// Device yang digunakan untuk inferensi.
    fn device_type(&self) -> DeviceType;

    fn model_info(&self) -> ModelInfo;
}

/// Trait for downscalers
#[async_trait]
pub trait Downscaler: Send + Sync {
    async fn downscale(&self, input: DownscalerInput) -> Result<DownscalerOutput>;

    fn info(&self) -> ModelInfo;
}

/// Grid prediction result
#[derive(Debug, Clone)]
pub struct GridPrediction {
    pub lat: f64,
    pub lon: f64,
    pub cell_id: u64,
    pub predictions: Vec<(WeatherVariable, f32)>,
    pub quantiles: Option<HashMap<WeatherVariable, QuantilePrediction>>,
    pub confidence: f32,
    pub model_version: String,
    /// Kontrak output Andronikos per cell
    pub lead_seconds: u32,
    pub probabilitas: Option<f32>,
    pub tingkat_siaga: Option<String>,
    pub anggota: usize,
    pub koreksi_bias: bool,
    pub sumber_id: Vec<String>,
    pub atribusi: Vec<String>,
    pub label_istilah: String,
}

/// Quantile prediction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantilePrediction {
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
    pub prob_rain: f32,
}

/// Downscaler input - owned types to avoid lifetime issues
#[derive(Debug, Clone)]
pub struct DownscalerInput {
    pub lat: f64,
    pub lon: f64,
    pub lead_seconds: u32,
    pub coarse_temp: f32,
    pub coarse_humidity: f32,
    pub coarse_precip: f32,
    pub coarse_wind_u: f32,
    pub coarse_wind_v: f32,
    pub coarse_pressure: f32,
    pub elevation: f32,
    pub land_mask: f32,
    pub month: u8,
    pub hour: u8,
}

/// Downscaler output
#[derive(Debug, Clone)]
pub struct DownscalerOutput {
    pub temp: QuantilePrediction,
    pub humidity: QuantilePrediction,
    pub precip: QuantilePrediction,
    pub wind_u: QuantilePrediction,
    pub wind_v: QuantilePrediction,
    pub pressure: QuantilePrediction,
}

/// Model info
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub num_params: usize,
    pub input_vars: Vec<WeatherVariable>,
    pub output_vars: Vec<WeatherVariable>,
    pub supported_lead_times: Vec<u32>,
    pub supported_resolutions: Vec<u32>,
}

/// Trainer trait
#[async_trait]
pub trait Trainer: Send + Sync {
    async fn train(
        &mut self,
        train_data: &[WeatherObservation],
        val_data: &[WeatherObservation],
    ) -> Result<TrainingResult>;
    async fn validate(&self, data: &[WeatherObservation]) -> Result<ValidationMetrics>;
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub train_loss: f32,
    pub val_loss: f32,
    pub epochs: usize,
    pub best_epoch: usize,
}

/// Validation metrics
#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    pub mae: f32,
    pub rmse: f32,
    pub bias: f32,
    pub correlation: f32,
}

/// Forecaster wrapper for dyn compatibility
#[derive(Clone)]
pub struct DynForecaster(pub Arc<dyn Forecaster>);

/// Downscaler wrapper for dyn compatibility
#[derive(Clone)]
pub struct DynDownscaler(pub Arc<dyn Downscaler>);
