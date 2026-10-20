//! # Core Traits
//!
//! Common traits for the weather intelligence engine.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::types::{
    Bbox, DataSource, FeatureVector, GeoCoord, GridPredictRequest, InferenceModelConfig,
    ModelPrediction, WeatherForecast, WeatherObservation, WeatherVariable,
};
use crate::Result;

/// Trait for weather data sources
#[async_trait]
pub trait WeatherSource: Send + Sync {
    fn name(&self) -> &'static str;
    fn source_type(&self) -> DataSource;
    fn provides_forecasts(&self) -> bool {
        false
    }
    fn provides_historical(&self) -> bool {
        false
    }
    fn rate_limit(&self) -> f64 {
        10.0
    }

    async fn fetch_observations(
        &self,
        region: Bbox,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<WeatherObservation>> + Send + Unpin>>;

    async fn fetch_forecast(&self, coord: GeoCoord, lead_seconds: u32) -> Result<WeatherForecast>;
}

/// Trait for grid systems
pub trait Grid: Send + Sync {
    type CellId: Copy + Eq + std::hash::Hash + Send + Sync;

    #[allow(clippy::wrong_self_convention)]
    fn from_lat_lon(&self, lat: f64, lon: f64) -> Result<Self::CellId>;
    fn center(&self, cell_id: Self::CellId) -> GeoCoord;
    fn neighbors(&self, cell_id: Self::CellId) -> Vec<Self::CellId>;
    fn resolution_meters(&self) -> f64;
    fn bbox(&self) -> Bbox;
    fn contains(&self, cell_id: Self::CellId) -> bool;
}

/// Trait for weather forecasters
#[async_trait]
pub trait Forecaster: Send + Sync {
    async fn predict_point(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
        variables: &[WeatherVariable],
    ) -> Result<HashMap<WeatherVariable, f32>>;

    async fn predict_grid(&self, request: GridPredictRequest) -> Result<Vec<GridPrediction>>;

    fn model_info(&self) -> ModelInfo;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPrediction {
    pub lat: f64,
    pub lon: f64,
    pub cell_id: u64,
    pub predictions: HashMap<WeatherVariable, f32>,
    pub quantiles: Option<HashMap<WeatherVariable, QuantilePrediction>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantilePrediction {
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub num_parameters: usize,
    pub supported_variables: Vec<WeatherVariable>,
    pub supported_lead_times: Vec<u32>,
    pub supported_resolutions: Vec<u8>,
}

/// Trait for downscaling models
#[async_trait]
pub trait Downscaler: Send + Sync {
    async fn downscale(
        &self,
        coarse_forecast: &WeatherForecast,
        features: &FeatureVector,
    ) -> Result<ModelPrediction>;

    fn model_info(&self) -> ModelInfo;
}

/// Trait for model training
#[async_trait]
pub trait Trainer: Send + Sync {
    async fn train(&mut self, data_path: &str) -> Result<TrainingResult>;
    async fn validate(&self, data_path: &str) -> Result<ValidationMetrics>;
    fn model_config(&self) -> InferenceModelConfig;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub final_epoch: usize,
    pub duration_secs: f64,
    pub best_epoch: usize,
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub rmse: HashMap<WeatherVariable, f32>,
    pub mae: HashMap<WeatherVariable, f32>,
    pub bias: HashMap<WeatherVariable, f32>,
    pub brier_score: HashMap<WeatherVariable, f32>,
    pub crps: Option<f32>,
}

/// Trait for verification
#[async_trait]
pub trait Verifier: Send + Sync {
    async fn verify(
        &self,
        forecasts: &[ModelPrediction],
        observations: &[WeatherObservation],
    ) -> Result<VerificationResult>;

    async fn verify_ensemble(
        &self,
        ensemble: &[Vec<ModelPrediction>],
        observations: &[WeatherObservation],
    ) -> Result<EnsembleVerificationResult>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub rmse: HashMap<WeatherVariable, f32>,
    pub mae: HashMap<WeatherVariable, f32>,
    pub bias: HashMap<WeatherVariable, f32>,
    pub brier: HashMap<WeatherVariable, f32>,
    pub reliability: HashMap<WeatherVariable, Vec<(f32, f32)>>, // (predicted_prob, observed_freq)
    pub sample_count: usize,
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleVerificationResult {
    pub crps: f32,
    pub spread: f32,
    pub spread_skill_ratio: f32,
    pub rank_histogram: Vec<usize>,
}

/// Trait for storage backends
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize>;
    async fn read_observations(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherObservation>>;
    async fn write_forecasts(&self, forecasts: Vec<WeatherForecast>) -> Result<usize>;
    async fn read_forecasts(
        &self,
        model_id: &str,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherForecast>>;
    async fn get_stats(&self) -> Result<StorageStats>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_size_bytes: u64,
    pub file_count: u64,
    pub observation_count: u64,
    pub forecast_count: u64,
}

/// Trait for cache backends
#[async_trait]
pub trait CacheBackend: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn set(&self, key: &str, value: &[u8], ttl_secs: u64) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn clear(&self) -> Result<()>;
    async fn stats(&self) -> Result<CacheStats>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
    pub entry_count: u64,
}

/// Trait for data validation
pub trait Validator {
    fn validate(&self, obs: &WeatherObservation) -> ValidationResult;
    fn validate_batch(&self, observations: &mut [WeatherObservation]) -> Vec<ValidationResult>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub observation: WeatherObservation,
    pub was_modified: bool,
    pub issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationIssue {
    TemperatureOutOfRange { value: f64, min: f64, max: f64 },
    PressureOutOfRange { value: f64, min: f64, max: f64 },
    HumidityOutOfRange { value: f64, min: f64, max: f64 },
    WindSpeedExcessive { value: f64, max: f64 },
    InconsistentWindComponents { u: f64, v: f64, speed: f64 },
    DuplicateTimestamp,
    InvalidCoordinate,
    NegativePrecipitation,
    MissingRequiredVariable { variable: WeatherVariable },
}

/// Trait for gap filling
pub trait GapFiller {
    fn fill_gaps(&self, observations: &mut Vec<WeatherObservation>);
    fn interpolate_temporal(&self, observations: &mut Vec<WeatherObservation>, max_gap_hours: i64);
}

/// Trait for feature engineering
pub trait FeatureEngineer {
    fn engineer_features(&self, observations: &[WeatherObservation]) -> Vec<FeatureVector>;
    fn feature_names(&self) -> Vec<String>;
}

use serde::{Deserialize, Serialize};
