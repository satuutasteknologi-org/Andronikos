//! # Configuration Management
//!
//! AetherForge uses Figment for flexible configuration from multiple sources:
//! - TOML files
//! - Environment variables
//! - CLI arguments
//!
//! Configuration is hierarchical with environment-specific overrides.

use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root configuration for AetherForge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub data: DataConfig,
    pub ml: MLConfig,
    pub storage: StorageConfig,
    pub api: ApiConfig,
    pub observability: ObservabilityConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            ml: MLConfig::default(),
            storage: StorageConfig::default(),
            api: ApiConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

/// Data ingestion and sync configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub sources: Vec<SourceConfig>,
    pub sync: SyncConfig,
    pub preprocessing: PreprocessingConfig,
    pub rate_limit: RateLimitConfig,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            sources: vec![
                SourceConfig::open_meteo(),
                SourceConfig::noaa(),
                SourceConfig::era5(),
            ],
            sync: SyncConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            rate_limit: RateLimitConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub name: String,
    pub source_type: SourceType,
    pub enabled: bool,
    pub priority: u8,
    pub api_key: Option<String>,
    pub base_url: String,
    pub params: HashMap<String, toml::Value>,
}

impl SourceConfig {
    pub fn open_meteo() -> Self {
        Self {
            name: "open_meteo".into(),
            source_type: SourceType::OpenMeteo,
            enabled: true,
            priority: 10,
            api_key: None,
            base_url: "https://archive-api.open-meteo.com/v1".into(),
            params: HashMap::new(),
        }
    }

    pub fn noaa() -> Self {
        Self {
            name: "noaa".into(),
            source_type: SourceType::NOAA,
            enabled: true,
            priority: 20,
            api_key: None,
            base_url: "https://www.ncdc.noaa.gov/cdo-web/api/v2".into(),
            params: HashMap::new(),
        }
    }

    pub fn era5() -> Self {
        Self {
            name: "era5".into(),
            source_type: SourceType::ERA5,
            enabled: true,
            priority: 5,
            api_key: None,
            base_url: "https://cds.climate.copernicus.eu/api/v2".into(),
            params: HashMap::new(),
        }
    }
}

use std::collections::HashMap;
use toml;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    OpenMeteo,
    OpenWeatherMap,
    NOAA,
    ERA5,
    GFS,
    ECMWF,
    Meteomatics,
    VisualCrossing,
    DWD,
    MeteoFrance,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub parallel_requests: usize,
    pub batch_size: usize,
    pub look_back_days: i64,
    pub look_ahead_days: i64,
    pub retry_attempts: u8,
    pub retry_delay_ms: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            parallel_requests: 16,
            batch_size: 1000,
            look_back_days: 365,
            look_ahead_days: 14,
            retry_attempts: 3,
            retry_delay_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub spatial_resolution: u8,
    pub temporal_resolution_seconds: u32, // Supports 30s, 60s, 300s, etc.
    pub quantization_bins: usize,
    pub missing_value_strategy: MissingValueStrategy,
    pub feature_engineering: FeatureEngineeringConfig,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            spatial_resolution: 7,           // ~10km resolution
            temporal_resolution_seconds: 60, // 1 minute default
            quantization_bins: 256,
            missing_value_strategy: MissingValueStrategy::PhysicsInformed,
            feature_engineering: FeatureEngineeringConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    FillNaN,
    LinearInterpolation,
    Kriging,
    PhysicsInformed,
    Drop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub enable_temporal_features: bool,
    pub enable_lagged_features: bool,
    pub enable_spatial_lag: bool,
    pub enable_interaction_terms: bool,
    pub lagged_seconds: Vec<u32>, // Supports sub-hour lags
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            enable_temporal_features: true,
            enable_lagged_features: true,
            enable_spatial_lag: true,
            enable_interaction_terms: false,
            lagged_seconds: vec![60, 180, 360, 720, 1440, 2880, 4320, 10080], // 1m, 3m, 6m, 12m, 24m, 48m, 72m, 168m
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: f64,
    pub burst_size: usize,
    pub per_source_limits: HashMap<String, (f64, usize)>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        let mut per_source_limits = HashMap::new();
        per_source_limits.insert("era5".into(), (2.0, 4)); // CDS API limits
        per_source_limits.insert("open_meteo".into(), (10.0, 20));
        per_source_limits.insert("noaa".into(), (5.0, 10));

        Self {
            requests_per_second: 50.0,
            burst_size: 100,
            per_source_limits,
        }
    }
}

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub inference: InferenceConfig,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            inference: InferenceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub use_residual: bool,
    pub graph_aggregation: GraphAggregationType,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: "GraphCast".into(),
            hidden_dim: 256,
            num_layers: 16,
            num_heads: 8,
            dropout: 0.1,
            use_residual: true,
            graph_aggregation: GraphAggregationType::MultiHash,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphAggregationType {
    Mean,
    Attention,
    MultiHash,
    Transformer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub checkpoint_interval: usize,
    pub validation_split: f32,
    pub gradient_clip: Option<f32>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 100,
            learning_rate: 1e-4,
            optimizer: OptimizerConfig::AdamW,
            scheduler: SchedulerConfig::CosineAnnealing,
            early_stopping: EarlyStoppingConfig::default(),
            checkpoint_interval: 1,
            validation_split: 0.2,
            gradient_clip: Some(1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerConfig {
    Adam,
    AdamW,
    SGD,
    LAMB,
    Lion,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SchedulerConfig {
    Step { step_size: usize, gamma: f32 },
    CosineAnnealing,
    ReduceOnPlateau { patience: usize, factor: f32 },
    OneCycle,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self::CosineAnnealing
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f32,
    pub monitor: MonitoringMetric,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_delta: 1e-4,
            monitor: MonitoringMetric::ValidationLoss,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitoringMetric {
    ValidationLoss,
    RMSE,
    CRPS,
    CustomSkillScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub batch_size: usize,
    pub use_quantization: bool,
    pub use_tensorrt: bool,
    pub warmup_iterations: usize,
    pub num_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            use_quantization: false,
            use_tensorrt: false,
            warmup_iterations: 10,
            num_threads: 8,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub raw_dir: PathBuf,
    pub processed_dir: PathBuf,
    pub models_dir: PathBuf,
    pub checkpoints_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub parquet: ParquetConfig,
    pub compression: CompressionConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        let base = PathBuf::from("/var/aetherforge");
        Self {
            data_dir: base.join("data"),
            raw_dir: base.join("data/raw"),
            processed_dir: base.join("data/processed"),
            models_dir: base.join("models"),
            checkpoints_dir: base.join("checkpoints"),
            cache_dir: base.join("cache"),
            parquet: ParquetConfig::default(),
            compression: CompressionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetConfig {
    pub row_group_size: usize,
    pub page_size: usize,
    pub dictionary_encoding: bool,
    pub bloom_filter: bool,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            row_group_size: 100_000,
            page_size: 8192,
            dictionary_encoding: true,
            bloom_filter: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub level: i32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Snappy,
    Gzip,
    Zstd,
    Lz4,
    Brotli,
}

/// API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_connections: usize,
    pub timeout_seconds: u64,
    pub cors: CorsConfig,
    pub rate_limit: ApiRateLimitConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            workers: 4,
            max_connections: 1000,
            timeout_seconds: 30,
            cors: CorsConfig::default(),
            rate_limit: ApiRateLimitConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    pub enabled: bool,
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_origins: vec!["*".into()],
            allowed_methods: vec!["GET".into(), "POST".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: usize,
    pub burst_size: usize,
}

impl Default for ApiRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 100,
            burst_size: 20,
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub logging: LoggingConfig,
    pub tracing: TracingConfig,
    pub metrics: MetricsConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            logging: LoggingConfig::default(),
            tracing: TracingConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
    pub directory: PathBuf,
    pub rotation: LogRotation,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: LogFormat::Json,
            output: LogOutput::Both,
            directory: PathBuf::from("/var/aetherforge/logs"),
            rotation: LogRotation::Daily,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Plain,
    Pretty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogOutput {
    Stdout,
    File,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogRotation {
    Never,
    Hourly,
    Daily,
    Weekly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub sample_rate: f64,
    pub exporter: TracingExporter,
    pub service_name: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 0.1,
            exporter: TracingExporter::Otlp,
            service_name: "aetherforge".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracingExporter {
    Jaeger,
    Zipkin,
    Otlp,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub port: u16,
    pub path: String,
    pub exporters: Vec<MetricsExporter>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
            path: "/metrics".into(),
            exporters: vec![MetricsExporter::Prometheus],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsExporter {
    Prometheus,
    Statsd,
    Influxdb,
}

impl Config {
    /// Load configuration from default locations
    pub fn load() -> Result<Self, figment::Error> {
        Self::load_fromfigment(Figment::new())
    }

    /// Load configuration with custom Figment
    pub fn load_fromfigment(figment: Figment) -> Result<Self, figment::Error> {
        figment
            .merge(Toml::file("aetherforge.toml").nested())
            .merge(Toml::file("/etc/aetherforge/config.toml").nested())
            .merge(Env::prefixed("AETHERFORGE_").split("_"))
            .join(Serialized::defaults(Config::default()))
            .extract()
    }

    /// Save configuration to file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), anyhow::Error> {
        let toml = toml::to_string_pretty(self)?;
        std::fs::write(path, toml)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.data.sources.len(), 3);
        assert_eq!(config.ml.model.hidden_dim, 256);
    }

    #[test]
    fn test_source_config() {
        let open_meteo = SourceConfig::open_meteo();
        assert_eq!(open_meteo.source_type, SourceType::OpenMeteo);
        assert!(open_meteo.enabled);
    }
}
