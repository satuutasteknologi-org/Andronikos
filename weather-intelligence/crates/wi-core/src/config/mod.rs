//! # Configuration Management
//!
//! Centralized configuration using Figment for multi-source config loading.

use crate::Result;
use figment::{
    providers::{Env, Format, Json, Toml, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the weather intelligence engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WeatherIntelligenceConfig {
    pub general: GeneralConfig,
    pub storage: StorageConfig,
    pub sync: SyncConfig,
    pub ml: MLConfig,
    pub api: APIConfig,
    pub sources: SourcesConfig,
    pub logging: LoggingConfig,
    pub verification: VerificationConfig,
}

impl WeatherIntelligenceConfig {
    /// Load configuration from file and environment
    pub fn load(config_path: Option<PathBuf>) -> Result<Self> {
        let mut figment = Figment::new()
            .merge(Toml::file("config.toml"))
            .merge(Yaml::file("config.yaml"))
            .merge(Json::file("config.json"))
            .merge(Env::prefixed("WI_").global())
            .merge(Env::prefixed("WEATHER_").global())
            .merge(Env::prefixed("AETHERFORGE_").global());

        if let Some(path) = config_path {
            figment = figment.merge(Toml::file(path));
        }

        figment.extract().map_err(|e| {
            crate::WIError::config_with_source("Failed to load configuration", anyhow::anyhow!(e))
        })
    }

    /// Save configuration to file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            crate::WIError::serialization_with_source(
                "Failed to serialize config",
                anyhow::anyhow!(e),
            )
        })?;
        std::fs::write(path, content).map_err(|e| {
            crate::WIError::storage_with_source("Failed to write config file", anyhow::anyhow!(e))
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub version: String,
    pub environment: Environment,
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("/var/weather-intelligence"))
            .join("weather-intelligence");

        Self {
            version: "0.1.0".to_string(),
            environment: Environment::Development,
            data_dir: data_dir.clone(),
            model_dir: data_dir.join("models"),
            cache_dir: data_dir.join("cache"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Development,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub base_path: PathBuf,
    pub compression_enabled: bool,
    pub compression_level: i32,
    pub parquet_enabled: bool,
    pub postgres_url: Option<String>,
    pub redis_url: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/var/weather-intelligence/data"),
            compression_enabled: true,
            compression_level: 3,
            parquet_enabled: true,
            postgres_url: None,
            redis_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub parallel_workers: usize,
    pub batch_size: usize,
    pub timeout_secs: u64,
    pub retry_attempts: u8,
    pub retry_delay_ms: u64,
    pub checkpoint_interval: usize,
    pub enable_validation: bool,
    pub enable_gap_filling: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            parallel_workers: 8,
            batch_size: 1000,
            timeout_secs: 300,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            checkpoint_interval: 100,
            enable_validation: true,
            enable_gap_filling: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub model_path: PathBuf,
    pub model_format: ModelFormat,
    pub device: DeviceType,
    pub batch_size: usize,
    pub max_concurrent_batches: usize,
    pub cache_enabled: bool,
    pub cache_ttl_secs: u64,
    pub confidence_threshold: f32,
    pub enable_quantization: bool,
    pub quantization_precision: QuantizationPrecision,
    pub graphcast_weights_path: Option<PathBuf>,
    pub graphcast_onnx_path: Option<PathBuf>,
    pub downscaler_onnx_path: Option<PathBuf>,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("/var/weather-intelligence/models/graphcast.bin"),
            model_format: ModelFormat::BurnNative,
            device: DeviceType::Cpu,
            batch_size: 512,
            max_concurrent_batches: 4,
            cache_enabled: true,
            cache_ttl_secs: 3600,
            confidence_threshold: 0.5,
            enable_quantization: false,
            quantization_precision: QuantizationPrecision::FP16,
            graphcast_weights_path: None,
            graphcast_onnx_path: None,
            downscaler_onnx_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    BurnNative,
    Onnx,
    PyTorch,
    TensorRT,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cpu,
    Cuda(u32),
    Metal,
    Wgpu(u32),
}

impl DeviceType {
    pub fn default_device() -> Self {
        Self::Cpu
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Wgpu(_))
    }

    pub fn device_id(&self) -> u32 {
        match self {
            Self::Cuda(id) | Self::Wgpu(id) => *id,
            _ => 0,
        }
    }

    pub fn available_gpu_devices() -> Vec<Self> {
        // GPU device detection done at wi-model layer where burn is available.
        // Default: return CPU only; wi-model will detect actual GPUs via burn.
        vec![Self::Cpu]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum QuantizationPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub cors_enabled: bool,
    pub rate_limit_rps: u32,
    pub request_timeout_secs: u64,
    pub max_request_size_mb: usize,
    pub enable_grpc: bool,
    pub grpc_port: u16,
}

impl Default for APIConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: 4,
            cors_enabled: true,
            rate_limit_rps: 100,
            request_timeout_secs: 30,
            max_request_size_mb: 10,
            enable_grpc: true,
            grpc_port: 8081,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcesConfig {
    pub open_meteo: SourceConfig,
    pub noaa: SourceConfig,
    pub era5: SourceConfig,
    pub gfs: SourceConfig,
    pub bmkg: SourceConfig,
    pub himawari: SourceConfig,
    pub weather_next: SourceConfig,
}

impl Default for SourcesConfig {
    fn default() -> Self {
        Self {
            open_meteo: SourceConfig {
                enabled: true,
                base_url: "https://api.open-meteo.com".to_string(),
                api_key: None,
                rate_limit: 10.0,
            },
            noaa: SourceConfig {
                enabled: false,
                base_url: "https://www.ncdc.noaa.gov/cdo-web/api/v2".to_string(),
                api_key: None,
                rate_limit: 5.0,
            },
            era5: SourceConfig {
                enabled: false,
                base_url: "".to_string(),
                api_key: None,
                rate_limit: 1.0,
            },
            gfs: SourceConfig {
                enabled: false,
                base_url: "".to_string(),
                api_key: None,
                rate_limit: 1.0,
            },
            bmkg: SourceConfig {
                enabled: false,
                base_url: "https://data.bmkg.go.id".to_string(),
                api_key: None,
                rate_limit: 1.0,
            },
            himawari: SourceConfig {
                enabled: false,
                base_url: "".to_string(),
                api_key: None,
                rate_limit: 1.0,
            },
            weather_next: SourceConfig {
                enabled: false,
                base_url: "".to_string(),
                api_key: None,
                rate_limit: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: Option<String>,
    pub rate_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub file_enabled: bool,
    pub file_path: Option<PathBuf>,
    pub json_output: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Json,
            file_enabled: true,
            file_path: None,
            json_output: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub enabled: bool,
    pub auto_run: bool,
    pub schedule: String, // cron expression
    pub baseline_sources: Vec<String>,
    pub metrics: Vec<String>,
    pub temporal_split: bool,
    pub spatial_split: bool,
    pub output_path: PathBuf,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_run: true,
            schedule: "0 6 * * *".to_string(), // Daily at 6 AM
            baseline_sources: vec!["open_meteo".to_string(), "bmkg".to_string()],
            metrics: vec![
                "mae".to_string(),
                "rmse".to_string(),
                "brier".to_string(),
                "crps".to_string(),
            ],
            temporal_split: true,
            spatial_split: true,
            output_path: PathBuf::from("/var/weather-intelligence/verification"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WeatherIntelligenceConfig::default();
        assert_eq!(config.general.environment, Environment::Development);
        assert_eq!(config.api.port, 8080);
        assert!(config.sources.open_meteo.enabled);
    }
}
