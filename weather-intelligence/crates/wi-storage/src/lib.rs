//! # Weather Intelligence Storage
//!
//! Multi-backend storage for weather data:
//! - Custom ATD binary format (high compression)
//! - Parquet/Arrow (analytics, training)
//! - PostgreSQL + PostGIS (stations, verification)
//! - Redis (caching)

#![forbid(unsafe_code)]

pub mod atd;
pub mod parquet;
pub mod postgres;
pub mod redis;

pub use atd::{AtdConfig, AtdStats, AtdStorage};
pub use parquet::{ParquetConfig, ParquetStorage};
pub use postgres::{PostgresConfig, PostgresStorage};
pub use redis::{RedisCache, RedisConfig};

use chrono::{DateTime, Utc};
use wi_core::{Bbox, Result, WeatherForecast, WeatherObservation};

/// Unified storage trait
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize>;
    async fn read_observations(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherObservation>>;
    async fn write_forecasts(&self, forecasts: Vec<WeatherForecast>) -> Result<usize>;
    async fn read_forecasts(
        &self,
        model_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
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

use serde::{Deserialize, Serialize};
