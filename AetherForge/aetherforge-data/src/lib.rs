//! # AetherForge Data Module
//! 
//! Data ingestion, synchronization, preprocessing, and storage for weather data.
//! 
//! ## Key Features
//! 
//! - **Multi-Source Support**: Open-Meteo, NOAA, ERA5, and more
//! - **Incremental Sync**: Checkpoint-based resume capability
//! - **Data Validation**: Physics-based validation and gap filling
//! - **Parquet Storage**: Hive-style partitioned storage
//! 
//! ## Usage
//! 
//! ```rust,no_run
//! use aetherforge_data::{SyncEngine, SyncEngineConfig};
//! use chrono::{Duration, Utc};
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = SyncEngineConfig::default();
//!     let engine = SyncEngine::with_config(config);
//!     
//!     let end = Utc::now();
//!     let start = end - Duration::days(7);
//!     
//!     let stats = engine.sync_all(start, end).await?;
//!     println!("Synced {} records", stats.records_written);
//!     
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![allow(dead_code)]

pub mod sources;
pub mod adapters;
pub mod sync;
pub mod storage;
pub mod preprocessing;
pub mod validation;
pub mod scheduler;
pub mod checkpoint;

pub use sources::{WeatherSource, RegionBbox, SourceRegistry, WeatherStream, ForecastStream};
pub use adapters::{OpenMeteoAdapter, NOAAAdapter, create_default_registry, create_registry_with_keys};
pub use sync::{SyncEngine, SyncEngineConfig, SyncStats};
pub use storage::{StorageManager, StorageConfig, StorageStats};
pub use preprocessing::{
    DataNormalizer, MissingValueHandler, TemporalSlotter, SpatialSlotter,
    FeatureEngine, AdaptiveQuantizer, ClimatologyStore, ClimatologyStats,
};
pub use validation::{WeatherValidator, GapFiller, ValidationResult, ValidationIssue};
pub use scheduler::{SyncScheduler, SchedulerConfig, SchedulerStats};
pub use checkpoint::{CheckpointStore, SyncCheckpoint};