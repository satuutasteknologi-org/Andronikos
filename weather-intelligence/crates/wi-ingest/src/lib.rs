//! # Weather Intelligence Ingestion
//!
//! Data ingestion, synchronization, preprocessing, and storage for weather data.

#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![allow(dead_code)]

pub mod adapters;
pub mod bias_correction;
pub mod checkpoint;
pub mod orchestrator;
pub mod parsers;
pub mod preprocessing;
pub mod scheduler;
pub mod sources;
pub mod sync;
pub mod validation;

pub use adapters::{
    create_default_registry, create_registry_with_keys, BMKGAdapter, HimawariAdapter, NOAAAdapter,
    OpenMeteoAdapter, WeatherNextAdapter,
};
pub use bias_correction::{
    AdditiveBiasCorrector, AdditiveCorrection, BiasCorrectionEngine, BiasKey, BiasUpdateConfig,
    BiasVariableGroup, CorrectedValue, RainBiasCorrector, RainCorrection,
};
pub use checkpoint::{CheckpointStore, SyncCheckpoint};
pub use orchestrator::{
    IngestOrchestrator, IngestStats, OrchestratorConfig, LABEL_INFORMASI_RISIKO,
};
pub use preprocessing::{AdaptiveQuantizer, ClimatologyStore, PreprocessingConfig, Preprocessor};
pub use scheduler::{SchedulerConfig, SchedulerStats, SyncScheduler};
pub use sources::{ForecastStream, RegionBbox, SourceRegistry, WeatherSource, WeatherStream};
pub use sync::{SyncEngine, SyncEngineConfig, SyncStats};
pub use validation::{GapFiller, ValidationIssue, ValidationResult, WeatherValidator};

/// Convenience type alias for SyncEngine with ParquetStorage
pub type ParquetSyncEngine = SyncEngine<wi_storage::ParquetStorage>;
