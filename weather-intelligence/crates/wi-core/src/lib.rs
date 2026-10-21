//! # Weather Intelligence Core
//!
//! Shared types, configuration, error handling, and traits for the
//! unified weather intelligence engine.

#![forbid(unsafe_code)]
#![warn(unreachable_pub, rust_2018_idioms)]
#![allow(missing_docs)]

pub mod config;
pub mod error;
pub mod traits;
pub mod types;
pub mod utils;

pub use config::{
    APIConfig, DeviceType, Environment, GeneralConfig, LoggingConfig, MLConfig, ModelFormat,
    QuantizationPrecision, SourceConfig, SourcesConfig, StorageConfig, SyncConfig,
    VerificationConfig, WeatherIntelligenceConfig,
};
pub use error::{Result, WIError};
pub use traits::*;
pub use types::{
    AnalysisType, AreaPredictRequest, BatchAreaResult, BatchGridPredictOutput,
    BatchGridPredictRequest, Bbox, Celsius, ClimateZone, DataCube, DataSource, Degrees,
    DependencyGroup, EnsembleMember, FeatureVector, ForecastProbability, FusionInput, GeoCoord,
    GridCell, GridCoord, GridPredictRequest, Hpa, Humidity, InferenceModelConfig, Kelvin, MeshEdge,
    Millimeters, ModelArchitecture, ModelPrediction, Mps, ObservedValue, PredictionValue,
    Probability, QualityFlag, QuantizationConfig, ReanalysisData, SourceId, SourceMetadata,
    SourceRegistry, SourceRole, SpatialIndex, SpatialTileConfig, TimeSeries, TimeSeriesValue,
    TrainingBatch, VerticalLevel, WeatherForecast, WeatherObservation, WeatherStation,
    WeatherVariable,
};

pub use chrono::{DateTime, Duration as ChronoDuration, Utc};
pub use geo::Coord;
pub use h3o::CellIndex;
pub use sha2::{Digest, Sha256};
