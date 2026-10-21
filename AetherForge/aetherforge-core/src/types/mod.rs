//! # Core Types for Weather Intelligence
//!
//! This module defines the fundamental data types used throughout AetherForge,
//! including weather observations, forecasts, geospatial primitives, and ML-related types.

use chrono::{DateTime, NaiveDate, Utc};
use geo::{Coord, Point};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bounding box type for spatial queries (min_lon, min_lat, max_lon, max_lat)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl Bbox {
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Self {
        Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        }
    }

    pub fn from_coords(c1: &Coord, c2: &Coord) -> Self {
        Self {
            min_lon: c1.x.min(c2.x),
            min_lat: c1.y.min(c2.y),
            max_lon: c1.x.max(c2.x),
            max_lat: c1.y.max(c2.y),
        }
    }
}

/// Geographic coordinate with optional altitude
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoCoord {
    pub lat: f64,
    pub lon: f64,
    pub altitude_m: Option<f64>,
}

impl GeoCoord {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self {
            lat,
            lon,
            altitude_m: None,
        }
    }

    pub fn with_altitude(lat: f64, lon: f64, altitude_m: f64) -> Self {
        Self {
            lat,
            lon,
            altitude_m: Some(altitude_m),
        }
    }

    pub fn to_point(&self) -> Point {
        Point::new(self.lon, self.lat)
    }

    pub fn to_geo_hash(&self, precision: u8) -> String {
        let latlng = match h3o::LatLng::new(self.lat, self.lon) {
            Ok(ll) => ll,
            Err(_) => return String::new(),
        };
        let resolution = match h3o::Resolution::try_from(precision) {
            Ok(r) => r,
            Err(_) => return String::new(),
        };
        let cell = latlng.to_cell(resolution);
        cell.to_string()
    }
}

/// Weather variables supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum WeatherVariable {
    Temperature = 0,
    RelativeHumidity = 1,
    Precipitation = 2,
    WindSpeed = 3,
    WindDirection = 4,
    WindU = 5, // U component (east-west)
    WindV = 6, // V component (north-south)
    Pressure = 7,
    SurfacePressure = 8,
    CloudCover = 9,
    DewPoint = 10,
    Visibility = 11,
    SolarRadiation = 12,
    UVIndex = 13,
    // Vertical levels
    GeopotentialHeight = 14,
    SpecificHumidity = 15,
    VerticalVelocity = 16,
}

impl WeatherVariable {
    pub fn units(&self) -> &'static str {
        match self {
            Self::Temperature => "K",
            Self::RelativeHumidity => "%",
            Self::Precipitation => "mm",
            Self::WindSpeed | Self::WindU | Self::WindV => "m/s",
            Self::WindDirection => "degrees",
            Self::Pressure | Self::SurfacePressure => "Pa",
            Self::CloudCover => "%",
            Self::DewPoint => "K",
            Self::Visibility => "m",
            Self::SolarRadiation => "W/m²",
            Self::UVIndex => "index",
            Self::GeopotentialHeight => "m",
            Self::SpecificHumidity => "kg/kg",
            Self::VerticalVelocity => "Pa/s",
        }
    }

    pub fn all() -> Vec<WeatherVariable> {
        vec![
            Self::Temperature,
            Self::RelativeHumidity,
            Self::Precipitation,
            Self::WindSpeed,
            Self::WindDirection,
            Self::WindU,
            Self::WindV,
            Self::Pressure,
            Self::SurfacePressure,
            Self::CloudCover,
            Self::DewPoint,
            Self::Visibility,
            Self::SolarRadiation,
            Self::UVIndex,
            Self::GeopotentialHeight,
            Self::SpecificHumidity,
            Self::VerticalVelocity,
        ]
    }
}

/// Atmospheric vertical level types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerticalLevel {
    Surface,
    PressureLevel(u16),
    HeightLevel(i32),
    ModelLevel(u8),
}

impl std::hash::Hash for VerticalLevel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Surface => 0u8.hash(state),
            Self::PressureLevel(p) => (1u8, p).hash(state),
            Self::HeightLevel(h) => (2u8, h).hash(state),
            Self::ModelLevel(m) => (3u8, m).hash(state),
        }
    }
}

impl VerticalLevel {
    pub fn pressure_hpa(&self) -> Option<f64> {
        match self {
            Self::PressureLevel(hpa) => Some(*hpa as f64),
            Self::Surface => Some(1013.25),
            _ => None,
        }
    }
}

/// Weather station information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherStation {
    pub id: String,
    pub name: String,
    pub coord: GeoCoord,
    pub elevation_m: f32,
    pub country_code: String,
    pub timezone: String,
    pub data_sources: Vec<DataSource>,
    pub record_started: Option<NaiveDate>,
    pub metadata: HashMap<String, String>,
}

/// Weather observation at a specific time and location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherObservation {
    pub station_id: Option<String>,
    pub coord: GeoCoord,
    pub time: DateTime<Utc>,
    pub variables: HashMap<WeatherVariable, ObservedValue>,
    pub quality_flag: QualityFlag,
    pub source: DataSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservedValue {
    pub value: f64,
    pub is_valid: bool,
    pub confidence: f32,
}

impl ObservedValue {
    pub fn new(value: f64, confidence: f32) -> Self {
        Self {
            value,
            is_valid: true,
            confidence,
        }
    }

    pub fn missing() -> Self {
        Self {
            value: f64::NAN,
            is_valid: false,
            confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityFlag {
    Valid,
    Suspect,
    Erroneous,
    Missing,
}

/// Data source identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataSource {
    OpenMeteo = 0,
    OpenWeatherMap = 1,
    NOAA = 2,
    ERA5 = 3,
    GFS = 4,
    ECMWF = 5,
    Meteomatics = 6,
    VisualCrossing = 7,
    DWD = 8,
    MeteoFrance = 9,
    UKMetOffice = 10,
    JMA = 11,
    CMA = 12,
    Synop = 13,
    Metar = 14,
    Satelite = 15,
    Radar = 16,
    UserProvided = 17,
}

impl DataSource {
    pub fn reliability_score(&self) -> f32 {
        match self {
            Self::ERA5 => 0.98,
            Self::ECMWF => 0.97,
            Self::GFS => 0.93,
            Self::NOAA => 0.95,
            Self::DWD => 0.94,
            Self::OpenMeteo => 0.88,
            Self::Meteomatics => 0.92,
            Self::MeteoFrance => 0.91,
            Self::UKMetOffice => 0.90,
            Self::JMA => 0.91,
            Self::CMA => 0.89,
            Self::OpenWeatherMap => 0.82,
            Self::VisualCrossing => 0.80,
            Self::Synop => 0.85,
            Self::Metar => 0.86,
            Self::Satelite => 0.87,
            Self::Radar => 0.83,
            Self::UserProvided => 0.50,
        }
    }
}

/// Grid cell for spatial data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCell {
    pub h3_index: u64,
    pub resolution: u8,
    pub bbox: Bbox,
    pub center: GeoCoord,
}

/// Weather forecast at a specific location and time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherForecast {
    pub coord: GeoCoord,
    pub init_time: DateTime<Utc>,
    pub valid_time: DateTime<Utc>,
    pub lead_seconds: u32, // Supports sub-minute granularity (30s, 1m, 5m, etc.)
    pub variables: HashMap<(WeatherVariable, VerticalLevel), f64>,
    pub ensemble_members: Option<u8>,
    pub model_id: String,
    pub probability: Option<ForecastProbability>,
}

impl WeatherForecast {
    pub fn lead_hours(&self) -> f32 {
        self.lead_seconds as f32 / 3600.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastProbability {
    pub threshold: f64,
    pub probability: f32,
}

/// Spatiotemporal data cube for ML processing
#[derive(Debug, Clone)]
pub struct DataCube {
    pub spatial_resolution: u8,
    pub temporal_resolution_seconds: u32, // Supports 30s, 60s, 300s, etc.
    pub vertical_levels: Vec<VerticalLevel>,
    pub variables: Vec<WeatherVariable>,
    pub bbox: Bbox,
    pub data: ndarray::Array4<f32>,
    pub timestamps: Vec<DateTime<Utc>>,
}

impl DataCube {
    pub fn new(
        spatial_resolution: u8,
        temporal_resolution_seconds: u32,
        vertical_levels: Vec<VerticalLevel>,
        variables: Vec<WeatherVariable>,
        bbox: Bbox,
        spatial_grid: (usize, usize),
        temporal_steps: usize,
    ) -> Self {
        let levels = vertical_levels.len().max(1);
        let vars = variables.len();
        let (lat_grid, lon_grid) = spatial_grid;

        let shape = (temporal_steps, levels, vars, lat_grid * lon_grid);
        let data = ndarray::Array4::zeros(shape);

        Self {
            spatial_resolution,
            temporal_resolution_seconds,
            vertical_levels,
            variables,
            bbox,
            data,
            timestamps: Vec::with_capacity(temporal_steps),
        }
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        let (t, l, v, s) = self.data.dim().clone();
        (t, l, v, s)
    }
}

/// Weather reanalysis data (historical)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReanalysisData {
    pub coord: GeoCoord,
    pub time: DateTime<Utc>,
    pub level: VerticalLevel,
    pub variables: HashMap<WeatherVariable, f64>,
    pub source: DataSource,
    pub analysis_type: AnalysisType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisType {
    Analysis,
    Forecast,
    Anomaly,
    Climatology,
}

/// Climate zone classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClimateZone {
    Tropical,
    Arid,
    Temperate,
    Continental,
    Polar,
    Oceanic,
    Mediterranean,
    Subtropical,
}

impl ClimateZone {
    pub fn from_lat_lon(lat: f64, lon: f64) -> Self {
        match lat.abs() {
            lat if lat < 23.5 => Self::Tropical,
            lat if lat >= 23.5 && lat < 35.0 => {
                if lon < -100.0 || lon > 60.0 {
                    Self::Arid
                } else {
                    Self::Subtropical
                }
            }
            lat if lat >= 35.0 && lat < 55.0 => Self::Temperate,
            lat if lat >= 55.0 && lat < 70.0 => {
                if lon < -100.0 || lon > 60.0 {
                    Self::Continental
                } else {
                    Self::Oceanic
                }
            }
            lat if lat >= 70.0 => Self::Polar,
            _ => Self::Temperate,
        }
    }
}

/// Feature vector for ML model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub coord: GeoCoord,
    pub time: DateTime<Utc>,
    pub lead_seconds: u32, // Supports sub-hourly granularity
    pub features: Vec<f32>,
    pub feature_names: Vec<String>,
}

impl FeatureVector {
    pub fn new(
        coord: GeoCoord,
        time: DateTime<Utc>,
        lead_seconds: u32,
        num_features: usize,
    ) -> Self {
        Self {
            coord,
            time,
            lead_seconds,
            features: vec![0.0; num_features],
            feature_names: Vec::with_capacity(num_features),
        }
    }

    pub fn lead_hours(&self) -> f32 {
        self.lead_seconds as f32 / 3600.0
    }

    pub fn set_feature(&mut self, name: &str, value: f32) {
        if let Some(idx) = self.feature_names.iter().position(|n| n == name) {
            self.features[idx] = value;
        } else {
            self.feature_names.push(name.to_string());
            self.features.push(value);
        }
    }
}

/// Model output prediction
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    pub coord: GeoCoord,
    pub valid_time: DateTime<Utc>,
    pub lead_seconds: u32, // Supports sub-hourly granularity
    pub variables: HashMap<WeatherVariable, PredictionValue>,
    pub confidence: f32,
    pub ensemble_std: Option<HashMap<WeatherVariable, f32>>,
}

impl ModelPrediction {
    pub fn lead_hours(&self) -> f32 {
        self.lead_seconds as f32 / 3600.0
    }
}

#[derive(Debug, Clone)]
pub struct PredictionValue {
    pub mean: f32,
    pub median: f32,
    pub percentile_05: f32,
    pub percentile_95: f32,
    pub std: f32,
}

/// Training batch for ML models
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub input: ndarray::Array4<f32>,
    pub target: ndarray::Array4<f32>,
    pub time_idx: usize,
    pub spatial_mask: ndarray::Array2<bool>,
}

/// Model configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceModelConfig {
    pub architecture: ModelArchitecture,
    pub input_resolution: u8,
    pub output_resolution: u8,
    pub lead_times_seconds: Vec<u32>, // Supports 30s, 60s, 300s, etc.
    pub vertical_levels: Vec<VerticalLevel>,
    pub ensemble_size: usize,
    pub quantization: QuantizationConfig,
}

impl InferenceModelConfig {
    pub fn lead_times_hours(&self) -> Vec<f32> {
        self.lead_times_seconds
            .iter()
            .map(|s| *s as f32 / 3600.0)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    GraphCast,
    FourCastNet,
    PanguWeather,
    NeuralGCM,
    ClimaX,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub precision: QuantizationPrecision,
    pub calibration_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

/// Spatial indexing result
#[derive(Debug, Clone)]
pub struct SpatialIndex {
    pub h3_cells: Vec<u64>,
    pub resolution: u8,
    pub grid_shape: (usize, usize),
}

impl SpatialIndex {
    pub fn from_bbox(bbox: Bbox, resolution: u8) -> Self {
        let h3_resolution = match h3o::Resolution::try_from(resolution) {
            Ok(r) => r,
            Err(_) => return Self::default(),
        };

        let min_latlng = match h3o::LatLng::new(bbox.min_lat, bbox.min_lon) {
            Ok(ll) => ll,
            Err(_) => return Self::default(),
        };
        let max_latlng = match h3o::LatLng::new(bbox.max_lat, bbox.max_lon) {
            Ok(ll) => ll,
            Err(_) => return Self::default(),
        };

        let min_cell = min_latlng.to_cell(h3_resolution);
        let max_cell = max_latlng.to_cell(h3_resolution);

        let mut h3_cells = Vec::new();
        h3_cells.push(u64::from(min_cell));
        h3_cells.push(u64::from(max_cell));

        Self {
            h3_cells,
            resolution,
            grid_shape: (2, 1),
        }
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self {
            h3_cells: Vec::new(),
            resolution: 0,
            grid_shape: (0, 0),
        }
    }
}

/// Time series for temporal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub coord: GeoCoord,
    pub variable: WeatherVariable,
    pub level: VerticalLevel,
    pub values: Vec<TimeSeriesValue>,
    pub sampling_period_seconds: u32, // Supports 30s, 60s, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesValue {
    pub time: DateTime<Utc>,
    pub value: f64,
    pub quality: QualityFlag,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_coord() {
        let coord = GeoCoord::new(52.52, 13.405);
        assert_eq!(coord.lat, 52.52);
        assert_eq!(coord.lon, 13.405);
    }

    #[test]
    fn test_climate_zone() {
        assert_eq!(ClimateZone::from_lat_lon(0.0, 0.0), ClimateZone::Tropical);
        assert_eq!(ClimateZone::from_lat_lon(55.0, 10.0), ClimateZone::Oceanic);
        assert_eq!(ClimateZone::from_lat_lon(75.0, 0.0), ClimateZone::Polar);
    }
}
