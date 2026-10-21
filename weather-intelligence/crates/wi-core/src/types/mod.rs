//! #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]Core Types for Weather Intelligence
//!
//! This module defines the fundamental data types used throughout the engine,
//! including weather observations, forecasts, geospatial primitives, and ML-related types.

use chrono::{DateTime, NaiveDate, Utc};
use geo::{Coord, Point};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use crate::error::WIError;

/// Physical unit newtypes for type safety
/// Temperature in Celsius
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Celsius(pub f32);

impl Celsius {
    pub fn new(value: f32) -> Self {
        Self(value)
    }
    pub fn from_kelvin(k: f32) -> Self {
        Self(k - 273.15)
    }
    pub fn to_kelvin(self) -> f32 {
        self.0 + 273.15
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Celsius {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}°C", self.0)
    }
}

impl Add for Celsius {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Celsius {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Celsius {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f32> for Celsius {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

/// Temperature in Kelvin
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Kelvin(pub f32);

impl Kelvin {
    pub fn new(value: f32) -> Self {
        Self(value)
    }
    pub fn from_celsius(c: Celsius) -> Self {
        Self(c.0 + 273.15)
    }
    pub fn to_celsius(self) -> Celsius {
        Celsius(self.0 - 273.15)
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Kelvin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} K", self.0)
    }
}

/// Wind speed in meters per second
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Mps(pub f32);

impl Mps {
    pub fn new(value: f32) -> Self {
        Self(value.max(0.0))
    }
    pub fn from_kmh(kmh: f32) -> Self {
        Self(kmh / 3.6)
    }
    pub fn to_kmh(self) -> f32 {
        self.0 * 3.6
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Mps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} m/s", self.0)
    }
}

/// Wind direction in degrees (0-360)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Degrees(pub f32);

impl Degrees {
    pub fn new(value: f32) -> Self {
        Self(value.rem_euclid(360.0))
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Degrees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}°", self.0)
    }
}

/// Pressure in hectopascals (hPa)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Hpa(pub f32);

impl Hpa {
    pub fn new(value: f32) -> Self {
        Self(value)
    }
    pub fn from_pa(pa: f32) -> Self {
        Self(pa / 100.0)
    }
    pub fn to_pa(self) -> f32 {
        self.0 * 100.0
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Hpa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} hPa", self.0)
    }
}

/// Relative humidity percentage (0-100)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Humidity(pub f32);

impl Humidity {
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 100.0))
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Humidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}%", self.0)
    }
}

/// Precipitation in millimeters
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Millimeters(pub f32);

impl Millimeters {
    pub fn new(value: f32) -> Self {
        Self(value.max(0.0))
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Millimeters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} mm", self.0)
    }
}

/// Probability (0.0 - 1.0)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Probability(pub f32);

impl Probability {
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    pub fn from_percent(pct: f32) -> Self {
        Self(pct / 100.0)
    }
    pub fn to_percent(self) -> f32 {
        self.0 * 100.0
    }
    pub fn value(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}%", self.0 * 100.0)
    }
}

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

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat && lon >= self.min_lon && lon <= self.max_lon
    }

    pub fn width_deg(&self) -> f64 {
        self.max_lon - self.min_lon
    }
    pub fn height_deg(&self) -> f64 {
        self.max_lat - self.min_lat
    }

    /// Indonesia bounding box
    pub fn indonesia() -> Self {
        Self::new(95.0, -11.0, 141.1, 6.1)
    }

    /// Global bounding box
    pub fn global() -> Self {
        Self::new(-180.0, -90.0, 180.0, 90.0)
    }
}

/// Request parameters for grid prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPredictRequest {
    pub bbox: Bbox,
    pub resolution: u8,
    pub lead_seconds: u32,
    pub variables: Vec<WeatherVariable>,
    /// Maksimum jumlah cell yang akan diproses (hindari 38,9M cell Indonesia penuh)
    pub max_cells: Option<usize>,
}

impl GridPredictRequest {
    pub fn new(
        bbox: Bbox,
        resolution: u8,
        lead_seconds: u32,
        variables: Vec<WeatherVariable>,
    ) -> Self {
        Self {
            bbox,
            resolution,
            lead_seconds,
            variables,
            max_cells: None,
        }
    }

    /// Batasi jumlah cell untuk pilot area (default 100 cell untuk target 1 menit).
    pub fn with_max_cells(mut self, max: usize) -> Self {
        self.max_cells = Some(max);
        self
    }

    /// Pilot area: Jakarta dan sekitarnya (batas konservatif untuk target 1 menit).
    /// Lat: -6.5 to -6.0, Lon: 106.5 to 107.0
    pub fn pilot_jakarta() -> Bbox {
        Bbox::new(106.5, -6.5, 107.0, -6.0)
    }

    /// Periksa apakah bbox berada dalam wilayah pilot yang diizinkan.
    pub fn is_within_pilot(&self) -> bool {
        let pilot = Self::pilot_jakarta();
        self.bbox.min_lon >= pilot.min_lon
            && self.bbox.max_lon <= pilot.max_lon
            && self.bbox.min_lat >= pilot.min_lat
            && self.bbox.max_lat <= pilot.max_lat
    }
}

/// Konfigurasi tiling spasial untuk area besar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialTileConfig {
    /// Ukuran tile dalam jumlah cell (default 500x500 = 250.000 cell per tile)
    pub tile_cells: usize,
    /// Overlap antar tile dalam cell (untuk konsistensi GNN edge di batas tile)
    pub overlap_cells: usize,
    /// Maksimum cell total yang boleh diproses dalam satu request
    pub max_total_cells: usize,
}

impl Default for SpatialTileConfig {
    fn default() -> Self {
        Self {
            tile_cells: 250_000,
            overlap_cells: 2,
            max_total_cells: 10_000_000,
        }
    }
}

/// Request untuk prediksi grid multi-area dengan batch inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchGridPredictRequest {
    /// Daftar area yang akan diprediksi
    pub areas: Vec<AreaPredictRequest>,
    /// Konfigurasi tiling untuk area besar
    pub tile_config: Option<SpatialTileConfig>,
    /// Perangkat komputasi (default CPU)
    pub device_type: Option<crate::config::DeviceType>,
    /// Maksimum total cell semua area
    pub max_total_cells: Option<usize>,
}

/// Satu area dalam batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaPredictRequest {
    pub area_id: String,
    pub bbox: Bbox,
    pub resolution: u8,
    pub lead_seconds: u32,
    pub variables: Vec<WeatherVariable>,
    pub max_cells: Option<usize>,
}

impl BatchGridPredictRequest {
    pub fn new(areas: Vec<AreaPredictRequest>) -> Self {
        Self {
            areas,
            tile_config: None,
            device_type: None,
            max_total_cells: None,
        }
    }

    pub fn with_device(mut self, device: crate::config::DeviceType) -> Self {
        self.device_type = Some(device);
        self
    }

    pub fn with_tile_config(mut self, config: SpatialTileConfig) -> Self {
        self.tile_config = Some(config);
        self
    }

    pub fn with_max_total_cells(mut self, max: usize) -> Self {
        self.max_total_cells = Some(max);
        self
    }

    /// Validasi total cell tidak melebihi batas.
    pub fn validate_total_cells(&self) -> Result<(), WIError> {
        let max = self.max_total_cells.unwrap_or(10_000_000);
        let mut total = 0usize;
        for area in &self.areas {
            let area_max = area.max_cells.unwrap_or(100_000);
            total += area_max;
        }
        if total > max {
            return Err(WIError::validation(format!(
                "Total cell melebihi batas maksimum: {} > {}",
                total, max
            )));
        }
        Ok(())
    }

    /// Hitung estimasi total cell dari semua area.
    pub fn estimated_total_cells(&self) -> usize {
        self.areas
            .iter()
            .map(|a| a.max_cells.unwrap_or(100_000))
            .sum()
    }
}

/// Hasil prediksi untuk satu cell (ringkasan untuk batch output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPrediction {
    pub lat: f64,
    pub lon: f64,
    pub cell_id: u64,
    pub predictions: Vec<(WeatherVariable, f32)>,
}

/// Hasil prediksi untuk satu area dalam batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAreaResult {
    pub area_id: String,
    pub predictions: Vec<CellPrediction>,
    pub cell_count: usize,
    pub inference_ms: u64,
}

/// Hasil batch prediction untuk banyak area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchGridPredictOutput {
    pub results: Vec<BatchAreaResult>,
    pub total_cells: usize,
    pub total_inference_ms: u64,
    pub device_used: crate::config::DeviceType,
}

impl BatchGridPredictOutput {
    pub fn new(device_used: crate::config::DeviceType) -> Self {
        Self {
            results: Vec::new(),
            total_cells: 0,
            total_inference_ms: 0,
            device_used,
        }
    }
}

/// Koordinat grid untuk pembangunan mesh edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridCoord {
    pub row: usize,
    pub col: usize,
}

/// Informasi edge untuk mesh GNN (source -> target).
#[derive(Debug, Clone)]
pub struct MeshEdge {
    pub source: usize,
    pub target: usize,
}
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

    pub fn distance_to(&self, other: &GeoCoord) -> f64 {
        // Haversine formula
        const R: f64 = 6371000.0; // Earth radius in meters
        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        R * c
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
    WindU = 5,
    WindV = 6,
    Pressure = 7,
    SurfacePressure = 8,
    CloudCover = 9,
    DewPoint = 10,
    Visibility = 11,
    SolarRadiation = 12,
    UVIndex = 13,
    GeopotentialHeight = 14,
    SpecificHumidity = 15,
    VerticalVelocity = 16,
    // Fire/hotspot variables (for peatland fire monitoring)
    FireRadiativePower = 17,
    FireBrightnessTemp = 18,
    FireBrightnessT31 = 19,
    FireConfidence = 20,
    FireCount = 21,
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
            Self::FireRadiativePower => "MW",
            Self::FireBrightnessTemp => "K",
            Self::FireBrightnessT31 => "K",
            Self::FireConfidence => "0-1",
            Self::FireCount => "count",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::RelativeHumidity => "Relative Humidity",
            Self::Precipitation => "Precipitation",
            Self::WindSpeed => "Wind Speed",
            Self::WindDirection => "Wind Direction",
            Self::WindU => "Wind U Component",
            Self::WindV => "Wind V Component",
            Self::Pressure => "Pressure",
            Self::SurfacePressure => "Surface Pressure",
            Self::CloudCover => "Cloud Cover",
            Self::DewPoint => "Dew Point",
            Self::Visibility => "Visibility",
            Self::SolarRadiation => "Solar Radiation",
            Self::UVIndex => "UV Index",
            Self::GeopotentialHeight => "Geopotential Height",
            Self::SpecificHumidity => "Specific Humidity",
            Self::VerticalVelocity => "Vertical Velocity",
            Self::FireRadiativePower => "Fire Radiative Power",
            Self::FireBrightnessTemp => "Fire Brightness Temperature",
            Self::FireBrightnessT31 => "Fire Brightness T31",
            Self::FireConfidence => "Fire Confidence",
            Self::FireCount => "Fire Count",
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
            Self::FireRadiativePower,
            Self::FireBrightnessTemp,
            Self::FireBrightnessT31,
            Self::FireConfidence,
            Self::FireCount,
        ]
    }

    pub fn surface_variables() -> Vec<WeatherVariable> {
        vec![
            Self::Temperature,
            Self::RelativeHumidity,
            Self::Precipitation,
            Self::WindSpeed,
            Self::WindDirection,
            Self::Pressure,
            Self::CloudCover,
            Self::DewPoint,
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

    pub fn standard_levels() -> Vec<VerticalLevel> {
        vec![
            Self::Surface,
            Self::PressureLevel(1000),
            Self::PressureLevel(925),
            Self::PressureLevel(850),
            Self::PressureLevel(700),
            Self::PressureLevel(600),
            Self::PressureLevel(500),
            Self::PressureLevel(400),
            Self::PressureLevel(300),
            Self::PressureLevel(250),
            Self::PressureLevel(200),
            Self::PressureLevel(150),
            Self::PressureLevel(100),
            Self::PressureLevel(50),
            Self::PressureLevel(10),
        ]
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
    BMKG = 18,
    Himawari = 19,
    WeatherNext = 20,
    FIRMS = 21,
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
            Self::BMKG => 0.95,
            Self::Himawari => 0.85,
            Self::WeatherNext => 0.90,
            Self::FIRMS => 0.92,
            Self::UserProvided => 0.50,
        }
    }

    pub fn requires_api_key(&self) -> bool {
        matches!(
            self,
            Self::NOAA
                | Self::ECMWF
                | Self::Meteomatics
                | Self::VisualCrossing
                | Self::WeatherNext
                | Self::FIRMS
        )
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
    pub lead_seconds: u32,
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
    pub temporal_resolution_seconds: u32,
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
        let (t, l, v, s) = self.data.dim();
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            lat if (23.5..35.0).contains(&lat) => {
                if lon < -100.0 || lon > 60.0 {
                    Self::Arid
                } else {
                    Self::Subtropical
                }
            }
            lat if (35.0..55.0).contains(&lat) => Self::Temperate,
            lat if (55.0..70.0).contains(&lat) => {
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

    pub fn temperature_bounds(&self) -> (f64, f64) {
        match self {
            Self::Polar => (180.0, 303.0),     // -93°C to 30°C
            Self::Tropical => (260.0, 330.0),  // -13°C to 57°C
            Self::Arid => (230.0, 340.0),      // -43°C to 67°C
            Self::Temperate => (200.0, 320.0), // -73°C to 47°C
            Self::Continental => (200.0, 320.0),
            Self::Oceanic => (200.0, 310.0),
            Self::Mediterranean => (220.0, 325.0),
            Self::Subtropical => (250.0, 330.0),
        }
    }
}

/// Feature vector for ML model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub coord: GeoCoord,
    pub time: DateTime<Utc>,
    pub lead_seconds: u32,
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

/// Model output prediction with quantiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub coord: GeoCoord,
    pub valid_time: DateTime<Utc>,
    pub lead_seconds: u32,
    pub variables: HashMap<WeatherVariable, PredictionValue>,
    pub confidence: f32,
    pub ensemble_std: Option<HashMap<WeatherVariable, f32>>,
}

impl ModelPrediction {
    pub fn lead_hours(&self) -> f32 {
        self.lead_seconds as f32 / 3600.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionValue {
    pub mean: f32,
    pub median: f32,
    pub percentile_05: f32,
    pub percentile_10: f32,
    pub percentile_25: f32,
    pub percentile_50: f32,
    pub percentile_75: f32,
    pub percentile_90: f32,
    pub percentile_95: f32,
    pub std: f32,
}

impl PredictionValue {
    pub fn from_quantiles(q10: f32, q50: f32, q90: f32, mean: f32, std: f32) -> Self {
        Self {
            mean,
            median: q50,
            std,
            percentile_05: q10 - 0.5 * (q90 - q10),
            percentile_10: q10,
            percentile_25: q10 + 0.5 * (q50 - q10),
            percentile_50: q50,
            percentile_75: q50 + 0.5 * (q90 - q50),
            percentile_90: q90,
            percentile_95: q90 + 0.5 * (q90 - q50),
        }
    }
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
    pub lead_times_seconds: Vec<u32>,
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
    DownscalerONNX,
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
#[derive(Debug, Clone, Default)]
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
        let h3_cells = vec![u64::from(min_cell), u64::from(max_cell)];
        Self {
            h3_cells,
            resolution,
            grid_shape: (2, 1),
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
    pub sampling_period_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesValue {
    pub time: DateTime<Utc>,
    pub value: f64,
    pub quality: QualityFlag,
}

// ============================================================================
// Source Registry & Fusion Types (Anti-Double Counting)
// ============================================================================

/// ID unik sumber data sesuai registri_variabel_v2_5.yaml
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum SourceId {
    /// NOAA GFS — dasar fisika deterministik
    MetGfs = 0,
    /// NOAA GEFS — dasar ensemble probabilistik
    MetGefs = 1,
    /// NOAA AIGFS — anggota AI tambahan (turunan GFS)
    MetAigfs = 2,
    /// NOAA AIGEFS (31 anggota) — anggota ensemble AI (turunan GEFS)
    MetAigefs = 3,
    /// NOAA HGEFS (62 anggota) — gabungan AIGEFS+GEFSv12, TIDAK independen
    MetHgefs = 4,
    /// BMKG prakiraan terbuka — anggota gabungan lokal
    MetBmkgFc = 5,
    /// BMKG peringatan dini (CAP) — pembanding, bukan input model
    MetBmkgNowcast = 6,
    /// GraphCast/GenCast — model AI sendiri, CC BY-NC-SA (diekspor komersial)
    MetGraphcast = 7,
    /// Open-Meteo — agregasi model lain, TIDAK independen, dikeluarkan dari fusion
    MetOpenMeteo = 8,
    /// ERA5 (Copernicus/ECMWF) — training + bias correction only, bukan real-time input
    RefEra5 = 9,
    /// BMKG pengamatan stasiun — verifikasi + koreksi bias
    ObsBmkgStn = 10,
    /// NOAA ISD — pengamatan stasiun global — verifikasi + koreksi bias
    ObsIsd = 11,
    /// GPM hujan satelit — modul spesifik (longsor/banjir)
    ObsGpm = 12,
    /// GSMaP hujan satelit — modul spesifik (longsor)
    ObsGsmap = 13,
    /// NASA FIRMS titik panas — asap
    FireFirms = 14,
    /// SiPongi+ titik panas — sensor sama dengan FIRMS, hitung sekali
    FireSipongi = 15,
    /// Hotspot BRIN tersaring — asap
    FireBrin = 16,
    /// Radar nowcasting — belum ada kemitraan
    NowcastRadar = 17,
    /// Sumber lain tidak terdaftar di atas
    Other(u16),
}

impl SourceId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MetGfs => "met.gfs",
            Self::MetGefs => "met.gefs",
            Self::MetAigfs => "met.aigfs",
            Self::MetAigefs => "met.aigefs",
            Self::MetHgefs => "met.hgefs",
            Self::MetBmkgFc => "met.bmkg_fc",
            Self::MetBmkgNowcast => "met.bmkg_nowcast",
            Self::MetGraphcast => "met.graphcast",
            Self::MetOpenMeteo => "met.openmeteo",
            Self::RefEra5 => "ref.era5",
            Self::ObsBmkgStn => "obs.bmkg_stn",
            Self::ObsIsd => "obs.isd",
            Self::ObsGpm => "obs.gpm",
            Self::ObsGsmap => "obs.gsmap",
            Self::FireFirms => "fire.firms",
            Self::FireSipongi => "fire.sipongi",
            Self::FireBrin => "fire.brin",
            Self::NowcastRadar => "nowcast.radar",
            Self::Other(_) => "source.other",
        }
    }
}

/// Peran sumber dalam alur penggabungan (fusion)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceRole {
    /// Dasar fisika deterministik (GFS, GEFS) — baseline weight tinggi
    BasePhysics,
    /// Dasar ensemble probabilistik (GEFS)
    BaseEnsemble,
    /// Anggota AI tambahan (AIGFS, AIGEFS) — bukan pengganti fisika
    AiMember,
    /// Anggota gabungan lokal (BMKG_fc)
    LocalMember,
    /// Pembanding / komparator, bukan input model (bmkg_nowcast, openmeteo)
    ComparatorOnly,
    /// Training + koreksi bias only, bukan real-time input (era5)
    TrainingOnly,
    /// Observasi untuk verifikasi + koreksi bias (bmkg_stn, isd)
    ObservationOnly,
    /// Diekspor dari fusion karena tidak independen atau lisensi komersial (graphcast, hgefs)
    Excluded,
    /// Titik panas / spesifik modul (firms, sipongi, brin, gpm, gsmap)
    ModuleSpecific,
}

/// Kelompok dependensi: sumber dalam kelompok yang sama berbagi komponen,
/// hanya satu anggota yang boleh menerima bobot non-nol dalam fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyGroup {
    /// Sumber independen, tidak berbagi komponen
    Independent,
    /// NOAA GEFS + AIGEFS + HGEFS — hgefs adalah gabungan, hanya satu yang boleh bobot
    NoaaGefsFamily,
    /// NASA FIRMS + SiPongi+ — sensor sama, hitung sekali
    FireHotspotFamily,
    /// Open-Meteo adalah agregasi NOAA/ECMWF/dll — dikeluarkan dari fusion
    OpenMeteoAggregator,
    /// ERA5 adalah reanalisis — hanya training/bias correction
    Era5Reanalysis,
    /// GraphCast bobot CC BY-NC-SA — diekspor komersial
    GraphCastNonCommercial,
}

/// Metadata sumber untuk registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub id: SourceId,
    pub role: SourceRole,
    pub dependency_group: DependencyGroup,
    pub reliability_score: f32,
    pub requires_written_permission: bool,
    pub commercial_allowed: bool,
    pub rate_limit_per_min: Option<u32>,
    pub attribution_text: &'static str,
}

impl SourceMetadata {
    pub fn is_fusion_eligible(&self) -> bool {
        matches!(
            self.role,
            SourceRole::BasePhysics
                | SourceRole::BaseEnsemble
                | SourceRole::AiMember
                | SourceRole::LocalMember
                | SourceRole::ModuleSpecific
        )
    }

    pub fn can_have_separate_weight(&self, other: &SourceMetadata) -> bool {
        // Sumber dalam dependency group yang sama (non-Independent) tidak boleh sama-sama memiliki bobot non-nol
        if self.dependency_group != DependencyGroup::Independent
            && self.dependency_group == other.dependency_group
        {
            return false;
        }
        true
    }
}

/// Registry sumber: melacak sumber yang aktif dan mencegah double-counting.
/// Tidak ada interior mutability yang bisa menyebabkan race condition —
/// semua operasi melalui immutable reference atauArc<tokio::sync::RwLock>.
#[derive(Debug, Clone, Default)]
pub struct SourceRegistry {
    sources: Vec<SourceMetadata>,
    selected: Vec<SourceId>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Daftarkan sumber metadata. Sumber dengan role Excluded tidak boleh dipilih untuk fusion.
    pub fn register(&mut self, metadata: SourceMetadata) {
        self.sources.push(metadata);
    }

    /// Pilih sumber untuk fusion. Mengembalikan error jika sumber tidak independen
    /// dengan yang sudah dipilih (double-counting).
    pub fn select(&mut self, source_id: SourceId) -> Result<(), WIError> {
        let meta = self
            .sources
            .iter()
            .find(|m| m.id == source_id)
            .ok_or_else(|| {
                WIError::validation(format!("Sumber tidak terdaftar: {:?}", source_id.as_str()))
            })?;

        if !meta.is_fusion_eligible() {
            return Err(WIError::validation(format!(
                "Sumber {} tidak eligible untuk fusion (role={:?})",
                source_id.as_str(),
                meta.role
            )));
        }

        // Cek double-counting dengan sumber yang sudah dipilih
        for &already in &self.selected {
            let other = self
                .sources
                .iter()
                .find(|m| m.id == already)
                .expect("registry internal consistency");
            if !meta.can_have_separate_weight(other) {
                return Err(WIError::validation(format!(
                    "Double-counting terdeteksi: {} dan {} share dependency group {:?}",
                    source_id.as_str(),
                    already.as_str(),
                    other.dependency_group
                )));
            }
        }

        self.selected.push(source_id);
        Ok(())
    }

    /// Daftar sumber yang sudah dipilih
    pub fn selected(&self) -> &[SourceId] {
        &self.selected
    }

    /// Ambil metadata berdasarkan ID
    pub fn metadata(&self, source_id: SourceId) -> Option<&SourceMetadata> {
        self.sources.iter().find(|m| m.id == source_id)
    }

    /// Jumlah sumber yang dipilih
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }
}

/// Anggota ensemble dengan sumber dan bobot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMember {
    pub source_id: SourceId,
    pub member_index: usize,
    pub weight: f32,
    pub version: String,
    pub bias_applied: bool,
}

/// Hasil fusion: kumpulan anggota ensemble dengan bobot yang tervalidasi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionInput {
    pub members: Vec<EnsembleMember>,
    pub total_weight: f32,
    pub sources_used: Vec<SourceId>,
}

impl FusionInput {
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            total_weight: 0.0,
            sources_used: Vec::new(),
        }
    }

    /// Tambah anggota ensemble. Weight harus >= 0. Total weight tidak boleh melebihi 1.0
    /// setelah penambahan (dipasang pada akhir via normalize).
    pub fn add_member(&mut self, member: EnsembleMember) -> Result<(), WIError> {
        if member.weight < 0.0 {
            return Err(WIError::validation(format!(
                "Weight anggota {} tidak boleh negatif",
                member.source_id.as_str()
            )));
        }
        let prospective_total = self.total_weight + member.weight;
        if prospective_total > 1.0 + 1e-6 {
            return Err(WIError::validation(format!(
                "Total weight melebihi 1.0: {:.6}",
                prospective_total
            )));
        }
        self.total_weight = prospective_total;
        let source_id = member.source_id;
        self.members.push(member);
        if !self.sources_used.contains(&source_id) {
            self.sources_used.push(source_id);
        }
        Ok(())
    }

    /// Normalisasi weight sehingga total = 1.0. Jika total_weight = 0, semua weight diatur 1/M.
    pub fn normalize_weights(&mut self) -> Result<(), WIError> {
        if self.members.is_empty() {
            return Err(WIError::validation(
                "Tidak ada anggota ensemble untuk dinormalisasi",
            ));
        }
        if self.total_weight <= 0.0 {
            let equal = 1.0 / self.members.len() as f32;
            for m in &mut self.members {
                m.weight = equal;
            }
            self.total_weight = 1.0;
        } else {
            for m in &mut self.members {
                m.weight /= self.total_weight;
            }
            self.total_weight = 1.0;
        }
        Ok(())
    }
}

impl Default for FusionInput {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newtypes() {
        let temp = Celsius::new(25.0);
        assert_eq!(temp.value(), 25.0);
        assert_eq!(temp.to_kelvin(), 298.15);

        let wind = Mps::new(10.0);
        assert_eq!(wind.to_kmh(), 36.0);

        let pressure = Hpa::new(1013.25);
        assert_eq!(pressure.to_pa(), 101325.0);

        let humidity = Humidity::new(150.0);
        assert_eq!(humidity.value(), 100.0);

        let prob = Probability::from_percent(75.0);
        assert_eq!(prob.value(), 0.75);
    }

    #[test]
    fn test_bbox() {
        let indonesia = Bbox::indonesia();
        assert!(indonesia.contains(-6.595, 106.806)); // Jakarta
        assert!(!indonesia.contains(51.5, -0.1)); // London
    }

    #[test]
    fn test_climate_zone() {
        assert_eq!(ClimateZone::from_lat_lon(0.0, 0.0), ClimateZone::Tropical);
        assert_eq!(
            ClimateZone::from_lat_lon(-6.595, 106.806),
            ClimateZone::Tropical
        );
        assert_eq!(ClimateZone::from_lat_lon(55.0, 10.0), ClimateZone::Oceanic);
    }

    #[test]
    fn test_source_registry_double_counting_noaa_gefs_family() {
        let mut registry = SourceRegistry::new();
        registry.register(SourceMetadata {
            id: SourceId::MetGfs,
            role: SourceRole::BasePhysics,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.93,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA",
        });
        registry.register(SourceMetadata {
            id: SourceId::MetGefs,
            role: SourceRole::BaseEnsemble,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA",
        });
        registry.register(SourceMetadata {
            id: SourceId::MetAigefs,
            role: SourceRole::AiMember,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA",
        });
        registry.register(SourceMetadata {
            id: SourceId::MetHgefs,
            role: SourceRole::Excluded,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "NOAA",
        });

        // GFS (independent) boleh dipilih
        assert!(registry.select(SourceId::MetGfs).is_ok());
        // GEFS (NoaaGefsFamily) boleh dipilih setelah GFS (group berbeda)
        assert!(registry.select(SourceId::MetGefs).is_ok());
        // AIGEFS (NoaaGefsFamily) TIDAK boleh — double counting dengan GEFS
        assert!(registry.select(SourceId::MetAigefs).is_err());
        // HGEFS (Excluded) TIDAK boleh — role Excluded
        assert!(registry.select(SourceId::MetHgefs).is_err());

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_source_registry_openmeteo_excluded() {
        let mut registry = SourceRegistry::new();
        registry.register(SourceMetadata {
            id: SourceId::MetOpenMeteo,
            role: SourceRole::ComparatorOnly,
            dependency_group: DependencyGroup::OpenMeteoAggregator,
            reliability_score: 0.88,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "Open-Meteo CC BY 4.0",
        });
        registry.register(SourceMetadata {
            id: SourceId::MetGfs,
            role: SourceRole::BasePhysics,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.93,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA",
        });

        // Open-Meteo adalah ComparatorOnly, tidak eligible untuk fusion
        assert!(registry.select(SourceId::MetOpenMeteo).is_err());
        // GFS boleh
        assert!(registry.select(SourceId::MetGfs).is_ok());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_source_registry_fire_hotspot_family() {
        let mut registry = SourceRegistry::new();
        registry.register(SourceMetadata {
            id: SourceId::FireFirms,
            role: SourceRole::ModuleSpecific,
            dependency_group: DependencyGroup::FireHotspotFamily,
            reliability_score: 0.92,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NASA FIRMS",
        });
        registry.register(SourceMetadata {
            id: SourceId::FireSipongi,
            role: SourceRole::ModuleSpecific,
            dependency_group: DependencyGroup::FireHotspotFamily,
            reliability_score: 0.85,
            requires_written_permission: true,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "Kementerian Kehutanan",
        });

        assert!(registry.select(SourceId::FireFirms).is_ok());
        // SiPongi+ berbagi sensor dengan FIRMS — tidak boleh dipilih sekaligus
        assert!(registry.select(SourceId::FireSipongi).is_err());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_fusion_input_normalize_equal_weights() {
        let mut input = FusionInput::new();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGfs,
                member_index: 0,
                weight: 0.0,
                version: "gfs-1.0".to_string(),
                bias_applied: false,
            })
            .unwrap();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGefs,
                member_index: 0,
                weight: 0.0,
                version: "gefs-1.0".to_string(),
                bias_applied: false,
            })
            .unwrap();

        input.normalize_weights().unwrap();
        assert!((input.total_weight - 1.0).abs() < 1e-6);
        for m in &input.members {
            assert!((m.weight - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fusion_input_rejects_negative_weight() {
        let mut input = FusionInput::new();
        let err = input.add_member(EnsembleMember {
            source_id: SourceId::MetGfs,
            member_index: 0,
            weight: -0.1,
            version: "gfs-1.0".to_string(),
            bias_applied: false,
        });
        assert!(err.is_err());
    }

    #[test]
    fn test_belum_diputuskan_error() {
        let err = WIError::belum_diputuskan("alpha", "belum ditetapkan lewat validasi");
        assert!(err.to_string().contains("belum diputuskan"));
        assert!(err.to_string().contains("alpha"));
    }
}
