//! # Binary Storage Layer - Custom Format Weather Data Storage
//! 
//! This module provides efficient storage for spatiotemporal weather data using
//! a custom binary format with zstd compression.
//!
//! Format: [magic:4][version:4][num_records:8][flags:4][compressed_data:N]

use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tracing::info;
use serde::{Serialize, Deserialize};

use aetherforge_core::{
    AetherError, GeoCoord, Result, WeatherObservation,
    DataSource, WeatherVariable, QualityFlag, ObservedValue,
};

const STORAGE_MAGIC: u32 = 0x41545448; // "ATTH" - AetherForge Weather Data
const STORAGE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub base_path: PathBuf,
    pub compression_enabled: bool,
    pub compression_level: i32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/var/aetherforge/data"),
            compression_enabled: true,
            compression_level: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    Zstd,
    Snappy,
    Gzip,
    Lz4,
    None,
}

impl CompressionType {
    fn extension(&self) -> &'static str {
        match self {
            CompressionType::Zstd => "zst",
            CompressionType::Snappy => "snappy",
            CompressionType::Gzip => "gz",
            CompressionType::Lz4 => "lz4",
            CompressionType::None => "raw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageCategory {
    Observations,
    Forecasts,
    Reanalysis,
}

impl StorageCategory {
    fn path(&self) -> &'static str {
        match self {
            StorageCategory::Observations => "observations",
            StorageCategory::Forecasts => "forecasts",
            StorageCategory::Reanalysis => "reanalysis",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObservationRecord {
    time: i64,
    lat: f64,
    lon: f64,
    altitude: f32,
    station_id: Option<u64>,
    source: u8,
    quality: u8,
    geohash: [u8; 7],
    variables: VariableStorage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct VariableStorage {
    temperature: Option<f32>,
    temperature_conf: Option<f32>,
    humidity: Option<f32>,
    humidity_conf: Option<f32>,
    precipitation: Option<f32>,
    precipitation_conf: Option<f32>,
    wind_speed: Option<f32>,
    wind_speed_conf: Option<f32>,
    wind_direction: Option<f32>,
    wind_direction_conf: Option<f32>,
    pressure: Option<f32>,
    pressure_conf: Option<f32>,
    cloud_cover: Option<f32>,
    cloud_cover_conf: Option<f32>,
}

impl VariableStorage {
    fn from_observation(obs: &WeatherObservation) -> Self {
        let mut vars = VariableStorage::default();
        
        for (var, val) in &obs.variables {
            match var {
                WeatherVariable::Temperature => {
                    vars.temperature = Some(val.value as f32);
                    vars.temperature_conf = Some(val.confidence as f32);
                },
                WeatherVariable::RelativeHumidity => {
                    vars.humidity = Some(val.value as f32);
                    vars.humidity_conf = Some(val.confidence as f32);
                },
                WeatherVariable::Precipitation => {
                    vars.precipitation = Some(val.value as f32);
                    vars.precipitation_conf = Some(val.confidence as f32);
                },
                WeatherVariable::WindSpeed => {
                    vars.wind_speed = Some(val.value as f32);
                    vars.wind_speed_conf = Some(val.confidence as f32);
                },
                WeatherVariable::WindDirection => {
                    vars.wind_direction = Some(val.value as f32);
                    vars.wind_direction_conf = Some(val.confidence as f32);
                },
                WeatherVariable::Pressure | WeatherVariable::SurfacePressure => {
                    vars.pressure = Some(val.value as f32);
                    vars.pressure_conf = Some(val.confidence as f32);
                },
                WeatherVariable::CloudCover => {
                    vars.cloud_cover = Some(val.value as f32);
                    vars.cloud_cover_conf = Some(val.confidence as f32);
                },
                _ => {}
            }
        }
        
        vars
    }
}

pub struct StorageManager {
    config: StorageConfig,
    write_buffer: RwLock<Vec<u8>>,
    record_count: RwLock<usize>,
}

impl StorageManager {
    pub fn new(config: StorageConfig) -> Self {
        std::fs::create_dir_all(&config.base_path).ok();
        
        Self {
            config,
            write_buffer: RwLock::new(Vec::with_capacity(1_000_000)),
            record_count: RwLock::new(0),
        }
    }
    
    pub async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize> {
        if observations.is_empty() {
            return Ok(0);
        }
        
        let records: Vec<ObservationRecord> = observations.iter()
            .map(|obs| self.observation_to_record(obs))
            .collect();
        
        let compressed = self.serialize_and_compress(&records)?;
        let count = records.len();
        
        let date = Utc::now();
        let path = self.config.base_path
            .join(StorageCategory::Observations.path())
            .join(format!("year={}", date.format("%Y")))
            .join(format!("month={}", date.format("%m")))
            .join(format!("day={}", date.format("%d")))
            .join(format!("{}_{}.atd", StorageCategory::Observations.path(), date.format("%H%M%S")));
        
        std::fs::create_dir_all(path.parent().unwrap())
            .map_err(|e| AetherError::storage("Failed to create directory").with_source(e))?;
        
        std::fs::write(&path, &compressed)
            .map_err(|e| AetherError::storage("Failed to write file").with_source(e))?;
        
        info!("Written {} observations to {:?}", count, path);
        Ok(count)
    }
    
    pub async fn read_observations(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<(f64, f64, f64, f64)>,
    ) -> Result<Vec<WeatherObservation>> {
        let mut all_observations = Vec::new();
        
        let base_path = self.config.base_path.join(StorageCategory::Observations.path());
        if !base_path.exists() {
            return Ok(all_observations);
        }
        
        for entry in walkdir::WalkDir::new(&base_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "atd"))
        {
            if let Ok(observations) = self.read_observations_from_file(entry.path(), start, end, bbox) {
                all_observations.extend(observations);
            }
        }
        
        Ok(all_observations)
    }
    
    fn read_observations_from_file(
        &self,
        path: &Path,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<(f64, f64, f64, f64)>,
    ) -> Result<Vec<WeatherObservation>> {
        let data = std::fs::read(path)
            .map_err(|e| AetherError::storage("Failed to read file").with_source(e))?;
        
        let records: Vec<ObservationRecord> = self.decompress_and_deserialize(&data)?;
        
        let observations = records.into_iter()
            .filter(|r| {
                let time = DateTime::from_timestamp(r.time, 0).unwrap_or_else(Utc::now);
                time >= start && time <= end
            })
            .filter(|r| {
                if let Some((min_lat, min_lon, max_lat, max_lon)) = bbox {
                    r.lat >= min_lat && r.lat <= max_lat && r.lon >= min_lon && r.lon <= max_lon
                } else {
                    true
                }
            })
            .map(|r| self.record_to_observation(r))
            .collect();
        
        Ok(observations)
    }
    
    fn observation_to_record(&self, obs: &WeatherObservation) -> ObservationRecord {
        let geohash_bytes = {
            let gh = obs.coord.to_geo_hash(6);
            let mut bytes = [0u8; 7];
            bytes.copy_from_slice(gh.as_bytes());
            bytes
        };
        
        let source_id = obs.source as u8;
        let quality_id = obs.quality_flag as u8;
        
        ObservationRecord {
            time: obs.time.timestamp(),
            lat: obs.coord.lat,
            lon: obs.coord.lon,
            altitude: obs.coord.altitude_m.unwrap_or(0.0) as f32,
            station_id: obs.station_id.as_ref().map(|s| s.len() as u64),
            source: source_id,
            quality: quality_id,
            geohash: geohash_bytes,
            variables: VariableStorage::from_observation(obs),
        }
    }
    
    fn record_to_observation(&self, record: ObservationRecord) -> WeatherObservation {
        let mut variables = std::collections::HashMap::new();
        
        if let Some(t) = record.variables.temperature {
            variables.insert(WeatherVariable::Temperature, ObservedValue::new(t as f64, record.variables.temperature_conf.unwrap_or(0.9)));
        }
        if let Some(h) = record.variables.humidity {
            variables.insert(WeatherVariable::RelativeHumidity, ObservedValue::new(h as f64, record.variables.humidity_conf.unwrap_or(0.9)));
        }
        if let Some(p) = record.variables.precipitation {
            variables.insert(WeatherVariable::Precipitation, ObservedValue::new(p as f64, record.variables.precipitation_conf.unwrap_or(0.9)));
        }
        if let Some(w) = record.variables.wind_speed {
            variables.insert(WeatherVariable::WindSpeed, ObservedValue::new(w as f64, record.variables.wind_speed_conf.unwrap_or(0.9)));
        }
        if let Some(w) = record.variables.wind_direction {
            variables.insert(WeatherVariable::WindDirection, ObservedValue::new(w as f64, record.variables.wind_direction_conf.unwrap_or(0.9)));
        }
        if let Some(p) = record.variables.pressure {
            variables.insert(WeatherVariable::Pressure, ObservedValue::new(p as f64, record.variables.pressure_conf.unwrap_or(0.9)));
        }
        if let Some(c) = record.variables.cloud_cover {
            variables.insert(WeatherVariable::CloudCover, ObservedValue::new(c as f64, record.variables.cloud_cover_conf.unwrap_or(0.9)));
        }
        
        let source = num_to_data_source(record.source);
        let quality = num_to_quality_flag(record.quality);
        
        WeatherObservation {
            station_id: record.station_id.map(|_| None).unwrap_or(None),
            coord: GeoCoord::with_altitude(record.lat, record.lon, record.altitude as f64),
            time: DateTime::from_timestamp(record.time, 0).unwrap_or_else(Utc::now),
            variables,
            quality_flag: quality,
            source,
        }
    }
    
    fn serialize_and_compress<T: Serialize>(&self, records: &[T]) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(12);
        
        result.extend_from_slice(&STORAGE_MAGIC.to_le_bytes());
        result.extend_from_slice(&STORAGE_VERSION.to_le_bytes());
        result.extend_from_slice(&(records.len() as u64).to_le_bytes());
        result.push(if self.config.compression_enabled { 1 } else { 0 });
        
        let data = bincode::serialize(records)
            .map_err(|e| AetherError::storage("Serialization failed").with_source(e))?;
        
        if self.config.compression_enabled {
            let compressed = zstd::encode_all(data.as_slice(), self.config.compression_level)
                .map_err(|e| AetherError::storage("Compression failed").with_source(e))?;
            result.extend(compressed);
        } else {
            result.extend(data);
        }
        
        Ok(result)
    }
}

fn num_to_data_source(n: u8) -> DataSource {
    match n {
        0 => DataSource::OpenMeteo,
        1 => DataSource::OpenWeatherMap,
        2 => DataSource::NOAA,
        3 => DataSource::ERA5,
        4 => DataSource::GFS,
        5 => DataSource::ECMWF,
        6 => DataSource::Meteomatics,
        7 => DataSource::VisualCrossing,
        8 => DataSource::DWD,
        9 => DataSource::MeteoFrance,
        10 => DataSource::UKMetOffice,
        11 => DataSource::JMA,
        12 => DataSource::CMA,
        13 => DataSource::Synop,
        14 => DataSource::Metar,
        15 => DataSource::Satelite,
        16 => DataSource::Radar,
        17 => DataSource::UserProvided,
        _ => DataSource::UserProvided,
    }
}

fn num_to_quality_flag(n: u8) -> QualityFlag {
    match n {
        0 => QualityFlag::Valid,
        1 => QualityFlag::Suspect,
        2 => QualityFlag::Erroneous,
        3 => QualityFlag::Missing,
        _ => QualityFlag::Valid,
    }
}

impl StorageManager {
    fn decompress_and_deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<Vec<T>> {
        if data.len() < 13 {
            return Err(AetherError::storage("File too small"));
        }
        
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != STORAGE_MAGIC {
            return Err(AetherError::storage("Invalid file format"));
        }
        
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != STORAGE_VERSION {
            return Err(AetherError::storage("Unsupported file version"));
        }
        
        let compressed = data[12] == 1;
        let payload = &data[13..];
        
        let decompressed = if compressed {
            zstd::decode_all(payload)
                .map_err(|e| AetherError::storage("Decompression failed").with_source(e))?
        } else {
            payload.to_vec()
        };
        
        let records: Vec<T> = bincode::deserialize(&decompressed)
            .map_err(|e| AetherError::storage("Deserialization failed").with_source(e))?;
        
        Ok(records)
    }
    
    pub fn get_storage_stats(&self) -> Result<StorageStats> {
        let base_path = &self.config.base_path;
        let mut total_size = 0u64;
        let mut file_count = 0u64;
        let observation_count = 0u64;
        
        for category in [StorageCategory::Observations, StorageCategory::Forecasts, StorageCategory::Reanalysis] {
            let category_path = base_path.join(category.path());
            if category_path.exists() {
                for entry in walkdir::WalkDir::new(&category_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map_or(false, |ext| ext == "atd"))
                {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                        file_count += 1;
                    }
                }
            }
        }
        
        Ok(StorageStats {
            total_size,
            file_count,
            observation_count,
        })
    }
    
    pub async fn flush(&self) -> Result<()> {
        let buffer = self.write_buffer.write();
        if !buffer.is_empty() {
            let date = Utc::now();
            let path = self.config.base_path
                .join(StorageCategory::Observations.path())
                .join(format!("buffer_{}.atd", date.format("%s")));
            
            std::fs::write(&path, &*buffer)
                .map_err(|e| AetherError::storage("Flush failed").with_source(e))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_size: u64,
    pub file_count: u64,
    pub observation_count: u64,
}

impl Default for StorageManager {
    fn default() -> Self {
        Self::new(StorageConfig::default())
    }
}

pub mod checkpoint {
    use super::*;
    
    pub struct CheckpointManager {
        checkpoint_path: PathBuf,
    }
    
    impl CheckpointManager {
        pub fn new(base_path: &Path) -> Self {
            Self {
                checkpoint_path: base_path.join("checkpoints"),
            }
        }
        
        pub fn save_checkpoint(&self, data: &[u8], checkpoint_id: &str) -> Result<PathBuf> {
            std::fs::create_dir_all(&self.checkpoint_path)
                .map_err(|e| AetherError::storage("Failed to create checkpoint dir").with_source(e))?;
            
            let path = self.checkpoint_path.join(format!("{}.ckpt", checkpoint_id));
            std::fs::write(&path, data)
                .map_err(|e| AetherError::storage("Failed to save checkpoint").with_source(e))?;
            
            info!("Saved checkpoint to {:?}", path);
            Ok(path)
        }
        
        pub fn load_checkpoint(&self, checkpoint_id: &str) -> Result<Vec<u8>> {
            let path = self.checkpoint_path.join(format!("{}.ckpt", checkpoint_id));
            std::fs::read(&path)
                .map_err(|e| AetherError::storage("Failed to load checkpoint").with_source(e))
        }
        
        pub fn list_checkpoints(&self) -> Result<Vec<String>> {
            let entries = std::fs::read_dir(&self.checkpoint_path)
                .map_err(|e| AetherError::storage("Failed to read checkpoint dir").with_source(e))?;
            
            let checkpoints: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "ckpt"))
                .filter_map(|e| e.file_name().to_str().map(|s| s.replace(".ckpt", "")))
                .collect();
            
            Ok(checkpoints)
        }
    }
}