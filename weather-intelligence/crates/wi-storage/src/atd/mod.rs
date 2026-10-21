//! # ATD Binary Storage
//!
//! Custom binary format with zstd compression for efficient weather data storage.
//!
//! Format: [MAGIC:4][VERSION:4][NUM_RECORDS:8][FLAGS:1][COMPRESSED_DATA:N]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::info;

use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, WIError, WeatherObservation,
    WeatherVariable,
};

const STORAGE_MAGIC: u32 = 0x41545448; // "ATTH" - AetherForge Weather Data
const STORAGE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct AtdConfig {
    pub base_path: PathBuf,
    pub compression_enabled: bool,
    pub compression_level: i32,
}

impl Default for AtdConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/var/weather-intelligence/data"),
            compression_enabled: true,
            compression_level: 3,
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
    wind_u: Option<f32>,
    wind_u_conf: Option<f32>,
    wind_v: Option<f32>,
    wind_v_conf: Option<f32>,
    dew_point: Option<f32>,
    dew_point_conf: Option<f32>,
    visibility: Option<f32>,
    visibility_conf: Option<f32>,
}

impl VariableStorage {
    fn from_observation(obs: &WeatherObservation) -> Self {
        let mut vars = VariableStorage::default();

        for (var, val) in &obs.variables {
            match var {
                WeatherVariable::Temperature => {
                    vars.temperature = Some(val.value as f32);
                    vars.temperature_conf = Some(val.confidence);
                }
                WeatherVariable::RelativeHumidity => {
                    vars.humidity = Some(val.value as f32);
                    vars.humidity_conf = Some(val.confidence);
                }
                WeatherVariable::Precipitation => {
                    vars.precipitation = Some(val.value as f32);
                    vars.precipitation_conf = Some(val.confidence);
                }
                WeatherVariable::WindSpeed => {
                    vars.wind_speed = Some(val.value as f32);
                    vars.wind_speed_conf = Some(val.confidence);
                }
                WeatherVariable::WindDirection => {
                    vars.wind_direction = Some(val.value as f32);
                    vars.wind_direction_conf = Some(val.confidence);
                }
                WeatherVariable::Pressure | WeatherVariable::SurfacePressure => {
                    vars.pressure = Some(val.value as f32);
                    vars.pressure_conf = Some(val.confidence);
                }
                WeatherVariable::CloudCover => {
                    vars.cloud_cover = Some(val.value as f32);
                    vars.cloud_cover_conf = Some(val.confidence);
                }
                WeatherVariable::WindU => {
                    vars.wind_u = Some(val.value as f32);
                    vars.wind_u_conf = Some(val.confidence);
                }
                WeatherVariable::WindV => {
                    vars.wind_v = Some(val.value as f32);
                    vars.wind_v_conf = Some(val.confidence);
                }
                WeatherVariable::DewPoint => {
                    vars.dew_point = Some(val.value as f32);
                    vars.dew_point_conf = Some(val.confidence);
                }
                WeatherVariable::Visibility => {
                    vars.visibility = Some(val.value as f32);
                    vars.visibility_conf = Some(val.confidence);
                }
                _ => {}
            }
        }

        vars
    }
}

pub struct AtdStorage {
    config: AtdConfig,
    _write_buffer: RwLock<Vec<u8>>,
    _record_count: RwLock<usize>,
}

impl AtdStorage {
    pub fn new(config: AtdConfig) -> Self {
        std::fs::create_dir_all(&config.base_path).ok();

        Self {
            config,
            _write_buffer: RwLock::new(Vec::with_capacity(1_000_000)),
            _record_count: RwLock::new(0),
        }
    }

    pub async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize> {
        if observations.is_empty() {
            return Ok(0);
        }

        let records: Vec<ObservationRecord> = observations
            .iter()
            .map(|obs| self.observation_to_record(obs))
            .collect();

        let compressed = self.serialize_and_compress(&records)?;
        let count = records.len();

        let date = Utc::now();
        let path = self
            .config
            .base_path
            .join("observations")
            .join(format!("year={}", date.format("%Y")))
            .join(format!("month={}", date.format("%m")))
            .join(format!("day={}", date.format("%d")))
            .join(format!("observations_{}.atd", date.format("%H%M%S")));

        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .map_err(|e| WIError::storage_with_source("Failed to create directory", e.into()))?;

        tokio::fs::write(&path, &compressed)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to write file", e.into()))?;

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

        let base_path = self.config.base_path.join("observations");
        if !base_path.exists() {
            return Ok(all_observations);
        }

        for entry in walkdir::WalkDir::new(&base_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "atd"))
        {
            if let Ok(observations) = self
                .read_observations_from_file(entry.path(), start, end, bbox)
                .await
            {
                all_observations.extend(observations);
            }
        }

        Ok(all_observations)
    }

    async fn read_observations_from_file(
        &self,
        path: &Path,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<(f64, f64, f64, f64)>,
    ) -> Result<Vec<WeatherObservation>> {
        let data = tokio::fs::read(path)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to read file", e.into()))?;

        let records: Vec<ObservationRecord> = self.decompress_and_deserialize(&data)?;

        let observations = records
            .into_iter()
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
        let mut variables = HashMap::new();

        macro_rules! add_var {
            ($var:expr, $val:expr, $conf:expr) => {
                if let Some(v) = $val {
                    variables.insert($var, ObservedValue::new(v as f64, $conf.unwrap_or(0.9)));
                }
            };
        }

        add_var!(
            WeatherVariable::Temperature,
            record.variables.temperature,
            record.variables.temperature_conf
        );
        add_var!(
            WeatherVariable::RelativeHumidity,
            record.variables.humidity,
            record.variables.humidity_conf
        );
        add_var!(
            WeatherVariable::Precipitation,
            record.variables.precipitation,
            record.variables.precipitation_conf
        );
        add_var!(
            WeatherVariable::WindSpeed,
            record.variables.wind_speed,
            record.variables.wind_speed_conf
        );
        add_var!(
            WeatherVariable::WindDirection,
            record.variables.wind_direction,
            record.variables.wind_direction_conf
        );
        add_var!(
            WeatherVariable::Pressure,
            record.variables.pressure,
            record.variables.pressure_conf
        );
        add_var!(
            WeatherVariable::CloudCover,
            record.variables.cloud_cover,
            record.variables.cloud_cover_conf
        );
        add_var!(
            WeatherVariable::WindU,
            record.variables.wind_u,
            record.variables.wind_u_conf
        );
        add_var!(
            WeatherVariable::WindV,
            record.variables.wind_v,
            record.variables.wind_v_conf
        );
        add_var!(
            WeatherVariable::DewPoint,
            record.variables.dew_point,
            record.variables.dew_point_conf
        );
        add_var!(
            WeatherVariable::Visibility,
            record.variables.visibility,
            record.variables.visibility_conf
        );

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
        result.push(if self.config.compression_enabled {
            1
        } else {
            0
        });

        let data = bincode::serialize(records)
            .map_err(|e| WIError::storage_with_source("Serialization failed", e.into()))?;

        if self.config.compression_enabled {
            let compressed = zstd::encode_all(data.as_slice(), self.config.compression_level)
                .map_err(|e| WIError::storage_with_source("Compression failed", e.into()))?;
            result.extend(compressed);
        } else {
            result.extend(data);
        }

        Ok(result)
    }

    fn decompress_and_deserialize<T: for<'de> Deserialize<'de>>(
        &self,
        data: &[u8],
    ) -> Result<Vec<T>> {
        if data.len() < 13 {
            return Err(WIError::storage("File too small"));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != STORAGE_MAGIC {
            return Err(WIError::storage("Invalid file format"));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != STORAGE_VERSION {
            return Err(WIError::storage("Unsupported file version"));
        }

        let compressed = data[12] == 1;
        let payload = &data[13..];

        let decompressed = if compressed {
            zstd::decode_all(payload)
                .map_err(|e| WIError::storage_with_source("Decompression failed", e.into()))?
        } else {
            payload.to_vec()
        };

        let records: Vec<T> = bincode::deserialize(&decompressed)
            .map_err(|e| WIError::storage_with_source("Deserialization failed", e.into()))?;

        Ok(records)
    }

    pub async fn get_stats(&self) -> Result<AtdStats> {
        let base_path = &self.config.base_path;
        let mut total_size = 0u64;
        let mut file_count = 0u64;

        for entry in walkdir::WalkDir::new(base_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "atd"))
        {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
                file_count += 1;
            }
        }

        Ok(AtdStats {
            total_size,
            file_count,
            observation_count: 0,
        })
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
        18 => DataSource::BMKG,
        19 => DataSource::Himawari,
        20 => DataSource::WeatherNext,
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

#[derive(Debug, Clone)]
pub struct AtdStats {
    pub total_size: u64,
    pub file_count: u64,
    pub observation_count: u64,
}

impl Default for AtdStorage {
    fn default() -> Self {
        Self::new(AtdConfig::default())
    }
}
