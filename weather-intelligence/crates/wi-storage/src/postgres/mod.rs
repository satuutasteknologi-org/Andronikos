//! # PostgreSQL + PostGIS Storage
//!
//! Relational storage for stations, verification, and metadata.

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use sqlx::{PgPool, Row};
use tracing::{debug, info};

use crate::StorageBackend;
use crate::StorageStats;
use wi_core::{
    Bbox, DataSource, QualityFlag, Result, WIError, WeatherForecast, WeatherObservation,
    WeatherStation, WeatherVariable,
};

/// Helper to convert sqlx errors to WIError
fn sqlx_err(e: sqlx::Error, ctx: &str) -> WIError {
    WIError::storage_with_source(ctx, e.into())
}

/// Parse weather variable name to WeatherVariable enum
fn parse_weather_variable(name: &str) -> Option<WeatherVariable> {
    match name {
        "Temperature" | "temperature" | "temp" | "t2m" => Some(WeatherVariable::Temperature),
        "RelativeHumidity" | "humidity" | "rh" | "rhum" => Some(WeatherVariable::RelativeHumidity),
        "Precipitation" | "precip" | "rain" | "tp" => Some(WeatherVariable::Precipitation),
        "WindSpeed" | "wind_speed" | "ws" | "wspd" => Some(WeatherVariable::WindSpeed),
        "WindDirection" | "wind_dir" | "wd" | "wdir" => Some(WeatherVariable::WindDirection),
        "WindU" | "u_component" | "u" => Some(WeatherVariable::WindU),
        "WindV" | "v_component" | "v" => Some(WeatherVariable::WindV),
        "Pressure" | "pressure" | "pres" | "msl" => Some(WeatherVariable::Pressure),
        "SurfacePressure" | "surface_pressure" | "sp" => Some(WeatherVariable::SurfacePressure),
        "CloudCover" | "cloud_cover" | "cc" | "tcc" => Some(WeatherVariable::CloudCover),
        "DewPoint" | "dew_point" | "td" | "dpt" => Some(WeatherVariable::DewPoint),
        "Visibility" | "visibility" | "vis" => Some(WeatherVariable::Visibility),
        "GeopotentialHeight" | "geopotential" | "gh" | "z" => {
            Some(WeatherVariable::GeopotentialHeight)
        }
        "SpecificHumidity" | "specific_humidity" | "q" => Some(WeatherVariable::SpecificHumidity),
        "VerticalVelocity" | "vertical_velocity" | "w" | "omega" => {
            Some(WeatherVariable::VerticalVelocity)
        }
        "SolarRadiation" | "solar_radiation" | "ssrd" => Some(WeatherVariable::SolarRadiation),
        "UVIndex" | "uv_index" | "uv" => Some(WeatherVariable::UVIndex),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct PostgresConfig {
    pub database_url: String,
    pub max_connections: u32,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost/weather_intelligence".to_string(),
            max_connections: 10,
        }
    }
}

pub struct PostgresStorage {
    pool: std::sync::Arc<PgPool>,
}

impl PostgresStorage {
    pub async fn new(config: PostgresConfig) -> Result<Self> {
        let pool = PgPool::connect(&config.database_url)
            .await
            .map_err(|e| sqlx_err(e, "Failed to connect to PostgreSQL"))?;

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to run migrations", e.into()))?;

        info!("PostgreSQL storage initialized with PostGIS");

        Ok(Self {
            pool: std::sync::Arc::new(pool),
        })
    }

    pub async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize> {
        if observations.is_empty() {
            return Ok(0);
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| sqlx_err(e, "Failed to begin transaction"))?;

        for obs in &observations {
            let vars_json = serde_json::to_value(&obs.variables)
                .unwrap_or(JsonValue::Object(serde_json::Map::new()));

            sqlx::query(
                r#"
                INSERT INTO observations (time, lat, lon, altitude, station_id, source, quality, variables, geom)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, ST_SetSRID(ST_MakePoint($3, $2), 4326)::geography)
                ON CONFLICT (time, lat, lon, station_id) DO UPDATE SET
                    variables = EXCLUDED.variables,
                    quality = EXCLUDED.quality,
                    altitude = EXCLUDED.altitude,
                    updated_at = NOW()
                "#,
            )
            .bind(obs.time)
            .bind(obs.coord.lat)
            .bind(obs.coord.lon)
            .bind(obs.coord.altitude_m)
            .bind(obs.station_id.as_deref())
            .bind(obs.source as i32)
            .bind(obs.quality_flag as i32)
            .bind(vars_json)
            .execute(&mut *tx)
            .await
            .map_err(|e| sqlx_err(e, "Failed to insert observation"))?;
        }

        tx.commit()
            .await
            .map_err(|e| sqlx_err(e, "Failed to commit transaction"))?;

        debug!("Wrote {} observations to PostgreSQL", observations.len());
        Ok(observations.len())
    }

    pub async fn read_observations(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherObservation>> {
        let (query, _use_bbox) = if bbox.is_some() {
            (
                r#"
                SELECT time, lat, lon, altitude, station_id, source, quality, variables
                FROM observations
                WHERE time >= $1 AND time <= $2
                AND ST_Within(geom, ST_MakeEnvelope($3, $4, $5, $6, 4326)::geography)
                ORDER BY time DESC LIMIT 100000
                "#,
                true,
            )
        } else {
            (
                r#"
                SELECT time, lat, lon, altitude, station_id, source, quality, variables
                FROM observations
                WHERE time >= $1 AND time <= $2
                ORDER BY time DESC LIMIT 100000
                "#,
                false,
            )
        };

        let mut q = sqlx::query(query);
        q = q.bind(start).bind(end);
        if let Some(bbox) = bbox {
            q = q
                .bind(bbox.min_lon)
                .bind(bbox.min_lat)
                .bind(bbox.max_lon)
                .bind(bbox.max_lat);
        }

        let rows = q
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| sqlx_err(e, "Failed to read observations"))?;

        let mut observations = Vec::with_capacity(rows.len());
        for row in rows {
            let time: DateTime<Utc> = row
                .try_get("time")
                .map_err(|e| sqlx_err(e, "Failed to get time"))?;
            let lat: f64 = row
                .try_get("lat")
                .map_err(|e| sqlx_err(e, "Failed to get lat"))?;
            let lon: f64 = row
                .try_get("lon")
                .map_err(|e| sqlx_err(e, "Failed to get lon"))?;
            let altitude: Option<f64> = row
                .try_get("altitude")
                .map_err(|e| sqlx_err(e, "Failed to get altitude"))?;
            let station_id: Option<String> = row
                .try_get("station_id")
                .map_err(|e| sqlx_err(e, "Failed to get station_id"))?;
            let source: i32 = row
                .try_get("source")
                .map_err(|e| sqlx_err(e, "Failed to get source"))?;
            let quality: i32 = row
                .try_get("quality")
                .map_err(|e| sqlx_err(e, "Failed to get quality"))?;
            let variables: JsonValue = row
                .try_get("variables")
                .map_err(|e| sqlx_err(e, "Failed to get variables"))?;

            let mut vars_map = std::collections::HashMap::new();
            if let JsonValue::Object(map) = variables {
                for (k, v) in map {
                    if let Some(var) = parse_weather_variable(&k) {
                        if let Some(val) = v.get("value").and_then(|x| x.as_f64()) {
                            let conf =
                                v.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                            let is_valid =
                                v.get("is_valid").and_then(|x| x.as_bool()).unwrap_or(true);
                            vars_map.insert(
                                var,
                                wi_core::types::ObservedValue {
                                    value: val,
                                    is_valid,
                                    confidence: conf,
                                },
                            );
                        }
                    }
                }
            }

            observations.push(WeatherObservation {
                station_id,
                coord: wi_core::types::GeoCoord {
                    lat,
                    lon,
                    altitude_m: altitude,
                },
                time,
                variables: vars_map,
                quality_flag: match quality {
                    0 => QualityFlag::Valid,
                    1 => QualityFlag::Suspect,
                    2 => QualityFlag::Erroneous,
                    _ => QualityFlag::Missing,
                },
                source: match source {
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
                },
            });
        }

        Ok(observations)
    }

    pub async fn write_station(&self, station: &WeatherStation) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO stations (id, name, lat, lon, elevation, country_code, timezone, metadata, geom)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, ST_SetSRID(ST_MakePoint($4, $3), 4326)::geography)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                elevation = EXCLUDED.elevation,
                country_code = EXCLUDED.country_code,
                timezone = EXCLUDED.timezone,
                metadata = EXCLUDED.metadata,
                geom = EXCLUDED.geom,
                updated_at = NOW()
            "#,
        )
        .bind(&station.id)
        .bind(&station.name)
        .bind(station.coord.lat)
        .bind(station.coord.lon)
        .bind(station.elevation_m)
        .bind(&station.country_code)
        .bind(&station.timezone)
        .bind(serde_json::to_value(&station.metadata).unwrap_or(JsonValue::Object(serde_json::Map::new())))
        .execute(&*self.pool)
        .await
        .map_err(|e| sqlx_err(e, "Failed to insert station"))?;

        Ok(())
    }

    pub async fn get_stations_in_bbox(&self, bbox: Bbox) -> Result<Vec<WeatherStation>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, lat, lon, elevation, country_code, timezone, metadata
            FROM stations
            WHERE ST_Within(geom, ST_MakeEnvelope($1, $2, $3, $4, 4326)::geography)
            "#,
        )
        .bind(bbox.min_lon)
        .bind(bbox.min_lat)
        .bind(bbox.max_lon)
        .bind(bbox.max_lat)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| sqlx_err(e, "Failed to query stations"))?;

        let mut stations = Vec::with_capacity(rows.len());
        for row in rows {
            stations.push(WeatherStation {
                id: row
                    .try_get("id")
                    .map_err(|e| sqlx_err(e, "Failed to get id"))?,
                name: row
                    .try_get("name")
                    .map_err(|e| sqlx_err(e, "Failed to get name"))?,
                coord: wi_core::types::GeoCoord {
                    lat: row
                        .try_get("lat")
                        .map_err(|e| sqlx_err(e, "Failed to get lat"))?,
                    lon: row
                        .try_get("lon")
                        .map_err(|e| sqlx_err(e, "Failed to get lon"))?,
                    altitude_m: row
                        .try_get("elevation")
                        .map_err(|e| sqlx_err(e, "Failed to get elevation"))?,
                },
                elevation_m: row
                    .try_get("elevation")
                    .map_err(|e| sqlx_err(e, "Failed to get elevation"))?,
                country_code: row
                    .try_get("country_code")
                    .map_err(|e| sqlx_err(e, "Failed to get country_code"))?,
                timezone: row
                    .try_get("timezone")
                    .map_err(|e| sqlx_err(e, "Failed to get timezone"))?,
                metadata: {
                    let meta_json: JsonValue = row
                        .try_get("metadata")
                        .map_err(|e| sqlx_err(e, "Failed to get metadata"))?;
                    serde_json::from_value(meta_json).unwrap_or_default()
                },
                data_sources: Vec::new(),
                record_started: None,
            });
        }

        Ok(stations)
    }

    pub async fn write_forecasts(&self, forecasts: Vec<WeatherForecast>) -> Result<usize> {
        if forecasts.is_empty() {
            return Ok(0);
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| sqlx_err(e, "Failed to begin transaction"))?;

        for fc in &forecasts {
            let prob_json = fc
                .probability
                .as_ref()
                .map(|p| serde_json::to_value(p).unwrap_or(JsonValue::Null))
                .unwrap_or(JsonValue::Null);
            let vars_json = serde_json::to_value(&fc.variables)
                .unwrap_or(JsonValue::Object(serde_json::Map::new()));

            sqlx::query(
                r#"
                INSERT INTO forecasts (model_id, init_time, valid_time, lead_seconds, lat, lon, variables, ensemble_members, probability, geom)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, ST_SetSRID(ST_MakePoint($6, $5), 4326)::geography)
                ON CONFLICT DO NOTHING
                "#,
            )
            .bind(&fc.model_id)
            .bind(fc.init_time)
            .bind(fc.valid_time)
            .bind(fc.lead_seconds as i32)
            .bind(fc.coord.lat)
            .bind(fc.coord.lon)
            .bind(vars_json)
            .bind(fc.ensemble_members.map(|x| x as i32))
            .bind(prob_json)
            .execute(&mut *tx)
            .await
            .map_err(|e| sqlx_err(e, "Failed to insert forecast"))?;
        }

        tx.commit()
            .await
            .map_err(|e| sqlx_err(e, "Failed to commit forecast transaction"))?;

        debug!("Wrote {} forecasts to PostgreSQL", forecasts.len());
        Ok(forecasts.len())
    }

    pub async fn read_forecasts(
        &self,
        model_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherForecast>> {
        let (query, _use_bbox) = if bbox.is_some() {
            (
                r#"
                SELECT model_id, init_time, valid_time, lead_seconds, lat, lon, variables, ensemble_members, probability
                FROM forecasts
                WHERE model_id = $1 AND init_time >= $2 AND init_time <= $3
                AND ST_Within(geom, ST_MakeEnvelope($4, $5, $6, $7, 4326)::geography)
                ORDER BY valid_time ASC LIMIT 10000
                "#,
                true,
            )
        } else {
            (
                r#"
                SELECT model_id, init_time, valid_time, lead_seconds, lat, lon, variables, ensemble_members, probability
                FROM forecasts
                WHERE model_id = $1 AND init_time >= $2 AND init_time <= $3
                ORDER BY valid_time ASC LIMIT 10000
                "#,
                false,
            )
        };

        let mut q = sqlx::query(query);
        q = q.bind(model_id).bind(start).bind(end);
        if let Some(bbox) = bbox {
            q = q
                .bind(bbox.min_lon)
                .bind(bbox.min_lat)
                .bind(bbox.max_lon)
                .bind(bbox.max_lat);
        }

        let rows = q
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| sqlx_err(e, "Failed to read forecasts"))?;

        let mut forecasts = Vec::with_capacity(rows.len());
        for row in rows {
            let model_id: String = row
                .try_get("model_id")
                .map_err(|e| sqlx_err(e, "Failed to get model_id"))?;
            let init_time: DateTime<Utc> = row
                .try_get("init_time")
                .map_err(|e| sqlx_err(e, "Failed to get init_time"))?;
            let valid_time: DateTime<Utc> = row
                .try_get("valid_time")
                .map_err(|e| sqlx_err(e, "Failed to get valid_time"))?;
            let lead_seconds: i32 = row
                .try_get("lead_seconds")
                .map_err(|e| sqlx_err(e, "Failed to get lead_seconds"))?;
            let lat: f64 = row
                .try_get("lat")
                .map_err(|e| sqlx_err(e, "Failed to get lat"))?;
            let lon: f64 = row
                .try_get("lon")
                .map_err(|e| sqlx_err(e, "Failed to get lon"))?;
            let variables: JsonValue = row
                .try_get("variables")
                .map_err(|e| sqlx_err(e, "Failed to get variables"))?;
            let ensemble_members: Option<i32> = row
                .try_get("ensemble_members")
                .map_err(|e| sqlx_err(e, "Failed to get ensemble_members"))?;
            let probability: JsonValue = row
                .try_get("probability")
                .map_err(|e| sqlx_err(e, "Failed to get probability"))?;

            let mut vars_map = std::collections::HashMap::new();
            if let JsonValue::Object(map) = variables {
                for (k, v) in map {
                    if let Some(var) = parse_weather_variable(&k) {
                        if let Some(val) = v.as_f64() {
                            vars_map.insert((var, wi_core::types::VerticalLevel::Surface), val);
                        }
                    }
                }
            }

            forecasts.push(WeatherForecast {
                coord: wi_core::types::GeoCoord::new(lat, lon),
                init_time,
                valid_time,
                lead_seconds: lead_seconds as u32,
                variables: vars_map,
                ensemble_members: ensemble_members.map(|x| x as u8),
                model_id,
                probability: serde_json::from_value(probability).ok(),
            });
        }

        Ok(forecasts)
    }
}

#[async_trait::async_trait]
impl StorageBackend for PostgresStorage {
    async fn write_observations(&self, observations: Vec<WeatherObservation>) -> Result<usize> {
        self.write_observations(observations).await
    }

    async fn read_observations(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherObservation>> {
        self.read_observations(start, end, bbox).await
    }

    async fn write_forecasts(&self, forecasts: Vec<WeatherForecast>) -> Result<usize> {
        self.write_forecasts(forecasts).await
    }

    async fn read_forecasts(
        &self,
        model_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bbox: Option<Bbox>,
    ) -> Result<Vec<WeatherForecast>> {
        self.read_forecasts(model_id, start, end, bbox).await
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let obs_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM observations")
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| sqlx_err(e, "Failed to get observation count"))?;

        let fc_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM forecasts")
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| sqlx_err(e, "Failed to get forecast count"))?;

        let size_bytes: i64 = sqlx::query_scalar("SELECT pg_database_size(current_database())")
            .fetch_one(&*self.pool)
            .await
            .unwrap_or(0);

        Ok(StorageStats {
            total_size_bytes: size_bytes as u64,
            file_count: 0,
            observation_count: obs_count as u64,
            forecast_count: fc_count as u64,
        })
    }
}
