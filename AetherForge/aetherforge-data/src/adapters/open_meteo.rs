//! # Open-Meteo API Adapter
//! 
//! Implementation for Open-Meteo's weather APIs including Archive and Forecast.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use chrono::{DateTime, Utc, Datelike, Timelike};
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::warn;

use aetherforge_core::{DataSource, GeoCoord, Result, WeatherObservation, WeatherVariable, ObservedValue, QualityFlag};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};

/// Open-Meteo API adapter
pub struct OpenMeteoAdapter {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    rate_limiter: Arc<RwLock<tokio::time::Instant>>,
    min_request_interval: Duration,
}

impl OpenMeteoAdapter {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            api_key,
            rate_limiter: Arc::new(RwLock::new(tokio::time::Instant::now())),
            min_request_interval: Duration::from_millis(100),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenMeteoResponse {
    hourly: Option<HourlyData>,
}

#[derive(Debug, Deserialize)]
struct HourlyData {
    time: Option<Vec<String>>,
    temperature_2m: Option<Vec<f64>>,
    relative_humidity_2m: Option<Vec<f64>>,
    precipitation: Option<Vec<f64>>,
    wind_speed_10m: Option<Vec<f64>>,
    wind_direction_10m: Option<Vec<f64>>,
    surface_pressure: Option<Vec<f64>>,
    cloud_cover: Option<Vec<f64>>,
}

struct ObservationIter {
    observations: Vec<WeatherObservation>,
    current_error: Option<String>,
    index: usize,
}

impl Stream for ObservationIter {
    type Item = Result<WeatherObservation>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(err) = self.current_error.take() {
            return Poll::Ready(Some(Err(aetherforge_core::AetherError::data_ingestion(&err))));
        }
        if self.index < self.observations.len() {
            let item = self.observations[self.index].clone();
            self.index += 1;
            Poll::Ready(Some(Ok(item)))
        } else {
            Poll::Ready(None)
        }
    }
}

impl OpenMeteoAdapter {
    async fn fetch_all_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<WeatherObservation> {
        let mut all_obs = Vec::new();
        let resolution = 0.5;
        let cells = region.subdivide(resolution);
        
        for cell in cells {
            let lat = (cell.min_lat + cell.max_lat) / 2.0;
            let lon = (cell.min_lon + cell.max_lon) / 2.0;
            
            {
                let mut last_req = self.rate_limiter.write().await;
                let now = tokio::time::Instant::now();
                let elapsed = now.duration_since(*last_req);
                if elapsed < self.min_request_interval {
                    tokio::time::sleep(self.min_request_interval - elapsed).await;
                }
                *last_req = tokio::time::Instant::now();
            }
            
            let url = {
                let hourly_vars = [
                    "temperature_2m", "relative_humidity_2m", "precipitation",
                    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
                ].join(",");
                
                let mut url = format!(
                    "{}/archive?latitude={}&longitude={}&start_date={}&end_date={}",
                    self.base_url,
                    lat,
                    lon,
                    start.format("%Y-%m-%d"),
                    end.format("%Y-%m-%d"),
                );
                url.push_str(&format!("&hourly={}", hourly_vars));
                url.push_str("&timezone=UTC&wind_speed_unit=ms");
                if let Some(ref key) = self.api_key {
                    url.push_str(&format!("&apikey={}", key));
                }
                url
            };

            match self.client.get(&url).send().await {
                Ok(response) => {
                    if let Ok(json) = response.json::<OpenMeteoResponse>().await {
                        let obs = parse_open_meteo_json(json, lat, lon);
                        all_obs.extend(obs);
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch for cell ({}, {}): {}", lat, lon, e);
                }
            }
            
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        
        all_obs
    }
}

#[async_trait]
impl WeatherSource for OpenMeteoAdapter {
    fn name(&self) -> &'static str {
        "Open-Meteo"
    }

    fn source_type(&self) -> DataSource {
        DataSource::OpenMeteo
    }

    fn provides_forecasts(&self) -> bool {
        true
    }

    fn provides_historical(&self) -> bool {
        true
    }

    fn rate_limit(&self) -> f64 {
        10.0
    }

    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        let observations = self.fetch_all_observations(region, start, end).await;
        let stream = ObservationIter {
            observations,
            current_error: None,
            index: 0,
        };
        Ok(Box::pin(stream))
    }
}

fn parse_open_meteo_json(json: OpenMeteoResponse, lat: f64, lon: f64) -> Vec<WeatherObservation> {
    let hourly = match json.hourly {
        Some(h) => h,
        None => return Vec::new(),
    };
    let times = match hourly.time {
        Some(t) => t,
        None => return Vec::new(),
    };
    
    let mut observations = Vec::new();
    
    for (i, time_str) in times.iter().enumerate() {
        let time = match DateTime::parse_from_rfc3339(time_str) {
            Ok(t) => t.with_timezone(&Utc),
            Err(_) => continue,
        };
        
        let mut variables = HashMap::new();
        
        if let Some(ref temps) = hourly.temperature_2m {
            if i < temps.len() {
                let val = temps[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::Temperature,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref humidity) = hourly.relative_humidity_2m {
            if i < humidity.len() {
                let val = humidity[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::RelativeHumidity,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref precip) = hourly.precipitation {
            if i < precip.len() {
                let val = precip[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::Precipitation,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref wind) = hourly.wind_speed_10m {
            if i < wind.len() {
                let val = wind[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::WindSpeed,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref wind_dir) = hourly.wind_direction_10m {
            if i < wind_dir.len() {
                let val = wind_dir[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::WindDirection,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref pressure) = hourly.surface_pressure {
            if i < pressure.len() {
                let val = pressure[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::SurfacePressure,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if let Some(ref cloud) = hourly.cloud_cover {
            if i < cloud.len() {
                let val = cloud[i];
                if val.is_finite() {
                    variables.insert(
                        WeatherVariable::CloudCover,
                        ObservedValue::new(val, 0.9),
                    );
                }
            }
        }
        
        if !variables.is_empty() {
            observations.push(WeatherObservation {
                station_id: None,
                coord: GeoCoord::new(lat, lon),
                time,
                variables,
                quality_flag: QualityFlag::Valid,
                source: DataSource::OpenMeteo,
            });
        }
    }
    
    observations
}