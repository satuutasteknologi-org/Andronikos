//! # Open-Meteo API Adapter
//!
//! Implementation for Open-Meteo's weather APIs including Archive and Forecast.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::warn;

use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, VerticalLevel, WeatherForecast,
    WeatherObservation, WeatherVariable,
};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};

/// Open-Meteo API adapter
pub struct OpenMeteoAdapter {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    rate_limiter: Arc<RwLock<tokio::time::Instant>>,
    min_request_interval: Duration,
}

impl Default for OpenMeteoAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMeteoAdapter {
    pub fn new() -> Self {
        Self::new_with_key(None)
    }

    pub fn new_with_key(api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.open-meteo.com".to_string(),
            api_key,
            rate_limiter: Arc::new(RwLock::new(tokio::time::Instant::now())),
            min_request_interval: Duration::from_millis(100), // 10 req/s
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
    dew_point_2m: Option<Vec<f64>>,
    visibility: Option<Vec<f64>>,
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
            return Poll::Ready(Some(Err(wi_core::WIError::data_ingestion(&err))));
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
        let step = 0.5; // ~50km cells for API calls
        let cells = region.subdivide(step);

        for cell in cells {
            let lat = (cell.min_lat + cell.max_lat) / 2.0;
            let lon = (cell.min_lon + cell.max_lon) / 2.0;

            // Rate limiting
            {
                let mut last_req = self.rate_limiter.write().await;
                let now = tokio::time::Instant::now();
                let elapsed = now.duration_since(*last_req);
                if elapsed < self.min_request_interval {
                    tokio::time::sleep(self.min_request_interval - elapsed).await;
                }
                *last_req = tokio::time::Instant::now();
            }

            let hourly_vars = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
                "cloud_cover",
                "dew_point_2m",
                "visibility",
            ]
            .join(",");

            let mut url = format!(
                "{}/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}",
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

    async fn fetch_forecast(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        let lead_hours = lead_seconds / 3600;
        let hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover",
            "dew_point_2m",
            "visibility",
        ]
        .join(",");

        let mut url = format!(
            "{}/v1/forecast?latitude={}&longitude={}&hourly={}",
            self.base_url, lat, lon, hourly_vars
        );
        url.push_str(&format!(
            "&forecast_days={}",
            (lead_hours / 24).clamp(1, 16)
        ));
        url.push_str("&timezone=UTC&wind_speed_unit=ms");
        if let Some(ref key) = self.api_key {
            url.push_str(&format!("&apikey={}", key));
        }

        let response = self.client.get(&url).send().await.map_err(|e| {
            wi_core::WIError::network_with_source("Forecast request failed", e.into())
        })?;

        let json: OpenMeteoResponse = response.json().await.map_err(|e| {
            wi_core::WIError::serialization_with_source("Failed to parse forecast", e.into())
        })?;

        let mut variables = HashMap::new();
        let hourly = json
            .hourly
            .ok_or_else(|| wi_core::WIError::data_ingestion("No hourly data"))?;

        if let Some(ref times) = hourly.time {
            let target_idx = (lead_hours as usize).min(times.len().saturating_sub(1));

            if let Some(ref temps) = hourly.temperature_2m {
                if target_idx < temps.len() {
                    variables.insert(
                        (WeatherVariable::Temperature, VerticalLevel::Surface),
                        temps[target_idx],
                    );
                }
            }
            if let Some(ref humidity) = hourly.relative_humidity_2m {
                if target_idx < humidity.len() {
                    variables.insert(
                        (WeatherVariable::RelativeHumidity, VerticalLevel::Surface),
                        humidity[target_idx],
                    );
                }
            }
            if let Some(ref precip) = hourly.precipitation {
                if target_idx < precip.len() {
                    variables.insert(
                        (WeatherVariable::Precipitation, VerticalLevel::Surface),
                        precip[target_idx],
                    );
                }
            }
            if let Some(ref wind) = hourly.wind_speed_10m {
                if target_idx < wind.len() {
                    variables.insert(
                        (WeatherVariable::WindSpeed, VerticalLevel::Surface),
                        wind[target_idx],
                    );
                }
            }
            if let Some(ref wind_dir) = hourly.wind_direction_10m {
                if target_idx < wind_dir.len() {
                    variables.insert(
                        (WeatherVariable::WindDirection, VerticalLevel::Surface),
                        wind_dir[target_idx],
                    );
                }
            }
            if let Some(ref pressure) = hourly.surface_pressure {
                if target_idx < pressure.len() {
                    variables.insert(
                        (WeatherVariable::SurfacePressure, VerticalLevel::Surface),
                        pressure[target_idx],
                    );
                }
            }
        }

        let valid_time = Utc::now() + chrono::Duration::seconds(lead_seconds as i64);

        Ok(WeatherForecast {
            coord: GeoCoord::new(lat, lon),
            init_time: Utc::now(),
            valid_time,
            lead_seconds,
            variables,
            ensemble_members: None,
            model_id: "open-meteo".to_string(),
            probability: None,
        })
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
            if i < temps.len() && temps[i].is_finite() {
                variables.insert(
                    WeatherVariable::Temperature,
                    ObservedValue::new(temps[i], 0.9),
                );
            }
        }

        if let Some(ref humidity) = hourly.relative_humidity_2m {
            if i < humidity.len() && humidity[i].is_finite() {
                variables.insert(
                    WeatherVariable::RelativeHumidity,
                    ObservedValue::new(humidity[i], 0.9),
                );
            }
        }

        if let Some(ref precip) = hourly.precipitation {
            if i < precip.len() && precip[i].is_finite() {
                variables.insert(
                    WeatherVariable::Precipitation,
                    ObservedValue::new(precip[i], 0.9),
                );
            }
        }

        if let Some(ref wind) = hourly.wind_speed_10m {
            if i < wind.len() && wind[i].is_finite() {
                variables.insert(WeatherVariable::WindSpeed, ObservedValue::new(wind[i], 0.9));
            }
        }

        if let Some(ref wind_dir) = hourly.wind_direction_10m {
            if i < wind_dir.len() && wind_dir[i].is_finite() {
                variables.insert(
                    WeatherVariable::WindDirection,
                    ObservedValue::new(wind_dir[i], 0.9),
                );
            }
        }

        if let Some(ref pressure) = hourly.surface_pressure {
            if i < pressure.len() && pressure[i].is_finite() {
                variables.insert(
                    WeatherVariable::SurfacePressure,
                    ObservedValue::new(pressure[i], 0.9),
                );
            }
        }

        if let Some(ref cloud) = hourly.cloud_cover {
            if i < cloud.len() && cloud[i].is_finite() {
                variables.insert(
                    WeatherVariable::CloudCover,
                    ObservedValue::new(cloud[i], 0.9),
                );
            }
        }

        if let Some(ref dew) = hourly.dew_point_2m {
            if i < dew.len() && dew[i].is_finite() {
                variables.insert(WeatherVariable::DewPoint, ObservedValue::new(dew[i], 0.9));
            }
        }

        if let Some(ref vis) = hourly.visibility {
            if i < vis.len() && vis[i].is_finite() {
                variables.insert(WeatherVariable::Visibility, ObservedValue::new(vis[i], 0.9));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = OpenMeteoAdapter::new();
        assert_eq!(adapter.name(), "Open-Meteo");
        assert_eq!(adapter.source_type(), DataSource::OpenMeteo);
    }
}
