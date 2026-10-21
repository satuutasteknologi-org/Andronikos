//! # NOAA API Adapter
//!
//! Implementation for NOAA NCEI API (ISD/GHCN data).

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, warn};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};
use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, WIError, WeatherForecast,
    WeatherObservation, WeatherVariable,
};

/// Rate limiter using token bucket algorithm
struct RateLimiter {
    tokens: RwLock<f64>,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: RwLock<Instant>,
}

impl RateLimiter {
    fn new(rate_per_second: f64) -> Self {
        Self {
            tokens: RwLock::new(rate_per_second),
            max_tokens: rate_per_second,
            refill_rate: rate_per_second,
            last_refill: RwLock::new(Instant::now()),
        }
    }

    async fn acquire(&self) {
        loop {
            let mut tokens = self.tokens.write().await;
            let mut last_refill = self.last_refill.write().await;
            let now = Instant::now();
            let elapsed = now.duration_since(*last_refill).as_secs_f64();
            *tokens = (*tokens + elapsed * self.refill_rate).min(self.max_tokens);
            *last_refill = now;

            if *tokens >= 1.0 {
                *tokens -= 1.0;
                return;
            }
            drop(tokens);
            drop(last_refill);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// NOAA API adapter
pub struct NOAAAdapter {
    client: Client,
    base_url: String,
    api_token: Option<String>,
    rate_limiter: Arc<RateLimiter>,
    semaphore: Arc<Semaphore>,
}

impl NOAAAdapter {
    /// Create new NOAA adapter
    ///
    /// # Arguments
    /// * `api_token` - Optional NOAA API token (5 req/s with token, 1 req/s without)
    pub fn new(api_token: Option<String>) -> Self {
        let rate = api_token.as_ref().map(|_| 5.0).unwrap_or(1.0);
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://www.ncei.noaa.gov/cdo-web/api/v2".to_string(),
            api_token,
            rate_limiter: Arc::new(RateLimiter::new(rate)),
            semaphore: Arc::new(Semaphore::new(4)), // max concurrent requests
        }
    }

    /// Build request with authentication
    fn build_request(&self, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.get(format!("{}{}", self.base_url, path));
        if let Some(ref token) = self.api_token {
            req = req.header("token", token);
        }
        req
    }

    /// Fetch stations within a bounding box
    async fn fetch_stations(&self, region: &RegionBbox) -> Result<Vec<NoaaStation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let mut all_stations = Vec::new();
        let mut offset = 1;
        let limit = 1000;

        loop {
            let url = format!(
                "{}?datasetid=GSOM&extent={},{},{},{}&limit={}&offset={}",
                "/stations",
                region.min_lat,
                region.min_lon,
                region.max_lat,
                region.max_lon,
                limit,
                offset
            );

            let response =
                self.build_request(&url).send().await.map_err(|e| {
                    WIError::network_with_source("Station request failed", e.into())
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(WIError::data_ingestion(format!(
                    "Station API error {}: {}",
                    status, text
                )));
            }

            let result: NoaaStationResponse = response.json().await.map_err(|e| {
                WIError::serialization_with_source("Failed to parse stations", e.into())
            })?;

            let results_len = result.results.len();
            all_stations.extend(result.results);
            if results_len < limit {
                break;
            }
            offset += limit;

            // Avoid rate limit
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        debug!("Fetched {} NOAA stations for region", all_stations.len());
        Ok(all_stations)
    }

    /// Fetch observations for a station
    async fn fetch_station_observations(
        &self,
        station_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<NoaaObservation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let mut all_obs = Vec::new();
        let mut offset = 1;
        let limit = 1000;

        let start_str = start.format("%Y-%m-%d").to_string();
        let end_str = end.format("%Y-%m-%d").to_string();

        loop {
            let url = format!(
                "{}?datasetid=GSOM&stationid={}&startdate={}&enddate={}&limit={}&offset={}&units=metric",
                "/data", station_id, start_str, end_str, limit, offset
            );

            let response = self.build_request(&url).send().await.map_err(|e| {
                WIError::network_with_source("Observation request failed", e.into())
            })?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(WIError::data_ingestion(format!(
                    "Observation API error {}: {}",
                    status, text
                )));
            }

            let result: NoaaDataResponse = response.json().await.map_err(|e| {
                WIError::serialization_with_source("Failed to parse observations", e.into())
            })?;

            let results_len = result.results.len();
            all_obs.extend(result.results);
            if results_len < limit {
                break;
            }
            offset += limit;

            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        Ok(all_obs)
    }
}

/// Stream of NOAA observations
struct NoaaObservationStream {
    observations: Vec<WeatherObservation>,
    index: usize,
    error: Option<String>,
}

impl Stream for NoaaObservationStream {
    type Item = Result<WeatherObservation>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(err) = self.error.take() {
            return Poll::Ready(Some(Err(WIError::data_ingestion(&err))));
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

/// NOAA Station response
#[derive(Debug, Deserialize)]
struct NoaaStationResponse {
    results: Vec<NoaaStation>,
}

/// NOAA Station
#[derive(Debug, Deserialize, Clone)]
struct NoaaStation {
    id: String,
    name: String,
    latitude: f64,
    longitude: f64,
    elevation: Option<f64>,
    elevation_unit: Option<String>,
    #[serde(default)]
    mindate: Option<String>,
    #[serde(default)]
    maxdate: Option<String>,
}

/// NOAA Data response
#[derive(Debug, Deserialize)]
struct NoaaDataResponse {
    results: Vec<NoaaObservation>,
}

/// NOAA Observation data point
#[derive(Debug, Deserialize, Clone)]
struct NoaaObservation {
    date: String,
    datatype: String,
    station: String,
    value: f64,
    attributes: Option<String>,
}

#[async_trait]
impl WeatherSource for NOAAAdapter {
    fn name(&self) -> &'static str {
        "NOAA"
    }
    fn source_type(&self) -> DataSource {
        DataSource::NOAA
    }
    fn provides_historical(&self) -> bool {
        true
    }
    fn provides_forecasts(&self) -> bool {
        false
    }
    fn rate_limit(&self) -> f64 {
        self.api_token.as_ref().map(|_| 5.0).unwrap_or(1.0)
    }

    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        // Fetch stations in region
        let stations = self.fetch_stations(&region).await?;
        if stations.is_empty() {
            debug!("No NOAA stations found in region");
            let stream = NoaaObservationStream {
                observations: Vec::new(),
                index: 0,
                error: None,
            };
            return Ok(Box::pin(stream));
        }

        let mut all_observations = Vec::new();

        // Fetch observations for each station
        for station in &stations {
            match self
                .fetch_station_observations(&station.id, start, end)
                .await
            {
                Ok(raw_obs) => {
                    let parsed = parse_noaa_observations(raw_obs, station);
                    all_observations.extend(parsed);
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch observations for station {}: {}",
                        station.id, e
                    );
                    // Continue with other stations
                }
            }
        }

        debug!(
            "Total NOAA observations fetched: {}",
            all_observations.len()
        );
        let stream = NoaaObservationStream {
            observations: all_observations,
            index: 0,
            error: None,
        };
        Ok(Box::pin(stream))
    }

    async fn fetch_forecast(
        &self,
        _lat: f64,
        _lon: f64,
        _lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        Err(WIError::model("NOAA adapter does not provide forecasts"))
    }
}

/// Parse NOAA raw observations into WeatherObservation
fn parse_noaa_observations(
    raw: Vec<NoaaObservation>,
    station: &NoaaStation,
) -> Vec<WeatherObservation> {
    // Group by date/time
    let mut grouped: HashMap<String, HashMap<String, NoaaObservation>> = HashMap::new();

    for obs in raw {
        grouped
            .entry(obs.date.clone())
            .or_default()
            .insert(obs.datatype.clone(), obs);
    }

    let mut observations = Vec::new();

    for (date_str, datatypes) in grouped {
        let time = match DateTime::parse_from_rfc3339(&format!("{}T00:00:00+00:00", date_str)) {
            Ok(t) => t.with_timezone(&Utc),
            Err(_) => continue,
        };

        let mut variables = HashMap::new();

        // Map NOAA datatypes to WeatherVariable
        // GSOM (Global Summary of the Month) datatypes:
        // TAVG = average temperature (tenths of °C)
        // TMAX = max temperature
        // TMIN = min temperature
        // PRCP = precipitation (tenths of mm)
        // SNOW = snowfall (mm)
        // SNWD = snow depth (mm)
        // AWND = average wind speed (tenths of m/s)
        // WDF2 = direction of fastest 2-min wind (degrees)
        // WDF5 = direction of fastest 5-sec wind (degrees)
        // PGTM = peak gust time
        // WSF2 = fastest 2-min wind speed (tenths of m/s)
        // WSF5 = fastest 5-sec wind speed (tenths of m/s)

        if let Some(obs) = datatypes.get("TAVG") {
            variables.insert(
                WeatherVariable::Temperature,
                ObservedValue::new(obs.value / 10.0 + 273.15, 0.85), // Convert to Kelvin
            );
        }
        if let Some(obs) = datatypes.get("TMAX") {
            variables.insert(
                WeatherVariable::Temperature,
                ObservedValue::new(obs.value / 10.0 + 273.15, 0.8),
            );
        }
        if let Some(obs) = datatypes.get("TMIN") {
            variables.insert(
                WeatherVariable::Temperature,
                ObservedValue::new(obs.value / 10.0 + 273.15, 0.8),
            );
        }
        if let Some(obs) = datatypes.get("PRCP") {
            variables.insert(
                WeatherVariable::Precipitation,
                ObservedValue::new(obs.value / 10.0, 0.85), // Convert to mm
            );
        }
        if let Some(obs) = datatypes.get("AWND") {
            variables.insert(
                WeatherVariable::WindSpeed,
                ObservedValue::new(obs.value / 10.0, 0.8), // Convert to m/s
            );
        }
        if let Some(obs) = datatypes.get("WDF2") {
            variables.insert(
                WeatherVariable::WindDirection,
                ObservedValue::new(obs.value, 0.8),
            );
        }
        if let Some(obs) = datatypes.get("WSF2") {
            variables.insert(
                WeatherVariable::WindSpeed,
                ObservedValue::new(obs.value / 10.0, 0.75),
            );
        }

        if !variables.is_empty() {
            let mut coord = GeoCoord::new(station.latitude, station.longitude);
            if let Some(elev) = station.elevation {
                coord.altitude_m = Some(elev);
            }

            observations.push(WeatherObservation {
                station_id: Some(station.id.clone()),
                coord,
                time,
                variables,
                quality_flag: QualityFlag::Valid,
                source: DataSource::NOAA,
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
        let adapter = NOAAAdapter::new(None);
        assert_eq!(adapter.name(), "NOAA");
        assert_eq!(adapter.source_type(), DataSource::NOAA);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10.0);
        // Just verify it compiles and creates
        assert_eq!(limiter.max_tokens, 10.0);
    }
}
