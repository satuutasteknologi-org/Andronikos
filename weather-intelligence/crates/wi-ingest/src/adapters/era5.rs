//! # ERA5 Reanalysis Adapter
//!
//! Implementation for ECMWF CDS API (ERA5 reanalysis data).
//! Used for training data and bias correction only — NOT for real-time forecasting or verification baseline.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, warn};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};
use wi_core::{DataSource, Result, WIError, WeatherForecast, WeatherObservation};

/// Rate limiter for CDS API (1 req/sec default, can be configured)
struct CdsRateLimiter {
    tokens: RwLock<f64>,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: RwLock<Instant>,
}

impl CdsRateLimiter {
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
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

/// ERA5 API adapter for CDS (Climate Data Store)
pub struct ERA5Adapter {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    rate_limiter: Arc<CdsRateLimiter>,
    semaphore: Arc<Semaphore>,
}

impl ERA5Adapter {
    /// Create new ERA5 adapter
    ///
    /// # Arguments
    /// * `api_key` - CDS API key (format: "uid:key"). Required for CDS API access.
    /// * `rate_limit` - Requests per second (default: 1.0 for CDS)
    pub fn new(api_key: Option<String>, rate_limit: Option<f64>) -> Self {
        let rate = rate_limit.unwrap_or(1.0);
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120)) // ERA5 requests can be slow
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://cds.climate.copernicus.eu/api/v2".to_string(),
            api_key,
            rate_limiter: Arc::new(CdsRateLimiter::new(rate)),
            semaphore: Arc::new(Semaphore::new(2)), // CDS allows limited concurrency
        }
    }

    /// Build authenticated request
    fn build_request(&self, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.post(format!("{}{}", self.base_url, path));
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        req
    }

    /// Submit ERA5 data request and wait for completion
    async fn submit_and_wait(&self, request: &Era5Request) -> Result<Era5JobResponse> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let _url = format!("{}/resources/retrieve", self.base_url);
        let response = self
            .build_request("/resources/retrieve")
            .json(request)
            .send()
            .await
            .map_err(|e| WIError::network_with_source("ERA5 request failed", e.into()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(WIError::data_ingestion(format!(
                "ERA5 API error {}: {}",
                status, text
            )));
        }

        let job_response: Era5JobResponse = response.json().await.map_err(|e| {
            WIError::serialization_with_source("Failed to parse ERA5 job response", e.into())
        })?;

        if job_response.state == "completed" {
            return Ok(job_response);
        }

        // Poll for completion
        self.poll_job(&job_response.job_id).await
    }

    /// Poll ERA5 job until completion
    async fn poll_job(&self, job_id: &str) -> Result<Era5JobResponse> {
        let max_wait = Duration::from_secs(3600); // 1 hour max
        let poll_interval = Duration::from_secs(30);
        let start = Instant::now();

        loop {
            if start.elapsed() > max_wait {
                return Err(WIError::data_ingestion("ERA5 job timeout"));
            }

            tokio::time::sleep(poll_interval).await;

            self.rate_limiter.acquire().await;
            let url = format!("{}/resources/{}", self.base_url, job_id);
            let response = self
                .build_request(&url)
                .send()
                .await
                .map_err(|e| WIError::network_with_source("ERA5 poll failed", e.into()))?;

            if !response.status().is_success() {
                continue;
            }

            let job: Era5JobResponse = response.json().await.map_err(|e| {
                WIError::serialization_with_source("Failed to parse ERA5 job status", e.into())
            })?;

            match job.state.as_str() {
                "completed" => return Ok(job),
                "failed" | "aborted" => {
                    return Err(WIError::data_ingestion(format!(
                        "ERA5 job {}: {}",
                        job_id, job.state
                    )));
                }
                _ => continue, // "queued", "running", "accepted"
            }
        }
    }

    /// Build ERA5 request for single-level or pressure-level data
    fn build_era5_request(
        &self,
        region: &RegionBbox,
        start: DateTime<Utc>,
        _end: DateTime<Utc>,
        pressure_levels: bool,
    ) -> Era5Request {
        let dataset = if pressure_levels {
            "reanalysis-era5-pressure-levels".to_string()
        } else {
            "reanalysis-era5-single-levels".to_string()
        };

        let mut request = Era5Request {
            dataset,
            ..Default::default()
        };

        // Variables
        if pressure_levels {
            request.variables = vec![
                "temperature".to_string(),
                "specific_humidity".to_string(),
                "u_component_of_wind".to_string(),
                "v_component_of_wind".to_string(),
                "geopotential".to_string(),
                "vertical_velocity".to_string(),
            ];
            request.pressure_level = Some(
                vec![
                    "1000", "925", "850", "700", "600", "500", "400", "300", "250", "200", "150",
                    "100", "50",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            );
        } else {
            request.variables = vec![
                "2m_temperature".to_string(),
                "2m_dewpoint_temperature".to_string(),
                "total_precipitation".to_string(),
                "10m_u_component_of_wind".to_string(),
                "10m_v_component_of_wind".to_string(),
                "mean_sea_level_pressure".to_string(),
                "surface_pressure".to_string(),
                "total_cloud_cover".to_string(),
            ];
        }

        // Time range
        let year = start.format("%Y").to_string();
        let month = start.format("%m").to_string();
        let day = start.format("%d").to_string();

        request.year = Some(year);
        request.month = Some(month);
        request.day = Some(day);
        request.time = Some(
            vec![
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Geographic area (CDS format: N, W, S, E)
        request.area = Some(vec![
            region.max_lat,
            region.min_lon,
            region.min_lat,
            region.max_lon,
        ]);

        // Grid resolution
        request.grid = Some(vec!["0.25".to_string(), "0.25".to_string()]); // 0.25° ≈ 25km
        request.format = Some("netcdf".to_string());

        request
    }

    /// Download and parse NetCDF data (simplified - production would use netcdf crate)
    async fn download_and_parse_netcdf(&self, url: &str) -> Result<Vec<WeatherObservation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| WIError::network_with_source("NetCDF download failed", e.into()))?;

        if !response.status().is_success() {
            return Err(WIError::data_ingestion("NetCDF download failed"));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| WIError::storage_with_source("Failed to read NetCDF", e.into()))?;

        // Parse NetCDF - simplified implementation
        // Production should use netcdf crate or convert to Zarr/Parquet first
        self.parse_netcdf_bytes(&bytes)
    }

    /// Parse NetCDF bytes ERA5 → Vec<WeatherObservation>
    /// Menggunakan crate `netcdf-reader 0.9` — pure Rust, tanpa C dependency.
    /// ⚠️ ERA5 adalah TRAINING ONLY — output tidak boleh masuk fusion real-time.
    fn parse_netcdf_bytes(&self, bytes: &[u8]) -> Result<Vec<WeatherObservation>> {
        // Default ke center Indonesia jika tidak ada koordinat konteks.
        // Caller idealnya memanggil parse_era5_netcdf_at(bytes, lat, lon) langsung.
        let (lat, lon) = (-2.5, 118.0); // Center Indonesia approximate
        crate::parsers::parse_era5_netcdf(bytes, lat, lon)
    }
}

/// ERA5 Request structure for CDS API
#[derive(Debug, Serialize, Deserialize, Default)]
struct Era5Request {
    dataset: String,
    variables: Vec<String>,
    pressure_level: Option<Vec<String>>,
    year: Option<String>,
    month: Option<String>,
    day: Option<String>,
    time: Option<Vec<String>>,
    area: Option<Vec<f64>>,
    grid: Option<Vec<String>>,
    format: Option<String>,
}

/// ERA5 Job Response
#[derive(Debug, Deserialize)]
struct Era5JobResponse {
    job_id: String,
    state: String,
    location: Option<String>,
    size: Option<u64>,
}

/// Stream of ERA5 observations
struct Era5ObservationStream {
    observations: Vec<WeatherObservation>,
    index: usize,
    error: Option<String>,
}

impl Stream for Era5ObservationStream {
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

#[async_trait]
impl WeatherSource for ERA5Adapter {
    fn name(&self) -> &'static str {
        "ERA5"
    }
    fn source_type(&self) -> DataSource {
        DataSource::ERA5
    }
    fn provides_historical(&self) -> bool {
        true
    }
    fn provides_forecasts(&self) -> bool {
        false
    }
    fn rate_limit(&self) -> f64 {
        1.0
    }

    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        // ERA5 is typically requested by month
        // For multi-month ranges, we'd need to split into monthly requests
        let mut all_observations = Vec::new();

        // Single-level data
        let request_sl = self.build_era5_request(&region, start, end, false);
        match self.submit_and_wait(&request_sl).await {
            Ok(job) => {
                if let Some(location) = job.location {
                    match self.download_and_parse_netcdf(&location).await {
                        Ok(obs) => all_observations.extend(obs),
                        Err(e) => warn!("Failed to parse single-level ERA5: {}", e),
                    }
                }
            }
            Err(e) => warn!("ERA5 single-level request failed: {}", e),
        }

        // Pressure-level data
        let request_pl = self.build_era5_request(&region, start, end, true);
        match self.submit_and_wait(&request_pl).await {
            Ok(job) => {
                if let Some(location) = job.location {
                    match self.download_and_parse_netcdf(&location).await {
                        Ok(obs) => all_observations.extend(obs),
                        Err(e) => warn!("Failed to parse pressure-level ERA5: {}", e),
                    }
                }
            }
            Err(e) => warn!("ERA5 pressure-level request failed: {}", e),
        }

        debug!(
            "Total ERA5 observations fetched: {}",
            all_observations.len()
        );
        let stream = Era5ObservationStream {
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
        Err(WIError::model(
            "ERA5 adapter does not provide forecasts - reanalysis only",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = ERA5Adapter::new(None, None);
        assert_eq!(adapter.name(), "ERA5");
        assert_eq!(adapter.source_type(), DataSource::ERA5);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = CdsRateLimiter::new(1.0);
        assert_eq!(limiter.max_tokens, 1.0);
    }

    #[test]
    fn test_build_request() {
        let adapter = ERA5Adapter::new(Some("test:key".to_string()), None);
        let request =
            adapter.build_era5_request(&RegionBbox::indonesia(), Utc::now(), Utc::now(), false);
        assert_eq!(request.dataset, "reanalysis-era5-single-levels");
        assert!(request.variables.contains(&"2m_temperature".to_string()));
    }
}
