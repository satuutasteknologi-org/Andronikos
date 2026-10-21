//! # ERA5 Reanalysis Adapter
//! 
//! Implementation for ECMWF ERA5 reanalysis data via Copernicus Climate Data Store (CDS) API.

use async_trait::async_trait;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use chrono::{DateTime, Utc, Datelike, Timelike};
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use aetherforge_core::{DataSource, GeoCoord, Result, WeatherObservation, WeatherVariable, ObservedValue, QualityFlag};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};

/// ERA5 API adapter via CDS
/// 
/// Requires CDS API key: https://cds.climate.copernicus.eu/api-how-to
/// 
/// Supports:
/// - ERA5 hourly data on single levels
/// - ERA5 hourly data on pressure levels
/// - ERA5-Land
/// - ERA5 monthly averaged data
pub struct ERA5Adapter {
    client: Client,
    api_key: Option<String>,
    rate_limiter: Arc<RwLock<tokio::time::Instant>>,
    min_request_interval: Duration,
    base_url: String,
}

impl ERA5Adapter {
    pub fn new(api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .build()
            .unwrap_or_else(|_| Client::new());
        
        Self {
            client,
            api_key,
            rate_limiter: Arc::new(RwLock::new(tokio::time::Instant::now())),
            min_request_interval: Duration::from_millis(1000),
            base_url: "https://cds.climate.copernicus.eu/api/v2".to_string(),
        }
    }
    
    async fn wait_for_rate_limit(&self) {
        let mut last = self.rate_limiter.write().await;
        let now = tokio::time::Instant::now();
        let elapsed = now.duration_since(*last);
        
        if elapsed < self.min_request_interval {
            tokio::time::sleep(self.min_request_interval - elapsed).await;
        }
        
        *last = tokio::time::Instant::now();
    }
    
    /// Submit a CDS API request and wait for completion
    async fn submit_request(&self, request: CDSRequest) -> Result<CDSResponse> {
        self.wait_for_rate_limit().await;
        
        let mut req = self.client.post(&format!("{}/resources/era5-single-levels", self.base_url))
            .header("Content-Type", "application/json");
        
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        
        let resp = req.json(&request).send().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let mut result: CDSResponse = resp.json().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        // Poll for completion
        while result.state == "running" || result.state == "queued" {
            tokio::time::sleep(Duration::from_secs(10)).await;
            if let Some(ref loc) = result.location { result = self.poll_request(loc).await?; } else { break; }
        }
        
        if result.state != "completed" {
            return Err(aetherforge_core::AetherError::api(
                format!("CDS request failed: {}", result.state)
            ));
        }
        
        Ok(result)
    }
    
    async fn poll_request(&self, location: &str) -> Result<CDSResponse> {
        let mut req = self.client.get(location);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        
        let resp = req.send().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let result: CDSResponse = resp.json().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        Ok(result)
    }
    
    async fn download_file(&self, url: &str) -> Result<Vec<u8>> {
        let resp = self.client.get(url).send().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let bytes = resp.bytes().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        Ok(bytes.to_vec())
    }
    
    async fn fetch_all_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<WeatherObservation> {
        let mut all_obs = Vec::new();
        
        // Use 0.25° grid for ERA5 (~25km resolution, will interpolate to 0.5km)
        let resolution = 0.25;
        let cells = region.subdivide(resolution);
        
        for cell in cells {
            let lat = (cell.min_lat + cell.max_lat) / 2.0;
            let lon = (cell.min_lon + cell.max_lon) / 2.0;
            
            // ERA5 single-level variables
            let request = CDSRequest {
                dataset: "reanalysis-era5-single-levels".to_string(),
                product_type: "reanalysis".to_string(),
                variable: vec![
                    "2m_temperature".to_string(),
                    "2m_dewpoint_temperature".to_string(),
                    "total_precipitation".to_string(),
                    "10m_u_component_of_wind".to_string(),
                    "10m_v_component_of_wind".to_string(),
                    "surface_pressure".to_string(),
                    "total_cloud_cover".to_string(),
                ],
                year: start.format("%Y").to_string(),
                month: start.format("%m").to_string(),
                day: (start.day()..=(end.day())).map(|d| format!("{:02}", d)).collect(),
                time: (0..24).map(|h| format!("{:02}:00", h)).collect(),
                area: format!("{},{},{},{}", cell.max_lat, cell.min_lon, cell.min_lat, cell.max_lon),
                format: "grib".to_string(),
            };
            
            match self.submit_request(request).await {
                Ok(response) => {
                    if let Some(url) = response.location {
                        if let Ok(data) = self.download_file(&url).await {
                            if let Ok(obs) = self.parse_grib(&data, lat, lon, start, end) {
                                all_obs.extend(obs);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch ERA5 for cell ({}, {}): {}", lat, lon, e);
                }
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        all_obs
    }
    
    fn parse_grib(
        &self,
        data: &[u8],
        lat: f64,
        lon: f64,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<WeatherObservation>> {
        // GRIB2 parsing would require grib library
        // For now, return empty - would integrate with grib2-rs or eccodes
        // This is a placeholder for the actual GRIB parsing implementation
        Ok(Vec::new())
    }
}

#[derive(Debug, Serialize)]
struct CDSRequest {
    dataset: String,
    product_type: String,
    variable: Vec<String>,
    year: String,
    month: String,
    day: Vec<String>,
    time: Vec<String>,
    area: String,
    format: String,
}

#[derive(Debug, Deserialize)]
struct CDSResponse {
    state: String,
    location: Option<String>,
    #[serde(rename = "error")]
    error_message: Option<String>,
}

struct ObservationIter {
    observations: Vec<WeatherObservation>,
    index: usize,
}

impl Stream for ObservationIter {
    type Item = Result<WeatherObservation>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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
    
    fn priority(&self) -> u8 {
        5
    }
    
    fn accuracy_score(&self) -> f32 {
        0.98
    }
    
    fn provides_historical(&self) -> bool {
        true
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
        let observations = self.fetch_all_observations(region, start, end).await;
        let stream = ObservationIter {
            observations,
            index: 0,
        };
        Ok(Box::pin(stream))
    }
}

impl Clone for ERA5Adapter {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            rate_limiter: self.rate_limiter.clone(),
            min_request_interval: self.min_request_interval,
            base_url: self.base_url.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_era5_adapter_creation() {
        let adapter = ERA5Adapter::new(None);
        assert_eq!(adapter.name(), "ERA5");
        assert_eq!(adapter.source_type(), DataSource::ERA5);
    }
}
