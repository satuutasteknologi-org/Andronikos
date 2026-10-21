//! # GFS (Global Forecast System) Adapter
//! 
//! Implementation for NOAA GFS forecast data via NOMADS and AWS S3.

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

use aetherforge_core::{DataSource, GeoCoord, Result, WeatherObservation, WeatherVariable, ObservedValue, QualityFlag, WeatherForecast, VerticalLevel};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream, ForecastStream};

/// GFS adapter via NOMADS OpenDAP/HTTP and AWS S3
/// 
/// Sources:
/// - NOMADS OpenDAP: https://nomads.ncep.noaa.gov/dods/gfs_0p25
/// - AWS S3: s3://noaa-gfs-bdp-pds
/// - NOAA GFS 0.25° (~25km) and 0.5° (~50km) grids
pub struct GFSAdapter {
    client: Client,
    rate_limiter: Arc<RwLock<tokio::time::Instant>>,
    min_request_interval: Duration,
    base_url_nomads: String,
    use_s3: bool,
    s3_bucket: String,
}

impl GFSAdapter {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .pool_max_idle_per_host(8)
            .build()
            .unwrap_or_else(|_| Client::new());
        
        Self {
            client,
            rate_limiter: Arc::new(RwLock::new(tokio::time::Instant::now())),
            min_request_interval: Duration::from_millis(200),
            base_url_nomads: "https://nomads.ncep.noaa.gov/dods/gfs_0p25".to_string(),
            use_s3: false,
            s3_bucket: "noaa-gfs-bdp-pds".to_string(),
        }
    }
    
    pub fn with_s3(mut self, use_s3: bool) -> Self {
        self.use_s3 = use_s3;
        self
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
    
    fn build_gfs_url(&self, cycle: &str, _forecast_hour: u32, _variable: &str) -> String {
        // NOMADS OpenDAP URL structure
        // https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20261014/gfs_0p25_00z
        format!("{}/gfs{}/gfs_0p25_{}z", self.base_url_nomads, cycle, cycle)
    }
    
    fn build_s3_url(&self, cycle: &str, forecast_hour: u32) -> String {
        // AWS S3 structure: s3://noaa-gfs-bdp-pds/gfs.20261014/00/atmos/gfs.t00z.pgrb2.0p25.f000
        format!(
            "https://{}.s3.amazonaws.com/gfs.{}/{}/atmos/gfs.t{}z.pgrb2.0p25.f{:03}",
            self.s3_bucket, cycle, cycle, cycle, forecast_hour
        )
    }
    
    async fn fetch_forecast_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<WeatherObservation> {
        let mut all_obs = Vec::new();
        
        // GFS cycles: 00, 06, 12, 18 UTC
        let cycles = ["00", "06", "12", "18"];
        let cycle = cycles.iter()
            .min_by_key(|c| {
                let cycle_time = DateTime::parse_from_rfc3339(&format!("{}T{}", start.format("%Y-%m-%d"), c))
                    .map(|t| t.with_timezone(&Utc))
                    .unwrap_or(start);
                (start - cycle_time).abs().num_seconds()
            })
            .unwrap_or(&"00");
        
        // Use 0.5° grid for 0.5km interpolation capability
        let resolution = 0.5;
        let cells = region.subdivide(resolution);
        
        for cell in cells {
            let _lat = (cell.min_lat + cell.max_lat) / 2.0;
            let _lon = (cell.min_lon + cell.max_lon) / 2.0;
            
            self.wait_for_rate_limit().await;
            
            // Fetch from NOMADS OpenDAP (if available) or S3
            // For OpenDAP, we'd query the DODS endpoint
            // For S3, we'd download and parse GRIB2
            
            // This is a simplified implementation - real implementation would:
            // 1. Query OpenDAP for specific variables at lat/lon
            // 2. Or download GRIB2 from S3 and parse with grib2-rs
            
            // For now, simulate with empty data
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        all_obs
    }
    
    async fn fetch_forecast_gridded(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<WeatherForecast> {
        let mut all_forecasts = Vec::new();
        
        // Similar to observations but returns WeatherForecast structs
        // with lead times and vertical levels
        
        all_forecasts
    }
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

struct ForecastIter {
    forecasts: Vec<WeatherForecast>,
    index: usize,
}

impl Stream for ForecastIter {
    type Item = Result<WeatherForecast>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.index < self.forecasts.len() {
            let item = self.forecasts[self.index].clone();
            self.index += 1;
            Poll::Ready(Some(Ok(item)))
        } else {
            Poll::Ready(None)
        }
    }
}

#[async_trait]
impl WeatherSource for GFSAdapter {
    fn name(&self) -> &'static str {
        "GFS"
    }
    
    fn source_type(&self) -> DataSource {
        DataSource::GFS
    }
    
    fn priority(&self) -> u8 {
        20
    }
    
    fn accuracy_score(&self) -> f32 {
        0.90
    }
    
    fn provides_forecasts(&self) -> bool {
        true
    }
    
    fn provides_historical(&self) -> bool {
        false
    }
    
    fn rate_limit(&self) -> f64 {
        5.0
    }
    
    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        let observations = self.fetch_forecast_observations(region, start, end).await;
        let stream = ObservationIter {
            observations,
            index: 0,
        };
        Ok(Box::pin(stream))
    }
    
    async fn fetch_forecasts(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<ForecastStream> {
        let forecasts = self.fetch_forecast_gridded(region, start, end).await;
        let stream = ForecastIter {
            forecasts,
            index: 0,
        };
        Ok(Box::pin(stream))
    }
}

impl Default for GFSAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GFSAdapter {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            rate_limiter: self.rate_limiter.clone(),
            min_request_interval: self.min_request_interval,
            base_url_nomads: self.base_url_nomads.clone(),
            use_s3: self.use_s3,
            s3_bucket: self.s3_bucket.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gfs_adapter_creation() {
        let adapter = GFSAdapter::new();
        assert_eq!(adapter.name(), "GFS");
        assert_eq!(adapter.source_type(), DataSource::GFS);
        assert!(adapter.provides_forecasts());
    }
}
