//! # GFS Forecast Adapter
//!
//! Implementation for NOAA NOMADS GFS (Global Forecast System) data.
//! Rate limit: ~120 hits/min/user (2021 notice). Centralize download on one node.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Timelike, Utc};
use futures::Stream;
use reqwest::Client;
use tokio::sync::{RwLock, Semaphore};
use tracing::warn;

use crate::sources::{ForecastStream, RegionBbox, WeatherSource, WeatherStream};
use wi_core::{DataSource, Result, WIError, WeatherForecast};

/// Rate limiter for NOMADS (120 req/min = 2 req/sec)
struct NomadsRateLimiter {
    tokens: RwLock<u32>,
    max_tokens: u32,
    last_refill: RwLock<Instant>,
}

impl NomadsRateLimiter {
    fn new() -> Self {
        Self {
            tokens: RwLock::new(120),
            max_tokens: 120,
            last_refill: RwLock::new(Instant::now()),
        }
    }

    async fn acquire(&self) {
        loop {
            let mut tokens = self.tokens.write().await;
            let mut last_refill = self.last_refill.write().await;
            let now = Instant::now();
            let elapsed_secs = now.duration_since(*last_refill).as_secs();

            if elapsed_secs >= 60 {
                *tokens = self.max_tokens;
                *last_refill = now;
            }

            if *tokens > 0 {
                *tokens -= 1;
                return;
            }
            drop(tokens);
            drop(last_refill);
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

/// GFS Adapter for NOMADS
pub struct GFSAdapter {
    client: Client,
    base_url: String,
    rate_limiter: Arc<NomadsRateLimiter>,
    semaphore: Arc<Semaphore>,
}

impl GFSAdapter {
    /// Create new GFS adapter
    ///
    /// NOMADS rate limit: ~120 requests per minute per user.
    /// DO NOT split across IPs to bypass - centralize on one node.
    pub fn new(_api_key: Option<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl".to_string(),
            rate_limiter: Arc::new(NomadsRateLimiter::new()),
            semaphore: Arc::new(Semaphore::new(4)),
        }
    }

    /// Build GFS forecast request URL
    fn build_gfs_url(&self, lat: f64, lon: f64, lead_hours: u32, cycle: &str) -> String {
        // NOMADS filter_gfs_0p25.pl parameters:
        // file=gfs.t{cycle}z.pgrb2.0p25.f{lead:03}
        // lev_1000_mb=on, lev_925_mb=on, etc.
        // var_TMP=on, var_RH=on, var_UGRD=on, var_VGRD=on, var_PRATE=on, var_PRES=on
        // subregion=on, leftlon={lon}, rightlon={lon}, toplat={lat}, bottomlat={lat}
        // dir=/gfs.{YYYYMMDD}/{cycle}/atmos

        let now = Utc::now();
        let date_str = now.format("%Y%m%d").to_string();
        let lead_str = format!("{:03}", lead_hours);

        let mut url = format!(
            "{}?file=gfs.t{}z.pgrb2.0p25.f{}&",
            self.base_url, cycle, lead_str
        );

        // Variables
        url.push_str("var_TMP=on&"); // Temperature
        url.push_str("var_RH=on&"); // Relative Humidity
        url.push_str("var_UGRD=on&"); // U wind component
        url.push_str("var_VGRD=on&"); // V wind component
        url.push_str("var_PRATE=on&"); // Precipitation rate
        url.push_str("var_PRES=on&"); // Pressure
        url.push_str("var_TCDC=on&"); // Total cloud cover

        // Levels - surface + pressure levels
        url.push_str("lev_surface=on&");
        url.push_str("lev_1000_mb=on&");
        url.push_str("lev_925_mb=on&");
        url.push_str("lev_850_mb=on&");
        url.push_str("lev_700_mb=on&");
        url.push_str("lev_600_mb=on&");
        url.push_str("lev_500_mb=on&");
        url.push_str("lev_400_mb=on&");
        url.push_str("lev_300_mb=on&");
        url.push_str("lev_250_mb=on&");
        url.push_str("lev_200_mb=on&");
        url.push_str("lev_150_mb=on&");
        url.push_str("lev_100_mb=on&");
        url.push_str("lev_50_mb=on&");

        // Subregion (point query)
        url.push_str(&format!(
            "subregion=on&leftlon={}&rightlon={}&toplat={}&bottomlat={}&",
            lon, lon, lat, lat
        ));

        // Directory
        url.push_str(&format!("dir=/gfs.{}/{}/atmos", date_str, cycle));

        url
    }

    /// Get available GFS cycles (00, 06, 12, 18 UTC)
    fn get_latest_cycle(&self) -> String {
        let now = Utc::now();
        let hour = now.hour();

        // GFS cycles: 00, 06, 12, 18 UTC
        // Data available ~3-4 hours after cycle
        let cycle = if (4..10).contains(&hour) {
            "00"
        } else if (10..16).contains(&hour) {
            "06"
        } else if (16..22).contains(&hour) {
            "12"
        } else {
            "18"
        };

        // If it's early in the cycle, use previous cycle
        if hour < 4 { "18" } else { cycle }.to_string()
    }

    /// Download and parse GRIB2 data
    async fn download_grib2(&self, url: &str) -> Result<Vec<u8>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| WIError::network_with_source("GFS GRIB2 download failed", e.into()))?;

        if !response.status().is_success() {
            return Err(WIError::data_ingestion(format!(
                "GFS download failed: {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| WIError::storage_with_source("Failed to read GRIB2", e.into()))?;

        Ok(bytes.to_vec())
    }

    /// Parse GRIB2 bytes → WeatherForecast menggunakan crate `grib 0.19`
    fn parse_grib2(
        &self,
        bytes: &[u8],
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        let now = chrono::Utc::now();
        let valid_time = now + chrono::Duration::seconds(lead_seconds as i64);
        crate::parsers::parse_grib2_bytes(bytes, lat, lon, lead_seconds, now, valid_time, "gfs")
    }
}

/// Stream of GFS forecasts
struct GfsForecastStream {
    forecasts: Vec<WeatherForecast>,
    index: usize,
    error: Option<String>,
}

impl Stream for GfsForecastStream {
    type Item = Result<WeatherForecast>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(err) = self.error.take() {
            return Poll::Ready(Some(Err(WIError::data_ingestion(&err))));
        }
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
    fn provides_historical(&self) -> bool {
        false
    }
    fn provides_forecasts(&self) -> bool {
        true
    }
    fn rate_limit(&self) -> f64 {
        2.0
    }

    async fn fetch_observations(
        &self,
        _region: RegionBbox,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        Err(WIError::model(
            "GFS adapter does not provide historical observations",
        ))
    }

    async fn fetch_forecast(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        let lead_hours = lead_seconds / 3600;
        let cycle = self.get_latest_cycle();

        let url = self.build_gfs_url(lat, lon, lead_hours, &cycle);

        let grib_bytes = self.download_grib2(&url).await?;
        let forecast = self.parse_grib2(&grib_bytes, lat, lon, lead_seconds)?;

        Ok(forecast)
    }
}

#[async_trait]
impl crate::sources::ForecastSource for GFSAdapter {
    async fn fetch_forecast_stream(
        &self,
        lat: f64,
        lon: f64,
        max_lead_hours: u32,
    ) -> Result<ForecastStream> {
        let cycle = self.get_latest_cycle();
        let mut forecasts = Vec::new();

        // Fetch multiple lead times (every 3 hours up to max)
        for lead_h in (3..=max_lead_hours).step_by(3) {
            let url = self.build_gfs_url(lat, lon, lead_h, &cycle);

            match self.download_grib2(&url).await {
                Ok(grib_bytes) => {
                    if let Ok(forecast) = self.parse_grib2(&grib_bytes, lat, lon, lead_h * 3600) {
                        forecasts.push(forecast);
                    }
                }
                Err(e) => warn!("Failed to fetch GFS lead {}h: {}", lead_h, e),
            }

            // Rate limiting between requests
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        let stream = GfsForecastStream {
            forecasts,
            index: 0,
            error: None,
        };
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = GFSAdapter::new(None);
        assert_eq!(adapter.name(), "GFS");
        assert_eq!(adapter.source_type(), DataSource::GFS);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = NomadsRateLimiter::new();
        assert_eq!(limiter.max_tokens, 120);
    }

    #[test]
    fn test_cycle_selection() {
        let adapter = GFSAdapter::new(None);
        // Test that cycle is one of 00, 06, 12, 18
        let cycle = adapter.get_latest_cycle();
        assert!(["00", "06", "12", "18"].contains(&cycle.as_str()));
    }

    #[test]
    fn test_url_building() {
        let adapter = GFSAdapter::new(None);
        let url = adapter.build_gfs_url(-6.595, 106.806, 24, "00");
        assert!(url.contains("var_TMP=on"));
        assert!(url.contains("lev_surface=on"));
        assert!(url.contains("subregion=on"));
    }
}
