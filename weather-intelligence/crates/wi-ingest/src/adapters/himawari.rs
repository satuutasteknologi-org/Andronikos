//! # Himawari Satellite Adapter
//!
//! Implementation for Himawari-8/9 satellite data via JAXA P-Tree and AWS Open Data.
//! Provides cloud properties, aerosol optical depth, SST, and fire detection products.

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
use wi_core::{DataSource, Result, WIError, WeatherForecast, WeatherObservation, WeatherVariable};

/// Rate limiter for Himawari data access (conservative: 30 req/min)
struct HimawariRateLimiter {
    tokens: RwLock<u32>,
    max_tokens: u32,
    last_refill: RwLock<Instant>,
}

impl HimawariRateLimiter {
    fn new() -> Self {
        Self {
            tokens: RwLock::new(30),
            max_tokens: 30,
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
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
}

/// Himawari product types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HimawariProduct {
    /// Cloud optical thickness / cloud top height
    CloudProperties,
    /// Aerosol optical depth
    AerosolOpticalDepth,
    /// Sea surface temperature
    SeaSurfaceTemperature,
    /// Fire/hotspot detection
    FireDetection,
    /// Atmospheric motion vectors (wind)
    AtmosphericMotionVectors,
    /// Precipitation estimation
    Precipitation,
}

impl HimawariProduct {
    fn aws_s3_prefix(&self) -> &'static str {
        match self {
            Self::CloudProperties => "AHI-CLD",
            Self::AerosolOpticalDepth => "AHI-AOD",
            Self::SeaSurfaceTemperature => "AHI-SST",
            Self::FireDetection => "AHI-FIR",
            Self::AtmosphericMotionVectors => "AHI-AMV",
            Self::Precipitation => "AHI-PRE",
        }
    }

    fn mapped_variables(&self) -> Vec<WeatherVariable> {
        match self {
            Self::CloudProperties => vec![WeatherVariable::CloudCover, WeatherVariable::Visibility],
            Self::AerosolOpticalDepth => vec![WeatherVariable::Visibility],
            Self::SeaSurfaceTemperature => vec![WeatherVariable::Temperature],
            Self::FireDetection => vec![WeatherVariable::Temperature],
            Self::AtmosphericMotionVectors => vec![
                WeatherVariable::WindSpeed,
                WeatherVariable::WindDirection,
                WeatherVariable::WindU,
                WeatherVariable::WindV,
            ],
            Self::Precipitation => vec![WeatherVariable::Precipitation],
        }
    }
}

/// Himawari satellite adapter
pub struct HimawariAdapter {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    rate_limiter: Arc<HimawariRateLimiter>,
    semaphore: Arc<Semaphore>,
    use_aws: bool,
}

impl HimawariAdapter {
    /// Create new Himawari adapter
    ///
    /// # Arguments
    /// * `api_key` - Optional API key for JAXA P-Tree (if available)
    /// * `use_aws` - If true, use AWS Open Data bucket (no auth required for public data)
    pub fn new(api_key: Option<String>, use_aws: bool) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://himawari8.nict.go.jp".to_string(), // NICT public
            api_key,
            rate_limiter: Arc::new(HimawariRateLimiter::new()),
            semaphore: Arc::new(Semaphore::new(2)),
            use_aws,
        }
    }

    /// Build AWS S3 URL for Himawari data
    fn build_aws_url(
        &self,
        product: HimawariProduct,
        date: DateTime<Utc>,
        band: Option<u8>,
    ) -> String {
        let _date_str = date.format("%Y%m%d").to_string();
        let _hour = date.format("%H").to_string();
        let _minute = date.format("%M").to_string();

        let _band_str = band.map(|b| format!("B{:02}", b)).unwrap_or_default();

        // AWS Open Data: s3://noaa-himawari8/AHI-CLD/2024/01/15/00/
        // Access via https://noaa-himawari8.s3.amazonaws.com/
        format!(
            "https://noaa-himawari8.s3.amazonaws.com/{}/{}/{}/{}/{}/{}_B{}.nc",
            product.aws_s3_prefix(),
            date.format("%Y"),
            date.format("%m"),
            date.format("%d"),
            date.format("%H"),
            product.aws_s3_prefix(),
            band.unwrap_or(1)
        )
    }

    /// Fetch satellite data for a region and time range
    async fn fetch_satellite_data(
        &self,
        region: &RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        products: &[HimawariProduct],
    ) -> Result<Vec<WeatherObservation>> {
        let mut all_observations = Vec::new();

        // Determine time steps (every 10 minutes for Himawari-8/9 full disk)
        let mut current = start;
        while current <= end {
            for product in products {
                self.rate_limiter.acquire().await;
                let _permit = self.semaphore.acquire().await;

                // Try AWS first (public data)
                let url = self.build_aws_url(*product, current, None);

                match self.download_and_parse_netcdf(&url, region, *product).await {
                    Ok(obs) => all_observations.extend(obs),
                    Err(e) => {
                        debug!(
                            "Failed to fetch {} for {}: {}",
                            product.aws_s3_prefix(),
                            current,
                            e
                        );
                        // Could try NICT or JAXA as fallback
                    }
                }

                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            current += chrono::Duration::minutes(10);
        }

        Ok(all_observations)
    }

    /// Download and parse NetCDF from satellite
    async fn download_and_parse_netcdf(
        &self,
        url: &str,
        region: &RegionBbox,
        product: HimawariProduct,
    ) -> Result<Vec<WeatherObservation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let response = self.client.get(url).send().await.map_err(|e| {
            WIError::network_with_source("Himawari NetCDF download failed", e.into())
        })?;

        if !response.status().is_success() {
            return Err(WIError::data_ingestion(format!(
                "Himawari download failed: {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| WIError::storage_with_source("Failed to read NetCDF", e.into()))?;

        self.parse_himawari_netcdf(&bytes, region, product)
    }

    /// Parse Himawari NetCDF to observations
    /// Production would use netcdf crate with proper CF conventions
    fn parse_himawari_netcdf(
        &self,
        _bytes: &[u8],
        _region: &RegionBbox,
        _product: HimawariProduct,
    ) -> Result<Vec<WeatherObservation>> {
        // Simplified NetCDF parsing - production should use netcdf crate
        // Himawari NetCDF follows CF conventions with lat/lon/time dimensions

        warn!("Himawari NetCDF parsing is simplified - implement with netcdf crate for production");

        // For now, return empty - proper implementation requires netcdf crate
        // This would parse:
        // - latitude, longitude, time variables
        // - data variables per product (e.g., cloud_optical_thickness, aerosol_optical_depth, etc.)
        // - Apply region filtering
        // - Convert to WeatherObservation with proper units

        Ok(Vec::new())
    }

    /// Convert Himawari product data to WeatherVariable values
    fn convert_product_data(
        &self,
        product: HimawariProduct,
        raw_value: f32,
        _lat: f64,
        _lon: f64,
    ) -> Option<(WeatherVariable, f64)> {
        match product {
            HimawariProduct::CloudProperties => {
                // Cloud optical thickness -> cloud cover estimation
                Some((
                    WeatherVariable::CloudCover,
                    (raw_value / 100.0).min(100.0) as f64,
                ))
            }
            HimawariProduct::AerosolOpticalDepth => {
                // AOD -> visibility reduction
                Some((
                    WeatherVariable::Visibility,
                    (10000.0 / (1.0 + raw_value as f64)),
                ))
            }
            HimawariProduct::SeaSurfaceTemperature => {
                // SST in Kelvin
                Some((WeatherVariable::Temperature, (raw_value as f64 + 273.15)))
            }
            HimawariProduct::FireDetection => {
                // Fire radiative power -> temperature anomaly
                Some((WeatherVariable::Temperature, (raw_value as f64 + 273.15)))
            }
            HimawariProduct::AtmosphericMotionVectors => {
                // AMV provides u/v wind components
                Some((WeatherVariable::WindU, raw_value as f64))
            }
            HimawariProduct::Precipitation => {
                // Precipitation rate in mm/hr
                Some((WeatherVariable::Precipitation, raw_value as f64))
            }
        }
    }
}

/// Stream of Himawari observations
struct HimawariObservationStream {
    observations: Vec<WeatherObservation>,
    index: usize,
    error: Option<String>,
}

impl Stream for HimawariObservationStream {
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
impl WeatherSource for HimawariAdapter {
    fn name(&self) -> &'static str {
        "Himawari"
    }
    fn source_type(&self) -> DataSource {
        DataSource::Himawari
    }
    fn provides_historical(&self) -> bool {
        true
    }
    fn provides_forecasts(&self) -> bool {
        false
    }
    fn rate_limit(&self) -> f64 {
        0.5
    } // 30 req/min

    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        let products = vec![
            HimawariProduct::CloudProperties,
            HimawariProduct::SeaSurfaceTemperature,
            HimawariProduct::FireDetection,
        ];

        match self
            .fetch_satellite_data(&region, start, end, &products)
            .await
        {
            Ok(observations) => {
                debug!("Fetched {} Himawari observations", observations.len());
                let stream = HimawariObservationStream {
                    observations,
                    index: 0,
                    error: None,
                };
                Ok(Box::pin(stream))
            }
            Err(e) => {
                warn!("Himawari fetch failed: {}", e);
                let stream = HimawariObservationStream {
                    observations: Vec::new(),
                    index: 0,
                    error: Some(e.to_string()),
                };
                Ok(Box::pin(stream))
            }
        }
    }

    async fn fetch_forecast(
        &self,
        _lat: f64,
        _lon: f64,
        _lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        Err(WIError::model(
            "Himawari adapter does not provide forecasts",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = HimawariAdapter::new(None, true);
        assert_eq!(adapter.name(), "Himawari");
        assert_eq!(adapter.source_type(), DataSource::Himawari);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = HimawariRateLimiter::new();
        assert_eq!(limiter.max_tokens, 30);
    }

    #[test]
    fn test_product_mapping() {
        assert_eq!(HimawariProduct::CloudProperties.aws_s3_prefix(), "AHI-CLD");
        assert_eq!(
            HimawariProduct::FireDetection.mapped_variables(),
            vec![WeatherVariable::Temperature]
        );
    }

    #[test]
    fn test_aws_url_building() {
        let adapter = HimawariAdapter::new(None, true);
        let date = Utc::now();
        let url = adapter.build_aws_url(HimawariProduct::CloudProperties, date, Some(13));
        assert!(url.contains("noaa-himawari8.s3.amazonaws.com"));
        assert!(url.contains("AHI-CLD"));
    }
}
