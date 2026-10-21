//! # NASA FIRMS Fire Hotspot Adapter
//!
//! Fetches active fire/hotspot data from NASA FIRMS (MODIS and VIIRS).
//! Supports both MODIS (1km) and VIIRS (375m) products.
//!
//! References:
//! - FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/
//! - MAP_KEY: Required for API access (free registration)

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use csv;
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::{RwLock, Semaphore};
use tracing::debug;

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};
use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, WIError, WeatherForecast,
    WeatherObservation, WeatherVariable,
};

/// Rate limiter using token bucket algorithm
struct RateLimiter {
    tokens: RwLock<f64>,
    max_tokens: f64,
    refill_rate: f64,
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

/// FIRMS products available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirmsProduct {
    ModisC61,    // MODIS Collection 6.1 (MCD14DL) - 1km
    ViirsSNPP,   // VIIRS S-NPP - 375m
    ViirsNOAA20, // VIIRS NOAA-20 - 375m
    ViirsNOAA21, // VIIRS NOAA-21 - 375m
}

impl FirmsProduct {
    fn api_name(&self) -> &'static str {
        match self {
            FirmsProduct::ModisC61 => "MODIS_C61",
            FirmsProduct::ViirsSNPP => "VIIRS_SNPP",
            FirmsProduct::ViirsNOAA20 => "VIIRS_NOAA20",
            FirmsProduct::ViirsNOAA21 => "VIIRS_NOAA21",
        }
    }

    #[allow(dead_code)]
    fn resolution_km(&self) -> f64 {
        match self {
            FirmsProduct::ModisC61 => 1.0,
            _ => 0.375,
        }
    }
}

/// Confidence filter levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirmsConfidence {
    Low,     // All confidences
    Nominal, // >= nominal (MODIS >=30, VIIRS nominal/high)
    High,    // High only (MODIS >=80, VIIRS high)
}

/// FIRMS fire hotspot adapter
pub struct FirmsAdapter {
    client: Client,
    map_key: String,
    base_url: String,
    rate_limiter: Arc<RateLimiter>,
    semaphore: Arc<Semaphore>,
    default_confidence: FirmsConfidence,
    default_product: FirmsProduct,
}

impl FirmsAdapter {
    /// Create new FIRMS adapter
    ///
    /// # Arguments
    /// * `map_key` - NASA FIRMS MAP_KEY (free registration at https://firms.modaps.eosdis.nasa.gov/api/)
    /// * `product` - Which satellite product to use
    /// * `confidence` - Minimum confidence level to filter
    pub fn new(
        map_key: String,
        product: FirmsProduct,
        confidence: FirmsConfidence,
    ) -> Result<Self> {
        if map_key.is_empty() {
            return Err(WIError::config("FIRMS MAP_KEY cannot be empty. Register at https://firms.modaps.eosdis.nasa.gov/api/"));
        }

        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|e| {
                    WIError::network_with_source("Failed to create HTTP client", e.into())
                })?,
            map_key,
            base_url: "https://firms.modaps.eosdis.nasa.gov/api/country/csv".to_string(),
            rate_limiter: Arc::new(RateLimiter::new(10.0)), // 10 req/s per FIRMS policy
            semaphore: Arc::new(Semaphore::new(2)),
            default_confidence: confidence,
            default_product: product,
        })
    }

    /// Build country request URL for Indonesia
    fn build_country_url(&self, country: &str, days: u32) -> String {
        format!(
            "{}/{}/{}/{}/{}",
            self.base_url,
            self.map_key,
            self.default_product.api_name(),
            country,
            days
        )
    }

    /// Build area request URL (bbox)
    fn build_area_url(&self, region: &RegionBbox, days: u32) -> String {
        format!(
            "{}/{}/{}/{},{},{},{}/{}",
            self.base_url,
            self.map_key,
            self.default_product.api_name(),
            region.min_lon,
            region.min_lat,
            region.max_lon,
            region.max_lat,
            days
        )
    }

    /// Parse FIRMS CSV response
    fn parse_csv(&self, csv_data: &str) -> Result<Vec<FirmsHotspot>> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let mut hotspots = Vec::new();

        for record in reader.deserialize() {
            let hotspot: FirmsCsvRecord = record.map_err(|e| {
                WIError::serialization_with_source("Failed to parse FIRMS CSV", e.into())
            })?;

            // Filter by confidence
            if !self.passes_confidence_filter(&hotspot) {
                continue;
            }

            // Filter by type (exclude non-vegetation fires)
            if hotspot.r#type == 2 || hotspot.r#type == 3 {
                // Type 2: Non-vegetation fire (gas flares, volcanoes)
                // Type 3: Static land source
                continue;
            }

            hotspots.push(FirmsHotspot {
                latitude: hotspot.latitude,
                longitude: hotspot.longitude,
                brightness: hotspot.brightness,
                scan: hotspot.scan,
                track: hotspot.track,
                acq_date: hotspot.acq_date,
                acq_time: hotspot.acq_time,
                satellite: hotspot.satellite,
                instrument: hotspot.instrument,
                confidence: hotspot.confidence,
                version: hotspot.version,
                bright_t31: hotspot.bright_t31,
                frp: hotspot.frp,
                daynight: hotspot.daynight,
                r#type: hotspot.r#type,
            });
        }

        Ok(hotspots)
    }

    fn passes_confidence_filter(&self, record: &FirmsCsvRecord) -> bool {
        match self.default_confidence {
            FirmsConfidence::Low => true,
            FirmsConfidence::Nominal => {
                // MODIS: confidence >= 30 (nominal)
                // VIIRS: confidence in ["nominal", "high"]
                match self.default_product {
                    FirmsProduct::ModisC61 => record.confidence.parse::<i32>().unwrap_or(0) >= 30,
                    _ => matches!(record.confidence.as_str(), "nominal" | "high"),
                }
            }
            FirmsConfidence::High => match self.default_product {
                FirmsProduct::ModisC61 => record.confidence.parse::<i32>().unwrap_or(0) >= 80,
                _ => record.confidence == "high",
            },
        }
    }

    /// Convert hotspots to WeatherObservation format
    fn hotspots_to_observations(&self, hotspots: Vec<FirmsHotspot>) -> Vec<WeatherObservation> {
        let mut observations = Vec::new();

        for hs in hotspots {
            // Parse acquisition datetime
            let time = match NaiveDateTime::parse_from_str(
                &format!("{} {:04}", hs.acq_date, hs.acq_time),
                "%Y-%m-%d %H%M",
            ) {
                Ok(dt) => DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc),
                Err(_) => continue,
            };

            let mut variables = HashMap::new();

            // Fire Radiative Power (MW) - key indicator of fire intensity
            if hs.frp > 0.0 {
                variables.insert(
                    WeatherVariable::FireRadiativePower,
                    ObservedValue::new(hs.frp as f64, 0.85),
                );
            }

            // Brightness temperature (Kelvin) - fire temperature
            variables.insert(
                WeatherVariable::FireBrightnessTemp,
                ObservedValue::new(hs.brightness as f64, 0.8),
            );

            // Brightness T31 (for MODIS)
            if hs.bright_t31 > 0.0 {
                variables.insert(
                    WeatherVariable::FireBrightnessT31,
                    ObservedValue::new(hs.bright_t31 as f64, 0.8),
                );
            }

            // Confidence as a numeric value
            let conf_val = match self.default_product {
                FirmsProduct::ModisC61 => hs.confidence.parse::<f32>().unwrap_or(0.0) / 100.0,
                _ => match hs.confidence.as_str() {
                    "high" => 0.9,
                    "nominal" => 0.7,
                    "low" => 0.3,
                    _ => 0.5,
                },
            };
            variables.insert(
                WeatherVariable::FireConfidence,
                ObservedValue::new(conf_val as f64, 0.9),
            );

            if !variables.is_empty() {
                let mut coord = GeoCoord::new(hs.latitude, hs.longitude);
                coord.altitude_m = None;

                observations.push(WeatherObservation {
                    station_id: Some(format!(
                        "{}_{}_{}",
                        hs.satellite, hs.instrument, hs.acq_date
                    )),
                    coord,
                    time,
                    variables,
                    quality_flag: QualityFlag::Valid,
                    source: DataSource::FIRMS,
                });
            }
        }

        observations
    }
}

/// FIRMS CSV record (matches FIRMS API output)
#[derive(Debug, Deserialize)]
struct FirmsCsvRecord {
    latitude: f64,
    longitude: f64,
    brightness: f32,
    scan: f32,
    track: f32,
    acq_date: String,
    acq_time: i32,
    satellite: String,
    instrument: String,
    confidence: String,
    version: String,
    bright_t31: f32,
    frp: f32,
    daynight: String,
    r#type: i32,
}

/// Parsed FIRMS hotspot
#[derive(Debug, Clone)]
struct FirmsHotspot {
    latitude: f64,
    longitude: f64,
    brightness: f32,
    scan: f32,
    track: f32,
    acq_date: String,
    acq_time: i32,
    satellite: String,
    instrument: String,
    confidence: String,
    version: String,
    bright_t31: f32,
    frp: f32,
    daynight: String,
    r#type: i32,
}

/// Stream of FIRMS observations
struct FirmsObservationStream {
    observations: Vec<WeatherObservation>,
    index: usize,
    error: Option<String>,
}

impl Stream for FirmsObservationStream {
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
impl WeatherSource for FirmsAdapter {
    fn name(&self) -> &'static str {
        "FIRMS"
    }
    fn source_type(&self) -> DataSource {
        DataSource::FIRMS
    }
    fn provides_historical(&self) -> bool {
        true
    }
    fn provides_forecasts(&self) -> bool {
        false
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
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        // Calculate days to fetch (FIRMS API uses days parameter)
        let days = (end - start).num_days().clamp(1, 10) as u32;

        // Use area request for bounding box
        let url = self.build_area_url(&region, days);

        debug!("Fetching FIRMS data from: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| WIError::network_with_source("FIRMS request failed", e.into()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(WIError::data_ingestion(format!(
                "FIRMS API error {}: {}",
                status, text
            )));
        }

        let csv_data = response
            .text()
            .await
            .map_err(|e| WIError::network_with_source("Failed to read FIRMS response", e.into()))?;

        let hotspots = self.parse_csv(&csv_data)?;
        debug!("Parsed {} FIRMS hotspots", hotspots.len());

        // Filter by time range
        let filtered: Vec<_> = hotspots
            .into_iter()
            .filter(|hs| {
                if let Ok(dt) = NaiveDateTime::parse_from_str(
                    &format!("{} {:04}", hs.acq_date, hs.acq_time),
                    "%Y-%m-%d %H%M",
                ) {
                    let dt = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
                    dt >= start && dt <= end
                } else {
                    false
                }
            })
            .collect();

        let observations = self.hotspots_to_observations(filtered);
        debug!(
            "Total FIRMS observations in time range: {}",
            observations.len()
        );

        let stream = FirmsObservationStream {
            observations,
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
        Err(WIError::model("FIRMS adapter does not provide forecasts"))
    }
}

impl FirmsAdapter {
    /// Create adapter for Indonesia with VIIRS 375m (recommended for peatland monitoring)
    pub fn indonesia_viirs(map_key: String) -> Result<Self> {
        Self::new(map_key, FirmsProduct::ViirsSNPP, FirmsConfidence::Nominal)
    }

    /// Create adapter for Indonesia with MODIS 1km (longer historical record)
    pub fn indonesia_modis(map_key: String) -> Result<Self> {
        Self::new(map_key, FirmsProduct::ModisC61, FirmsConfidence::Nominal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = FirmsAdapter::new(
            "test_key".to_string(),
            FirmsProduct::ModisC61,
            FirmsConfidence::Low,
        )
        .unwrap();
        assert_eq!(adapter.name(), "FIRMS");
        assert_eq!(adapter.source_type(), DataSource::FIRMS);
    }

    #[test]
    fn test_confidence_filter_modis() {
        let adapter = FirmsAdapter::new(
            "test_key".to_string(),
            FirmsProduct::ModisC61,
            FirmsConfidence::Nominal,
        )
        .unwrap();

        let record = FirmsCsvRecord {
            latitude: 0.0,
            longitude: 0.0,
            brightness: 300.0,
            scan: 1.0,
            track: 1.0,
            acq_date: "2024-01-01".to_string(),
            acq_time: 1200,
            satellite: "Terra".to_string(),
            instrument: "MODIS".to_string(),
            confidence: "80".to_string(),
            version: "1.0".to_string(),
            bright_t31: 290.0,
            frp: 50.0,
            daynight: "D".to_string(),
            r#type: 0,
        };

        assert!(adapter.passes_confidence_filter(&record));

        let record_low = FirmsCsvRecord {
            confidence: "20".to_string(),
            ..record
        };
        assert!(!adapter.passes_confidence_filter(&record_low));
    }

    #[test]
    fn test_confidence_filter_viirs() {
        let adapter = FirmsAdapter::new(
            "test_key".to_string(),
            FirmsProduct::ViirsSNPP,
            FirmsConfidence::Nominal,
        )
        .unwrap();

        let record = FirmsCsvRecord {
            latitude: 0.0,
            longitude: 0.0,
            brightness: 300.0,
            scan: 1.0,
            track: 1.0,
            acq_date: "2024-01-01".to_string(),
            acq_time: 1200,
            satellite: "SNPP".to_string(),
            instrument: "VIIRS".to_string(),
            confidence: "nominal".to_string(),
            version: "1.0".to_string(),
            bright_t31: 290.0,
            frp: 50.0,
            daynight: "D".to_string(),
            r#type: 0,
        };

        assert!(adapter.passes_confidence_filter(&record));

        let record_low = FirmsCsvRecord {
            confidence: "low".to_string(),
            ..record
        };
        assert!(!adapter.passes_confidence_filter(&record_low));
    }
}
