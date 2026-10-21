//! # Weather Data Sources
//!
//! Traits and implementations for weather data sources.

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use wi_core::{Bbox, DataSource, Result, WeatherForecast, WeatherObservation};

/// Bounding box for a region
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegionBbox {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

impl RegionBbox {
    pub fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Self {
        Self {
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        }
    }

    pub fn global() -> Self {
        Self::new(-90.0, -180.0, 90.0, 180.0)
    }

    pub fn indonesia() -> Self {
        Self::new(-11.0, 95.0, 6.1, 141.1)
    }

    pub fn subdivide(&self, step_deg: f64) -> Vec<Self> {
        let mut regions = Vec::new();
        let mut lat = self.min_lat;
        while lat < self.max_lat {
            let mut lon = self.min_lon;
            let next_lat = (lat + step_deg).min(self.max_lat);
            while lon < self.max_lon {
                let next_lon = (lon + step_deg).min(self.max_lon);
                regions.push(Self::new(lat, lon, next_lat, next_lon));
                lon = next_lon;
            }
            lat = next_lat;
        }
        regions
    }

    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat && lon >= self.min_lon && lon <= self.max_lon
    }
}

impl From<RegionBbox> for Bbox {
    fn from(r: RegionBbox) -> Self {
        Bbox::new(r.min_lon, r.min_lat, r.max_lon, r.max_lat)
    }
}

/// Stream of weather observations
pub type WeatherStream = Pin<Box<dyn Stream<Item = Result<WeatherObservation>> + Send + 'static>>;

/// Stream of weather forecasts
pub type ForecastStream = Pin<Box<dyn Stream<Item = Result<WeatherForecast>> + Send + 'static>>;

/// Trait for weather data sources
#[async_trait]
pub trait WeatherSource: Send + Sync {
    fn name(&self) -> &'static str;
    fn source_type(&self) -> DataSource;
    fn provides_forecasts(&self) -> bool {
        false
    }
    fn provides_historical(&self) -> bool {
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
    ) -> Result<WeatherStream>;

    async fn fetch_forecast(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast>;
}

/// Trait for forecast-only sources (like GFS)
#[async_trait]
pub trait ForecastSource: Send + Sync {
    async fn fetch_forecast_stream(
        &self,
        lat: f64,
        lon: f64,
        max_lead_hours: u32,
    ) -> Result<ForecastStream>;
}

/// Registry of weather sources
#[derive(Default)]
pub struct SourceRegistry {
    sources: Vec<Box<dyn WeatherSource>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, source: Box<dyn WeatherSource>) {
        self.sources.push(source);
    }

    pub fn get(&self, name: &str) -> Option<&dyn WeatherSource> {
        self.sources
            .iter()
            .find(|s| s.name() == name)
            .map(|s| s.as_ref())
    }

    pub fn all(&self) -> Vec<&dyn WeatherSource> {
        self.sources.iter().map(|s| s.as_ref()).collect()
    }

    pub fn sorted_by_priority(&self) -> Vec<&dyn WeatherSource> {
        let mut sources = self.all();
        sources.sort_by(|a, b| {
            let pa = a.source_type().reliability_score();
            let pb = b.source_type().reliability_score();
            pb.partial_cmp(&pa).unwrap()
        });
        sources
    }
}

use serde::{Deserialize, Serialize};
