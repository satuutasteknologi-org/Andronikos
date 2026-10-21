//! # Data Source Abstraction
//! 
//! Core trait for implementing weather data sources with incremental fetching capability.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use futures::Stream;
use std::pin::Pin;

use aetherforge_core::{
    DataSource, GeoCoord, Result, WeatherObservation, WeatherForecast,
};

/// Stream of weather data
pub type WeatherStream = Pin<Box<dyn Stream<Item = Result<WeatherObservation>> + Send>>;

/// Forecast stream
pub type ForecastStream = Pin<Box<dyn Stream<Item = Result<WeatherForecast>> + Send>>;

/// Bounding box for region of interest
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionBbox {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

impl RegionBbox {
    pub fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Self {
        Self { min_lat, min_lon, max_lat, max_lon }
    }
    
    pub fn global() -> Self {
        Self {
            min_lat: -90.0,
            min_lon: -180.0,
            max_lat: 90.0,
            max_lon: 180.0,
        }
    }
    
    pub fn contains(&self, coord: GeoCoord) -> bool {
        coord.lat >= self.min_lat && coord.lat <= self.max_lat &&
        coord.lon >= self.min_lon && coord.lon <= self.max_lon
    }
    
    /// Subdivide region into grid cells
    pub fn subdivide(&self, resolution_deg: f64) -> Vec<RegionBbox> {
        let mut cells = Vec::new();
        
        let mut lat = self.min_lat;
        while lat < self.max_lat {
            let mut lon = self.min_lon;
            while lon < self.max_lon {
                cells.push(RegionBbox {
                    min_lat: lat,
                    min_lon: lon,
                    max_lat: (lat + resolution_deg).min(self.max_lat),
                    max_lon: (lon + resolution_deg).min(self.max_lon),
                });
                lon += resolution_deg;
            }
            lat += resolution_deg;
        }
        
        cells
    }
}

/// Core trait for weather data sources
/// 
/// Implement this trait to add support for a new weather data provider.
/// Each source must provide:
/// - Incremental fetching capability
/// - Priority and accuracy scoring
/// - Rate limiting compliance
#[async_trait]
pub trait WeatherSource: Send + Sync {
    /// Human-readable name
    fn name(&self) -> &'static str;
    
    /// Source type identifier
    fn source_type(&self) -> DataSource;
    
    /// Priority (1 = highest, higher numbers = lower priority)
    fn priority(&self) -> u8 {
        50
    }
    
    /// Estimated accuracy score (0.0 - 1.0)
    fn accuracy_score(&self) -> f32 {
        0.8
    }
    
    /// Whether this source provides forecast data
    fn provides_forecasts(&self) -> bool {
        false
    }
    
    /// Whether this source provides reanalysis/historical data
    fn provides_historical(&self) -> bool {
        false
    }
    
    /// Fetch observations incrementally for a region and time range
    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream>;
    
    /// Fetch forecasts for a region and time range
    async fn fetch_forecasts(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<ForecastStream> {
        let _ = (region, start, end);
        Ok(Box::pin(futures::stream::empty()))
    }
    
    /// Check if source is available
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
    
    /// Get rate limit (requests per second)
    fn rate_limit(&self) -> f64 {
        10.0
    }
    
    /// Generate batch time ranges for efficient chunked fetching
    fn batch_intervals(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        max_hours_per_batch: i64,
    ) -> Vec<(DateTime<Utc>, DateTime<Utc>)> {
        let mut intervals = Vec::new();
        let mut current = start;
        
        while current < end {
            let next = (current + Duration::hours(max_hours_per_batch)).min(end);
            intervals.push((current, next));
            current = next;
        }
        
        intervals
    }
}

/// Source configuration
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub name: String,
    pub enabled: bool,
    pub priority: u8,
    pub accuracy_score: f32,
    pub rate_limit: f64,
    pub api_key: Option<String>,
    pub base_url: String,
    pub extra_params: std::collections::HashMap<String, String>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            enabled: true,
            priority: 50,
            accuracy_score: 0.8,
            rate_limit: 10.0,
            api_key: None,
            base_url: String::new(),
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// Source registry for managing multiple sources
pub struct SourceRegistry {
    sources: std::collections::HashMap<DataSource, Box<dyn WeatherSource>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self {
            sources: std::collections::HashMap::new(),
        }
    }
    
    pub fn register<S: WeatherSource + 'static>(&mut self, source: S) {
        self.sources.insert(source.source_type(), Box::new(source));
    }
    
    pub fn get(&self, source_type: DataSource) -> Option<&dyn WeatherSource> {
        self.sources.get(&source_type).map(|b| b.as_ref())
    }
    
    pub fn all(&self) -> Vec<&dyn WeatherSource> {
        self.sources.values().map(|b| b.as_ref()).collect()
    }
    
    pub fn sorted_by_priority(&self) -> Vec<&dyn WeatherSource> {
        let mut sources: Vec<_> = self.all();
        sources.sort_by_key(|s| s.priority());
        sources
    }
    
    pub fn enabled(&self) -> Vec<&dyn WeatherSource> {
        self.all()
    }
}

impl Default for SourceRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_region_bbox_global() {
        let global = RegionBbox::global();
        assert!(global.contains(GeoCoord::new(52.52, 13.405)));
        assert!(global.contains(GeoCoord::new(-33.87, 151.21)));
    }
    
    #[test]
    fn test_subdivision() {
        let region = RegionBbox::new(0.0, 0.0, 10.0, 10.0);
        let cells = region.subdivide(5.0);
        assert_eq!(cells.len(), 4);
    }
}