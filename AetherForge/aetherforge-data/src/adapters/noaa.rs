//! # NOAA API Adapter
//! 
//! Implementation for NOAA climate data APIs (ISD, ISD-Lite, GHCN-Daily, NCEI).

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

/// NOAA API adapter
/// 
/// Supports:
/// - Integrated Surface Database (ISD) - hourly station data
/// - ISD-Lite - simplified hourly data
/// - Global Historical Climatology Network Daily (GHCN-Daily)
/// - NCEI Climate Data Online (CDO) API
/// - Automated Surface Observing System (ASOS)
pub struct NOAAAdapter {
    client: Client,
    api_token: Option<String>,
    rate_limiter: Arc<RwLock<tokio::time::Instant>>,
    min_request_interval: Duration,
    use_isd_lite: bool,
}

impl NOAAAdapter {
    pub fn new(api_token: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(8)
            .build()
            .unwrap_or_else(|_| Client::new());
        
        Self {
            client,
            api_token,
            rate_limiter: Arc::new(RwLock::new(tokio::time::Instant::now())),
            min_request_interval: Duration::from_millis(200),
            use_isd_lite: true,
        }
    }
    
    pub fn with_isd_lite(mut self, use_isd_lite: bool) -> Self {
        self.use_isd_lite = use_isd_lite;
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
    
    /// Fetch station metadata from ISD
    async fn fetch_isd_stations(&self, region: RegionBbox) -> Result<Vec<ISDStation>> {
        // ISD station inventory is available at:
        // https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
        let url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv";
        
        let resp = self.client.get(url).send().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let text = resp.text().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let mut stations = Vec::new();
        
        for line in text.lines().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 10 {
                let lat = parts[6].parse::<f64>().unwrap_or(0.0);
                let lon = parts[7].parse::<f64>().unwrap_or(0.0);
                
                if region.contains(GeoCoord::new(lat, lon)) {
                    stations.push(ISDStation {
                        usaf: parts[0].to_string(),
                        wban: parts[1].to_string(),
                        name: parts[2].to_string(),
                        country: parts[3].to_string(),
                        state: parts[4].to_string(),
                        icao: parts[5].to_string(),
                        lat,
                        lon,
                        elevation: parts[8].parse::<f64>().unwrap_or(0.0),
                        begin: parts[9].to_string(),
                        end: parts[10].to_string(),
                    });
                }
            }
        }
        
        Ok(stations)
    }
    
    /// Fetch ISD-Lite hourly data for a station
    async fn fetch_isd_lite_data(
        &self,
        station: &ISDStation,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<WeatherObservation>> {
        let mut all_obs = Vec::new();
        
        // ISD-Lite files: https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{USAF}-{WBAN}-{year}.gz
        let start_year = start.year();
        let end_year = end.year();
        
        for year in start_year..=end_year {
            let url = format!(
                "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{}/{:06}-{:05}-{}.gz",
                year, station.usaf, station.wban, year
            );
            
            self.wait_for_rate_limit().await;
            
            let resp = self.client.get(&url).send().await;
            
            if let Ok(response) = resp {
                if response.status().is_success() {
                    let bytes = response.bytes().await
                        .map_err(|e| aetherforge_core::AetherError::from(e))?;
                    
                    // Decompress gzip
                    use flate2::read::GzDecoder;
                    use std::io::Read;
                    let mut decoder = GzDecoder::new(&bytes[..]);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed)
                        .map_err(|e| aetherforge_core::AetherError::api(e.to_string()))?;
                    
                    let text = String::from_utf8_lossy(&decompressed);
                    
                    for line in text.lines() {
                        if let Some(obs) = self.parse_isd_lite_line(line, station, year) {
                            if obs.time >= start && obs.time <= end {
                                all_obs.push(obs);
                            }
                        }
                    }
                }
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        Ok(all_obs)
    }
    
    fn parse_isd_lite_line(&self, line: &str, station: &ISDStation, year: i32) -> Option<WeatherObservation> {
        // ISD-Lite format: YEAR MONTH DAY HOUR MINUTE TEMP DEWPOINT PRESSURE WIND_DIR WIND_SPEED CLOUD_CLOUD CEILING PRECIP_1HR PRECIP_6HR
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 14 {
            return None;
        }
        
        let month = parts[1].parse::<u32>().ok()?;
        let day = parts[2].parse::<u32>().ok()?;
        let hour = parts[3].parse::<u32>().ok()?;
        
        let time = DateTime::parse_from_rfc3339(&format!("{}-{:02}-{:02}T{:02}:00:00Z", year, month, day, hour))
            .ok()?.with_timezone(&Utc);
        
        let mut variables = HashMap::new();
        
        // Temperature (tenths of Celsius)
        if let Ok(t) = parts[5].parse::<i16>() {
            if t != -9999 {
                variables.insert(
                    WeatherVariable::Temperature,
                    ObservedValue::new(t as f64 / 10.0 + 273.15, 0.95),
                );
            }
        }
        
        // Dewpoint (tenths of Celsius)
        if let Ok(d) = parts[6].parse::<i16>() {
            if d != -9999 {
                variables.insert(
                    WeatherVariable::DewPoint,
                    ObservedValue::new(d as f64 / 10.0 + 273.15, 0.93),
                );
            }
        }
        
        // Pressure (hPa * 10)
        if let Ok(p) = parts[7].parse::<i16>() {
            if p != -9999 {
                variables.insert(
                    WeatherVariable::Pressure,
                    ObservedValue::new(p as f64 * 10.0, 0.95),
                );
            }
        }
        
        // Wind direction (degrees)
        if let Ok(wd) = parts[8].parse::<i16>() {
            if wd != -9999 {
                variables.insert(
                    WeatherVariable::WindDirection,
                    ObservedValue::new(wd as f64, 0.90),
                );
            }
        }
        
        // Wind speed (m/s * 10)
        if let Ok(ws) = parts[9].parse::<i16>() {
            if ws != -9999 {
                variables.insert(
                    WeatherVariable::WindSpeed,
                    ObservedValue::new(ws as f64 / 10.0, 0.90),
                );
            }
        }
        
        // Cloud cover (oktas)
        if let Ok(cc) = parts[10].parse::<i16>() {
            if cc != -9999 && cc <= 8 {
                variables.insert(
                    WeatherVariable::CloudCover,
                    ObservedValue::new(cc as f64 * 12.5, 0.85),
                );
            }
        }
        
        // 1-hour precipitation (mm * 10)
        if let Ok(p1) = parts[12].parse::<i16>() {
            if p1 != -9999 {
                variables.insert(
                    WeatherVariable::Precipitation,
                    ObservedValue::new(p1 as f64 / 10.0, 0.85),
                );
            }
        }
        
        if variables.is_empty() {
            return None;
        }
        
        Some(WeatherObservation {
            station_id: Some(format!("{}-{}", station.usaf, station.wban)),
            coord: GeoCoord::with_altitude(station.lat, station.lon, station.elevation),
            time,
            variables,
            quality_flag: QualityFlag::Valid,
            source: DataSource::NOAA,
        })
    }
    
    /// Fetch GHCN-Daily data
    async fn fetch_ghcn_daily(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<WeatherObservation>> {
        let mut all_obs = Vec::new();
        
        // GHCN-Daily stations
        let stations_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt";
        
        let resp = self.client.get(stations_url).send().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let text = resp.text().await
            .map_err(|e| aetherforge_core::AetherError::from(e))?;
        
        let mut stations = Vec::new();
        
        for line in text.lines() {
            if line.len() >= 38 {
                let id = line[0..11].trim().to_string();
                let lat = line[12..20].trim().parse::<f64>().unwrap_or(0.0);
                let lon = line[21..30].trim().parse::<f64>().unwrap_or(0.0);
                let elevation = line[31..37].trim().parse::<f64>().unwrap_or(0.0);
                
                if region.contains(GeoCoord::new(lat, lon)) {
                    stations.push(GHCNStation { id, lat, lon, elevation });
                }
            }
        }
        
        // GHCN-Daily data files are per-station
        // https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{ID}.dly
        for station in stations.iter().take(50) {
            self.wait_for_rate_limit().await;
            
            let url = format!("https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{}.dly", station.id);
            
            let resp = self.client.get(&url).send().await;
            
            if let Ok(response) = resp {
                if response.status().is_success() {
                    let text = response.text().await
                        .map_err(|e| aetherforge_core::AetherError::from(e))?;
                    
                    for line in text.lines() {
                        if let Some(obs) = self.parse_ghcn_daily_line(line, station, start, end) {
                            all_obs.push(obs);
                        }
                    }
                }
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        Ok(all_obs)
    }
    
    fn parse_ghcn_daily_line(
        &self,
        line: &str,
        station: &GHCNStation,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Option<WeatherObservation> {
        if line.len() < 21 {
            return None;
        }
        
        let id = &line[0..11];
        let year = line[11..15].parse::<i32>().ok()?;
        let month = line[15..17].parse::<u32>().ok()?;
        let element = &line[17..21];
        
        // Parse daily values (each value is 5 chars + 1 char flag + 1 char flag + 1 char flag)
        let mut values = Vec::new();
        let mut pos = 21;
        
        for day in 1..=31 {
            if pos + 8 <= line.len() {
                let value_str = &line[pos..pos+5];
                let mflag = &line[pos+5..pos+6];
                let qflag = &line[pos+6..pos+7];
                let sflag = &line[pos+7..pos+8];
                
                if let Ok(v) = value_str.parse::<i32>() {
                    if v != -9999 && qflag == " " && mflag == " " {
                        values.push((day, v));
                    }
                }
                pos += 8;
            }
        }
        
        if values.is_empty() {
            return None;
        }
        
        let mut variables = HashMap::new();
        let mut last_time: Option<DateTime<Utc>> = None;
        
        for (day, value) in values {
            let time = DateTime::parse_from_rfc3339(&format!("{}-{:02}-{:02}T12:00:00Z", year, month, day))
                .ok()?.with_timezone(&Utc);
            
            last_time = Some(time);
            
            if time < start || time > end {
                continue;
            }
            
            match element {
                "TMAX" => {
                    variables.insert(
                        WeatherVariable::Temperature,
                        ObservedValue::new(value as f64 / 10.0 + 273.15, 0.95),
                    );
                }
                "TMIN" => {
                    variables.insert(
                        WeatherVariable::Temperature,
                        ObservedValue::new(value as f64 / 10.0 + 273.15, 0.95),
                    );
                }
                "PRCP" => {
                    variables.insert(
                        WeatherVariable::Precipitation,
                        ObservedValue::new(value as f64 / 10.0, 0.90),
                    );
                }
                "SNOW" => {
                    // Snow depth in mm
                }
                "SNWD" => {
                    // Snow water equivalent
                }
                "AWND" => {
                    variables.insert(
                        WeatherVariable::WindSpeed,
                        ObservedValue::new(value as f64 / 10.0, 0.90),
                    );
                }
                _ => {}
            }
        }
        
        if variables.is_empty() {
            return None;
        }
        
        Some(WeatherObservation {
            station_id: Some(station.id.clone()),
            coord: GeoCoord::with_altitude(station.lat, station.lon, station.elevation),
            time: last_time.unwrap_or_else(|| Utc::now()),
            variables,
            quality_flag: QualityFlag::Valid,
            source: DataSource::NOAA,
        })
    }
    
    async fn fetch_all_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<WeatherObservation> {
        let mut all_obs = Vec::new();
        
        // Try ISD-Lite first (most complete for hourly data)
        if self.use_isd_lite {
            if let Ok(stations) = self.fetch_isd_stations(region).await {
                for station in stations.iter().take(100) {
                    if let Ok(obs) = self.fetch_isd_lite_data(station, start, end).await {
                        all_obs.extend(obs);
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
        
        // Also try GHCN-Daily for daily data
        if let Ok(obs) = self.fetch_ghcn_daily(region, start, end).await {
            all_obs.extend(obs);
        }
        
        all_obs
    }
}

#[derive(Debug)]
struct ISDStation {
    usaf: String,
    wban: String,
    name: String,
    country: String,
    state: String,
    icao: String,
    lat: f64,
    lon: f64,
    elevation: f64,
    begin: String,
    end: String,
}

#[derive(Debug)]
struct GHCNStation {
    id: String,
    lat: f64,
    lon: f64,
    elevation: f64,
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
impl WeatherSource for NOAAAdapter {
    fn name(&self) -> &'static str {
        "NOAA"
    }
    
    fn source_type(&self) -> DataSource {
        DataSource::NOAA
    }
    
    fn priority(&self) -> u8 {
        15
    }
    
    fn accuracy_score(&self) -> f32 {
        0.95
    }
    
    fn provides_forecasts(&self) -> bool {
        false
    }
    
    fn provides_historical(&self) -> bool {
        true
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
        let observations = self.fetch_all_observations(region, start, end).await;
        let stream = ObservationIter {
            observations,
            index: 0,
        };
        Ok(Box::pin(stream))
    }
}

impl Clone for NOAAAdapter {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            api_token: self.api_token.clone(),
            rate_limiter: self.rate_limiter.clone(),
            min_request_interval: self.min_request_interval,
            use_isd_lite: self.use_isd_lite,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_noaa_adapter_creation() {
        let adapter = NOAAAdapter::new(None);
        assert_eq!(adapter.name(), "NOAA");
        assert_eq!(adapter.source_type(), DataSource::NOAA);
    }
    
    #[test]
    fn test_isd_lite_parsing() {
        let adapter = NOAAAdapter::new(None);
        let station = ISDStation {
            usaf: "725030".to_string(),
            wban: "14732".to_string(),
            name: "NEW YORK CENTRAL PARK".to_string(),
            country: "US".to_string(),
            state: "NY".to_string(),
            icao: "KNYC".to_string(),
            lat: 40.78,
            lon: -73.97,
            elevation: 39.0,
            begin: "2000".to_string(),
            end: "2023".to_string(),
        };
        
        // Sample ISD-Lite line
        let line = "2023 01 01 00 00  -25  -30  1013   270    50     0     0     0     0";
        let obs = adapter.parse_isd_lite_line(line, &station, 2023);
        assert!(obs.is_some());
    }
}
