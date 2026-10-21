//! # BMKG API Adapter
//!
//! Implementation for BMKG (Indonesian Meteorological Agency) API.
//! Must attribute BMKG per UU 31/2009. Commercial use requires written permission.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::Stream;
use reqwest::Client;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, warn};

use crate::sources::{RegionBbox, WeatherSource, WeatherStream};
use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, VerticalLevel, WIError,
    WeatherForecast, WeatherObservation, WeatherVariable,
};

/// Rate limiter for BMKG (60 req/min = 1 req/sec)
struct BmkgRateLimiter {
    tokens: RwLock<u32>,
    max_tokens: u32,
    last_refill: RwLock<Instant>,
}

impl BmkgRateLimiter {
    fn new() -> Self {
        Self {
            tokens: RwLock::new(60),
            max_tokens: 60,
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
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

/// BMKG API adapter
pub struct BMKGAdapter {
    client: Client,
    base_url: String,
    rate_limiter: Arc<BmkgRateLimiter>,
    semaphore: Arc<Semaphore>,
}

impl BMKGAdapter {
    /// Create new BMKG adapter
    ///
    /// BMKG requires attribution per UU 31/2009. Commercial use requires written permission.
    /// Rate limit: 60 requests per minute per IP.
    pub fn new(_api_key: Option<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("weather-intelligence/0.1 (+https://github.com/weather-intelligence; BMKG attribution required)")
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://data.bmkg.go.id".to_string(),
            rate_limiter: Arc::new(BmkgRateLimiter::new()),
            semaphore: Arc::new(Semaphore::new(2)), // conservative concurrency
        }
    }

    /// Fetch station metadata
    async fn fetch_stations(&self) -> Result<Vec<BmkgStation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let url = format!("{}/DataMKG/TEWS/autogempa.json", self.base_url);
        // Note: BMKG station list endpoint may vary, this is a common pattern
        // Actual implementation would need to discover the correct endpoint

        let response =
            self.client.get(&url).send().await.map_err(|e| {
                WIError::network_with_source("BMKG station request failed", e.into())
            })?;

        if !response.status().is_success() {
            // Try alternative endpoint
            let alt_url = format!("{}/DataMKG/TEWS/gempadirasakan.json", self.base_url);
            let alt_response = self.client.get(&alt_url).send().await.map_err(|e| {
                WIError::network_with_source("BMKG alt station request failed", e.into())
            })?;

            if !alt_response.status().is_success() {
                return Err(WIError::data_ingestion(
                    "BMKG station endpoints unavailable",
                ));
            }
        }

        // For now, return known Indonesian stations
        // In production, parse from BMKG's station list API
        Ok(get_known_indonesian_stations())
    }

    /// Fetch observations for a station
    async fn fetch_station_observations(
        &self,
        station_id: &str,
        start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Result<Vec<BmkgObservation>> {
        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        // BMKG historical data endpoints
        // This is a placeholder - actual BMKG API endpoints need to be discovered
        // Common patterns:
        // - /DataMKG/TEWS/autogempa.json (earthquake)
        // - /DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml (forecast)
        // - /DataMKG/MEWS/DigitalForecast/DigitalForecast-{region}.xml (regional forecast)

        let mut all_obs = Vec::new();

        // Try to fetch daily observations
        // BMKG typically provides forecast data, not historical observations via public API
        // Historical data may require different access

        let date_str = start.format("%Y%m%d").to_string();
        let url = format!(
            "{}/DataMKG/MEWS/DigitalForecast/DigitalForecast-{}.xml",
            self.base_url, date_str
        );

        let response =
            self.client.get(&url).send().await.map_err(|e| {
                WIError::network_with_source("BMKG forecast request failed", e.into())
            })?;

        if response.status().is_success() {
            let xml_text = response.text().await.map_err(|e| {
                WIError::serialization_with_source("Failed to read BMKG response", e.into())
            })?;

            if let Ok(forecast) = parse_bmkg_xml(&xml_text) {
                // Convert forecast to observations for the station
                for fc in forecast {
                    if fc.station_id == station_id {
                        all_obs.extend(fc.observations);
                    }
                }
            }
        }

        Ok(all_obs)
    }
}

/// Stream of BMKG observations
struct BmkgObservationStream {
    observations: Vec<WeatherObservation>,
    index: usize,
    error: Option<String>,
}

impl Stream for BmkgObservationStream {
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

/// BMKG Station
#[derive(Debug, Clone)]
struct BmkgStation {
    id: String,
    name: String,
    latitude: f64,
    longitude: f64,
    elevation: Option<f64>,
    province: String,
}

/// BMKG Observation
#[derive(Debug, Clone)]
struct BmkgObservation {
    station_id: String,
    time: DateTime<Utc>,
    variables: HashMap<WeatherVariable, ObservedValue>,
}

/// BMKG Forecast entry
struct BmkgForecast {
    station_id: String,
    observations: Vec<BmkgObservation>,
}

/// Get known Indonesian weather stations
fn get_known_indonesian_stations() -> Vec<BmkgStation> {
    vec![
        BmkgStation {
            id: "96745".to_string(),
            name: "Jakarta/Soekarno-Hatta".to_string(),
            latitude: -6.1256,
            longitude: 106.655,
            elevation: Some(8.0),
            province: "Jakarta".to_string(),
        },
        BmkgStation {
            id: "96747".to_string(),
            name: "Jakarta/Halim".to_string(),
            latitude: -6.2667,
            longitude: 106.883,
            elevation: Some(25.0),
            province: "Jakarta".to_string(),
        },
        BmkgStation {
            id: "96783".to_string(),
            name: "Bandung/Husein".to_string(),
            latitude: -6.900,
            longitude: 107.583,
            elevation: Some(740.0),
            province: "Jawa Barat".to_string(),
        },
        BmkgStation {
            id: "96825".to_string(),
            name: "Semarang/Achmad Yani".to_string(),
            latitude: -6.972,
            longitude: 110.375,
            elevation: Some(3.0),
            province: "Jawa Tengah".to_string(),
        },
        BmkgStation {
            id: "96945".to_string(),
            name: "Surabaya/Juanda".to_string(),
            latitude: -7.379,
            longitude: 112.787,
            elevation: Some(3.0),
            province: "Jawa Timur".to_string(),
        },
        BmkgStation {
            id: "97001".to_string(),
            name: "Denpasar/Ngurah Rai".to_string(),
            latitude: -8.748,
            longitude: 115.167,
            elevation: Some(4.0),
            province: "Bali".to_string(),
        },
        BmkgStation {
            id: "97051".to_string(),
            name: "Mataram/Lombok".to_string(),
            latitude: -8.560,
            longitude: 116.095,
            elevation: Some(18.0),
            province: "Nusa Tenggara Barat".to_string(),
        },
        BmkgStation {
            id: "97185".to_string(),
            name: "Makassar/Sultan Hasanuddin".to_string(),
            latitude: -5.062,
            longitude: 119.554,
            elevation: Some(14.0),
            province: "Sulawesi Selatan".to_string(),
        },
        BmkgStation {
            id: "97225".to_string(),
            name: "Manado/Sam Ratulangi".to_string(),
            latitude: 1.542,
            longitude: 124.926,
            elevation: Some(80.0),
            province: "Sulawesi Utara".to_string(),
        },
        BmkgStation {
            id: "97410".to_string(),
            name: "Jayapura/Sentani".to_string(),
            latitude: -2.576,
            longitude: 140.516,
            elevation: Some(88.0),
            province: "Papua".to_string(),
        },
        BmkgStation {
            id: "96000".to_string(),
            name: "Medan/Kualanamu".to_string(),
            latitude: 3.642,
            longitude: 98.885,
            elevation: Some(20.0),
            province: "Sumatera Utara".to_string(),
        },
        BmkgStation {
            id: "96155".to_string(),
            name: "Padang/Tabing".to_string(),
            latitude: -0.874,
            longitude: 100.351,
            elevation: Some(3.0),
            province: "Sumatera Barat".to_string(),
        },
        BmkgStation {
            id: "96275".to_string(),
            name: "Palembang/Sultan Mahmud Badaruddin".to_string(),
            latitude: -2.900,
            longitude: 104.699,
            elevation: Some(10.0),
            province: "Sumatera Selatan".to_string(),
        },
        BmkgStation {
            id: "96355".to_string(),
            name: "Bengkulu/Fatmawati".to_string(),
            latitude: -3.864,
            longitude: 102.340,
            elevation: Some(8.0),
            province: "Bengkulu".to_string(),
        },
        BmkgStation {
            id: "96455".to_string(),
            name: "Lampung/Radin Inten".to_string(),
            latitude: -5.242,
            longitude: 105.179,
            elevation: Some(40.0),
            province: "Lampung".to_string(),
        },
    ]
}

/// Parse BMKG XML forecast data
fn parse_bmkg_xml(xml_text: &str) -> Result<Vec<BmkgForecast>> {
    // BMKG provides DigitalForecast in XML format
    // This is a simplified parser - production would use quick-xml
    let mut forecasts = Vec::new();

    // Quick XML parsing for BMKG DigitalForecast structure
    // The XML contains <area> elements with id, latitude, longitude
    // Each area has <parameter> elements for different variables
    // Each parameter has <timerange> with datetime and <value>

    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml_text);
    reader.trim_text(true);

    let mut current_area_id = String::new();
    let mut current_param = String::new();
    let mut current_timerange_datetime = String::new();
    let mut current_value = String::new();
    let mut in_timerange = false;
    let mut in_value = false;
    let mut area_obs: HashMap<String, HashMap<WeatherVariable, HashMap<String, f64>>> =
        HashMap::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"area" => {
                    current_area_id = String::new();
                    let _current_area_lat = 0.0;
                    let _current_area_lon = 0.0;
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"id" => {
                                current_area_id = String::from_utf8_lossy(&attr.value).to_string()
                            }
                            b"latitude" => {
                                let _ = String::from_utf8_lossy(&attr.value).parse::<f64>();
                            }
                            b"longitude" => {
                                let _ = String::from_utf8_lossy(&attr.value).parse::<f64>();
                            }
                            _ => {}
                        }
                    }
                }
                b"parameter" => {
                    current_param = String::new();
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"id" {
                            current_param = String::from_utf8_lossy(&attr.value).to_string();
                        }
                    }
                }
                b"timerange" => {
                    in_timerange = true;
                    current_timerange_datetime = String::new();
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"datetime" {
                            current_timerange_datetime =
                                String::from_utf8_lossy(&attr.value).to_string();
                        }
                    }
                }
                b"value" => {
                    in_value = true;
                    current_value.clear();
                }
                _ => {}
            },
            Ok(Event::Text(e)) => {
                if in_value {
                    current_value.push_str(&e.unescape().unwrap_or_default());
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"value" => {
                        if in_timerange
                            && !current_timerange_datetime.is_empty()
                            && !current_value.is_empty()
                        {
                            if let Ok(val) = current_value.parse::<f64>() {
                                let var = map_bmkg_param(&current_param);
                                if var != WeatherVariable::Temperature {
                                    // skip unknown
                                    area_obs
                                        .entry(current_area_id.clone())
                                        .or_default()
                                        .entry(var)
                                        .or_default()
                                        .insert(current_timerange_datetime.clone(), val);
                                }
                            }
                        }
                        in_value = false;
                    }
                    b"timerange" => {
                        in_timerange = false;
                    }
                    b"area" => {
                        // Process completed area
                        if let Some(params) = area_obs.remove(&current_area_id) {
                            let mut obs_by_time: HashMap<
                                String,
                                HashMap<WeatherVariable, ObservedValue>,
                            > = HashMap::new();

                            for (var, time_values) in params {
                                for (time_str, val) in time_values {
                                    if DateTime::parse_from_rfc3339(&format!(
                                        "{}T00:00:00+00:00",
                                        time_str
                                    ))
                                    .is_ok()
                                    {
                                        obs_by_time
                                            .entry(time_str)
                                            .or_default()
                                            .insert(var, ObservedValue::new(val, 0.85));
                                    }
                                }
                            }

                            for (time_str, variables) in obs_by_time {
                                if !variables.is_empty() {
                                    if let Ok(time) = DateTime::parse_from_rfc3339(&format!(
                                        "{}T00:00:00+00:00",
                                        time_str
                                    )) {
                                        forecasts.push(BmkgForecast {
                                            station_id: current_area_id.clone(),
                                            observations: vec![BmkgObservation {
                                                station_id: current_area_id.clone(),
                                                time: time.with_timezone(&Utc),
                                                variables,
                                            }],
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(WIError::serialization_with_source(
                    "XML parse error",
                    e.into(),
                ))
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(forecasts)
}

/// Map BMKG parameter ID to WeatherVariable
fn map_bmkg_param(param: &str) -> WeatherVariable {
    match param {
        "T" | "Temp" => WeatherVariable::Temperature,
        "RH" | "Humidity" => WeatherVariable::RelativeHumidity,
        "RR" | "Rain" | "Precipitation" => WeatherVariable::Precipitation,
        "WS" | "WindSpeed" => WeatherVariable::WindSpeed,
        "WD" | "WindDir" => WeatherVariable::WindDirection,
        "PP" | "Pressure" => WeatherVariable::Pressure,
        "CC" | "CloudCover" => WeatherVariable::CloudCover,
        _ => WeatherVariable::Temperature, // default
    }
}

#[async_trait]
impl WeatherSource for BMKGAdapter {
    fn name(&self) -> &'static str {
        "BMKG"
    }
    fn source_type(&self) -> DataSource {
        DataSource::BMKG
    }
    fn provides_historical(&self) -> bool {
        true
    }
    fn provides_forecasts(&self) -> bool {
        true
    }
    fn rate_limit(&self) -> f64 {
        1.0
    } // 60/min = 1/sec

    async fn fetch_observations(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        // Fetch stations
        let stations = self.fetch_stations().await?;

        // Filter stations within region
        let filtered_stations: Vec<_> = stations
            .into_iter()
            .filter(|s| region.contains(s.latitude, s.longitude))
            .collect();

        if filtered_stations.is_empty() {
            debug!("No BMKG stations found in region");
            let stream = BmkgObservationStream {
                observations: Vec::new(),
                index: 0,
                error: None,
            };
            return Ok(Box::pin(stream));
        }

        let mut all_observations = Vec::new();

        // Fetch observations for each station in region
        for station in &filtered_stations {
            match self
                .fetch_station_observations(&station.id, start, end)
                .await
            {
                Ok(raw_obs) => {
                    let parsed = parse_bmkg_observations(raw_obs, station);
                    all_observations.extend(parsed);
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch observations for BMKG station {}: {}",
                        station.id, e
                    );
                }
            }
        }

        debug!(
            "Total BMKG observations fetched: {}",
            all_observations.len()
        );
        let stream = BmkgObservationStream {
            observations: all_observations,
            index: 0,
            error: None,
        };
        Ok(Box::pin(stream))
    }

    async fn fetch_forecast(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        // Find nearest station
        let stations = self.fetch_stations().await?;
        let nearest = stations.iter().min_by(|a, b| {
            let da = (a.latitude - lat).powi(2) + (a.longitude - lon).powi(2);
            let db = (b.latitude - lat).powi(2) + (b.longitude - lon).powi(2);
            da.partial_cmp(&db).unwrap()
        });

        if let Some(station) = nearest {
            // Fetch BMKG digital forecast
            let url = format!(
                "{}/DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml",
                self.base_url
            );

            self.rate_limiter.acquire().await;
            let _permit = self.semaphore.acquire().await;

            let response = self.client.get(&url).send().await.map_err(|e| {
                WIError::network_with_source("BMKG forecast request failed", e.into())
            })?;

            if response.status().is_success() {
                let xml_text = response.text().await.map_err(|e| {
                    WIError::serialization_with_source("Failed to read BMKG forecast", e.into())
                })?;

                if let Ok(forecasts) = parse_bmkg_xml(&xml_text) {
                    // Find forecast for this station
                    for fc in forecasts {
                        if fc.station_id == station.id {
                            if let Some(obs) = fc.observations.first() {
                                let mut variables = HashMap::new();
                                for (var, val) in &obs.variables {
                                    variables.insert((*var, VerticalLevel::Surface), val.value);
                                }

                                return Ok(WeatherForecast {
                                    coord: GeoCoord::new(station.latitude, station.longitude),
                                    init_time: Utc::now(),
                                    valid_time: obs.time,
                                    lead_seconds,
                                    variables,
                                    ensemble_members: None,
                                    model_id: "bmkg-digital-forecast".to_string(),
                                    probability: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        Err(WIError::model("BMKG forecast not available for location"))
    }
}

/// Parse BMKG raw observations into WeatherObservation
fn parse_bmkg_observations(
    raw: Vec<BmkgObservation>,
    station: &BmkgStation,
) -> Vec<WeatherObservation> {
    let mut observations = Vec::new();

    for obs in raw {
        let mut coord = GeoCoord::new(station.latitude, station.longitude);
        if let Some(elev) = station.elevation {
            coord.altitude_m = Some(elev);
        }

        observations.push(WeatherObservation {
            station_id: Some(station.id.clone()),
            coord,
            time: obs.time,
            variables: obs.variables,
            quality_flag: QualityFlag::Valid,
            source: DataSource::BMKG,
        });
    }

    observations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = BMKGAdapter::new(None);
        assert_eq!(adapter.name(), "BMKG");
        assert_eq!(adapter.source_type(), DataSource::BMKG);
    }

    #[test]
    fn test_bmkg_rate_limiter() {
        let limiter = BmkgRateLimiter::new();
        assert_eq!(limiter.max_tokens, 60);
    }

    #[test]
    fn test_station_list() {
        let stations = get_known_indonesian_stations();
        assert!(!stations.is_empty());
        assert!(stations.iter().any(|s| s.id == "96745")); // Jakarta
    }

    #[test]
    fn test_param_mapping() {
        assert_eq!(map_bmkg_param("T"), WeatherVariable::Temperature);
        assert_eq!(map_bmkg_param("RR"), WeatherVariable::Precipitation);
        assert_eq!(map_bmkg_param("WS"), WeatherVariable::WindSpeed);
    }
}
