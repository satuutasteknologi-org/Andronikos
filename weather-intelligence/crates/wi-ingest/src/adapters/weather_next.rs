//! # WeatherNext (Google DeepMind) Adapter
//!
//! Implementation for WeatherNext / GraphCast API (Google DeepMind).
//! Currently allowlist-only - requires explicit access grant from Google.
//! This adapter is structured for when API access becomes available.

use std::collections::HashMap;
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
use tracing::warn;

use crate::sources::{ForecastStream, RegionBbox, WeatherSource, WeatherStream};
use wi_core::{
    DataSource, GeoCoord, Result, VerticalLevel, WIError, WeatherForecast, WeatherVariable,
};

/// Rate limiter for WeatherNext API (very restrictive: allowlist only)
struct WeatherNextRateLimiter {
    tokens: RwLock<u32>,
    max_tokens: u32,
    last_refill: RwLock<Instant>,
}

impl WeatherNextRateLimiter {
    fn new() -> Self {
        Self {
            tokens: RwLock::new(10), // Very conservative
            max_tokens: 10,
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
            tokio::time::sleep(Duration::from_secs(6)).await;
        }
    }
}

/// WeatherNext model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherNextModel {
    /// GraphCast (original DeepMind model)
    GraphCast,
    /// WeatherNext Gen (successor with improved resolution)
    WeatherNextGen,
    /// WeatherNext Gen with ensemble
    WeatherNextGenEnsemble,
}

impl WeatherNextModel {
    fn model_id(&self) -> &'static str {
        match self {
            Self::GraphCast => "graphcast",
            Self::WeatherNextGen => "weathernext-gen",
            Self::WeatherNextGenEnsemble => "weathernext-gen-ensemble",
        }
    }

    fn supports_ensemble(&self) -> bool {
        matches!(self, Self::WeatherNextGenEnsemble)
    }

    fn max_lead_hours(&self) -> u32 {
        match self {
            Self::GraphCast => 240,      // 10 days
            Self::WeatherNextGen => 360, // 15 days
            Self::WeatherNextGenEnsemble => 360,
        }
    }
}

/// WeatherNext API adapter
pub struct WeatherNextAdapter {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: WeatherNextModel,
    rate_limiter: Arc<WeatherNextRateLimiter>,
    semaphore: Arc<Semaphore>,
}

impl WeatherNextAdapter {
    /// Create new WeatherNext adapter
    ///
    /// # Arguments
    /// * `api_key` - Required API key from Google DeepMind allowlist
    /// * `model` - Model variant to use (default: WeatherNextGen)
    pub fn new(api_key: Option<String>, model: Option<WeatherNextModel>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://weathernext.googleapis.com/v1".to_string(), // Hypothetical endpoint
            api_key,
            model: model.unwrap_or(WeatherNextModel::WeatherNextGen),
            rate_limiter: Arc::new(WeatherNextRateLimiter::new()),
            semaphore: Arc::new(Semaphore::new(1)), // Very conservative
        }
    }

    /// Build authenticated request
    fn build_request(&self, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.post(format!("{}{}", self.base_url, path));
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        req.header("Content-Type", "application/json")
    }

    /// Build forecast request payload
    fn build_forecast_payload(
        &self,
        lat: f64,
        lon: f64,
        lead_hours: u32,
        variables: &[WeatherVariable],
    ) -> WeatherNextForecastRequest {
        let vars: Vec<String> = variables.iter().map(|v| format!("{:?}", v)).collect();

        WeatherNextForecastRequest {
            model: self.model.model_id().to_string(),
            location: Location { lat, lon },
            lead_hours,
            variables: vars,
            ensemble: self.model.supports_ensemble(),
        }
    }
}

/// Location for WeatherNext API
#[derive(Debug, Serialize, Deserialize)]
struct Location {
    lat: f64,
    lon: f64,
}

/// WeatherNext forecast request
#[derive(Debug, Serialize, Deserialize)]
struct WeatherNextForecastRequest {
    model: String,
    location: Location,
    lead_hours: u32,
    variables: Vec<String>,
    ensemble: bool,
}

/// WeatherNext forecast response
#[derive(Debug, Deserialize)]
struct WeatherNextForecastResponse {
    predictions: Vec<WeatherNextPrediction>,
    ensemble_members: Option<Vec<WeatherNextEnsembleMember>>,
    model_version: String,
    generated_at: String,
}

/// Single prediction point
#[derive(Debug, Deserialize)]
struct WeatherNextPrediction {
    valid_time: String,
    variables: HashMap<String, f64>,
    uncertainties: Option<HashMap<String, f64>>,
}

/// Ensemble member
#[derive(Debug, Deserialize)]
struct WeatherNextEnsembleMember {
    member_id: u32,
    predictions: Vec<WeatherNextPrediction>,
}

/// Stream of WeatherNext forecasts
struct WeatherNextForecastStream {
    forecasts: Vec<WeatherForecast>,
    index: usize,
    error: Option<String>,
}

impl Stream for WeatherNextForecastStream {
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
impl WeatherSource for WeatherNextAdapter {
    fn name(&self) -> &'static str {
        "WeatherNext"
    }
    fn source_type(&self) -> DataSource {
        DataSource::WeatherNext
    }
    fn provides_historical(&self) -> bool {
        false
    }
    fn provides_forecasts(&self) -> bool {
        true
    }
    fn rate_limit(&self) -> f64 {
        1.0 / 6.0
    } // 10 req/min

    async fn fetch_observations(
        &self,
        _region: RegionBbox,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Result<WeatherStream> {
        Err(WIError::model(
            "WeatherNext adapter does not provide historical observations",
        ))
    }

    async fn fetch_forecast(
        &self,
        lat: f64,
        lon: f64,
        lead_seconds: u32,
    ) -> Result<WeatherForecast> {
        if self.api_key.is_none() {
            return Err(WIError::config(
                "WeatherNext requires API key from Google DeepMind allowlist",
            ));
        }

        let lead_hours = lead_seconds / 3600;
        if lead_hours > self.model.max_lead_hours() {
            return Err(WIError::model(format!(
                "Lead time {}h exceeds maximum {}h for model {:?}",
                lead_hours,
                self.model.max_lead_hours(),
                self.model
            )));
        }

        self.rate_limiter.acquire().await;
        let _permit = self.semaphore.acquire().await;

        let variables = WeatherVariable::surface_variables();
        let payload = self.build_forecast_payload(lat, lon, lead_hours, &variables);

        let response = self
            .build_request("/forecast")
            .json(&payload)
            .send()
            .await
            .map_err(|e| WIError::network_with_source("WeatherNext request failed", e.into()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(WIError::data_ingestion(format!(
                "WeatherNext API error {}: {}",
                status, text
            )));
        }

        let forecast_response: WeatherNextForecastResponse =
            response.json().await.map_err(|e| {
                WIError::serialization_with_source("Failed to parse WeatherNext response", e.into())
            })?;

        // Convert first prediction to WeatherForecast
        if let Some(pred) = forecast_response.predictions.first() {
            let mut variables = HashMap::new();

            for (var_name, value) in &pred.variables {
                if let Some(var) = parse_weather_variable(var_name) {
                    variables.insert((var, VerticalLevel::Surface), *value);
                }
            }

            let valid_time = DateTime::parse_from_rfc3339(&pred.valid_time)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now() + chrono::Duration::seconds(lead_seconds as i64));

            Ok(WeatherForecast {
                coord: GeoCoord::new(lat, lon),
                init_time: Utc::now(),
                valid_time,
                lead_seconds,
                variables,
                ensemble_members: forecast_response
                    .ensemble_members
                    .as_ref()
                    .map(|m| m.len() as u8),
                model_id: format!(
                    "{}-{}",
                    self.model.model_id(),
                    forecast_response.model_version
                ),
                probability: None,
            })
        } else {
            Err(WIError::data_ingestion(
                "No predictions in WeatherNext response",
            ))
        }
    }
}

#[async_trait]
impl crate::sources::ForecastSource for WeatherNextAdapter {
    async fn fetch_forecast_stream(
        &self,
        lat: f64,
        lon: f64,
        max_lead_hours: u32,
    ) -> Result<ForecastStream> {
        if self.api_key.is_none() {
            return Err(WIError::config(
                "WeatherNext requires API key from Google DeepMind allowlist",
            ));
        }

        let max_lead = max_lead_hours.min(self.model.max_lead_hours());
        let mut forecasts = Vec::new();

        // Fetch at 6-hour intervals
        for lead_h in (6..=max_lead).step_by(6) {
            match self.fetch_forecast(lat, lon, lead_h * 3600).await {
                Ok(fc) => forecasts.push(fc),
                Err(e) => warn!("Failed to fetch WeatherNext lead {}h: {}", lead_h, e),
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        let stream = WeatherNextForecastStream {
            forecasts,
            index: 0,
            error: None,
        };
        Ok(Box::pin(stream))
    }
}

/// Parse WeatherNext variable name to WeatherVariable
fn parse_weather_variable(name: &str) -> Option<WeatherVariable> {
    match name {
        "Temperature" | "temperature" | "temp" | "t2m" => Some(WeatherVariable::Temperature),
        "RelativeHumidity" | "humidity" | "rh" | "rhum" => Some(WeatherVariable::RelativeHumidity),
        "Precipitation" | "precip" | "rain" | "tp" => Some(WeatherVariable::Precipitation),
        "WindSpeed" | "wind_speed" | "ws" | "wspd" => Some(WeatherVariable::WindSpeed),
        "WindDirection" | "wind_dir" | "wd" | "wdir" => Some(WeatherVariable::WindDirection),
        "WindU" | "u_component" | "u" => Some(WeatherVariable::WindU),
        "WindV" | "v_component" | "v" => Some(WeatherVariable::WindV),
        "Pressure" | "pressure" | "pres" | "msl" => Some(WeatherVariable::Pressure),
        "SurfacePressure" | "surface_pressure" | "sp" => Some(WeatherVariable::SurfacePressure),
        "CloudCover" | "cloud_cover" | "cc" | "tcc" => Some(WeatherVariable::CloudCover),
        "DewPoint" | "dew_point" | "td" | "dpt" => Some(WeatherVariable::DewPoint),
        "Visibility" | "visibility" | "vis" => Some(WeatherVariable::Visibility),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = WeatherNextAdapter::new(None, None);
        assert_eq!(adapter.name(), "WeatherNext");
        assert_eq!(adapter.source_type(), DataSource::WeatherNext);
    }

    #[test]
    fn test_model_variants() {
        assert_eq!(WeatherNextModel::GraphCast.max_lead_hours(), 240);
        assert_eq!(WeatherNextModel::WeatherNextGen.max_lead_hours(), 360);
        assert!(!WeatherNextModel::GraphCast.supports_ensemble());
        assert!(WeatherNextModel::WeatherNextGenEnsemble.supports_ensemble());
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = WeatherNextRateLimiter::new();
        assert_eq!(limiter.max_tokens, 10);
    }
}
