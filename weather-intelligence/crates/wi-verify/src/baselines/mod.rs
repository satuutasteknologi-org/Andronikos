//! # Baseline Forecasts
//!
//! Standard baselines for comparison: persistence, climatology, raw model.

use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use std::collections::HashMap;

use wi_core::{GeoCoord, Result, WIError, WeatherObservation, WeatherVariable};

/// Baseline forecast trait
pub trait BaselineForecast: Send + Sync {
    fn name(&self) -> &'static str;
    fn predict(
        &self,
        coord: GeoCoord,
        valid_time: DateTime<Utc>,
        lead_seconds: u32,
    ) -> Result<HashMap<WeatherVariable, f32>>;
}

/// Persistence forecast (last observation)
pub struct PersistenceBaseline {
    observations: HashMap<String, WeatherObservation>,
}

impl Default for PersistenceBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistenceBaseline {
    pub fn new() -> Self {
        Self {
            observations: HashMap::new(),
        }
    }

    pub fn update(&mut self, obs: WeatherObservation) {
        if let Some(id) = &obs.station_id {
            self.observations.insert(id.clone(), obs);
        }
    }
}

impl BaselineForecast for PersistenceBaseline {
    fn name(&self) -> &'static str {
        "Persistence"
    }

    fn predict(
        &self,
        coord: GeoCoord,
        _valid_time: DateTime<Utc>,
        _lead_seconds: u32,
    ) -> Result<HashMap<WeatherVariable, f32>> {
        let mut best_dist = f64::INFINITY;
        let mut best_obs = None;

        for obs in self.observations.values() {
            let dist = coord.distance_to(&obs.coord);
            if dist < best_dist {
                best_dist = dist;
                best_obs = Some(obs);
            }
        }

        let mut predictions = HashMap::new();
        if let Some(obs) = best_obs {
            for (var, val) in &obs.variables {
                if val.is_valid {
                    predictions.insert(*var, val.value as f32);
                }
            }
        }

        Ok(predictions)
    }
}

/// Climatology baseline (historical average)
pub struct ClimatologyBaseline {
    climatology: HashMap<(WeatherVariable, u8, u8), f32>,
}

impl Default for ClimatologyBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl ClimatologyBaseline {
    pub fn new() -> Self {
        Self {
            climatology: HashMap::new(),
        }
    }

    pub fn train(&mut self, observations: &[WeatherObservation]) {
        let mut sums: HashMap<(WeatherVariable, u8, u8), (f64, u64)> = HashMap::new();

        for obs in observations {
            let month = obs.time.month() as u8;
            let hour = obs.time.hour() as u8;

            for (var, val) in &obs.variables {
                if val.is_valid {
                    let key = (*var, month, hour);
                    let entry = sums.entry(key).or_insert((0.0, 0));
                    entry.0 += val.value;
                    entry.1 += 1;
                }
            }
        }

        for (key, (sum, count)) in sums {
            self.climatology.insert(key, (sum / count as f64) as f32);
        }
    }
}

impl BaselineForecast for ClimatologyBaseline {
    fn name(&self) -> &'static str {
        "Climatology"
    }

    fn predict(
        &self,
        _coord: GeoCoord,
        valid_time: DateTime<Utc>,
        _lead_seconds: u32,
    ) -> Result<HashMap<WeatherVariable, f32>> {
        let month = valid_time.month() as u8;
        let hour = valid_time.hour() as u8;

        let mut predictions = HashMap::new();

        for var in WeatherVariable::surface_variables() {
            if let Some(&value) = self.climatology.get(&(var, month, hour)) {
                predictions.insert(var, value);
            }
        }

        Ok(predictions)
    }
}

/// Raw model output baseline (no downscaling)
pub struct RawModelBaseline {
    _model_name: String,
}

impl RawModelBaseline {
    pub fn new(model_name: String) -> Self {
        Self {
            _model_name: model_name,
        }
    }
}

impl BaselineForecast for RawModelBaseline {
    fn name(&self) -> &'static str {
        "RawModel"
    }

    fn predict(
        &self,
        _coord: GeoCoord,
        _valid_time: DateTime<Utc>,
        _lead_seconds: u32,
    ) -> Result<HashMap<WeatherVariable, f32>> {
        Err(WIError::model(
            "RawModelBaseline requires model integration",
        ))
    }
}

/// Baseline comparer for verification
pub struct BaselineComparer {
    baselines: Vec<Box<dyn BaselineForecast>>,
}

impl BaselineComparer {
    pub fn new() -> Self {
        Self {
            baselines: Vec::new(),
        }
    }

    pub fn add_baseline(&mut self, baseline: Box<dyn BaselineForecast>) {
        self.baselines.push(baseline);
    }

    pub async fn compare(
        &self,
        forecasts: &[HashMap<WeatherVariable, f32>],
        observations: &[WeatherObservation],
        lead_seconds: u32,
    ) -> Result<BaselineComparison> {
        let mut results = HashMap::new();

        for baseline in &self.baselines {
            let mut baseline_forecasts = Vec::new();

            for obs in observations {
                let pred = baseline.predict(
                    obs.coord,
                    obs.time + Duration::seconds(lead_seconds as i64),
                    lead_seconds,
                )?;
                baseline_forecasts.push(pred);
            }

            let metrics = self.calculate_metrics(&baseline_forecasts, observations)?;
            results.insert(baseline.name().to_string(), metrics);
        }

        let target_metrics = self.calculate_metrics(forecasts, observations)?;
        results.insert("target".to_string(), target_metrics);

        Ok(BaselineComparison {
            baseline_metrics: results,
            lead_seconds,
            sample_count: observations.len(),
        })
    }

    fn calculate_metrics(
        &self,
        forecasts: &[HashMap<WeatherVariable, f32>],
        observations: &[WeatherObservation],
    ) -> Result<BaselineMetrics> {
        let mut metrics = BaselineMetrics::default();

        for var in WeatherVariable::surface_variables() {
            let fcst_vals: Vec<f32> = forecasts
                .iter()
                .filter_map(|f| f.get(&var).copied())
                .collect();
            let obs_vals: Vec<f32> = observations
                .iter()
                .filter_map(|o| {
                    o.variables
                        .get(&var)
                        .filter(|v| v.is_valid)
                        .map(|v| v.value as f32)
                })
                .collect();

            if fcst_vals.len() == obs_vals.len() && !fcst_vals.is_empty() {
                let det = crate::metrics::MetricCalculator::deterministic(&fcst_vals, &obs_vals);
                metrics.deterministic.insert(var, det);
            }
        }

        Ok(metrics)
    }
}

impl Default for BaselineComparer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_metrics: HashMap<String, BaselineMetrics>,
    pub lead_seconds: u32,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub deterministic: HashMap<WeatherVariable, crate::metrics::DeterministicMetrics>,
    pub probabilistic: HashMap<WeatherVariable, crate::metrics::ProbabilisticMetrics>,
}

use serde::{Deserialize, Serialize};
