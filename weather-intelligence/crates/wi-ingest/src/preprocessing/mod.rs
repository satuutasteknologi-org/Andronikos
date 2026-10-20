//! # Data Preprocessing
//!
//! Normalization, slotting, feature engineering, and quantization.

use chrono::{Datelike, Timelike};
use wi_grid::Grid;

use std::collections::HashMap;

use wi_core::types::{FeatureVector, WeatherObservation, WeatherVariable};

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub enable_normalization: bool,
    pub enable_climatology: bool,
    pub enable_feature_engineering: bool,
    pub quantization_bits: u8,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_normalization: true,
            enable_climatology: true,
            enable_feature_engineering: true,
            quantization_bits: 16,
        }
    }
}

/// Preprocessing pipeline
pub struct Preprocessor {
    config: PreprocessingConfig,
    climatology: HashMap<(WeatherVariable, u8, u8), f64>,
    grid: wi_grid::FixedGrid,
}

impl Preprocessor {
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            climatology: HashMap::new(),
            grid: wi_grid::FixedGrid::indonesia(),
        }
    }

    /// Normalize observation values
    pub fn normalize(&self, obs: &mut WeatherObservation) {
        if !self.config.enable_normalization {
            return;
        }

        for (var, val) in &mut obs.variables {
            if val.is_valid {
                // Apply variable-specific normalization
                val.value = self.normalize_var(*var, val.value);
            }
        }
    }

    fn normalize_var(&self, var: WeatherVariable, value: f64) -> f64 {
        match var {
            WeatherVariable::Temperature => (value - 273.15) / 50.0, // Convert K to C, normalize
            WeatherVariable::RelativeHumidity => value / 100.0,
            WeatherVariable::Precipitation => (value + 1.0).ln() / 10.0,
            WeatherVariable::WindSpeed => value / 50.0,
            WeatherVariable::Pressure | WeatherVariable::SurfacePressure => {
                (value - 90000.0) / 20000.0
            }
            _ => value,
        }
    }

    /// Denormalize prediction values
    pub fn denormalize(&self, var: WeatherVariable, value: f64) -> f64 {
        match var {
            WeatherVariable::Temperature => value * 50.0 + 273.15,
            WeatherVariable::RelativeHumidity => value * 100.0,
            WeatherVariable::Precipitation => (value * 10.0).exp() - 1.0,
            WeatherVariable::WindSpeed => value * 50.0,
            WeatherVariable::Pressure | WeatherVariable::SurfacePressure => {
                value * 20000.0 + 90000.0
            }
            _ => value,
        }
    }

    /// Compute climatology from observations
    pub fn compute_climatology(&mut self, observations: &[WeatherObservation]) {
        if !self.config.enable_climatology {
            return;
        }

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
            self.climatology.insert(key, sum / count as f64);
        }
    }

    /// Fill missing values with climatology
    pub fn fill_climatology(&self, obs: &mut WeatherObservation) {
        if !self.config.enable_climatology {
            return;
        }

        let month = obs.time.month() as u8;
        let hour = obs.time.hour() as u8;

        for var in WeatherVariable::surface_variables() {
            if !obs.variables.contains_key(&var) || !obs.variables[&var].is_valid {
                if let Some(&clim_value) = self.climatology.get(&(var, month, hour)) {
                    obs.variables
                        .insert(var, wi_core::types::ObservedValue::new(clim_value, 0.5));
                }
            }
        }
    }

    /// Slot observations into grid cells
    pub fn slot_to_grid(
        &self,
        observations: &[WeatherObservation],
    ) -> HashMap<wi_grid::FixedCellId, Vec<WeatherObservation>> {
        let mut grid_map: HashMap<wi_grid::FixedCellId, Vec<WeatherObservation>> = HashMap::new();

        for obs in observations {
            if let Ok(cell) = self.grid.from_lat_lon(obs.coord.lat, obs.coord.lon) {
                grid_map.entry(cell).or_default().push(obs.clone());
            }
        }

        grid_map
    }

    /// Aggregate observations per cell (mean)
    pub fn aggregate_grid(
        &self,
        grid_map: HashMap<wi_grid::FixedCellId, Vec<WeatherObservation>>,
    ) -> HashMap<wi_grid::FixedCellId, WeatherObservation> {
        let mut result = HashMap::new();

        for (cell, obs_list) in grid_map {
            if obs_list.is_empty() {
                continue;
            }

            let mut aggregated = obs_list[0].clone();
            let _n = obs_list.len() as f64;

            for (var, val) in &mut aggregated.variables {
                if val.is_valid {
                    let sum: f64 = obs_list
                        .iter()
                        .filter_map(|o| o.variables.get(var))
                        .filter(|v| v.is_valid)
                        .map(|v| v.value)
                        .sum();
                    let count = obs_list
                        .iter()
                        .filter(|o| o.variables.get(var).is_some_and(|v| v.is_valid))
                        .count() as f64;
                    if count > 0.0 {
                        val.value = sum / count;
                        val.confidence = 1.0 - (1.0 - val.confidence) / count.sqrt() as f32;
                    }
                }
            }

            result.insert(cell, aggregated);
        }

        result
    }

    /// Feature engineering: add derived features
    pub fn engineer_features(&self, obs: &WeatherObservation) -> FeatureVector {
        let mut features: HashMap<String, f64> = HashMap::new();

        // Basic variables
        for var in WeatherVariable::surface_variables() {
            if let Some(val) = obs.variables.get(&var) {
                if val.is_valid {
                    features.insert(var.display_name().to_string(), val.value);
                }
            }
        }

        // Derived features
        if let (Some(temp), Some(hum)) = (
            obs.variables.get(&WeatherVariable::Temperature),
            obs.variables.get(&WeatherVariable::RelativeHumidity),
        ) {
            if temp.is_valid && hum.is_valid {
                // Dew point
                let td = temp.value - ((100.0 - hum.value) / 5.0);
                features.insert("dew_point".to_string(), td);

                // Heat index (simplified)
                if temp.value > 300.0 {
                    // > 27C
                    let hi = temp.value + 0.5 * (hum.value / 100.0) * (temp.value - 273.15);
                    features.insert("heat_index".to_string(), hi);
                }
            }
        }

        // Wind features
        if let (Some(ws), Some(wd)) = (
            obs.variables.get(&WeatherVariable::WindSpeed),
            obs.variables.get(&WeatherVariable::WindDirection),
        ) {
            if ws.is_valid && wd.is_valid {
                let (u, v) = wi_core::utils::wind_components(ws.value, wd.value);
                features.insert("wind_u".to_string(), u);
                features.insert("wind_v".to_string(), v);
            }
        }

        // Temporal features
        let month = obs.time.month() as f64;
        let hour = obs.time.hour() as f64;
        let day_of_year = obs.time.ordinal0() as f64;

        features.insert(
            "month_sin".to_string(),
            (month * 2.0 * std::f64::consts::PI / 12.0).sin(),
        );
        features.insert(
            "month_cos".to_string(),
            (month * 2.0 * std::f64::consts::PI / 12.0).cos(),
        );
        features.insert(
            "hour_sin".to_string(),
            (hour * 2.0 * std::f64::consts::PI / 24.0).sin(),
        );
        features.insert(
            "hour_cos".to_string(),
            (hour * 2.0 * std::f64::consts::PI / 24.0).cos(),
        );
        features.insert(
            "doy_sin".to_string(),
            (day_of_year * 2.0 * std::f64::consts::PI / 365.0).sin(),
        );
        features.insert(
            "doy_cos".to_string(),
            (day_of_year * 2.0 * std::f64::consts::PI / 365.0).cos(),
        );

        // Spatial features
        features.insert("lat".to_string(), obs.coord.lat);
        features.insert("lon".to_string(), obs.coord.lon);
        if let Some(alt) = obs.coord.altitude_m {
            features.insert("altitude".to_string(), alt);
        }

        // Create FeatureVector with proper constructor
        let mut fv = FeatureVector::new(obs.coord, obs.time, 0, features.len());
        for (name, value) in features {
            fv.feature_names.push(name);
            fv.features.push(value as f32);
        }
        fv
    }

    /// Quantize values to reduce precision for storage
    pub fn quantize(&self, value: f64) -> u16 {
        let max_val = (1u64 << self.config.quantization_bits) - 1;
        (value * max_val as f64).clamp(0.0, max_val as f64) as u16
    }

    pub fn dequantize(&self, value: u16) -> f64 {
        let max_val = (1u64 << self.config.quantization_bits) - 1;
        value as f64 / max_val as f64
    }
}

/// Adaptive quantization for storage
pub struct AdaptiveQuantizer {
    pub min_val: f64,
    pub max_val: f64,
    pub bits: u8,
}

impl AdaptiveQuantizer {
    pub fn new(bits: u8) -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            bits,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
    }

    pub fn quantize(&self, value: f64) -> u16 {
        if self.max_val <= self.min_val {
            return 0;
        }
        let range = self.max_val - self.min_val;
        let normalized = (value - self.min_val) / range;
        let max_val = (1u64 << self.bits) - 1;
        (normalized * max_val as f64).clamp(0.0, max_val as f64) as u16
    }

    pub fn dequantize(&self, value: u16) -> f64 {
        let max_val = (1u64 << self.bits) - 1;
        let normalized = value as f64 / max_val as f64;
        self.min_val + normalized * (self.max_val - self.min_val)
    }
}

/// Climatology storage for bias correction
pub struct ClimatologyStore {
    data: HashMap<(WeatherVariable, u8, u8), f64>, // (var, month, hour) -> mean
}

impl Default for ClimatologyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ClimatologyStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn update(&mut self, var: WeatherVariable, month: u8, hour: u8, value: f64) {
        let key = (var, month, hour);
        self.data
            .entry(key)
            .and_modify(|v| *v = (*v + value) / 2.0)
            .or_insert(value);
    }

    pub fn get(&self, var: WeatherVariable, month: u8, hour: u8) -> Option<f64> {
        self.data.get(&(var, month, hour)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use wi_core::types::{GeoCoord, ObservedValue};

    #[test]
    fn test_normalize_denormalize() {
        let config = PreprocessingConfig::default();
        let preprocessor = Preprocessor::new(config);

        let temp_k = 300.0; // 26.85 C
        let norm = preprocessor.normalize_var(WeatherVariable::Temperature, temp_k);
        let denorm = preprocessor.denormalize(WeatherVariable::Temperature, norm);
        assert!((denorm - temp_k).abs() < 0.1);
    }

    #[test]
    fn test_feature_engineering() {
        let config = PreprocessingConfig::default();
        let preprocessor = Preprocessor::new(config);

        let mut obs = WeatherObservation {
            station_id: Some("test".to_string()),
            coord: GeoCoord::new(-6.595, 106.806),
            time: Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap(),
            variables: HashMap::new(),
            quality_flag: wi_core::types::QualityFlag::Valid,
            source: wi_core::types::DataSource::OpenMeteo,
        };

        obs.variables
            .insert(WeatherVariable::Temperature, ObservedValue::new(300.0, 0.9));
        obs.variables.insert(
            WeatherVariable::RelativeHumidity,
            ObservedValue::new(80.0, 0.9),
        );
        obs.variables
            .insert(WeatherVariable::WindSpeed, ObservedValue::new(5.0, 0.9));
        obs.variables.insert(
            WeatherVariable::WindDirection,
            ObservedValue::new(90.0, 0.9),
        );

        let features = preprocessor.engineer_features(&obs);

        assert!(features.feature_names.iter().any(|n| n == "dew_point"));
        assert!(features.feature_names.iter().any(|n| n == "wind_u"));
        assert!(features.feature_names.iter().any(|n| n == "wind_v"));
        assert!(features.feature_names.iter().any(|n| n == "month_sin"));
    }
}
