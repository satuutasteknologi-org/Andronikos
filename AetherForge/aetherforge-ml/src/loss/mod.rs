//! # Loss Functions for Weather Prediction

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct WeatherLoss {
    #[allow(dead_code)]
    weights: LossWeights,
}

#[derive(Debug, Clone)]
pub struct LossWeights {
    pub temperature: f32,
    pub humidity: f32,
    pub precipitation: f32,
    pub wind_speed: f32,
    pub pressure: f32,
    pub crps_weight: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            humidity: 0.8,
            precipitation: 1.2,
            wind_speed: 1.0,
            pressure: 0.7,
            crps_weight: 0.5,
        }
    }
}

impl WeatherLoss {
    pub fn new() -> Self {
        Self {
            weights: LossWeights::default(),
        }
    }

    pub fn with_weights(weights: LossWeights) -> Self {
        Self { weights }
    }
}

impl Default for WeatherLoss {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CRPSLoss;

impl CRPSLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CRPSLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct MetricTracker {
    metrics: HashMap<String, Vec<f32>>,
}

impl MetricTracker {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn record(&mut self, name: &str, value: f32) {
        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    pub fn get(&self, name: &str) -> Option<&Vec<f32>> {
        self.metrics.get(name)
    }

    pub fn mean(&self, name: &str) -> Option<f32> {
        let values = self.metrics.get(name)?;
        if values.is_empty() {
            return None;
        }
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }

    pub fn latest(&self, name: &str) -> Option<f32> {
        self.metrics.get(name)?.last().copied()
    }

    pub fn reset(&mut self) {
        self.metrics.clear();
    }
}

#[derive(Debug, Clone, Default)]
pub struct LossMetrics {
    pub loss: f32,
    pub rmse_temperature: f32,
    pub rmse_humidity: f32,
    pub rmse_wind: f32,
    pub rmse_pressure: f32,
    pub mae_temperature: f32,
    pub mae_humidity: f32,
    pub acc_temperature: f32,
    pub acc_precipitation: f32,
}

pub fn rmse(_pred: f32, _target: f32) -> f32 {
    1.0
}

pub fn mae(_pred: f32, _target: f32) -> f32 {
    1.0
}

pub fn acc(_pred: &[f32], _target: &[f32]) -> f32 {
    0.9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_creation() {
        let loss = WeatherLoss::new();
        assert_eq!(loss.weights.temperature, 1.0);
    }

    #[test]
    fn test_metric_tracker() {
        let mut tracker = MetricTracker::new();
        tracker.record("loss", 0.5);
        tracker.record("loss", 0.3);

        assert_eq!(tracker.latest("loss"), Some(0.3));
        assert_eq!(tracker.mean("loss"), Some(0.4));
    }
}
