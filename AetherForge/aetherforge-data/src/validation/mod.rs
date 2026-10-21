//! # Data Validation and Cleaning
//!
//! Physics-based validation and cleaning for weather observations.

use std::collections::HashMap;

use aetherforge_core::{
    ClimateZone, ObservedValue, QualityFlag, WeatherObservation, WeatherVariable,
};

/// Validation result with optional corrections
#[derive(Debug)]
pub struct ValidationResult {
    pub observation: WeatherObservation,
    pub was_modified: bool,
    pub issues: Vec<ValidationIssue>,
}

/// Issues found during validation
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    TemperatureOutOfRange { value: f64, min: f64, max: f64 },
    PressureOutOfRange { value: f64, min: f64, max: f64 },
    HumidityOutOfRange { value: f64, min: f64, max: f64 },
    WindSpeedExcessive { value: f64, max: f64 },
    InconsistentWindComponents { u: f64, v: f64, speed: f64 },
    DuplicateTimestamp,
    InvalidCoordinate,
    NegativePrecipitation,
}

/// Physics-based validation for weather observations
pub struct WeatherValidator {
    strict_mode: bool,
    climate_adjustment: bool,
}

impl WeatherValidator {
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            climate_adjustment: true,
        }
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    pub fn validate(&self, obs: &WeatherObservation) -> ValidationResult {
        let mut issues = Vec::new();
        let mut modified = false;
        let mut vars = obs.variables.clone();
        let climate_zone = ClimateZone::from_lat_lon(obs.coord.lat, obs.coord.lon);

        if let Some(temp) = vars.get(&WeatherVariable::Temperature) {
            let (min_temp, max_temp) = self.temperature_bounds(climate_zone);
            if temp.value < min_temp || temp.value > max_temp {
                issues.push(ValidationIssue::TemperatureOutOfRange {
                    value: temp.value,
                    min: min_temp,
                    max: max_temp,
                });
                if self.strict_mode {
                    vars.insert(WeatherVariable::Temperature, ObservedValue::missing());
                }
            }
        }

        if let Some(humidity) = vars.get(&WeatherVariable::RelativeHumidity) {
            if humidity.value < 0.0 || humidity.value > 100.0 {
                issues.push(ValidationIssue::HumidityOutOfRange {
                    value: humidity.value,
                    min: 0.0,
                    max: 100.0,
                });
                if humidity.value < 0.0 {
                    modified = true;
                }
                vars.insert(
                    WeatherVariable::RelativeHumidity,
                    ObservedValue::new(humidity.value.clamp(0.0, 100.0), humidity.confidence),
                );
            }
        }

        if let Some(pressure) = vars.get(&WeatherVariable::Pressure) {
            if pressure.value < 80000.0 || pressure.value > 110000.0 {
                issues.push(ValidationIssue::PressureOutOfRange {
                    value: pressure.value,
                    min: 80000.0,
                    max: 110000.0,
                });
            }
        }

        if let (
            Some(&ObservedValue { value: wind_u, .. }),
            Some(&ObservedValue { value: wind_v, .. }),
        ) = (
            vars.get(&WeatherVariable::WindU),
            vars.get(&WeatherVariable::WindV),
        ) {
            let computed_speed = (wind_u.powi(2) + wind_v.powi(2)).sqrt();

            if let Some(&ObservedValue {
                value: reported_speed,
                ..
            }) = vars.get(&WeatherVariable::WindSpeed)
            {
                let diff = (computed_speed - reported_speed).abs();
                if diff > 5.0 {
                    issues.push(ValidationIssue::InconsistentWindComponents {
                        u: wind_u,
                        v: wind_v,
                        speed: reported_speed,
                    });

                    vars.insert(
                        WeatherVariable::WindSpeed,
                        ObservedValue::new(computed_speed, 0.85),
                    );
                    modified = true;
                }
            }

            if computed_speed > 100.0 {
                issues.push(ValidationIssue::WindSpeedExcessive {
                    value: computed_speed,
                    max: 100.0,
                });
            }
        }

        if let Some(precip) = vars.get(&WeatherVariable::Precipitation) {
            if precip.value < 0.0 {
                issues.push(ValidationIssue::NegativePrecipitation);
                vars.insert(
                    WeatherVariable::Precipitation,
                    ObservedValue::new(0.0, precip.confidence),
                );
                modified = true;
            }
        }

        let quality_flag = if issues.is_empty() {
            QualityFlag::Valid
        } else if issues
            .iter()
            .any(|i| matches!(i, ValidationIssue::InvalidCoordinate))
        {
            QualityFlag::Erroneous
        } else {
            QualityFlag::Suspect
        };

        ValidationResult {
            observation: WeatherObservation {
                station_id: obs.station_id.clone(),
                coord: obs.coord,
                time: obs.time,
                variables: vars,
                quality_flag,
                source: obs.source,
            },
            was_modified: modified,
            issues,
        }
    }

    fn temperature_bounds(&self, zone: ClimateZone) -> (f64, f64) {
        match zone {
            ClimateZone::Polar => (-273.15, 30.0),
            ClimateZone::Tropical => (260.0, 330.0),
            ClimateZone::Arid => (230.0, 340.0),
            ClimateZone::Temperate => (200.0, 320.0),
            ClimateZone::Continental => (200.0, 320.0),
            ClimateZone::Oceanic => (200.0, 310.0),
            ClimateZone::Mediterranean => (220.0, 325.0),
            ClimateZone::Subtropical => (250.0, 330.0),
        }
    }
}

impl Default for WeatherValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Physics-informed gap filling
pub struct GapFiller {
    climatology: HashMap<WeatherVariable, (f64, f64)>,
}

impl GapFiller {
    pub fn new() -> Self {
        let mut climatology = HashMap::new();

        climatology.insert(WeatherVariable::Temperature, (280.0, 15.0));
        climatology.insert(WeatherVariable::RelativeHumidity, (70.0, 20.0));
        climatology.insert(WeatherVariable::Precipitation, (2.0, 5.0));
        climatology.insert(WeatherVariable::WindSpeed, (5.0, 3.0));
        climatology.insert(WeatherVariable::Pressure, (101325.0, 1000.0));
        climatology.insert(WeatherVariable::CloudCover, (50.0, 30.0));

        Self { climatology }
    }

    pub fn fill_gaps(&self, observations: &mut Vec<WeatherObservation>) {
        for obs in observations.iter_mut() {
            for var in WeatherVariable::all() {
                if !obs.variables.contains_key(&var) {
                    if let Some((mean, _std)) = self.climatology.get(&var) {
                        obs.variables.insert(var, ObservedValue::new(*mean, 0.3));
                    }
                }
            }
        }
    }

    pub fn interpolate_temporal(
        &self,
        observations: &mut Vec<WeatherObservation>,
        max_gap_hours: i64,
    ) {
        observations.sort_by_key(|o| o.time);

        let mut i = 0;
        while i < observations.len() {
            let start = i;

            while i < observations.len() && observations[i].variables.values().all(|v| v.is_valid) {
                i += 1;
            }

            if i < observations.len() {
                let mut end = i + 1;
                let gap_start_time = observations[i].time;

                while end < observations.len() {
                    let gap = (observations[end].time - gap_start_time).num_hours();
                    if gap > max_gap_hours {
                        break;
                    }
                    if observations[end].variables.values().all(|v| v.is_valid) {
                        break;
                    }
                    end += 1;
                }

                if end < observations.len() && end > start + 1 {
                    let t0 = observations[start].time;
                    let t1 = observations[end].time;
                    let dt = (t1 - t0).num_seconds() as f64;

                    for j in (i + 1)..end {
                        let t = observations[j].time;
                        let alpha = (t - t0).num_seconds() as f64 / dt;

                        for var in WeatherVariable::all() {
                            let v0_value = observations[start].variables.get(&var).map(|v| v.value);
                            let v1_value = observations[end].variables.get(&var).map(|v| v.value);

                            if let (Some(v0), Some(v1)) = (v0_value, v1_value) {
                                if let Some(vj) = observations[j].variables.get_mut(&var) {
                                    if !vj.is_valid {
                                        let interpolated = v0 + alpha * (v1 - v0);
                                        let confidence = vj.confidence * (1.0 - alpha.abs() as f32);
                                        *vj = ObservedValue::new(interpolated, confidence);
                                    }
                                }
                            }
                        }
                    }
                }

                i = end;
            }
        }
    }
}

impl Default for GapFiller {
    fn default() -> Self {
        Self::new()
    }
}

/// Wind vector consistency check
pub fn compute_wind_components(speed: f64, direction_deg: f64) -> (f64, f64) {
    let dir_rad = direction_deg.to_radians();
    let u = -speed * dir_rad.sin();
    let v = -speed * dir_rad.cos();
    (u, v)
}

pub fn compute_wind_speed(u: f64, v: f64) -> f64 {
    (u.powi(2) + v.powi(2)).sqrt()
}

pub fn compute_wind_direction(u: f64, v: f64) -> f64 {
    let dir_rad = v.atan2(u);
    let deg = dir_rad.to_degrees();
    if deg < 0.0 {
        deg + 360.0
    } else {
        deg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_components() {
        let (u, v) = compute_wind_components(10.0, 0.0);
        assert!((v - (-10.0)).abs() < 0.1);

        let (u, v) = compute_wind_components(10.0, 90.0);
        assert!((u - (-10.0)).abs() < 0.1);
    }

    #[test]
    fn test_validator() {
        let validator = WeatherValidator::new();

        let mut obs = WeatherObservation {
            station_id: None,
            coord: GeoCoord::new(52.52, 13.405),
            time: Utc::now(),
            variables: std::collections::HashMap::new(),
            quality_flag: QualityFlag::Valid,
            source: DataSource::ERA5,
        };

        obs.variables
            .insert(WeatherVariable::Temperature, ObservedValue::new(300.0, 0.9));
        obs.variables.insert(
            WeatherVariable::RelativeHumidity,
            ObservedValue::new(50.0, 0.9),
        );

        let result = validator.validate(&obs);
        assert!(result.issues.is_empty());
    }
}
