//! # Data Validation and Cleaning
//!
//! Physics-based validation and cleaning for weather observations.

use chrono::{Datelike, Timelike};
use std::collections::HashMap;

use wi_core::{ClimateZone, ObservedValue, QualityFlag, WeatherObservation, WeatherVariable};

/// Validation result with optional corrections
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub observation: WeatherObservation,
    pub modified: bool,
    pub issues: Vec<ValidationIssue>,
}

/// Validation issue types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    TemperatureOutOfRange { value: f64, min: f64, max: f64 },
    HumidityOutOfRange { value: f64 },
    PressureOutOfRange { value: f64 },
    WindSpeedExcessive { value: f64 },
    WindComponentsInconsistent { speed: f64, u: f64, v: f64 },
    PrecipitationNegative { value: f64 },
    MissingRequiredVariables,
    InvalidCoordinates { lat: f64, lon: f64 },
    DuplicateTimestamp,
}

impl ValidationIssue {
    pub fn severity(&self) -> QualityFlag {
        match self {
            ValidationIssue::TemperatureOutOfRange { .. } => QualityFlag::Suspect,
            ValidationIssue::HumidityOutOfRange { .. } => QualityFlag::Suspect,
            ValidationIssue::PressureOutOfRange { .. } => QualityFlag::Suspect,
            ValidationIssue::WindSpeedExcessive { .. } => QualityFlag::Suspect,
            ValidationIssue::WindComponentsInconsistent { .. } => QualityFlag::Erroneous,
            ValidationIssue::PrecipitationNegative { .. } => QualityFlag::Erroneous,
            ValidationIssue::MissingRequiredVariables => QualityFlag::Suspect,
            ValidationIssue::InvalidCoordinates { .. } => QualityFlag::Erroneous,
            ValidationIssue::DuplicateTimestamp => QualityFlag::Suspect,
        }
    }
}

/// Physics-based weather validator
pub struct WeatherValidator {
    temp_bounds: HashMap<ClimateZone, (f64, f64)>,
}

impl Default for WeatherValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl WeatherValidator {
    pub fn new() -> Self {
        let mut temp_bounds = HashMap::new();
        temp_bounds.insert(ClimateZone::Tropical, (263.15, 313.15));
        temp_bounds.insert(ClimateZone::Subtropical, (253.15, 318.15));
        temp_bounds.insert(ClimateZone::Temperate, (233.15, 313.15));
        temp_bounds.insert(ClimateZone::Continental, (213.15, 293.15));
        temp_bounds.insert(ClimateZone::Polar, (193.15, 273.15));
        temp_bounds.insert(ClimateZone::Arid, (253.15, 323.15));
        temp_bounds.insert(ClimateZone::Mediterranean, (263.15, 313.15));
        temp_bounds.insert(ClimateZone::Oceanic, (273.15, 298.15));

        Self { temp_bounds }
    }

    pub fn validate(&self, mut obs: WeatherObservation) -> ValidationResult {
        let mut issues = Vec::new();

        let zone = ClimateZone::from_lat_lon(obs.coord.lat, obs.coord.lon);
        let (t_min, t_max) = self.temp_bounds.get(&zone).unwrap_or(&(193.15, 323.15));

        if let Some(temp) = obs.variables.get(&WeatherVariable::Temperature) {
            if temp.is_valid && (temp.value < *t_min || temp.value > *t_max) {
                issues.push(ValidationIssue::TemperatureOutOfRange {
                    value: temp.value,
                    min: *t_min,
                    max: *t_max,
                });
            }
        }

        if let Some(hum) = obs.variables.get(&WeatherVariable::RelativeHumidity) {
            if hum.is_valid && (hum.value < 0.0 || hum.value > 100.0) {
                issues.push(ValidationIssue::HumidityOutOfRange { value: hum.value });
            }
        }

        if let Some(pres) = obs.variables.get(&WeatherVariable::Pressure) {
            if pres.is_valid && (pres.value < 30000.0 || pres.value > 110000.0) {
                issues.push(ValidationIssue::PressureOutOfRange { value: pres.value });
            }
        }

        if let Some(ws) = obs.variables.get(&WeatherVariable::WindSpeed) {
            if ws.is_valid && ws.value > 100.0 {
                issues.push(ValidationIssue::WindSpeedExcessive { value: ws.value });
            }
        }

        if let (Some(ws), Some(wd), Some(wu), Some(wv)) = (
            obs.variables.get(&WeatherVariable::WindSpeed),
            obs.variables.get(&WeatherVariable::WindDirection),
            obs.variables.get(&WeatherVariable::WindU),
            obs.variables.get(&WeatherVariable::WindV),
        ) {
            if ws.is_valid && wd.is_valid && wu.is_valid && wv.is_valid {
                let (calc_u, calc_v) = wind_components(ws.value, wd.value);
                let diff_u = (calc_u - wu.value).abs();
                let diff_v = (calc_v - wv.value).abs();
                if diff_u > 1.0 || diff_v > 1.0 {
                    issues.push(ValidationIssue::WindComponentsInconsistent {
                        speed: ws.value,
                        u: wu.value,
                        v: wv.value,
                    });
                }
            }
        }

        if let Some(precip) = obs.variables.get(&WeatherVariable::Precipitation) {
            if precip.is_valid && precip.value < 0.0 {
                issues.push(ValidationIssue::PrecipitationNegative {
                    value: precip.value,
                });
            }
        }

        if obs.coord.lat < -90.0
            || obs.coord.lat > 90.0
            || obs.coord.lon < -180.0
            || obs.coord.lon > 180.0
        {
            issues.push(ValidationIssue::InvalidCoordinates {
                lat: obs.coord.lat,
                lon: obs.coord.lon,
            });
        }

        let quality = if issues
            .iter()
            .any(|i| i.severity() == QualityFlag::Erroneous)
        {
            QualityFlag::Erroneous
        } else if issues.iter().any(|i| i.severity() == QualityFlag::Suspect) {
            QualityFlag::Suspect
        } else {
            QualityFlag::Valid
        };

        obs.quality_flag = quality;

        ValidationResult {
            observation: obs,
            modified: !issues.is_empty(),
            issues,
        }
    }
}

/// Climatology-based gap filling and temporal interpolation
pub struct GapFiller {
    climatology: HashMap<(WeatherVariable, u8, u8), f64>,
}

impl Default for GapFiller {
    fn default() -> Self {
        Self::new()
    }
}

impl GapFiller {
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
            self.climatology.insert(key, sum / count as f64);
        }
    }

    pub fn fill_missing(&self, obs: &mut WeatherObservation) {
        let month = obs.time.month() as u8;
        let hour = obs.time.hour() as u8;

        for var in WeatherVariable::surface_variables() {
            if let std::collections::hash_map::Entry::Vacant(e) = obs.variables.entry(var) {
                if let Some(&clim_value) = self.climatology.get(&(var, month, hour)) {
                    e.insert(ObservedValue::new(clim_value, 0.5));
                }
            }
        }
    }

    pub fn interpolate(
        &self,
        before: &WeatherObservation,
        after: &WeatherObservation,
        target_time: chrono::DateTime<chrono::Utc>,
    ) -> WeatherObservation {
        let t_before = before.time.timestamp() as f64;
        let t_after = after.time.timestamp() as f64;
        let t_target = target_time.timestamp() as f64;

        if t_after <= t_before {
            return before.clone();
        }

        let weight = (t_target - t_before) / (t_after - t_before);
        let weight = weight.clamp(0.0, 1.0) as f32;

        let mut result = before.clone();
        result.time = target_time;

        for (var, val_before) in &before.variables {
            if let Some(val_after) = after.variables.get(var) {
                if val_before.is_valid && val_after.is_valid {
                    let interp =
                        val_before.value + weight as f64 * (val_after.value - val_before.value);
                    let conf =
                        val_before.confidence * (1.0 - weight) + val_after.confidence * weight;
                    result
                        .variables
                        .insert(*var, ObservedValue::new(interp, conf));
                }
            }
        }

        result
    }
}

pub fn wind_components(speed: f64, direction: f64) -> (f64, f64) {
    let dir_rad = direction.to_radians();
    let u = -speed * dir_rad.sin();
    let v = -speed * dir_rad.cos();
    (u, v)
}

pub fn wind_speed_direction(u: f64, v: f64) -> (f64, f64) {
    let speed = (u * u + v * v).sqrt();
    let dir_rad = (-u).atan2(-v);
    let direction = dir_rad.to_degrees();
    let direction = if direction < 0.0 {
        direction + 360.0
    } else {
        direction
    };
    (speed, direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use std::collections::HashMap;
    use wi_core::types::{GeoCoord, ObservedValue};

    #[test]
    fn test_wind_components() {
        let (u, v) = wind_components(10.0, 90.0);
        assert!((u + 10.0).abs() < 0.1);
        assert!(v.abs() < 0.1);

        let (u, v) = wind_components(10.0, 0.0);
        assert!(u.abs() < 0.1);
        assert!((v + 10.0).abs() < 0.1);
    }

    #[test]
    fn test_wind_speed_direction() {
        let (speed, dir) = wind_speed_direction(-10.0, 0.0);
        assert!((speed - 10.0).abs() < 0.1);
        assert!((dir - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_validator() {
        let validator = WeatherValidator::new();

        let mut obs = WeatherObservation {
            station_id: Some("test".to_string()),
            coord: GeoCoord::new(0.0, 0.0),
            time: Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap(),
            variables: HashMap::new(),
            quality_flag: QualityFlag::Valid,
            source: wi_core::types::DataSource::OpenMeteo,
        };

        obs.variables
            .insert(WeatherVariable::Temperature, ObservedValue::new(300.0, 0.9));
        obs.variables.insert(
            WeatherVariable::RelativeHumidity,
            ObservedValue::new(80.0, 0.9),
        );
        obs.variables
            .insert(WeatherVariable::Pressure, ObservedValue::new(101325.0, 0.9));

        let result = validator.validate(obs);
        assert_eq!(result.observation.quality_flag, QualityFlag::Valid);
    }

    #[test]
    fn test_gap_filler() {
        let mut filler = GapFiller::new();

        let mut obs = WeatherObservation {
            station_id: Some("test".to_string()),
            coord: GeoCoord::new(0.0, 0.0),
            time: Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap(),
            variables: HashMap::new(),
            quality_flag: QualityFlag::Valid,
            source: wi_core::types::DataSource::OpenMeteo,
        };

        // Train with Temperature AND RelativeHumidity so both get climatology entries
        obs.variables
            .insert(WeatherVariable::Temperature, ObservedValue::new(300.0, 0.9));
        obs.variables.insert(
            WeatherVariable::RelativeHumidity,
            ObservedValue::new(80.0, 0.9),
        );

        filler.train(&[obs.clone()]);

        // Now test filling: remove RelativeHumidity and see if it gets filled
        obs.variables.remove(&WeatherVariable::RelativeHumidity);
        filler.fill_missing(&mut obs);

        assert!(obs
            .variables
            .contains_key(&WeatherVariable::RelativeHumidity));
        let rh = obs
            .variables
            .get(&WeatherVariable::RelativeHumidity)
            .unwrap();
        assert!((rh.value - 80.0).abs() < 1.0); // Should be filled with climatology value
    }
}
