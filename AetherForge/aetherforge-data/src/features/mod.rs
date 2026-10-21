//! # Feature Engineering for Weather ML
//!
//! Advanced feature engineering including physics-derived features.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use aetherforge_core::{ClimateZone, GeoCoord, ObservedValue, WeatherVariable};

/// Feature names for ML input
pub const SURFACE_FEATURES: &[&str] = &[
    "temperature_2m",
    "relative_humidity",
    "dewpoint",
    "precipitation",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
    "cloud_cover",
    "shortwave_radiation",
];

/// Atmospheric level features (per pressure level)
pub const ATMOSPHERE_FEATURES: &[&str] = &[
    "temperature",
    "geopotential_height",
    "specific_humidity",
    "u_wind",
    "v_wind",
    "vertical_velocity",
];

/// Pressure levels (hPa) for atmospheric data
pub const PRESSURE_LEVELS: &[u16] = &[
    1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100,
];

/// Build feature vector from observations
pub fn build_surface_features(
    obs: &HashMap<WeatherVariable, ObservedValue>,
    coord: GeoCoord,
    time: DateTime<Utc>,
) -> Vec<f32> {
    let mut features = Vec::with_capacity(SURFACE_FEATURES.len() + 8);

    for var_name in SURFACE_FEATURES {
        let var = match *var_name {
            "temperature_2m" => WeatherVariable::Temperature,
            "relative_humidity" => WeatherVariable::RelativeHumidity,
            "dewpoint" => WeatherVariable::DewPoint,
            "precipitation" => WeatherVariable::Precipitation,
            "surface_pressure" | "pressure" => WeatherVariable::Pressure,
            "wind_speed" => WeatherVariable::WindSpeed,
            "wind_direction" => WeatherVariable::WindDirection,
            "cloud_cover" => WeatherVariable::CloudCover,
            "shortwave_radiation" => WeatherVariable::SolarRadiation,
            _ => continue,
        };

        if let Some(val) = obs.get(&var) {
            features.push(val.value as f32);
        } else {
            features.push(0.0);
        }
    }

    features.push(coord.lat as f32 / 90.0);
    features.push(coord.lon as f32 / 180.0);
    features.push((coord.altitude_m.unwrap_or(0.0) as f32) / 5000.0);

    features.push((time.hour() as f32 / 24.0) * 2.0 * std::f32::consts::PI);
    features.push(((time.hour() as f32 / 24.0) * 2.0 * std::f32::consts::PI).cos());
    features.push((time.month() as f32 / 12.0) * 2.0 * std::f32::consts::PI);
    features.push(((time.month() as f32 / 12.0) * 2.0 * std::f32::consts::PI).cos());

    let climate = ClimateZone::from_lat_lon(coord.lat, coord.lon);
    features.push(climate as u8 as f32 / 8.0);

    features
}

/// Compute derived physics features
pub fn compute_derived_features(
    obs: &HashMap<WeatherVariable, ObservedValue>,
) -> HashMap<String, f32> {
    let mut derived = HashMap::new();

    if let (
        Some(&ObservedValue { value: temp, .. }),
        Some(&ObservedValue {
            value: humidity, ..
        }),
    ) = (
        obs.get(&WeatherVariable::Temperature),
        obs.get(&WeatherVariable::RelativeHumidity),
    ) {
        let dewpoint = compute_dewpoint(temp, humidity);
        derived.insert("estimated_dewpoint".to_string(), dewpoint as f32);

        let heat_index = compute_heat_index(temp, humidity);
        derived.insert("heat_index".to_string(), heat_index as f32);
    }

    if let (Some(&ObservedValue { value: u, .. }), Some(&ObservedValue { value: v, .. })) = (
        obs.get(&WeatherVariable::WindU),
        obs.get(&WeatherVariable::WindV),
    ) {
        let wind_speed = (u.powi(2) + v.powi(2)).sqrt();
        let wind_dir = v.atan2(u).to_degrees() % 360.0;

        derived.insert("computed_wind_speed".to_string(), wind_speed as f32);
        derived.insert("computed_wind_direction".to_string(), wind_dir as f32);
    }

    if let Some(&ObservedValue {
        value: pressure, ..
    }) = obs.get(&WeatherVariable::Pressure)
    {
        derived.insert("pressure_hpa".to_string(), (pressure / 100.0) as f32);
    }

    derived
}

/// Compute dewpoint from temperature and relative humidity (Magnus formula)
fn compute_dewpoint(temperature_k: f64, humidity: f64) -> f64 {
    let temp_c = temperature_k - 273.15;
    let gamma = humidity / 100.0 * (17.62 * temp_c / (243.12 + temp_c)).exp();
    243.12 * gamma.ln() / (17.62 - gamma.ln()) + 273.15
}

/// Compute heat index from temperature and humidity
fn compute_heat_index(temperature_k: f64, humidity: f64) -> f64 {
    let temp_c = temperature_k - 273.15;
    let t = temp_c;
    let r = humidity;

    if t < 20.0 {
        return temperature_k;
    }

    let hi = -8.78469475556 + 1.61139411 * t + 2.33854883889 * r
        - 0.14611605 * t * r
        - 0.012308094 * t * t
        - 0.0164248277778 * r * r
        + 0.000002211908 * t * t * r
        + 0.000000425497 * t * r * r;

    (hi + 273.15).max(temperature_k)
}

/// Lagged feature builder
pub fn build_lagged_features(
    history: &[(DateTime<Utc>, HashMap<WeatherVariable, ObservedValue>)],
    target_time: DateTime<Utc>,
    lag_hours: &[u16],
) -> Vec<f32> {
    let mut lagged = Vec::new();

    for &lag in lag_hours {
        let lag_time = target_time - chrono::Duration::hours(lag as i64);

        let closest = history
            .iter()
            .min_by_key(|(t, _)| (t.timestamp() - lag_time.timestamp()).abs());

        if let Some((_, obs)) = closest {
            for var in [
                WeatherVariable::Temperature,
                WeatherVariable::RelativeHumidity,
                WeatherVariable::Precipitation,
                WeatherVariable::WindSpeed,
            ] {
                if let Some(val) = obs.get(&var) {
                    lagged.push(val.value as f32);
                } else {
                    lagged.push(0.0);
                }
            }
        } else {
            lagged.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
    }

    lagged
}

/// Rolling statistics over time window
pub fn compute_rolling_stats(values: &[f64], window_hours: u16) -> RollingStats {
    if values.is_empty() {
        return RollingStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            trend: 0.0,
        };
    }

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let trend = if n > 1 {
        let half = n / 2;
        let first_half: f64 = values[..half].iter().sum::<f64>() / half as f64;
        let second_half: f64 = values[half..].iter().sum::<f64>() / (n - half) as f64;
        second_half - first_half
    } else {
        0.0
    };

    RollingStats {
        mean,
        std,
        min,
        max,
        trend,
    }
}

#[derive(Debug, Clone)]
pub struct RollingStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub trend: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_rolling_stats(&values, 24);

        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.std - 1.41).abs() < 0.1);
    }
}
