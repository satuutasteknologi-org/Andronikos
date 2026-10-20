//! # Utility Functions
//!
//! Common utility functions for the weather intelligence engine.

use chrono::{DateTime, Duration, Utc};
use h3o::{CellIndex, LatLng, Resolution};
use std::collections::HashMap;

/// Convert latitude/longitude to H3 cell index
pub fn lat_lon_to_h3(lat: f64, lon: f64, resolution: u8) -> Option<CellIndex> {
    let resolution = Resolution::try_from(resolution).ok()?;
    let latlng = LatLng::new(lat, lon).ok()?;
    Some(latlng.to_cell(resolution))
}

/// Convert H3 cell index to latitude/longitude (center)
pub fn h3_to_lat_lon(cell: CellIndex) -> (f64, f64) {
    let latlng: LatLng = cell.into();
    (latlng.lat(), latlng.lng())
}

/// Get H3 cell boundary as polygon coordinates
pub fn h3_cell_boundary(cell: CellIndex) -> Vec<(f64, f64)> {
    cell.boundary()
        .iter()
        .map(|ll| (ll.lat(), ll.lng()))
        .collect()
}

/// Get H3 cells within a bounding box
pub fn h3_cells_in_bbox(
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
    resolution: u8,
) -> Vec<CellIndex> {
    let resolution = match Resolution::try_from(resolution) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let min_latlng = match LatLng::new(min_lat, min_lon) {
        Ok(ll) => ll,
        Err(_) => return Vec::new(),
    };
    let max_latlng = match LatLng::new(max_lat, max_lon) {
        Ok(ll) => ll,
        Err(_) => return Vec::new(),
    };

    let min_cell = min_latlng.to_cell(resolution);
    let max_cell = max_latlng.to_cell(resolution);
    let distance = min_cell.grid_distance(max_cell).unwrap_or(0) as u32;

    min_cell.grid_disk::<Vec<CellIndex>>(distance)
}

/// Haversine distance between two coordinates in meters
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371000.0; // Earth radius in meters
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
}

/// Bearing from point 1 to point 2 in degrees
pub fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let y = dlon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();
    let bearing = y.atan2(x).to_degrees();
    (bearing + 360.0) % 360.0
}

/// Wind vector components from speed and direction
pub fn wind_components(speed: f64, direction_deg: f64) -> (f64, f64) {
    let dir_rad = direction_deg.to_radians();
    let u = -speed * dir_rad.sin(); // East-West (positive = eastward)
    let v = -speed * dir_rad.cos(); // North-South (positive = northward)
    (u, v)
}

/// Wind speed and direction from components
pub fn wind_speed_direction(u: f64, v: f64) -> (f64, f64) {
    let speed = (u.powi(2) + v.powi(2)).sqrt();
    let dir_rad = v.atan2(u);
    let direction = dir_rad.to_degrees();
    (speed, (direction + 360.0) % 360.0)
}

/// Linear interpolation between two values
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Clamp value to range
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Convert Unix timestamp to DateTime
pub fn timestamp_to_datetime(ts: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(ts, 0).unwrap_or_else(Utc::now)
}

/// Convert DateTime to Unix timestamp
pub fn datetime_to_timestamp(dt: DateTime<Utc>) -> i64 {
    dt.timestamp()
}

/// Round timestamp to nearest interval (seconds)
pub fn round_timestamp(dt: DateTime<Utc>, interval_secs: u32) -> DateTime<Utc> {
    let ts = dt.timestamp();
    let interval = interval_secs as i64;
    let rounded = (ts as f64 / interval as f64).round() as i64 * interval;
    timestamp_to_datetime(rounded)
}

/// Generate time range with step
pub fn time_range(start: DateTime<Utc>, end: DateTime<Utc>, step: Duration) -> Vec<DateTime<Utc>> {
    let mut times = Vec::new();
    let mut current = start;
    while current <= end {
        times.push(current);
        current += step;
    }
    times
}

/// Parse ISO8601 datetime string
pub fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Format datetime as ISO8601
pub fn format_datetime(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

/// Compute statistics for a slice of f64 values
pub fn compute_stats(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let sum: f64 = sorted.iter().sum();
    let mean = sum / sorted.len() as f64;

    let variance: f64 =
        sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
    let std = variance.sqrt();

    (min, max, mean, std, variance)
}

/// Compute percentiles
pub fn compute_percentiles(values: &[f64], percentiles: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0; percentiles.len()];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    percentiles
        .iter()
        .map(|&p| {
            let idx = (p * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

/// Running statistics for streaming computation
#[derive(Debug, Clone, Default)]
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squared differences
    min: f64,
    max: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    pub fn min(&self) -> f64 {
        self.min
    }
    pub fn max(&self) -> f64 {
        self.max
    }
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// Simple LRU cache for predictions
pub struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: Vec<K>,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.map.get(key).cloned() {
            // Move to front
            self.order.retain(|k| k != key);
            self.order.insert(0, key.clone());
            Some(value)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.map.len() >= self.capacity {
            if let Some(lru) = self.order.pop() {
                self.map.remove(&lru);
            }
        }
        self.order.insert(0, key.clone());
        self.map.insert(key, value);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine() {
        let dist = haversine_distance(52.52, 13.405, 48.8566, 2.3522); // Berlin to Paris
        assert!((dist - 878_000.0).abs() < 10_000.0);
    }

    #[test]
    fn test_wind_components() {
        let (u, v) = wind_components(10.0, 0.0); // North wind
        assert!((v + 10.0).abs() < 0.1);
        assert!(u.abs() < 0.1);

        let (u, v) = wind_components(10.0, 90.0); // East wind
        assert!((u + 10.0).abs() < 0.1);
        assert!(v.abs() < 0.1);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.add(v);
        }
        assert_eq!(stats.mean(), 3.0);
        assert_eq!(stats.count(), 5);
        assert!((stats.std_dev() - 1.5811).abs() < 0.01);
    }
}
