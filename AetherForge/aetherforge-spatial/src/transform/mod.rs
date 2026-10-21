//! # Coordinate Transformation Utilities
//!
//! Precise coordinate transformations between different coordinate systems.

use serde::{Deserialize, Serialize};

/// Coordinate system types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateSystem {
    WGS84,
    ETRS89,
    NAD83,
    GRS80,
}

impl Default for CoordinateSystem {
    fn default() -> Self {
        Self::WGS84
    }
}

/// Coordinate transformer for different coordinate systems and projections
pub struct CoordinateTransformer {
    source_system: CoordinateSystem,
    target_system: CoordinateSystem,
}

impl CoordinateTransformer {
    pub fn new(source: CoordinateSystem, target: CoordinateSystem) -> Self {
        Self {
            source_system: source,
            target_system: target,
        }
    }

    pub fn transform(&self, lat: f64, lon: f64, alt: Option<f64>) -> (f64, f64, Option<f64>) {
        match (self.source_system, self.target_system) {
            (CoordinateSystem::WGS84, CoordinateSystem::WGS84) => (lat, lon, alt),
            (CoordinateSystem::WGS84, CoordinateSystem::ETRS89) => {
                self.wgs84_to_etrs89(lat, lon, alt)
            }
            _ => (lat, lon, alt),
        }
    }

    fn wgs84_to_etrs89(&self, lat: f64, lon: f64, alt: Option<f64>) -> (f64, f64, Option<f64>) {
        let dlat = -0.000000062; // Simplified transformation
        let dlon = 0.000000042;
        let dat = -0.000000048;

        (
            lat + dlat * 3600.0,
            lon + dlon * 3600.0,
            alt.map(|a| a + dat),
        )
    }

    pub fn transform_bulk(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points
            .iter()
            .map(|(lat, lon)| {
                let (tlat, tlon, _) = self.transform(*lat, *lon, None);
                (tlat, tlon)
            })
            .collect()
    }
}

/// Convert lat/lon to Cartesian (ECEF) coordinates
pub fn lat_lon_to_ecef(lat: f64, lon: f64, alt: f64) -> (f64, f64, f64) {
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    const A: f64 = 6378137.0; // WGS84 semi-major axis
    const F: f64 = 1.0 / 298.257223563; // WGS84 flattening
    const E2: f64 = 2.0 * F - F * F;

    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();

    let x = (n + alt) * cos_lat * cos_lon;
    let y = (n + alt) * cos_lat * sin_lon;
    let z = (n * (1.0 - E2) + alt) * sin_lat;

    (x, y, z)
}

/// Convert ECEF to lat/lon
pub fn ecef_to_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    const A: f64 = 6378137.0;
    const F: f64 = 1.0 / 298.257223563;
    const E2: f64 = 2.0 * F - F * F;

    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();
    let mut lat = z.atan2(p * (1.0 - E2));

    for _ in 0..5 {
        let sin_lat = lat.sin();
        let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
        lat = (z + E2 * n * sin_lat).atan2(p);
    }

    let sin_lat = lat.sin();
    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
    let alt = p / lat.cos() - n;

    (lat.to_degrees(), lon.to_degrees(), alt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecef_roundtrip() {
        let (x, y, z) = lat_lon_to_ecef(52.52, 13.405, 100.0);
        let (lat, lon, alt) = ecef_to_lat_lon(x, y, z);

        assert!((lat - 52.52).abs() < 0.0001);
        assert!((lon - 13.405).abs() < 0.0001);
        assert!((alt - 100.0).abs() < 0.001);
    }
}
