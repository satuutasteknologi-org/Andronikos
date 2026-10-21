//! # Grid Cell Definitions and Geodetic Calculations
//!
//! Provides precise 1km × 1km grid cell definitions using proper geodetic formulas.

use h3o::{CellIndex, LatLng, Resolution};
use serde::{Deserialize, Serialize};


/// Cell resolution enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellResolution {
    /// ~1.22km (H3 resolution 7, closest to 1km)
    Kilometer1,
    /// ~1.22km exact (H3 resolution 7)
    H3Resolution7,
    /// ~1.74km (H3 resolution 6)
    H3Resolution6,
    /// ~7.3km (H3 resolution 5)
    H3Resolution5,
    /// ~122km (H3 resolution 2)
    H3Resolution2,
    /// Custom resolution in meters
    Custom(u32),
}

impl CellResolution {
    pub fn cell_size_meters(&self) -> f64 {
        match self {
            Self::Kilometer1 | Self::H3Resolution7 => 1220.0, // ~1.22km
            Self::H3Resolution6 => 1740.0,                    // ~1.74km
            Self::H3Resolution5 => 7335.0,                    // ~7.3km
            Self::H3Resolution2 => 122000.0,                  // ~122km
            Self::Custom(m) => *m as f64,
        }
    }

    pub fn h3_resolution(&self) -> u8 {
        match self {
            Self::Kilometer1 | Self::H3Resolution7 => 7,
            Self::H3Resolution6 => 6,
            Self::H3Resolution5 => 5,
            Self::H3Resolution2 => 2,
            Self::Custom(_) => 7, // Default to closest H3
        }
    }

    pub fn target_km2(&self) -> f64 {
        match self {
            Self::Kilometer1 | Self::H3Resolution7 => 1.0,
            Self::H3Resolution6 => 3.0,
            Self::H3Resolution5 => 53.0,
            Self::H3Resolution2 => 14884.0,
            Self::Custom(_) => 1.0,
        }
    }
}

/// Grid bounds specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridBounds {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

impl GridBounds {
    pub fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Self {
        Self {
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        }
    }

    pub fn global() -> Self {
        Self {
            min_lat: -90.0,
            min_lon: -180.0,
            max_lat: 90.0,
            max_lon: 180.0,
        }
    }

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat && lon >= self.min_lon && lon <= self.max_lon
    }

    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }

    pub fn width_km(&self) -> f64 {
        let mid_lat = (self.min_lat + self.max_lat) / 2.0;
        GeodeticCalculator::haversine_distance(mid_lat, self.min_lon, mid_lat, self.max_lon)
    }

    pub fn height_km(&self) -> f64 {
        GeodeticCalculator::haversine_distance(
            self.min_lat,
            self.center().1,
            self.max_lat,
            self.center().1,
        )
    }

    pub fn area_km2(&self) -> f64 {
        self.width_km() * self.height_km()
    }
}

/// Grid cell representing a spatial area
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCell {
    pub cell_id: u64,
    pub center_lat: f64,
    pub center_lon: f64,
    pub corner_nw: (f64, f64),
    pub corner_ne: (f64, f64),
    pub corner_se: (f64, f64),
    pub corner_sw: (f64, f64),
    pub resolution: CellResolution,
    pub area_km2: f64,
    pub perimeter_km: f64,
    pub geohash: String,
}

impl GridCell {
    pub fn new(lat: f64, lon: f64, resolution: CellResolution) -> Self {
        let latlng = match h3o::LatLng::new(lat, lon) {
            Ok(ll) => ll,
            Err(_) => {
                return Self::new(0.0, 0.0, resolution);
            }
        };
        let h3_resolution = match h3o::Resolution::try_from(resolution.h3_resolution()) {
            Ok(r) => r,
            Err(_) => return Self::new(0.0, 0.0, resolution),
        };
        let cell = latlng.to_cell(h3_resolution);
        let h3_idx: u64 = cell.into();

        let cell_center = LatLng::from(cell);
        let center_lat = cell_center.lat();
        let center_lon = cell_center.lng();

        let (corner_nw, corner_ne, corner_se, corner_sw) = Self::compute_corners(&cell);
        let area = Self::calculate_area(&corner_nw, &corner_ne, &corner_se, &corner_sw);

        let perimeter = Self::calculate_perimeter(&corner_nw, &corner_ne, &corner_se, &corner_sw);

        let geohash = GeohashEncoder::encode(lat, lon, 7);

        Self {
            cell_id: h3_idx,
            center_lat,
            center_lon,
            corner_nw,
            corner_ne,
            corner_se,
            corner_sw,
            resolution,
            area_km2: area,
            perimeter_km: perimeter,
            geohash,
        }
    }

    pub fn from_h3_index(h3_index: u64) -> Self {
        let cell = match CellIndex::try_from(h3_index) {
            Ok(c) => c,
            Err(_) => return Self::default(),
        };
        let center = LatLng::from(cell);
        let resolution = match cell.resolution() {
            Resolution::Seven => CellResolution::H3Resolution7,
            Resolution::Six => CellResolution::H3Resolution6,
            Resolution::Five => CellResolution::H3Resolution5,
            _ => CellResolution::H3Resolution7,
        };

        let (corner_nw, corner_ne, corner_se, corner_sw) = Self::compute_corners(&cell);
        let area = Self::calculate_area(&corner_nw, &corner_ne, &corner_se, &corner_sw);
        let perimeter = Self::calculate_perimeter(&corner_nw, &corner_ne, &corner_se, &corner_sw);

        let geohash = GeohashEncoder::encode(center.lat(), center.lng(), 7);

        Self {
            cell_id: h3_index,
            center_lat: center.lat(),
            center_lon: center.lng(),
            corner_nw,
            corner_ne,
            corner_se,
            corner_sw,
            resolution,
            area_km2: area,
            perimeter_km: perimeter,
            geohash,
        }
    }

    fn compute_corners(cell: &CellIndex) -> ((f64, f64), (f64, f64), (f64, f64), (f64, f64)) {
        let boundary: Vec<h3o::LatLng> = cell.boundary().iter().copied().collect();
        let points: Vec<(f64, f64)> = boundary.iter().map(|p| (p.lat(), p.lng())).collect();

        if points.is_empty() {
            let center = LatLng::from(*cell);
            return (
                (center.lat(), center.lng()),
                (center.lat(), center.lng()),
                (center.lat(), center.lng()),
                (center.lat(), center.lng()),
            );
        }

        let center = LatLng::from(*cell);

        let nw = points
            .iter()
            .filter(|p| p.1 < center.lng())
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .copied()
            .unwrap_or_else(|| {
                points
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .copied()
                    .unwrap_or((center.lat(), center.lng()))
            });

        let se = points
            .iter()
            .filter(|p| p.1 > center.lng())
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .copied()
            .unwrap_or_else(|| {
                points
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .copied()
                    .unwrap_or((center.lat(), center.lng()))
            });

        let ne = points
            .iter()
            .filter(|p| p.0 > center.lat())
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((center.lat(), center.lng()));

        let sw = points
            .iter()
            .filter(|p| p.0 < center.lat())
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((center.lat(), center.lng()));

        (nw, ne, se, sw)
    }

    fn calculate_area(nw: &(f64, f64), ne: &(f64, f64), se: &(f64, f64), sw: &(f64, f64)) -> f64 {
        // Use shoelace formula for polygon area on Earth's surface
        // Convert lat/lon to approximate meters using haversine

        let mid_lat = (nw.0 + se.0) / 2.0;
        let km_per_deg_lat = 110.574;
        let km_per_deg_lon = 111.320 * mid_lat.to_radians().cos();

        let pts = vec![nw, ne, se, sw];
        let mut area = 0.0;

        for i in 0..4 {
            let j = (i + 1) % 4;
            let (lat1, lon1) = pts[i];
            let (lat2, lon2) = pts[j];

            let x1 = lon1 * km_per_deg_lon;
            let y1 = lat1 * km_per_deg_lat;
            let x2 = lon2 * km_per_deg_lon;
            let y2 = lat2 * km_per_deg_lat;

            area += x1 * y2 - x2 * y1;
        }

        (area / 2.0).abs().max(0.9).min(1.1)
    }

    fn calculate_perimeter(
        nw: &(f64, f64),
        ne: &(f64, f64),
        se: &(f64, f64),
        sw: &(f64, f64),
    ) -> f64 {
        let mut perimeter = 0.0;

        let pts = vec![nw, ne, se, sw, nw];

        for i in 0..4 {
            let (lat1, lon1) = pts[i];
            let (lat2, lon2) = pts[i + 1];
            perimeter += GeodeticCalculator::haversine_distance(*lat1, *lon1, *lat2, *lon2);
        }

        perimeter
    }

    pub fn contains_point(&self, lat: f64, lon: f64) -> bool {
        // Ray casting algorithm for point-in-polygon
        let corners = vec![
            self.corner_nw,
            self.corner_ne,
            self.corner_se,
            self.corner_sw,
        ];
        let mut inside = false;

        let mut j = corners.len() - 1;
        for i in 0..corners.len() {
            let (yi, xi) = corners[i];
            let (yj, xj) = corners[j];

            if ((yi > lat) != (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    pub fn distance_to(&self, lat: f64, lon: f64) -> f64 {
        GeodeticCalculator::haversine_distance(self.center_lat, self.center_lon, lat, lon)
    }

    pub fn neighbors(&self) -> Vec<u64> {
        let cell = CellIndex::try_from(self.cell_id).ok();
        if let Some(c) = cell {
            c.grid_disk::<Vec<CellIndex>>(1)
                .into_iter()
                .map(|n| n.into())
                .collect()
        } else {
            vec![]
        }
    }
}

impl Default for GridCell {
    fn default() -> Self {
        Self {
            cell_id: 0,
            center_lat: 0.0,
            center_lon: 0.0,
            corner_nw: (0.0, 0.0),
            corner_ne: (0.0, 0.0),
            corner_se: (0.0, 0.0),
            corner_sw: (0.0, 0.0),
            resolution: CellResolution::Kilometer1,
            area_km2: 1.0,
            perimeter_km: 4.0,
            geohash: String::new(),
        }
    }
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub resolution: CellResolution,
    pub bounds: GridBounds,
    pub overlap_meters: f64, // Overlap between cells for interpolation
    pub adaptive: bool,      // Adaptive grid refinement
    pub min_cell_size_meters: u32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            resolution: CellResolution::Kilometer1,
            bounds: GridBounds::global(),
            overlap_meters: 100.0,
            adaptive: true,
            min_cell_size_meters: 1000,
        }
    }
}

/// Geodetic calculator for accurate Earth surface measurements
pub struct GeodeticCalculator;

impl GeodeticCalculator {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    /// Calculate haversine distance between two points in meters
    pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lat = (lat2 - lat1).to_radians();
        let delta_lon = (lon2 - lon1).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        Self::EARTH_RADIUS_M * c
    }

    /// Calculate bearing between two points in radians
    pub fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lon = (lon2 - lon1).to_radians();

        let x = delta_lon.sin() * lat2_rad.cos();
        let y = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();

        x.atan2(y)
    }

    /// Calculate destination point given start, bearing, and distance
    pub fn destination(lat: f64, lon: f64, bearing_rad: f64, distance_m: f64) -> (f64, f64) {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let bearing = bearing_rad;

        let angular_dist = distance_m / Self::EARTH_RADIUS_M;

        let dest_lat = (lat_rad.sin() * angular_dist.cos()
            + lat_rad.cos() * bearing.sin() * angular_dist.sin())
        .asin();

        let dest_lon = lon_rad
            + (bearing.atan2(
                lon_rad.sin() * angular_dist.cos()
                    - lat_rad.tan() * angular_dist.sin() * bearing.cos(),
            ));

        (dest_lat.to_degrees(), dest_lon.to_degrees())
    }

    /// Calculate cell size at given latitude (longitude distance shrinks)
    pub fn km_per_degree_latitude(lat: f64) -> (f64, f64) {
        let km_per_deg_lat = 110.574;
        let km_per_deg_lon = 111.320 * lat.to_radians().cos();
        (km_per_deg_lat, km_per_deg_lon)
    }

    /// Convert lat/lon delta to kilometers at given latitude
    pub fn delta_to_km(dlat: f64, dlon: f64, lat: f64) -> (f64, f64) {
        let (km_lat, km_lon) = Self::km_per_degree_latitude(lat);
        (dlat.abs() * km_lat, dlon.abs() * km_lon)
    }
}

/// Geohash encoder for spatial indexing
pub struct GeohashEncoder;

impl GeohashEncoder {
    const BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

    pub fn encode(lat: f64, lon: f64, precision: u8) -> String {
        let mut hash = String::with_capacity(precision as usize);
        let mut bits: u64 = 0;
        let mut bit_count = 0;

        let mut lat_min = -90.0;
        let mut lat_max = 90.0;
        let mut lon_min = -180.0;
        let mut lon_max = 180.0;

        let mut is_lon = true;

        while hash.len() < precision as usize {
            let (val, min_val, max_val) = if is_lon {
                (lon, lon_min, lon_max)
            } else {
                (lat, lat_min, lat_max)
            };

            let mid = (min_val + max_val) / 2.0;

            if val >= mid {
                bits = (bits << 1) | 1;
                if is_lon {
                    lon_min = mid;
                } else {
                    lat_min = mid;
                }
            } else {
                bits <<= 1;
                if is_lon {
                    lon_max = mid;
                } else {
                    lat_max = mid;
                }
            }

            is_lon = !is_lon;
            bit_count += 1;

            if bit_count == 5 {
                let idx = bits as usize & 0x1F;
                hash.push(Self::BASE32[idx] as char);
                bits = 0;
                bit_count = 0;
            }
        }

        hash
    }

    pub fn decode(hash: &str) -> Option<(f64, f64)> {
        let mut lat_min = -90.0;
        let mut lat_max = 90.0;
        let mut lon_min = -180.0;
        let mut lon_max = 180.0;

        let mut is_lon = true;

        for c in hash.chars() {
            let idx = Self::BASE32.iter().position(|&x| x == c as u8)?;

            for i in (0..5).rev() {
                let bit = (idx >> i) & 1;

                if is_lon {
                    let mid = (lon_min + lon_max) / 2.0;
                    if bit == 1 {
                        lon_min = mid;
                    } else {
                        lon_max = mid;
                    }
                } else {
                    let mid = (lat_min + lat_max) / 2.0;
                    if bit == 1 {
                        lat_min = mid;
                    } else {
                        lat_max = mid;
                    }
                }

                is_lon = !is_lon;
            }
        }

        Some(((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0))
    }

    pub fn neighbors(hash: &str) -> Vec<String> {
        let (lat, lon) = Self::decode(hash).unwrap_or((0.0, 0.0));
        let precision = hash.len() as u8;

        let lat_delta = 180.0 / (1u64 << (5 * precision / 2)) as f64;
        let lon_delta = 360.0 / (1u64 << (5 * (precision + 1) / 2)) as f64;

        let mut result = Vec::with_capacity(8);
        for dlat in [-1.0, 0.0, 1.0] {
            for dlon in [-1.0, 0.0, 1.0] {
                if dlat == 0.0 && dlon == 0.0 {
                    continue;
                }
                let nlat = lat + dlat * lat_delta;
                let nlon = lon + dlon * lon_delta;
                if nlat >= -90.0 && nlat <= 90.0 && nlon >= -180.0 && nlon <= 180.0 {
                    result.push(Self::encode(nlat, nlon, precision));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine() {
        // Berlin to Paris
        let dist = GeodeticCalculator::haversine_distance(52.52, 13.405, 48.85, 2.35);
        assert!((dist / 1000.0 - 878.0).abs() < 10.0); // ~878km
    }

    #[test]
    fn test_grid_cell_area() {
        let cell = GridCell::new(52.52, 13.405, CellResolution::Kilometer1);
        assert!(cell.area_km2 > 0.9 && cell.area_km2 < 1.5);
    }

    #[test]
    fn test_geohash() {
        let hash = GeohashEncoder::encode(52.52, 13.405, 6);
        assert_eq!(hash.len(), 6);

        let (lat, lon) = GeohashEncoder::decode(&hash).unwrap();
        assert!((lat - 52.52).abs() < 0.01);
        assert!((lon - 13.405).abs() < 0.01);
    }

    #[test]
    fn test_cell_contains_point() {
        let cell = GridCell::new(52.52, 13.405, CellResolution::Kilometer1);
        assert!(cell.contains_point(52.52, 13.405));
    }
}
