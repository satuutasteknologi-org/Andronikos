//! # Spatial Indexing - H3 and Grid-based spatial indexing
//!
//! Provides hierarchical spatial indexing using H3 and custom grid systems
//! for efficient spatial queries and lookups.

use std::collections::{HashMap, HashSet};

use h3o::{CellIndex, LatLng};


use super::grid::{CellResolution, GeodeticCalculator, GridBounds, GridCell};

/// Spatial index trait for different indexing strategies
pub trait SpatialIndexer: Send + Sync {
    fn insert(&mut self, lat: f64, lon: f64, data_id: u64);
    fn query_point(&self, lat: f64, lon: f64) -> Option<u64>;
    fn query_radius(&self, lat: f64, lon: f64, radius_m: f64) -> Vec<u64>;
    fn query_box(&self, min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Vec<u64>;
    fn nearest(&self, lat: f64, lon: f64, k: usize) -> Vec<(u64, f64)>;
}

/// H3-based hierarchical spatial indexer
pub struct H3Indexer {
    resolution: u8,
    cells: HashMap<u64, Vec<u64>>,
    point_to_cell: HashMap<String, u64>,
}

impl H3Indexer {
    pub fn new(resolution: u8) -> Self {
        Self {
            resolution,
            cells: HashMap::new(),
            point_to_cell: HashMap::new(),
        }
    }

    pub fn from_resolution(resolution: CellResolution) -> Self {
        Self::new(resolution.h3_resolution())
    }

    pub fn insert_point(&mut self, lat: f64, lon: f64, data_id: u64) {
        let latlng = match h3o::LatLng::new(lat, lon) {
            Ok(ll) => ll,
            Err(_) => return,
        };
        let resolution = match h3o::Resolution::try_from(self.resolution) {
            Ok(r) => r,
            Err(_) => return,
        };
        let h3_idx = latlng.to_cell(resolution).into();

        self.cells.entry(h3_idx).or_default().push(data_id);

        let key = format!("{},{}", (lat * 1e6) as i64, (lon * 1e6) as i64);
        self.point_to_cell.insert(key, h3_idx);
    }

    pub fn get_cell_for_point(&self, lat: f64, lon: f64) -> Option<u64> {
        let latlng = match h3o::LatLng::new(lat, lon) {
            Ok(ll) => ll,
            Err(_) => return None,
        };
        let resolution = match h3o::Resolution::try_from(self.resolution) {
            Ok(r) => r,
            Err(_) => return None,
        };
        let h3_idx: u64 = latlng.to_cell(resolution).into();

        if self.cells.contains_key(&h3_idx) {
            Some(h3_idx)
        } else {
            None
        }
    }

    pub fn get_data_in_cell(&self, h3_idx: u64) -> Option<&Vec<u64>> {
        self.cells.get(&h3_idx)
    }

    pub fn get_neighbors(&self, lat: f64, lon: f64, rings: u8) -> Vec<u64> {
        let latlng = match h3o::LatLng::new(lat, lon) {
            Ok(ll) => ll,
            Err(_) => return vec![],
        };
        let resolution = match h3o::Resolution::try_from(self.resolution) {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        let cell = latlng.to_cell(resolution);

        let neighbors: Vec<u64> = cell
            .grid_disk::<Vec<CellIndex>>(rings as u32)
            .into_iter()
            .map(|n| n.into())
            .collect();
        neighbors
    }

    pub fn query_radius(&self, lat: f64, lon: f64, radius_m: f64) -> Vec<u64> {
        let rings = Self::meters_to_h3_rings(radius_m, lat);
        self.get_neighbors(lat, lon, rings)
    }

    fn meters_to_h3_rings(meters: f64, _lat: f64) -> u8 {
        if meters < 1000.0 {
            1
        } else if meters < 5000.0 {
            2
        } else if meters < 10000.0 {
            3
        } else if meters < 25000.0 {
            4
        } else if meters < 50000.0 {
            5
        } else if meters < 100000.0 {
            6
        } else {
            7
        }
    }

    pub fn query_box(&self, min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Vec<u64> {
        let mut results = HashSet::new();

        let mut lat = min_lat;
        while lat <= max_lat {
            let mut lon = min_lon;
            while lon <= max_lon {
                if let Some(h3_idx) = self.get_cell_for_point(lat, lon) {
                    results.insert(h3_idx);
                }
                lon += 0.01;
            }
            lat += 0.01;
        }

        results.into_iter().collect()
    }

    pub fn nearest_k(&self, lat: f64, lon: f64, k: usize) -> Vec<(u64, f64)> {
        let candidates = self.get_neighbors(lat, lon, 3);

        let mut distances: Vec<(u64, f64)> = candidates
            .iter()
            .filter_map(|&h3_idx| {
                let cell = CellIndex::try_from(h3_idx).ok()?;
                let center = LatLng::from(cell);
                let dist =
                    GeodeticCalculator::haversine_distance(lat, lon, center.lat(), center.lng());
                Some((h3_idx, dist))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.point_to_cell.clear();
    }
}

impl Default for H3Indexer {
    fn default() -> Self {
        Self::new(7) // ~1.22km resolution
    }
}

/// Custom 1km × 1km grid indexer
pub struct GridIndexer {
    resolution: CellResolution,
    bounds: GridBounds,
    cells: HashMap<(i32, i32), Vec<u64>>,
    cell_data: HashMap<u64, GridCell>,
    lat_to_row: HashMap<i32, i32>,
    lon_to_col: HashMap<i32, i32>,
    row_to_lat: HashMap<i32, f64>,
    col_to_lon: HashMap<i32, f64>,
}

impl GridIndexer {
    pub fn new(resolution: CellResolution, bounds: GridBounds) -> Self {
        let mut indexer = Self {
            resolution,
            bounds: bounds.clone(),
            cells: HashMap::new(),
            cell_data: HashMap::new(),
            lat_to_row: HashMap::new(),
            lon_to_col: HashMap::new(),
            row_to_lat: HashMap::new(),
            col_to_lon: HashMap::new(),
        };

        indexer.initialize_grid();
        indexer
    }

    fn initialize_grid(&mut self) {
        let (km_per_deg_lat, km_per_deg_lon) = GeodeticCalculator::km_per_degree_latitude(
            (self.bounds.min_lat + self.bounds.max_lat) / 2.0,
        );

        let target_cell_size_km = self.resolution.target_km2().sqrt();

        let lat_step = target_cell_size_km / km_per_deg_lat;
        let lon_step = target_cell_size_km / km_per_deg_lon;

        let mut row = 0i32;
        let mut lat = self.bounds.max_lat;
        while lat > self.bounds.min_lat {
            self.row_to_lat.insert(row, lat - lat_step / 2.0);
            self.lat_to_row.insert((lat * 1e6) as i32, row);
            lat -= lat_step;
            row += 1;
        }

        let mut col = 0i32;
        let mut lon = self.bounds.min_lon;
        while lon < self.bounds.max_lon {
            self.col_to_lon.insert(col, lon + lon_step / 2.0);
            self.lon_to_col.insert((lon * 1e6) as i32, col);
            lon += lon_step;
            col += 1;
        }
    }

    pub fn insert(&mut self, lat: f64, lon: f64, data_id: u64) {
        let row = self.get_row(lat);
        let col = self.get_col(lon);
        let key = (row, col);

        let cell_key = self.cell_key(row, col);
        self.cells.entry(key).or_default().push(data_id);

        if !self.cell_data.contains_key(&cell_key) {
            let center_lat = self.row_to_lat.get(&row).copied().unwrap_or(lat);
            let center_lon = self.col_to_lon.get(&col).copied().unwrap_or(lon);
            let cell = GridCell::new(center_lat, center_lon, self.resolution);
            self.cell_data.insert(cell_key, cell);
        }
    }

    fn get_row(&self, lat: f64) -> i32 {
        let lat_key = (lat * 1e6) as i32;

        let mut best_row = 0i32;
        let mut best_diff = i32::MAX;

        for (&key, _) in &self.row_to_lat {
            let diff = (key - lat_key).abs();
            if diff < best_diff {
                best_diff = diff;
                best_row = self.lat_to_row.get(&key).copied().unwrap_or(0);
            }
        }

        best_row
    }

    fn get_col(&self, lon: f64) -> i32 {
        let lon_key = (lon * 1e6) as i32;

        let mut best_col = 0i32;
        let mut best_diff = i32::MAX;

        for (&key, _) in &self.col_to_lon {
            let diff = (key - lon_key).abs();
            if diff < best_diff {
                best_diff = diff;
                best_col = self.lon_to_col.get(&key).copied().unwrap_or(0);
            }
        }

        best_col
    }

    fn cell_key(&self, row: i32, col: i32) -> u64 {
        ((row as u64) << 32) | (col as u64 & 0xFFFFFFFF)
    }

    pub fn query_point(&self, lat: f64, lon: f64) -> Option<u64> {
        let row = self.get_row(lat);
        let col = self.get_col(lon);

        self.cells.get(&(row, col)).and_then(|v| v.first().copied())
    }

    pub fn query_box(&self, min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Vec<u64> {
        let min_row = self.get_row(max_lat);
        let max_row = self.get_row(min_lat);
        let min_col = self.get_col(min_lon);
        let max_col = self.get_col(max_lon);

        let mut results = HashSet::new();

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                if let Some(data_ids) = self.cells.get(&(row, col)) {
                    results.extend(data_ids.iter().copied());
                }
            }
        }

        results.into_iter().collect()
    }

    pub fn query_radius(&self, lat: f64, lon: f64, radius_m: f64) -> Vec<u64> {
        let radius_deg_lat = radius_m / 110574.0;
        let radius_deg_lon = radius_m / (111320.0 * lat.to_radians().cos());

        let min_lat = (lat - radius_deg_lat).max(self.bounds.min_lat);
        let max_lat = (lat + radius_deg_lat).min(self.bounds.max_lat);
        let min_lon = (lon - radius_deg_lon).max(self.bounds.min_lon);
        let max_lon = (lon + radius_deg_lon).min(self.bounds.max_lon);

        self.query_box(min_lat, min_lon, max_lat, max_lon)
    }

    pub fn nearest_k(&self, lat: f64, lon: f64, k: usize) -> Vec<(u64, f64)> {
        let candidates = self.query_radius(lat, lon, 50000.0);

        let mut distances: Vec<(u64, f64)> = candidates
            .iter()
            .map(|&data_id| {
                let cell = self
                    .cell_data
                    .get(&data_id)
                    .cloned()
                    .unwrap_or_else(|| GridCell::new(lat, lon, self.resolution));
                let dist = GeodeticCalculator::haversine_distance(
                    lat,
                    lon,
                    cell.center_lat,
                    cell.center_lon,
                );
                (data_id, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    pub fn get_cell_at(&self, lat: f64, lon: f64) -> Option<GridCell> {
        let row = self.get_row(lat);
        let col = self.get_col(lon);
        let key = self.cell_key(row, col);

        self.cell_data.get(&key).cloned()
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.cell_data.clear();
    }

    pub fn get_cell_by_id(&self, id: u64) -> Option<GridCell> {
        self.cell_data.get(&id).cloned()
    }
}

/// Unified spatial index combining H3 and custom grid
pub struct SpatialIndex {
    h3_index: H3Indexer,
    grid_index: GridIndexer,
    resolution: CellResolution,
}

impl SpatialIndex {
    pub fn new(resolution: CellResolution, bounds: GridBounds) -> Self {
        Self {
            h3_index: H3Indexer::from_resolution(resolution),
            grid_index: GridIndexer::new(resolution, bounds),
            resolution,
        }
    }

    pub fn insert(&mut self, lat: f64, lon: f64, data_id: u64) {
        self.h3_index.insert_point(lat, lon, data_id);
        self.grid_index.insert(lat, lon, data_id);
    }

    pub fn query_point(&self, lat: f64, lon: f64) -> Option<u64> {
        self.h3_index.get_cell_for_point(lat, lon).and_then(|h3| {
            self.h3_index
                .get_data_in_cell(h3)
                .and_then(|v| v.first().copied())
        })
    }

    pub fn query_radius(&self, lat: f64, lon: f64, radius_m: f64) -> Vec<u64> {
        let h3_results = self.h3_index.query_radius(lat, lon, radius_m);

        let mut all_data: HashSet<u64> = HashSet::new();
        for h3 in h3_results {
            if let Some(data) = self.h3_index.get_data_in_cell(h3) {
                all_data.extend(data.iter().copied());
            }
        }

        all_data.into_iter().collect()
    }

    pub fn query_box(&self, min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Vec<u64> {
        self.grid_index
            .query_box(min_lat, min_lon, max_lat, max_lon)
    }

    pub fn nearest_k(&self, lat: f64, lon: f64, k: usize) -> Vec<(u64, f64)> {
        self.h3_index.nearest_k(lat, lon, k)
    }

    pub fn get_cell_for_point(&self, lat: f64, lon: f64) -> Option<GridCell> {
        self.grid_index.get_cell_at(lat, lon)
    }

    pub fn resolution(&self) -> CellResolution {
        self.resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h3_insert_query() {
        let mut indexer = H3Indexer::new(7);
        indexer.insert_point(52.52, 13.405, 1);
        indexer.insert_point(52.53, 13.41, 2);

        assert!(indexer.get_cell_for_point(52.52, 13.405).is_some());
    }

    #[test]
    fn test_grid_query_radius() {
        let bounds = GridBounds::new(50.0, 10.0, 55.0, 20.0);
        let mut indexer = GridIndexer::new(CellResolution::Kilometer1, bounds);

        indexer.insert(52.52, 13.405, 1);
        indexer.insert(52.53, 13.41, 2);

        let results = indexer.query_radius(52.52, 13.405, 10000.0);
        assert!(results.len() >= 1);
    }
}
