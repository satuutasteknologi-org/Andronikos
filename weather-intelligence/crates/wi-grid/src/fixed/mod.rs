//! # Fixed-Degree Grid Implementation
//!
//! Regular latitude/longitude grid with fixed step size.
//! Designed for Indonesia 0.5km grid (0.0045° step ≈ 500m).

use std::fmt::Debug;

use crate::traits::{Grid, GridCellId};
use wi_core::{Bbox, GeoCoord, Result, WIError};

/// Cell ID for fixed grid (row, col packed into u64)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FixedCellId {
    /// Row index in the grid
    pub row: u32,
    /// Column index in the grid
    pub col: u32,
}

impl FixedCellId {
    /// Create a new FixedCellId
    pub fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }
    /// Encode row and col into a single u64
    pub fn encode(&self) -> u64 {
        ((self.row as u64) << 32) | (self.col as u64)
    }
    /// Decode a u64 back into FixedCellId
    pub fn decode(val: u64) -> Self {
        Self {
            row: (val >> 32) as u32,
            col: val as u32,
        }
    }
}

impl GridCellId for FixedCellId {
    fn to_u64(&self) -> u64 {
        self.encode()
    }
    fn from_u64(val: u64) -> Self {
        Self::decode(val)
    }
}

/// Fixed-degree regular grid
#[derive(Debug, Clone)]
pub struct FixedGrid {
    /// Grid step size in degrees
    step_deg: f64,
    /// Minimum latitude
    lat_min: f64,
    /// Maximum latitude
    lat_max: f64,
    /// Minimum longitude
    lon_min: f64,
    /// Maximum longitude
    lon_max: f64,
    /// Number of rows
    n_rows: u32,
    /// Number of columns
    n_cols: u32,
    /// Bounding box of the grid
    bbox: Bbox,
}

impl FixedGrid {
    /// Step in degrees (0.0045° ≈ 500m at equator)
    pub const INDONESIA_STEP_DEG: f64 = 0.0045;

    /// Indonesia bounding box minimum latitude
    pub const INDONESIA_LAT_MIN: f64 = -11.0;
    /// Indonesia bounding box maximum latitude
    pub const INDONESIA_LAT_MAX: f64 = 6.1;
    /// Indonesia bounding box minimum longitude
    pub const INDONESIA_LON_MIN: f64 = 95.0;
    /// Indonesia bounding box maximum longitude
    pub const INDONESIA_LON_MAX: f64 = 141.1;

    /// Create new fixed grid
    pub fn new(step_deg: f64, lat_min: f64, lat_max: f64, lon_min: f64, lon_max: f64) -> Self {
        let n_rows = ((lat_max - lat_min) / step_deg).ceil() as u32;
        let n_cols = ((lon_max - lon_min) / step_deg).ceil() as u32;
        Self {
            step_deg,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            n_rows,
            n_cols,
            bbox: Bbox::new(lon_min, lat_min, lon_max, lat_max),
        }
    }

    /// Create Indonesia 0.5km grid
    pub fn indonesia() -> Self {
        Self::new(
            Self::INDONESIA_STEP_DEG,
            Self::INDONESIA_LAT_MIN,
            Self::INDONESIA_LAT_MAX,
            Self::INDONESIA_LON_MIN,
            Self::INDONESIA_LON_MAX,
        )
    }

    /// Create Indonesia grid with custom step
    pub fn indonesia_with_step(step_deg: f64) -> Self {
        Self::new(
            step_deg,
            Self::INDONESIA_LAT_MIN,
            Self::INDONESIA_LAT_MAX,
            Self::INDONESIA_LON_MIN,
            Self::INDONESIA_LON_MAX,
        )
    }

    /// Create global grid with given step
    pub fn global(step_deg: f64) -> Self {
        Self::new(step_deg, -90.0, 90.0, -180.0, 180.0)
    }

    /// Get the grid step size in degrees
    pub fn step_deg(&self) -> f64 {
        self.step_deg
    }
    /// Number of rows in the grid
    pub fn n_rows(&self) -> u32 {
        self.n_rows
    }
    /// Number of columns in the grid
    pub fn n_cols(&self) -> u32 {
        self.n_cols
    }
    /// Total number of cells in the grid
    pub fn total_cells(&self) -> u64 {
        self.n_rows as u64 * self.n_cols as u64
    }

    /// Convert lat/lon to row/col indices
    pub fn lat_lon_to_indices(&self, lat: f64, lon: f64) -> Option<(u32, u32)> {
        if lat < self.lat_min || lat >= self.lat_max || lon < self.lon_min || lon >= self.lon_max {
            return None;
        }
        let row = ((lat - self.lat_min) / self.step_deg).floor() as u32;
        let col = ((lon - self.lon_min) / self.step_deg).floor() as u32;
        Some((row.min(self.n_rows - 1), col.min(self.n_cols - 1)))
    }

    /// Get cell center from row/col
    ///
    /// Returns the geographic center coordinate of the cell at the given row/col.
    pub fn indices_to_center(&self, row: u32, col: u32) -> GeoCoord {
        let lat = self.lat_min + (row as f64 + 0.5) * self.step_deg;
        let lon = self.lon_min + (col as f64 + 0.5) * self.step_deg;
        GeoCoord::new(lat, lon)
    }

    /// Get row/col from cell ID
    ///
    /// Returns the (row, col) tuple for the given FixedCellId.
    pub fn cell_to_indices(&self, cell: FixedCellId) -> (u32, u32) {
        (cell.row, cell.col)
    }

    /// Get cell bounds (min_lat, min_lon, max_lat, max_lon)
    ///
    /// Returns the geographic bounds of the given cell.
    pub fn cell_bounds(&self, cell: FixedCellId) -> (f64, f64, f64, f64) {
        let row = cell.row;
        let col = cell.col;
        let min_lat = self.lat_min + row as f64 * self.step_deg;
        let max_lat = min_lat + self.step_deg;
        let min_lon = self.lon_min + col as f64 * self.step_deg;
        let max_lon = min_lon + self.step_deg;
        (min_lat, min_lon, max_lat, max_lon)
    }

    /// Get neighboring cells (4-connected or 8-connected)
    ///
    /// Returns a list of neighboring cell IDs. If `diagonal` is true, includes
    /// diagonal neighbors (8-connected), otherwise only cardinal directions (4-connected).
    pub fn neighbors(&self, cell: FixedCellId, diagonal: bool) -> Vec<FixedCellId> {
        let mut neighbors = Vec::new();
        let (row, col) = (cell.row as i32, cell.col as i32);

        let directions = if diagonal {
            vec![
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        } else {
            vec![(-1, 0), (1, 0), (0, -1), (0, 1)]
        };

        for (dr, dc) in directions {
            let nr = row + dr;
            let nc = col + dc;
            if nr >= 0 && nr < self.n_rows as i32 && nc >= 0 && nc < self.n_cols as i32 {
                neighbors.push(FixedCellId::new(nr as u32, nc as u32));
            }
        }
        neighbors
    }

    /// Iterate over all cells in a bounding box
    ///
    /// Returns an iterator over all cell IDs within the given bounding box.
    /// Uses zero-allocation flat_map for efficiency.
    pub fn cells_in_bbox_iter(&self, bbox: Bbox) -> impl Iterator<Item = FixedCellId> + use<'_> {
        let start = self
            .lat_lon_to_indices(bbox.min_lat, bbox.min_lon)
            .unwrap_or((0, 0));
        let end = self
            .lat_lon_to_indices(bbox.max_lat, bbox.max_lon)
            .unwrap_or((self.n_rows - 1, self.n_cols - 1));

        (start.0..=end.0.min(self.n_rows - 1)).flat_map(move |row| {
            (start.1..=end.1.min(self.n_cols - 1)).map(move |col| FixedCellId::new(row, col))
        })
    }
}

impl Grid for FixedGrid {
    type CellId = FixedCellId;

    fn from_lat_lon(&self, lat: f64, lon: f64) -> Result<Self::CellId> {
        if !lat.is_finite() || !lon.is_finite() {
            return Err(WIError::spatial("Coordinates not finite"));
        }
        if lat < self.lat_min || lat >= self.lat_max || lon < self.lon_min || lon >= self.lon_max {
            return Err(WIError::spatial(format!(
                "Coordinate ({}, {}) outside grid bounds [{}:{}, {}:{}]",
                lat, lon, self.lat_min, self.lat_max, self.lon_min, self.lon_max
            )));
        }
        let (row, col) = self.lat_lon_to_indices(lat, lon).unwrap();
        Ok(FixedCellId::new(row, col))
    }

    fn center(&self, cell_id: Self::CellId) -> GeoCoord {
        self.indices_to_center(cell_id.row, cell_id.col)
    }

    fn neighbors(&self, cell_id: Self::CellId) -> Vec<Self::CellId> {
        self.neighbors(cell_id, false)
    }

    fn resolution_meters(&self) -> f64 {
        // Approximate at equator: 1° ≈ 111km
        self.step_deg * 111_000.0
    }

    fn bbox(&self) -> Bbox {
        self.bbox
    }

    fn contains(&self, cell_id: Self::CellId) -> bool {
        cell_id.row < self.n_rows && cell_id.col < self.n_cols
    }

    fn cells_in_bbox(&self, bbox: Bbox) -> Vec<Self::CellId> {
        self.cells_in_bbox_iter(bbox).collect()
    }

    fn cell_bbox(&self, cell_id: Self::CellId) -> Bbox {
        let (min_lat, min_lon, max_lat, max_lon) = self.cell_bounds(cell_id);
        Bbox::new(min_lon, min_lat, max_lon, max_lat)
    }

    fn cell_count(&self) -> u64 {
        self.total_cells()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Grid;

    #[test]
    fn test_fixed_grid_indonesia() {
        let grid = FixedGrid::indonesia();
        assert_eq!(grid.step_deg(), 0.0045);
        // lat: -11.0 to 6.1 = 17.1 / 0.0045 = 3800 (exact with ceil)
        // lon: 95.0 to 141.1 = 46.1 / 0.0045 = 10244.44... -> 10245 with ceil
        // But due to floating point precision, may be 3801/10244
        assert!(grid.n_rows() >= 3800 && grid.n_rows() <= 3801);
        assert!(grid.n_cols() >= 10244 && grid.n_cols() <= 10245);
        // Total cells should be approximately 38.9M
        assert!(grid.total_cells() >= 38_900_000 && grid.total_cells() <= 39_000_000);
    }

    #[test]
    fn test_fixed_grid_roundtrip() {
        let grid = FixedGrid::indonesia();
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let center = grid.center(cell);
        assert!((center.lat + 6.595).abs() < grid.step_deg());
        assert!((center.lon - 106.806).abs() < grid.step_deg());
    }

    #[test]
    fn test_fixed_grid_center_exact() {
        let grid = FixedGrid::indonesia();
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let center = grid.center(cell);
        let cell2 = grid.from_lat_lon(center.lat, center.lon).unwrap();
        assert_eq!(cell, cell2);
    }

    #[test]
    fn test_fixed_grid_neighbors() {
        let grid = FixedGrid::indonesia();
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let neighbors = Grid::neighbors(&grid, cell);
        assert_eq!(neighbors.len(), 4); // 4-connected
        let neighbors8 = grid.neighbors(cell, true);
        assert_eq!(neighbors8.len(), 8); // 8-connected
    }

    #[test]
    fn test_fixed_grid_out_of_bounds() {
        let grid = FixedGrid::indonesia();
        assert!(grid.from_lat_lon(51.5, -0.1).is_err()); // London
        assert!(grid.from_lat_lon(-20.0, 100.0).is_err()); // Outside Indonesia
    }

    #[test]
    fn test_fixed_grid_cell_count() {
        let grid = FixedGrid::indonesia_with_step(0.01); // ~1km
        assert!(grid.total_cells() < 10_000_000);
    }
}
