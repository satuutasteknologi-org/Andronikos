//! # H3 Hexagonal Grid Implementation
//!
//! Uses the h3o crate for H3 hexagonal hierarchical spatial indexing.

use h3o::{CellIndex, LatLng, Resolution};

use crate::traits::{Grid, GridCellId};
use wi_core::{Bbox, GeoCoord, Result, WIError};

/// H3 Grid implementation
#[derive(Debug, Clone)]
pub struct H3Grid {
    resolution: u8,
    h3_resolution: Resolution,
    bbox: Bbox,
}

impl H3Grid {
    /// Create new H3 grid at specified resolution (0-15)
    pub fn new(resolution: u8) -> Self {
        let h3_resolution = Resolution::try_from(resolution).expect("Invalid H3 resolution");
        Self {
            resolution,
            h3_resolution,
            bbox: Bbox::global(),
        }
    }

    /// Create H3 grid for Indonesia region
    pub fn indonesia(resolution: u8) -> Self {
        let mut grid = Self::new(resolution);
        grid.bbox = Bbox::indonesia();
        grid
    }

    /// Get H3 resolution
    pub fn resolution(&self) -> u8 {
        self.resolution
    }

    /// Get approximate cell area in km²
    pub fn cell_area_km2(&self) -> f64 {
        match self.resolution {
            0 => 4250546.8,
            1 => 607220.9,
            2 => 86745.8,
            3 => 12392.3,
            4 => 1770.3,
            5 => 252.9,
            6 => 36.1,
            7 => 5.16,
            8 => 0.737,
            9 => 0.105,
            10 => 0.015,
            11 => 0.00214,
            12 => 0.000306,
            13 => 0.0000437,
            14 => 0.00000624,
            15 => 0.000000891,
            _ => 0.0,
        }
    }

    /// Get cell boundary as polygon
    pub fn cell_boundary(&self, cell: CellIndex) -> Vec<GeoCoord> {
        cell.boundary()
            .iter()
            .map(|ll| GeoCoord::new(ll.lat(), ll.lng()))
            .collect()
    }

    /// Get cells within distance k of a center cell
    pub fn disk(&self, center: CellIndex, k: u64) -> Vec<CellIndex> {
        center.grid_disk::<Vec<CellIndex>>(k.try_into().unwrap())
    }

    /// Get parent cell at lower resolution
    pub fn parent(&self, cell: CellIndex, target_resolution: u8) -> Option<CellIndex> {
        if target_resolution >= self.resolution {
            return None;
        }
        let target_res = match Resolution::try_from(target_resolution) {
            Ok(r) => r,
            Err(_) => return None,
        };
        cell.parent(target_res)
    }

    /// Get child cells at higher resolution
    pub fn children(&self, cell: CellIndex, target_resolution: u8) -> Vec<CellIndex> {
        if target_resolution <= self.resolution {
            return vec![];
        }
        let target_res = match Resolution::try_from(target_resolution) {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        cell.children(target_res).collect()
    }
}

impl Grid for H3Grid {
    type CellId = CellIndex;

    fn from_lat_lon(&self, lat: f64, lon: f64) -> Result<Self::CellId> {
        let latlng = LatLng::new(lat, lon).map_err(|_| WIError::spatial("Invalid coordinates"))?;
        Ok(latlng.to_cell(self.h3_resolution))
    }

    fn center(&self, cell_id: Self::CellId) -> GeoCoord {
        let latlng: LatLng = cell_id.into();
        GeoCoord::new(latlng.lat(), latlng.lng())
    }

    fn neighbors(&self, cell_id: Self::CellId) -> Vec<Self::CellId> {
        cell_id
            .grid_disk::<Vec<CellIndex>>(1)
            .into_iter()
            .filter(|c| *c != cell_id)
            .collect()
    }

    fn resolution_meters(&self) -> f64 {
        match self.resolution {
            0 => 1_107_712.6,
            1 => 418_676.0,
            2 => 158_244.0,
            3 => 59_810.0,
            4 => 22_588.0,
            5 => 8_511.0,
            6 => 3_209.0,
            7 => 1_208.0,
            8 => 454.0,
            9 => 171.0,
            10 => 64.3,
            11 => 24.2,
            12 => 9.1,
            13 => 3.4,
            14 => 1.3,
            15 => 0.5,
            _ => 0.0,
        }
    }

    fn bbox(&self) -> Bbox {
        self.bbox
    }

    fn contains(&self, cell_id: Self::CellId) -> bool {
        let center = self.center(cell_id);
        self.bbox.contains(center.lat, center.lon)
    }

    fn cells_in_bbox(&self, bbox: Bbox) -> Vec<Self::CellId> {
        let min_latlng = match LatLng::new(bbox.min_lat, bbox.min_lon) {
            Ok(ll) => ll,
            Err(_) => return Vec::new(),
        };
        let max_latlng = match LatLng::new(bbox.max_lat, bbox.max_lon) {
            Ok(ll) => ll,
            Err(_) => return Vec::new(),
        };

        let min_cell = min_latlng.to_cell(self.h3_resolution);
        let max_cell = max_latlng.to_cell(self.h3_resolution);
        let distance = min_cell.grid_distance(max_cell).unwrap_or(0) as u32;

        let cells = min_cell.grid_disk::<Vec<CellIndex>>(distance);
        cells
            .into_iter()
            .filter(|c| {
                let center = self.center(*c);
                bbox.contains(center.lat, center.lon)
            })
            .collect()
    }

    fn cell_bbox(&self, cell_id: Self::CellId) -> Bbox {
        let boundary = cell_id.boundary();
        let min_lat = boundary
            .iter()
            .map(|ll| ll.lat())
            .fold(f64::INFINITY, f64::min);
        let max_lat = boundary
            .iter()
            .map(|ll| ll.lat())
            .fold(f64::NEG_INFINITY, f64::max);
        let min_lon = boundary
            .iter()
            .map(|ll| ll.lng())
            .fold(f64::INFINITY, f64::min);
        let max_lon = boundary
            .iter()
            .map(|ll| ll.lng())
            .fold(f64::NEG_INFINITY, f64::max);
        Bbox::new(min_lon, min_lat, max_lon, max_lat)
    }

    fn cell_count(&self) -> u64 {
        // Estimate based on bbox area
        let area_km2 = (self.bbox.max_lat - self.bbox.min_lat)
            * 111.0
            * (self.bbox.max_lon - self.bbox.min_lon)
            * 111.0
            * self.bbox.min_lat.to_radians().cos();
        let cell_area = self.cell_area_km2();
        (area_km2 / cell_area) as u64
    }
}

impl GridCellId for CellIndex {
    fn to_u64(&self) -> u64 {
        u64::from(*self)
    }
    fn from_u64(val: u64) -> Self {
        CellIndex::try_from(val).expect("Invalid H3 cell index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Grid;

    #[test]
    fn test_h3_grid_creation() {
        let grid = H3Grid::new(7);
        assert_eq!(grid.resolution(), 7);
        assert_eq!(grid.resolution_meters(), 1208.0);
    }

    #[test]
    fn test_h3_roundtrip() {
        let grid = H3Grid::new(7);
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let center = grid.center(cell);
        assert!((center.lat + 6.595).abs() < 0.01);
        assert!((center.lon - 106.806).abs() < 0.01);
    }

    #[test]
    fn test_h3_neighbors() {
        let grid = H3Grid::new(7);
        let cell = grid.from_lat_lon(0.0, 0.0).unwrap();
        let neighbors = grid.neighbors(cell);
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn test_h3_children() {
        let grid = H3Grid::new(7);
        let cell = grid.from_lat_lon(0.0, 0.0).unwrap();
        let children = grid.children(cell, 8);
        // Parent at res 7 should have 7 children at res 8
        assert_eq!(children.len(), 7);
        // Each child should be contained in the parent
        for child in &children {
            assert!(grid
                .cell_bbox(*child)
                .contains(grid.center(*child).lat, grid.center(*child).lon));
        }
    }
}
