//! # Grid Traits
//!
//! Common traits for grid systems.

use std::fmt::Debug;
use wi_core::{Bbox, GeoCoord, Result};

/// Unique identifier for a grid cell
pub trait GridCellId: Copy + Clone + Eq + std::hash::Hash + Debug + Send + Sync {
    /// Convert cell ID to u64 for storage/serialization
    fn to_u64(&self) -> u64;
    /// Create cell ID from u64
    fn from_u64(val: u64) -> Self;
}

/// Trait for grid systems
pub trait Grid: Send + Sync {
    /// Cell ID type for this grid
    type CellId: GridCellId;

    /// Convert lat/lon to cell ID
    #[allow(clippy::wrong_self_convention)]
    fn from_lat_lon(&self, lat: f64, lon: f64) -> Result<Self::CellId>;

    /// Get center coordinate of a cell
    fn center(&self, cell_id: Self::CellId) -> GeoCoord;

    /// Get neighboring cell IDs
    fn neighbors(&self, cell_id: Self::CellId) -> Vec<Self::CellId>;

    /// Grid resolution in meters (approximate)
    fn resolution_meters(&self) -> f64;

    /// Grid bounding box
    fn bbox(&self) -> Bbox;

    /// Check if cell is within grid bounds
    fn contains(&self, cell_id: Self::CellId) -> bool;

    /// Get all cells within a bounding box
    fn cells_in_bbox(&self, bbox: Bbox) -> Vec<Self::CellId>;

    /// Get cell bounding box
    fn cell_bbox(&self, cell_id: Self::CellId) -> Bbox;

    /// Total number of cells (approximate)
    fn cell_count(&self) -> u64;
}

impl GridCellId for u64 {
    fn to_u64(&self) -> u64 {
        *self
    }
    fn from_u64(val: u64) -> Self {
        val
    }
}

impl GridCellId for u32 {
    fn to_u64(&self) -> u64 {
        *self as u64
    }
    fn from_u64(val: u64) -> Self {
        val as u32
    }
}
