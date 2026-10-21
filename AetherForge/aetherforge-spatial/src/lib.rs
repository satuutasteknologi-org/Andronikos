//! # AetherForge Spatial Engine - Single Source of Truth
//!
//! This module provides the definitive spatial resolution system for weather data,
//! supporting 1km × 1km grid resolution with intelligent algorithms.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                          SpatialEngine (SSOT)                                │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
//! │  │ Geodetic    │  │ H3 Spatial    │  │ Interpolation    │  │ Coordinate   │  │
//! │  │ Grid        │  │ Indexing      │  │ Engine           │  │ Transform    │  │
//! │  │ Calculator  │  │               │  │                  │  │               │  │
//! │  └─────────────┘  └──────────────┘  └──────────────────┘  └──────────────┘  │
//! │                                                                              │
//! │  ┌────────────────────────────────────────────────────────────────────────┐ │
//! │  │                 GridCell (1km × 1km precision)                          │ │
//! │  └────────────────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 1km Grid Specification
//!
//! - Uses WGS84 ellipsoid for accurate Earth modeling
//! - Grid cells are computed using proper geodetic distance formulas
//! - Supports global coverage with consistent cell sizing
//! - H3 resolution 7 provides ~1.22km cells (closest H3 to 1km)
//! - Custom grid generation for exact 1km × 1km when required

#![forbid(unsafe_code)]

pub mod engine;
pub mod grid;
pub mod indexing;
pub mod interpolation;
pub mod transform;

pub use engine::SpatialEngine;
pub use grid::GeodeticCalculator;
pub use grid::{CellResolution, GridBounds, GridCell, GridConfig};
pub use indexing::{GridIndexer, H3Indexer, SpatialIndex};
pub use interpolation::{InterpolationEngine, InterpolationMethod, SpatialInterpolator};
pub use transform::{CoordinateSystem, CoordinateTransformer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_creation() {
        let cell = GridCell::new(52.52, 13.405, CellResolution::Kilometer1);
        assert!(cell.area_km2() > 0.99 && cell.area_km2() < 1.01);
    }

    #[test]
    fn test_global_grid() {
        let engine = SpatialEngine::new();
        let cells = engine.create_global_grid(CellResolution::Kilometer1);
        assert!(cells.len() > 50000000); // ~51M km2 / 1km2 ≈ 51M cells
    }
}
