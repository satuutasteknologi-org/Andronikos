//! # Weather Intelligence Grid Systems
//!
//! Provides multiple grid implementations for spatial indexing:
//! - H3 hexagonal hierarchical grid
//! - Fixed-degree regular grid (for Indonesia 0.5km)

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod fixed;
pub mod h3;
pub mod traits;

pub use fixed::{FixedCellId, FixedGrid};
pub use h3::H3Grid;
pub use traits::{Grid, GridCellId};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Grid;

    #[test]
    fn test_h3_grid() {
        let grid = H3Grid::new(7);
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let center = grid.center(cell);
        assert!((center.lat + 6.595).abs() < 0.01);
        assert!((center.lon - 106.806).abs() < 0.01);
    }

    #[test]
    fn test_fixed_grid() {
        let grid = FixedGrid::indonesia();
        let cell = grid.from_lat_lon(-6.595, 106.806).unwrap();
        let center = grid.center(cell);
        assert!((center.lat + 6.595).abs() < 0.01);
        assert!((center.lon - 106.806).abs() < 0.01);
    }
}
