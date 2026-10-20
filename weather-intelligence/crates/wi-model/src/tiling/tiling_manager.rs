//! # Auto-Tiling Manager for Large-Scale Inference
//!
//! Membagi area besar (38,9M cell Indonesia 0.5km) menjadi multiple tiles
//! untuk diproses secara parallel dengan memori terbatas.
//! Target: < 1 menit untuk seluruh Indonesia 0.5km.

use serde::{Deserialize, Serialize};
use tracing::info;
use wi_core::types::{AreaPredictRequest, Bbox, CellPrediction};
use wi_core::{DeviceType, Result, WIError};
use wi_grid::{FixedGrid, Grid};

/// Konfigurasi tiling untuk area besar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingConfig {
    /// Maksimum cell per tile (default 2.5M untuk GPU memori aman)
    pub max_cells_per_tile: usize,
    /// Overlap antar tile dalam derajat (untuk konsistensi GNN edge di batas)
    pub overlap_deg: f64,
    /// Maksimum concurrent tile processing
    pub max_concurrent_tiles: usize,
    /// Target waktu total dalam detik (default 60s = 1 menit)
    pub target_total_seconds: u64,
    /// Perangkat komputasi default
    pub device_type: DeviceType,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            max_cells_per_tile: 2_500_000, // 2.5M cell per tile untuk GPU memori aman
            overlap_deg: 0.009,            // ~1km overlap
            max_concurrent_tiles: 4,       // 4 tile parallel
            target_total_seconds: 60,      // 1 menit
            device_type: DeviceType::default_device(),
        }
    }
}

/// Satu tile hasil pembagian area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    pub tile_id: String,
    pub bbox: Bbox,
    pub cell_count: usize,
    pub row_index: usize,
    pub col_index: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

/// Hasil prediksi per tile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileResult {
    pub tile_id: String,
    pub predictions: Vec<CellPrediction>,
    pub cell_count: usize,
    pub inference_ms: u64,
    pub device_used: DeviceType,
}

/// Manager untuk pembagian dan koordinasi tiling.
#[derive(Debug, Clone)]
pub struct TilingManager {
    config: TilingConfig,
}

impl TilingManager {
    pub fn new(config: TilingConfig) -> Self {
        Self { config }
    }

    /// Pembagi area besar menjadi multiple tiles berdasarkan bounding box dan resolusi.
    /// Untuk Indonesia 0.5km (38,9M cell), akan menghasilkan ~16 tiles @ 2.5M cell each.
    pub fn split_area_into_tiles(
        &self,
        area_id: &str,
        bbox: &Bbox,
        resolution: u8,
    ) -> Result<Vec<Tile>> {
        let step_deg = match resolution {
            0 => FixedGrid::INDONESIA_STEP_DEG, // 0.0045° ≈ 500m
            1 => 0.009,
            2 => 0.045,
            3 => 0.225,
            _ => FixedGrid::INDONESIA_STEP_DEG,
        };

        // Hitung dimensi grid untuk seluruh area
        let grid = FixedGrid::new(
            step_deg,
            FixedGrid::INDONESIA_LAT_MIN,
            FixedGrid::INDONESIA_LAT_MAX,
            FixedGrid::INDONESIA_LON_MIN,
            FixedGrid::INDONESIA_LON_MAX,
        );

        // Ambil semua cell dalam bounding box
        let all_cells: Vec<_> = grid.cells_in_bbox(*bbox).into_iter().collect();

        if all_cells.is_empty() {
            return Err(WIError::validation(format!(
                "Area {}: tidak ada cell dalam bounding box",
                area_id
            )));
        }

        let total_cells = all_cells.len();
        info!(
            "Area {}: total {} cell, resolusi {}",
            area_id, total_cells, resolution
        );

        // Jika cell_count <= max_cells_per_tile, satu tile saja
        if total_cells <= self.config.max_cells_per_tile {
            return Ok(vec![Tile {
                tile_id: format!("{}-tile-000", area_id),
                bbox: *bbox,
                cell_count: total_cells,
                row_index: 0,
                col_index: 0,
                total_rows: 1,
                total_cols: 1,
            }]);
        }

        // Hitung grid dimensi tile
        let lat_range = bbox.max_lat - bbox.min_lat;
        let lon_range = bbox.max_lon - bbox.min_lon;

        // Estimasi cell per derajat
        let _cells_per_deg_lat = 1.0 / step_deg;
        let _cells_per_deg_lon = 1.0 / step_deg;

        // Hitung jumlah tile yang diperlukan
        let total_cells_f64 = total_cells as f64;
        let num_tiles_needed =
            (total_cells_f64 / self.config.max_cells_per_tile as f64).ceil() as usize;

        // Buat grid 2D untuk tiling
        let tile_cols = (num_tiles_needed as f64).sqrt().ceil() as usize;
        let tile_rows = (num_tiles_needed as f64 / tile_cols as f64).ceil() as usize;

        let tile_lat_height = lat_range / tile_rows as f64;
        let tile_lon_width = lon_range / tile_cols as f64;

        let mut tiles = Vec::new();
        let mut tile_idx = 0;

        for row in 0..tile_rows {
            for col in 0..tile_cols {
                if tile_idx >= num_tiles_needed {
                    break;
                }

                // Hitung bbox tile dengan overlap
                let tile_min_lat = bbox.min_lat + row as f64 * tile_lat_height;
                let tile_max_lat = (bbox.min_lat + (row + 1) as f64 * tile_lat_height)
                    .min(bbox.max_lat)
                    .max(tile_min_lat + self.config.overlap_deg);
                let tile_min_lon = bbox.min_lon + col as f64 * tile_lon_width;
                let tile_max_lon = (bbox.min_lon + (col + 1) as f64 * tile_lon_width)
                    .min(bbox.max_lon)
                    .max(tile_min_lon + self.config.overlap_deg);

                // Hitung cell count untuk tile ini
                let tile_grid = FixedGrid::new(
                    step_deg,
                    FixedGrid::INDONESIA_LAT_MIN,
                    FixedGrid::INDONESIA_LAT_MAX,
                    FixedGrid::INDONESIA_LON_MIN,
                    FixedGrid::INDONESIA_LON_MAX,
                );
                let tile_cells = tile_grid
                    .cells_in_bbox(Bbox::new(
                        tile_min_lon,
                        tile_min_lat,
                        tile_max_lon,
                        tile_max_lat,
                    ))
                    .len();

                if tile_cells == 0 {
                    tile_idx += 1;
                    continue;
                }

                tiles.push(Tile {
                    tile_id: format!("{}-tile-{:03}", area_id, tile_idx),
                    bbox: Bbox::new(tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat),
                    cell_count: tile_cells,
                    row_index: row,
                    col_index: col,
                    total_rows: tile_rows,
                    total_cols: tile_cols,
                });

                tile_idx += 1;
            }
            if tile_idx >= num_tiles_needed {
                break;
            }
        }

        info!(
            "Area {}: dibagi menjadi {} tile, total {} cell",
            area_id,
            tiles.len(),
            tiles.iter().map(|t| t.cell_count).sum::<usize>()
        );

        Ok(tiles)
    }

    /// Buat BatchGridPredictRequest untuk satu tile.
    pub fn create_tile_request(
        &self,
        tile: &Tile,
        _area_id: &str,
        lead_seconds: u32,
        variables: &[wi_core::types::WeatherVariable],
    ) -> AreaPredictRequest {
        AreaPredictRequest {
            area_id: tile.tile_id.clone(),
            bbox: tile.bbox,
            resolution: 0, // 0.5km
            lead_seconds,
            variables: variables.to_vec(),
            max_cells: Some(tile.cell_count),
        }
    }

    pub fn config(&self) -> &TilingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wi_core::types::WeatherVariable;

    #[test]
    fn test_split_small_area_single_tile() {
        let manager = TilingManager::new(TilingConfig::default());
        let bbox = Bbox::new(106.5, -6.5, 107.0, -6.0); // Jakarta kecil

        let tiles = manager.split_area_into_tiles("jakarta", &bbox, 0).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].tile_id, "jakarta-tile-000");
    }

    #[test]
    fn test_tiling_config_default() {
        let config = TilingConfig::default();
        assert_eq!(config.max_cells_per_tile, 2_500_000);
        assert_eq!(config.max_concurrent_tiles, 4);
        assert_eq!(config.target_total_seconds, 60);
    }

    #[test]
    fn test_create_tile_request() {
        let manager = TilingManager::new(TilingConfig::default());
        let bbox = Bbox::new(106.5, -6.5, 107.0, -6.0);
        let tiles = manager.split_area_into_tiles("test", &bbox, 0).unwrap();

        let vars = vec![WeatherVariable::Temperature];
        let req = manager.create_tile_request(&tiles[0], "test", 3600, &vars);

        assert_eq!(req.area_id, "test-tile-000");
        assert_eq!(req.lead_seconds, 3600);
        assert_eq!(req.variables, vars);
    }
}
