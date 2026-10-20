//! # Tile Coordinator - Distributed Inference for 38.9M Cell
//!
//! Koordinasi inferensi distributed across multiple tiles dengan GPU support,
//! batch processing, dan guard waktu < 60 detik.

use crate::burn::GraphCastForecaster;
use crate::tiling::tiling_manager::{TileResult, TilingConfig, TilingManager};
use crate::traits::Forecaster;
use wi_core::types::{
    BatchAreaResult, BatchGridPredictOutput, Bbox, CellPrediction, WeatherVariable,
};
use wi_core::{DeviceType, Result};

/// Hasil koordinasi inference untuk seluruh area.
#[derive(Debug, Clone)]
pub struct CoordinatedInferenceResult {
    pub area_id: String,
    pub total_cells: usize,
    pub total_inference_ms: u64,
    pub device_used: DeviceType,
    pub tile_results: Vec<TileResult>,
    pub merged_predictions: Vec<CellPrediction>,
    pub completed_within_target: bool,
}

/// Coordinator untuk inferensi distributed multi-tile.
pub struct TileCoordinator {
    forecaster: GraphCastForecaster,
    tiling_manager: TilingManager,
    _max_concurrent: usize,
}

impl TileCoordinator {
    pub fn new(forecaster: GraphCastForecaster, config: TilingConfig) -> Self {
        let _max_concurrent = config.max_concurrent_tiles;
        Self {
            forecaster,
            tiling_manager: TilingManager::new(config),
            _max_concurrent,
        }
    }

    /// Inferensi distributed untuk satu area besar (bisa 38.9M cell).
    /// Akan otomatis split ke multiple tiles dan process secara parallel.
    /// Guard: gagal jika total waktu > target_total_seconds.
    pub async fn infer_area_distributed(
        &self,
        area_id: &str,
        bbox: &Bbox,
        resolution: u8,
        lead_seconds: u32,
        variables: &[WeatherVariable],
    ) -> Result<CoordinatedInferenceResult> {
        self.forecaster
            .infer_area_distributed(area_id, bbox, resolution, lead_seconds, variables)
            .await
    }

    /// Inferensi batch untuk banyak area sekaligus.
    pub async fn infer_areas_distributed(
        &self,
        areas: Vec<(String, Bbox, u8, u32, Vec<WeatherVariable>)>,
    ) -> Result<Vec<CoordinatedInferenceResult>> {
        let mut results = Vec::new();

        for (area_id, bbox, resolution, lead_seconds, variables) in areas {
            let result = self
                .infer_area_distributed(&area_id, &bbox, resolution, lead_seconds, &variables)
                .await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Buat BatchGridPredictOutput dari CoordinatedInferenceResult (untuk kompatibilitas API lama).
    pub fn to_batch_output(&self, result: &CoordinatedInferenceResult) -> BatchGridPredictOutput {
        let mut area_results = Vec::new();
        let mut total_cells = 0;
        let mut total_ms = 0;

        for tile_result in &result.tile_results {
            area_results.push(BatchAreaResult {
                area_id: tile_result.tile_id.clone(),
                predictions: tile_result.predictions.clone(),
                cell_count: tile_result.cell_count,
                inference_ms: tile_result.inference_ms,
            });
            total_cells += tile_result.cell_count;
            total_ms += tile_result.inference_ms;
        }

        BatchGridPredictOutput {
            results: area_results,
            total_cells,
            total_inference_ms: total_ms,
            device_used: result.device_used,
        }
    }

    pub fn tiling_manager(&self) -> &TilingManager {
        &self.tiling_manager
    }

    pub fn config(&self) -> &TilingConfig {
        self.tiling_manager.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_config() {
        let config = TilingConfig {
            max_cells_per_tile: 1_000_000,
            max_concurrent_tiles: 2,
            target_total_seconds: 30,
            ..Default::default()
        };
        assert_eq!(config.max_cells_per_tile, 1_000_000);
        assert_eq!(config.max_concurrent_tiles, 2);
        assert_eq!(config.target_total_seconds, 30);
    }
}
