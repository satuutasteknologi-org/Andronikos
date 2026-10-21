//! # Spatial Engine - Single Source of Truth
//!
//! The definitive spatial operations engine for AetherForge weather data.
//! Provides:
//! - Global 1km × 1km grid management
//! - Intelligent spatial indexing and queries
//! - Multi-resolution support (H3 + custom grid)
//! - Spatial interpolation and prediction

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use aetherforge_core::{AetherError, Result, WeatherObservation};

use super::grid::{CellResolution, GridBounds, GridCell};
use super::indexing::{GridIndexer, H3Indexer};
use super::interpolation::{
    InterpolationConfig, InterpolationEngine, ObservationPoint,
};

/// Main spatial engine - single source of truth for all spatial operations
pub struct SpatialEngine {
    config: SpatialEngineConfig,

    // Spatial indices
    h3_index: Arc<RwLock<H3Indexer>>,
    grid_index: Arc<RwLock<GridIndexer>>,

    // Interpolation engines per variable
    interpolation_engines: RwLock<HashMap<String, Arc<InterpolationEngine>>>,

    // Grid cache
    global_grid: RwLock<Option<Vec<GridCell>>>,
    regional_grids: RwLock<HashMap<String, Vec<GridCell>>>,

    // Statistics
    stats: RwLock<SpatialStats>,
}

#[derive(Debug, Clone)]
pub struct SpatialEngineConfig {
    pub default_resolution: CellResolution,
    pub global_bounds: GridBounds,
    pub search_radius_meters: f64,
    pub enable_adaptive_resolution: bool,
    pub min_stations_for_adaptive: usize,
    pub cache_grids: bool,
}

impl Default for SpatialEngineConfig {
    fn default() -> Self {
        Self {
            default_resolution: CellResolution::Kilometer1,
            global_bounds: GridBounds::global(),
            search_radius_meters: 50_000.0,
            enable_adaptive_resolution: true,
            min_stations_for_adaptive: 10,
            cache_grids: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpatialStats {
    pub total_observations: usize,
    pub cells_populated: usize,
    pub last_update: Option<chrono::DateTime<chrono::Utc>>,
    pub query_count: u64,
    pub interpolation_count: u64,
    pub cache_hits: u64,
}

/// Spatial engine error types
#[derive(Debug, thiserror::Error)]
pub enum SpatialError {
    #[error("No observations available for interpolation")]
    NoObservations,

    #[error("Insufficient observations: {0} < {1}")]
    InsufficientObservations(usize, usize),

    #[error("Invalid coordinate: lat={lat}, lon={lon}")]
    InvalidCoordinate { lat: f64, lon: f64 },

    #[error("Grid computation failed: {0}")]
    GridComputationFailed(String),
}

impl SpatialEngine {
    pub fn new() -> Self {
        Self::with_config(SpatialEngineConfig::default())
    }

    pub fn with_config(config: SpatialEngineConfig) -> Self {
        let h3 = H3Indexer::from_resolution(config.default_resolution);
        let grid = GridIndexer::new(config.default_resolution, config.global_bounds.clone());

        Self {
            config: config.clone(),
            h3_index: Arc::new(RwLock::new(h3)),
            grid_index: Arc::new(RwLock::new(grid)),
            interpolation_engines: RwLock::new(HashMap::new()),
            global_grid: RwLock::new(None),
            regional_grids: RwLock::new(HashMap::new()),
            stats: RwLock::new(SpatialStats::default()),
        }
    }

    // =========================================================================
    // Observation Management
    // =========================================================================

    pub fn add_observation(&self, lat: f64, lon: f64, value: f64, variable: &str) {
        let obs_idx = {
            let mut stats = self.stats.write();
            stats.total_observations += 1;
            stats.total_observations - 1
        } as u64;

        self.h3_index.write().insert_point(lat, lon, obs_idx);
        self.grid_index.write().insert(lat, lon, obs_idx);

        let obs_point = ObservationPoint::with_quality(lat, lon, value, 1.0);

        let mut engines = self.interpolation_engines.write();
        let _engine = engines
            .entry(variable.to_string())
            .or_insert_with(|| Arc::new(InterpolationEngine::new(InterpolationConfig::default())));

        // Get mutable reference through the RwLock inside Arc
        if let Some(eng) = engines.get_mut(variable) {
            Arc::get_mut(eng).map(|e| e.add_observation(obs_point));
        }
    }

    pub fn add_observations(&self, observations: &[(f64, f64, f64)], variable: &str) {
        for &(lat, lon, value) in observations {
            self.add_observation(lat, lon, value, variable);
        }

        let mut stats = self.stats.write();
        stats.last_update = Some(chrono::Utc::now());
    }

    pub fn add_weather_observations(
        &self,
        obs: &[WeatherObservation],
        variable: aetherforge_core::WeatherVariable,
    ) {
        let var_name = format!("{:?}", variable);

        for weather_obs in obs {
            if let Some(value) = weather_obs.variables.get(&variable) {
                if value.is_valid {
                    self.add_observation(
                        weather_obs.coord.lat,
                        weather_obs.coord.lon,
                        value.value,
                        &var_name,
                    );
                }
            }
        }
    }

    // =========================================================================
    // Grid Operations
    // =========================================================================

    pub fn create_global_grid(&self, resolution: CellResolution) -> Vec<GridCell> {
        if self.config.cache_grids {
            if let Some(cache) = self.global_grid.read().as_ref() {
                if cache
                    .first()
                    .map(|c| c.resolution == resolution)
                    .unwrap_or(false)
                {
                    return cache.clone();
                }
            }
        }

        let mut cells = Vec::new();
        let bounds = &self.config.global_bounds;

        let cell_size_deg = resolution.target_km2().sqrt() / 110.0;

        let mut lat = bounds.min_lat;
        while lat <= bounds.max_lat {
            let mut lon = bounds.min_lon;
            while lon <= bounds.max_lon {
                let cell = GridCell::new(lat, lon, resolution);
                cells.push(cell);
                lon += cell_size_deg;
            }
            lat += cell_size_deg;
        }

        if self.config.cache_grids {
            *self.global_grid.write() = Some(cells.clone());
        }

        cells
    }

    pub fn get_cell_for_point(&self, lat: f64, lon: f64) -> Option<GridCell> {
        self.grid_index
            .read()
            .get_cell_at(lat, lon)
            .or_else(|| Some(GridCell::new(lat, lon, self.config.default_resolution)))
    }

    pub fn get_neighbors(&self, lat: f64, lon: f64, rings: u8) -> Vec<GridCell> {
        let h3_index = self.h3_index.read();
        let neighbors = h3_index.get_neighbors(lat, lon, rings);

        neighbors.into_iter().map(GridCell::from_h3_index).collect()
    }

    // =========================================================================
    // Query Operations
    // =========================================================================

    pub fn query_point(&self, lat: f64, lon: f64) -> Option<u64> {
        {
            let mut stats = self.stats.write();
            stats.query_count += 1;
        }

        self.h3_index.read().get_cell_for_point(lat, lon)
    }

    pub fn query_radius(&self, lat: f64, lon: f64, radius_m: f64) -> Vec<(f64, f64, f64)> {
        let h3_index = self.h3_index.read();
        let cell_ids = h3_index.query_radius(lat, lon, radius_m);

        let mut results = Vec::new();
        for cell_id in cell_ids {
            if let Some(data_ids) = h3_index.get_data_in_cell(cell_id) {
                for data_id in data_ids {
                    results.push(((*data_id) as f64, cell_id as f64, 0.0));
                }
            }
        }

        results
    }

    pub fn query_box(
        &self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Vec<GridCell> {
        let grid_index = self.grid_index.read();
        let data_ids = grid_index.query_box(min_lat, min_lon, max_lat, max_lon);

        data_ids
            .iter()
            .filter_map(|&id| grid_index.get_cell_by_id(id))
            .collect()
    }

    // =========================================================================
    // Interpolation Operations
    // =========================================================================

    pub fn interpolate(&self, lat: f64, lon: f64, variable: &str) -> f64 {
        {
            let mut stats = self.stats.write();
            stats.interpolation_count += 1;
        }

        let engines = self.interpolation_engines.read();
        if let Some(engine) = engines.get(variable) {
            engine.interpolate(lat, lon)
        } else {
            280.0
        }
    }

    pub fn interpolate_grid(&self, grid: &[GridCell], variable: &str) -> HashMap<u64, f64> {
        let engines = self.interpolation_engines.read();
        let engine = engines.get(variable);

        let mut results = HashMap::new();

        if let Some(eng) = engine {
            for cell in grid {
                let value = eng.interpolate(cell.center_lat, cell.center_lon);
                results.insert(cell.cell_id, value);
            }
        }

        results
    }

    // =========================================================================
    // 1km Grid Generation (Precise)
    // =========================================================================

    pub fn generate_1km_grid(&self, bounds: &GridBounds) -> Vec<GridCell> {
        let cells = self.create_global_grid(CellResolution::Kilometer1);

        cells
            .into_iter()
            .filter(|cell| {
                cell.center_lat >= bounds.min_lat
                    && cell.center_lat <= bounds.max_lat
                    && cell.center_lon >= bounds.min_lon
                    && cell.center_lon <= bounds.max_lon
            })
            .collect()
    }

    pub fn generate_1km_grid_parallel(
        &self,
        bounds: &GridBounds,
        num_threads: usize,
    ) -> Vec<GridCell> {
        let all_cells = self.create_global_grid(CellResolution::Kilometer1);

        let chunk_size = all_cells.len() / num_threads;
        let mut handles = Vec::new();

        for (_i, chunk) in all_cells.chunks(chunk_size).enumerate() {
            let filtered: Vec<GridCell> = chunk
                .iter()
                .filter(|cell| {
                    cell.center_lat >= bounds.min_lat
                        && cell.center_lat <= bounds.max_lat
                        && cell.center_lon >= bounds.min_lon
                        && cell.center_lon <= bounds.max_lon
                })
                .cloned()
                .collect();

            handles.push(filtered);
        }

        handles.into_iter().flatten().collect()
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    pub fn stats(&self) -> SpatialStats {
        self.stats.read().clone()
    }

    pub fn update_stats(&self) {
        let mut stats = self.stats.write();

        let grid = self.grid_index.read();
        stats.cells_populated = grid.cell_count();
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    pub fn validate_coordinate(lat: f64, lon: f64) -> Result<()> {
        if lat < -90.0 || lat > 90.0 {
            return Err(AetherError::geospatial(format!(
                "Invalid latitude: {lat} (must be -90 to 90)"
            )));
        }
        if lon < -180.0 || lon > 180.0 {
            return Err(AetherError::geospatial(format!(
                "Invalid longitude: {lon} (must be -180 to 180)"
            )));
        }
        Ok(())
    }

    pub fn resolution(&self) -> CellResolution {
        self.config.default_resolution
    }

    pub fn clear(&self) {
        // For Arc<RwLock<T>>, we need to replace the Arc contents
        // Since we can't directly write to the inner RwLock through Arc,
        // we recreate the indices
        let _new_h3 = H3Indexer::from_resolution(self.config.default_resolution);
        let _new_grid = GridIndexer::new(
            self.config.default_resolution,
            self.config.global_bounds.clone(),
        );

        // Use Arc::ptr_map to create new Arcs with the new values
        // Since h3_index and grid_index are Arc, we can't easily replace inner values
        // Instead, we clear the existing ones through their methods
        self.h3_index.write().clear();
        self.grid_index.write().clear();

        self.interpolation_engines.write().clear();
        *self.global_grid.write() = None;
        self.regional_grids.write().clear();

        let mut stats = self.stats.write();
        *stats = SpatialStats::default();
    }
}

impl Default for SpatialEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = SpatialEngine::new();
        assert_eq!(engine.stats().total_observations, 0);
    }

    #[test]
    fn test_add_observations() {
        let engine = SpatialEngine::new();

        engine.add_observation(52.52, 13.405, 280.0, "Temperature");
        engine.add_observation(52.53, 13.41, 281.0, "Temperature");

        assert_eq!(engine.stats().total_observations, 2);
    }

    #[test]
    fn test_interpolation() {
        let engine = SpatialEngine::new();

        engine.add_observation(52.0, 13.0, 280.0, "Temperature");
        engine.add_observation(53.0, 13.0, 285.0, "Temperature");

        let interpolated = engine.interpolate(52.5, 13.0, "Temperature");
        assert!(interpolated > 280.0 && interpolated < 285.0);
    }

    #[test]
    fn test_1km_grid_generation() {
        let engine = SpatialEngine::new();
        let bounds = GridBounds::new(52.0, 12.0, 53.0, 14.0);

        let grid = engine.generate_1km_grid(&bounds);
        assert!(!grid.is_empty());
    }
}
