//! # ML Engine Core - Continuous Inference Architecture

pub mod pipeline;
pub mod model_manager;
pub mod batch_processor;
pub mod cache;

pub use pipeline::InferencePipeline;
pub use model_manager::{ModelManager, ModelVersion, ModelConfig};
pub use batch_processor::{BatchProcessor, BatchConfig, ProcessingResult};
pub use cache::{PredictionCache, CacheConfig};

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PredictionRequest {
    pub grid_cells: Vec<GridCellInput>,
    pub lead_seconds: u32,
    pub variables: Vec<String>,
    pub priority: RequestPriority,
}

impl PredictionRequest {
    pub fn new(grid_cells: Vec<GridCellInput>, lead_seconds: u32) -> Self {
        Self {
            grid_cells,
            lead_seconds,
            variables: vec!["Temperature".to_string()],
            priority: RequestPriority::Normal,
        }
    }
    
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for RequestPriority {
    fn default() -> Self { Self::Normal }
}

#[derive(Debug, Clone)]
pub struct GridCellInput {
    pub cell_id: u64,
    pub lat: f64,
    pub lon: f64,
    pub features: Vec<f32>,
    pub timestamp_ms: i64,
}

impl GridCellInput {
    pub fn new(lat: f64, lon: f64, features: Vec<f32>) -> Self {
        Self {
            cell_id: Self::compute_cell_id(lat, lon),
            lat,
            lon,
            features,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }
    
    fn compute_cell_id(lat: f64, lon: f64) -> u64 {
        let lat_int = ((lat + 90.0) * 10000.0) as u64;
        let lon_int = ((lon + 180.0) * 10000.0) as u64;
        (lat_int << 32) | lon_int
    }
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub cell_id: u64,
    pub predictions: HashMap<String, f32>,
    pub confidence: f32,
    pub latency_ms: u64,
    pub model_version: String,
}

impl PredictionResult {
    pub fn new(cell_id: u64, predictions: HashMap<String, f32>) -> Self {
        Self {
            cell_id,
            predictions,
            confidence: 0.9,
            latency_ms: 0,
            model_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prediction_request() {
        let input = GridCellInput::new(52.52, 13.405, vec![1.0, 2.0, 3.0]);
        let request = PredictionRequest::new(vec![input], 3600);
        
        assert_eq!(request.lead_seconds, 3600);
        assert_eq!(request.grid_cells.len(), 1);
    }
    
    #[test]
    fn test_cell_id_computation() {
        let cell1 = GridCellInput::compute_cell_id(52.52, 13.405);
        let cell2 = GridCellInput::compute_cell_id(52.52, 13.405);
        assert_eq!(cell1, cell2);
        
        let cell3 = GridCellInput::compute_cell_id(52.53, 13.405);
        assert_ne!(cell1, cell3);
    }
}