//! # Weather Intelligence Models
//!
//! Dual-path ML inference:
//! - Burn native: GraphCast global model
//! - ONNX Runtime: Regional downscaling models

#![forbid(unsafe_code)]
#![allow(missing_docs)]

pub mod burn;
pub mod fusion;
pub mod onnx;
pub mod registry;
pub mod tiling;
pub mod traits;

pub use burn::{GraphCastBatchOutput, GraphCastConfig, GraphCastModel};
pub use fusion::{FusedVariable, FusionEngine, FusionOutput};
pub use onnx::OnnxDownscaler;
pub use registry::{ModelRegistry, ModelVersion};
pub use tiling::{CoordinatedInferenceResult, Tile, TileCoordinator, TilingConfig, TilingManager};
pub use traits::{
    Downscaler, DownscalerInput, DownscalerOutput, DynDownscaler, DynForecaster, Forecaster,
    GridPrediction, ModelInfo, QuantilePrediction,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_registry() {
        // TODO: Add tests
    }
}
