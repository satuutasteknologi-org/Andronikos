//! # AetherForge ML Core - Graph Neural Network Weather Prediction
//! 
//! This module implements the GraphCast-inspired architecture for weather forecasting,
//! combining Graph Neural Networks with Spatiotemporal Attention mechanisms.
//! 
//! ## Architecture Overview
//! 
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────┐
//! │                         AetherGraphCast Architecture                          │
//! ├──────────────────────────────────────────────────────────────────────────────┤
//! │  Input Encoder                                                                │
//! │  ├── Surface variables (temperature, humidity, pressure, wind, precip)     │
//! │  ├── Atmospheric variables (13 pressure levels × 6 variables)               │
//! │  ├── Latitude/longitude positional encoding                                  │
//! │  └── Temporal encoding                                                        │
//! │                                      ↓                                        │
//! │  Graph Processor (12 GNN layers)                                              │
//! │  ├── Edge update with message passing                                         │
//! │  ├── Edge attention (multi-head)                                             │
//! │  ├── Node aggregation                                                        │
//! │  └── Residual connections + LayerNorm                                         │
//! │                                      ↓                                        │
//! │  Temporal Transformer (3 multi-head attention layers)                        │
//! │                                      ↓                                        │
//! │  Multi-Task Decoder                                                           │
//! │  ├── Temperature prediction                                                   │
//! │  ├── Humidity prediction                                                     │
//! │  ├── Precipitation prediction                                                │
//! │  ├── Wind speed prediction                                                    │
//! │  └── Surface pressure prediction                                             │
//! └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//! 
//! ## Model Variants
//! 
//! - **GraphCast**: Full 3D atmospheric model (default)
//! - **Pangu-Weather**: Hierarchical temporal model
//! - **FourCastNet**: Fourier-based global model
//! 
//! ## Training Features
//! 
//! - Multi-task weighted loss
//! - CRPS (Continuous Ranked Probability Score) for probabilistic predictions
//! - Early stopping with patience
//! - Learning rate scheduling (warmup + cosine decay)
//! - Gradient clipping

#![forbid(unsafe_code)]

pub mod models;
pub mod layers;
pub mod training;
pub mod loss;
pub mod graph;
pub mod inference;

pub use models::{AetherGraphCast, AetherPredictor, DefaultPredictor, GraphCastConfig, GraphCastOutput, GridPrediction, ModelConfig, PipelinePredictor};
pub use layers::*;
pub use training::{ModelTrainer, TrainingConfig, TrainingState, TrainingResult};
pub use loss::{WeatherLoss, CRPSLoss, MetricTracker, LossMetrics, rmse, mae, acc};
pub use inference::{InferencePipeline, InferenceConfig, GridPredictionRequest, GridPredictionResult, PipelineStats};
