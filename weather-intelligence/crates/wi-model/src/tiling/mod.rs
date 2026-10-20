//! # Tiling Module
//!
//! Auto-tiling dan distributed inference untuk skala besar (38.9M cell Indonesia 0.5km).

pub mod tile_coordinator;
pub mod tiling_manager;

pub use tile_coordinator::{CoordinatedInferenceResult, TileCoordinator};
pub use tiling_manager::{Tile, TileResult, TilingConfig, TilingManager};
