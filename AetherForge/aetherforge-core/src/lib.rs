//! # AetherForge Core Library
//!
//! Weather Intelligence Engine - Pure Rust ML-powered weather forecasting system.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        AetherForge System                           │
//  ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
//! │  │   CLI/API    │  │  Data Sync   │  │  ML Engine   │  │ Storage  │ │
//! │  │  Interface   │  │    Engine    │  │  (GNN+ATT)   │  │ (Parquet)│ │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │
//! │         │                 │                   │               │      │
//! │  ┌──────┴─────────────────┴───────────────────┴───────────────┴────┐ │
//! │  │                    aetherforge-core Types & Traits              │ │
//! │  └──────────────────────────────────────────────────────────────────┘ │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Modules
//!
//! - `types`: Core data types for weather data (stations, observations, forecasts)
//! - `config`: Configuration management with Figment
//! - `error`: Unified error handling
//! - `utils`: Utility functions (geohash, time, etc.)

#![forbid(unsafe_code)]
#![warn(unreachable_pub, rust_2018_idioms)]
#![doc(html_root_url = "https://docs.rs/aetherforge-core")]

pub mod config;
pub mod error;
pub mod types;
pub mod utils;

pub use config::*;
pub use error::*;
pub use types::*;

// Re-export commonly used items
pub use chrono::{DateTime, Duration as ChronoDuration, Utc};
pub use geo::Coord;
pub use h3o::CellIndex;
pub use sha2::{Digest, Sha256};
