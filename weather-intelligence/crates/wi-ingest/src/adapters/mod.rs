//! # Source Adapters
//!
//! Implementations for various weather API providers.

pub mod bmkg;
pub mod era5;
pub mod firms;
pub mod gfs;
pub mod himawari;
pub mod noaa;
pub mod open_meteo;
pub mod weather_next;

pub use bmkg::BMKGAdapter;
pub use era5::ERA5Adapter;
pub use firms::FirmsAdapter;
pub use gfs::GFSAdapter;
pub use himawari::HimawariAdapter;
pub use noaa::NOAAAdapter;
pub use open_meteo::OpenMeteoAdapter;
pub use weather_next::WeatherNextAdapter;

use crate::sources::SourceRegistry;

/// Create default source registry with Open-Meteo
pub fn create_default_registry() -> SourceRegistry {
    let mut registry = SourceRegistry::new();
    registry.register(Box::new(OpenMeteoAdapter::new()));
    registry
}

/// Create registry with API keys
pub fn create_registry_with_keys(
    open_meteo_key: Option<String>,
    noaa_token: Option<String>,
    bmkg_key: Option<String>,
    weather_next_key: Option<String>,
) -> SourceRegistry {
    let mut registry = SourceRegistry::new();
    registry.register(Box::new(OpenMeteoAdapter::new_with_key(open_meteo_key)));
    if noaa_token.is_some() {
        registry.register(Box::new(NOAAAdapter::new(noaa_token)));
    }
    if bmkg_key.is_some() {
        registry.register(Box::new(BMKGAdapter::new(bmkg_key)));
    }
    if weather_next_key.is_some() {
        registry.register(Box::new(WeatherNextAdapter::new(weather_next_key, None)));
    }
    registry
}
