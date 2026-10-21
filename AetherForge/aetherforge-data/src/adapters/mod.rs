//! # Data Source Adapters
//! 
//! Implementations for various weather data providers.

pub mod open_meteo;
pub mod noaa;
pub mod era5;
pub mod gfs;

pub use open_meteo::OpenMeteoAdapter;
pub use noaa::NOAAAdapter;
pub use era5::ERA5Adapter;
pub use gfs::GFSAdapter;

use crate::sources::SourceRegistry;

/// Create default source registry with all supported sources
pub fn create_default_registry() -> SourceRegistry {
    let mut registry = SourceRegistry::new();
    
    registry.register(OpenMeteoAdapter::new("https://api.open-meteo.com", None));
    registry.register(NOAAAdapter::new(None));
    registry.register(ERA5Adapter::new(None));
    registry.register(GFSAdapter::new());
    
    registry
}

/// Create registry with API keys
pub fn create_registry_with_keys(
    open_meteo_key: Option<String>,
    noaa_token: Option<String>,
    era5_key: Option<String>,
    use_gfs_s3: bool,
) -> SourceRegistry {
    let mut registry = SourceRegistry::new();
    
    registry.register(OpenMeteoAdapter::new("https://api.open-meteo.com", open_meteo_key));
    registry.register(NOAAAdapter::new(noaa_token));
    registry.register(ERA5Adapter::new(era5_key));
    registry.register(GFSAdapter::new().with_s3(use_gfs_s3));
    
    registry
}
