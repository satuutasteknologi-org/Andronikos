//! AetherForge Data Module Binary
//! 
//! Entry point for data sync operations

use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("AetherForge Data Engine v{}", env!("CARGO_PKG_VERSION"));
    
    let storage_config = aetherforge_data::StorageConfig {
        base_path: PathBuf::from("/tmp/aetherforge/data"),
        compression_enabled: true,
        compression_level: 3,
    };
    let _storage = Arc::new(aetherforge_data::StorageManager::new(storage_config));
    
    let sync_engine = aetherforge_data::SyncEngine::new();
    
    info!("Data engine initialized");
    info!("Syncing data...");
    
    let end = chrono::Utc::now();
    let start = end - chrono::Duration::days(1);
    
    let stats = sync_engine.sync_all(start, end).await?;
    
    info!("Sync complete: {} records fetched, {} validated, {} written, {} failed", 
          stats.records_fetched, stats.records_validated, stats.records_written, stats.records_failed);
    
    Ok(())
}