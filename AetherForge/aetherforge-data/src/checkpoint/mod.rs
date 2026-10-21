//! # Sync Checkpoint Management
//! 
//! Handles incremental sync state with durable checkpoints.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{debug, info};

use aetherforge_core::AetherError;

/// Checkpoint data for a sync job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCheckpoint {
    pub source_name: String,
    pub region_key: String,
    pub last_sync_time: DateTime<Utc>,
    pub last_synced_record: Option<String>,
    pub records_synced: u64,
    pub checksum: String,
    pub version: u32,
}

impl SyncCheckpoint {
    pub fn new(source_name: &str, region_key: &str) -> Self {
        Self {
            source_name: source_name.to_string(),
            region_key: region_key.to_string(),
            last_sync_time: Utc::now(),
            last_synced_record: None,
            records_synced: 0,
            checksum: String::new(),
            version: 1,
        }
    }
    
    pub fn with_record(mut self, record_id: &str, records: u64) -> Self {
        self.last_synced_record = Some(record_id.to_string());
        self.records_synced = records;
        self.checksum = Self::compute_checksum(record_id);
        self.last_sync_time = Utc::now();
        self
    }
    
    fn compute_checksum(record_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(record_id.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }
    
    pub fn is_valid(&self) -> bool {
        !self.checksum.is_empty()
    }
}

/// Checkpoint store for managing sync state
pub struct CheckpointStore {
    checkpoint_dir: PathBuf,
    checkpoints: Arc<AsyncRwLock<HashMap<String, SyncCheckpoint>>>,
    dirty: Arc<AsyncRwLock<Vec<String>>>,
}

impl CheckpointStore {
    pub fn new(data_dir: PathBuf) -> Self {
        let checkpoint_dir = data_dir.join("checkpoints");
        fs::create_dir_all(&checkpoint_dir).ok();
        
        Self {
            checkpoint_dir,
            checkpoints: Arc::new(AsyncRwLock::new(HashMap::new())),
            dirty: Arc::new(AsyncRwLock::new(Vec::new())),
        }
    }
    
    fn checkpoint_path(&self, key: &str) -> PathBuf {
        let encoded = Self::encode_key(key);
        self.checkpoint_dir.join(format!("{}.json", encoded))
    }
    
    fn encode_key(key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{:x}", hasher.finalize())[..32].to_string()
    }
    
    /// Load all checkpoints from disk
    pub async fn load_all(&self) -> Result<(), AetherError> {
        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| AetherError::storage("Failed to read checkpoint dir").with_source(e))?;
        
        let mut checkpoints = self.checkpoints.write().await;
        
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                let content = fs::read_to_string(&path)
                    .map_err(|e| AetherError::storage("Failed to read checkpoint").with_source(e))?;
                
                let checkpoint: SyncCheckpoint = serde_json::from_str(&content)
                    .map_err(|e| AetherError::serialization("Failed to parse checkpoint").with_source(e))?;
                
                let key = format!("{}:{}", checkpoint.source_name, checkpoint.region_key);
                checkpoints.insert(key, checkpoint);
            }
        }
        
        info!("Loaded {} checkpoints", checkpoints.len());
        Ok(())
    }
    
    /// Get checkpoint for a source/region
    pub async fn get(&self, source: &str, region: &str) -> Option<SyncCheckpoint> {
        let key = format!("{}:{}", source, region);
        self.checkpoints.read().await.get(&key).cloned()
    }
    
    /// Update checkpoint after successful sync
    pub async fn update(&self, checkpoint: SyncCheckpoint) -> Result<(), AetherError> {
        let key = format!("{}:{}", checkpoint.source_name, checkpoint.region_key);
        
        {
            let mut checkpoints = self.checkpoints.write().await;
            checkpoints.insert(key.clone(), checkpoint.clone());
        }
        
        {
            let mut dirty = self.dirty.write().await;
            if !dirty.contains(&key) {
                dirty.push(key.clone());
            }
        }
        
        self.persist(&checkpoint).await?;
        
        Ok(())
    }
    
    /// Persist checkpoint to disk
    async fn persist(&self, checkpoint: &SyncCheckpoint) -> Result<(), AetherError> {
        let key = format!("{}:{}", checkpoint.source_name, checkpoint.region_key);
        let path = self.checkpoint_path(&key);
        
        let json = serde_json::to_string_pretty(checkpoint)
            .map_err(|e| AetherError::serialization("Failed to serialize checkpoint").with_source(e))?;
        
        fs::write(&path, json)
            .map_err(|e| AetherError::storage("Failed to write checkpoint").with_source(e))?;
        
        debug!("Persisted checkpoint to {:?}", path);
        
        Ok(())
    }
    
    /// Flush all dirty checkpoints
    pub async fn flush_all(&self) -> Result<(), AetherError> {
        let dirty: Vec<String> = {
            let mut d = self.dirty.write().await;
            std::mem::take(&mut d)
        };
        
        let checkpoints = self.checkpoints.read().await;
        
        for key in dirty {
            if let Some(checkpoint) = checkpoints.get(&key) {
                self.persist(checkpoint).await?;
            }
        }
        
        Ok(())
    }
    
    /// Check if we can resume from checkpoint
    pub async fn can_resume(&self, source: &str, region: &str) -> bool {
        if let Some(cp) = self.get(source, region).await {
            let hours_since_sync = (Utc::now() - cp.last_sync_time).num_hours();
            hours_since_sync < 24
        } else {
            false
        }
    }
    
    /// Get the last sync time for a source
    pub async fn last_sync_time(&self, source: &str) -> Option<DateTime<Utc>> {
        let checkpoints = self.checkpoints.read().await;
        
        checkpoints
            .values()
            .filter(|cp| cp.source_name == source)
            .max_by_key(|cp| cp.last_sync_time)
            .map(|cp| cp.last_sync_time)
    }
    
    /// Clear all checkpoints (for full resync)
    pub async fn clear_all(&self) -> Result<(), AetherError> {
        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| AetherError::storage("Failed to read checkpoint dir").with_source(e))?;
        
        for entry in entries.flatten() {
            fs::remove_file(entry.path())
                .map_err(|e| AetherError::storage("Failed to remove checkpoint").with_source(e))?;
        }
        
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.clear();
        
        info!("Cleared all checkpoints");
        Ok(())
    }
}

impl Default for CheckpointStore {
    fn default() -> Self {
        Self::new(PathBuf::from("/var/aetherforge/data"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_checkpoint_persist() {
        let temp_dir = tempdir().unwrap();
        let store = CheckpointStore::new(temp_dir.path().to_path_buf());
        
        let checkpoint = SyncCheckpoint::new("open_meteo", "global")
            .with_record("record_12345", 1000);
        
        store.update(checkpoint.clone()).await.unwrap();
        
        let loaded = store.get("open_meteo", "global").await;
        assert!(loaded.is_some());
        
        let loaded = loaded.unwrap();
        assert_eq!(loaded.records_synced, 1000);
    }
    
    #[tokio::test]
    async fn test_resume() {
        let temp_dir = tempdir().unwrap();
        let store = CheckpointStore::new(temp_dir.path().to_path_buf());
        
        let can_resume = store.can_resume("open_meteo", "global").await;
        assert!(!can_resume);
    }
}