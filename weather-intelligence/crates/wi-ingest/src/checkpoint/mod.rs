//! # Sync Checkpoint Management
//!
//! Incremental sync checkpoints with integrity verification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use wi_core::Result;
use wi_core::WIError;

/// Sync checkpoint for resume capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCheckpoint {
    pub source: String,
    pub region_key: String,
    pub last_sync_time: DateTime<Utc>,
    pub last_record_id: String,
    pub last_record_index: u64,
    pub records_processed: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub checksum: String,
}

impl SyncCheckpoint {
    pub fn new(source: &str, region_key: &str) -> Self {
        let now = Utc::now();
        let mut cp = Self {
            source: source.to_string(),
            region_key: region_key.to_string(),
            last_sync_time: now,
            last_record_id: String::new(),
            last_record_index: 0,
            records_processed: 0,
            created_at: now,
            updated_at: now,
            checksum: String::new(),
        };
        cp.update_checksum();
        cp
    }

    pub fn with_record(mut self, record_id: &str, index: u64) -> Self {
        self.last_record_id = record_id.to_string();
        self.last_record_index = index;
        self.records_processed += 1;
        self.updated_at = Utc::now();
        self.update_checksum();
        self
    }

    pub fn with_time(mut self, time: DateTime<Utc>) -> Self {
        self.last_sync_time = time;
        self.updated_at = Utc::now();
        self.update_checksum();
        self
    }

    fn update_checksum(&mut self) {
        let data = format!(
            "{}:{}:{}:{}:{}",
            self.source,
            self.region_key,
            self.last_record_id,
            self.last_record_index,
            self.records_processed
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        self.checksum = format!("{:x}", hasher.finalize());
    }

    pub fn is_valid(&self) -> bool {
        let data = format!(
            "{}:{}:{}:{}:{}",
            self.source,
            self.region_key,
            self.last_record_id,
            self.last_record_index,
            self.records_processed
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let computed = format!("{:x}", hasher.finalize());
        computed == self.checksum
    }
}

/// Checkpoint store for persistence
#[derive(Debug)]
pub struct CheckpointStore {
    checkpoints: RwLock<HashMap<String, SyncCheckpoint>>,
    persist_path: Option<PathBuf>,
}

impl CheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_path(path: PathBuf) -> Self {
        let mut store = Self {
            persist_path: Some(path),
            checkpoints: RwLock::new(HashMap::new()),
        };
        store.load().ok();
        store
    }

    /// Get checkpoint for source and region
    pub async fn get(&self, source: &str, region_key: &str) -> Option<SyncCheckpoint> {
        let key = Self::make_key(source, region_key);
        self.checkpoints.read().get(&key).cloned()
    }

    /// Update checkpoint
    pub async fn update(&self, checkpoint: SyncCheckpoint) -> Result<()> {
        let key = Self::make_key(&checkpoint.source, &checkpoint.region_key);
        self.checkpoints.write().insert(key, checkpoint.clone());

        // Persist if path configured
        if let Some(ref path) = self.persist_path {
            self.persist(path).await?;
        }

        Ok(())
    }

    /// Delete checkpoint
    pub async fn delete(&self, source: &str, region_key: &str) -> Result<()> {
        let key = Self::make_key(source, region_key);
        self.checkpoints.write().remove(&key);

        if let Some(ref path) = self.persist_path {
            self.persist(path).await?;
        }

        Ok(())
    }

    /// List all checkpoints
    pub async fn list(&self) -> Vec<SyncCheckpoint> {
        self.checkpoints.read().values().cloned().collect()
    }

    /// Clear all checkpoints for a source (force full resync)
    pub async fn clear_source(&self, source: &str) -> Result<()> {
        {
            let mut checkpoints = self.checkpoints.write();
            checkpoints.retain(|_, cp| cp.source != source);
        }

        if let Some(ref path) = self.persist_path {
            self.persist(path).await?;
        }

        Ok(())
    }

    fn make_key(source: &str, region_key: &str) -> String {
        format!("{}:{}", source, region_key)
    }

    async fn persist(&self, path: &Path) -> Result<()> {
        let checkpoints = self.checkpoints.read().clone();
        let data = serde_json::to_vec_pretty(&checkpoints).map_err(|e| {
            WIError::serialization_with_source("Failed to serialize checkpoints", e.into())
        })?;

        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .map_err(|e| {
                WIError::storage_with_source("Failed to create checkpoint dir", e.into())
            })?;

        tokio::fs::write(path, data)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to write checkpoints", e.into()))?;

        Ok(())
    }

    fn load(&mut self) -> Result<()> {
        if let Some(ref path) = self.persist_path {
            if path.exists() {
                let data = std::fs::read(path).map_err(|e| {
                    WIError::storage_with_source("Failed to read checkpoints", e.into())
                })?;

                let checkpoints: HashMap<String, SyncCheckpoint> = serde_json::from_slice(&data)
                    .map_err(|e| {
                        WIError::serialization_with_source("Failed to parse checkpoints", e.into())
                    })?;

                *self.checkpoints.write() = checkpoints;
            }
        }
        Ok(())
    }
}

impl Default for CheckpointStore {
    fn default() -> Self {
        Self {
            checkpoints: RwLock::new(HashMap::new()),
            persist_path: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let cp = SyncCheckpoint::new("open_meteo", "region_1");
        assert_eq!(cp.source, "open_meteo");
        assert_eq!(cp.region_key, "region_1");
        assert!(cp.is_valid());
    }

    #[test]
    fn test_checkpoint_with_record() {
        let cp = SyncCheckpoint::new("open_meteo", "region_1").with_record("record_100", 100);
        assert_eq!(cp.last_record_id, "record_100");
        assert_eq!(cp.last_record_index, 100);
        assert_eq!(cp.records_processed, 1);
        assert!(cp.is_valid());
    }

    #[test]
    fn test_checkpoint_corruption_detection() {
        let mut cp = SyncCheckpoint::new("open_meteo", "region_1").with_record("record_100", 100);

        // Corrupt the data
        cp.last_record_index = 999;
        assert!(!cp.is_valid());
    }
}
