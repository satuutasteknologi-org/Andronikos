//! # Model Manager - Versioning, Hot-Reload, and Caching
//! 
//! Manages multiple model versions with:
//! - Hot-reload for zero-downtime updates
//! - Model versioning and rollback
//! - Read-write locks for concurrent access
//! - Memory-efficient model caching

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn, error};

use burn::module::Module;
use burn::prelude::Backend;

use crate::models::AetherGraphCast;
use crate::models::GraphCastConfig;

/// Model version metadata
#[derive(Debug, Clone)]
pub struct ModelVersion {
    pub version_id: String,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    pub file_path: Option<String>,
    pub config: GraphCastConfig,
    pub is_active: bool,
    pub inference_count: u64,
    pub avg_inference_time_ms: f64,
}

impl ModelVersion {
    pub fn new(version_id: &str, config: GraphCastConfig) -> Self {
        Self {
            version_id: version_id.to_string(),
            loaded_at: chrono::Utc::now(),
            file_path: None,
            config,
            is_active: false,
            inference_count: 0,
            avg_inference_time_ms: 0.0,
        }
    }
}

/// Model configuration for the manager
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub max_versions: usize,
    pub enable_hot_reload: bool,
    pub model_dir: String,
    pub auto_save_interval_secs: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_versions: 5,
            enable_hot_reload: true,
            model_dir: "/var/aetherforge/models".to_string(),
            auto_save_interval_secs: 300,
        }
    }
}

/// Thread-safe model manager with versioning support
pub struct ModelManager<B: Backend> {
    config: ModelConfig,
    
    // Active model for inference
    active_model: Arc<RwLock<Option<Arc<AetherGraphCast<B>>>>,
    
    // Model version history
    versions: Arc<RwLock<VecDeque<ModelVersion>>>,
    
    // Current active version ID
    active_version_id: Arc<RwLock<Option<String>>>,
    
    // Semaphore for read-write locking
    rw_semaphore: Arc<Semaphore>,
    
    // Model loading semaphore (prevent too many loads)
    load_semaphore: Arc<Semaphore>,
}

impl<B: Backend> ModelManager<B> {
    /// Create new model manager
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            active_model: Arc::new(RwLock::new(None)),
            versions: Arc::new(RwLock::new(VecDeque::new())),
            active_version_id: Arc::new(RwLock::new(None)),
            rw_semaphore: Arc::new(Semaphore::new(10)),
            load_semaphore: Arc::new(Semaphore::new(1)),
        }
    }
    
    /// Load a model into the manager
    pub async fn load_model(
        &self,
        version_id: &str,
        config: GraphCastConfig,
        model: AetherGraphCast<B>,
    ) -> Result<(), String> {
        // Acquire load permit (only one load at a time)
        let _load_permit = self.load_semaphore.acquire().await
            .map_err(|_| "Failed to acquire load permit")?;
        
        let mut versions = self.versions.write();
        
        // Check if version already exists
        if versions.iter().any(|v| v.version_id == version_id) {
            warn!("Model version {} already loaded", version_id);
            return Ok(());
        }
        
        // Create version metadata
        let mut version = ModelVersion::new(version_id, config.clone());
        version.is_active = true;
        
        // Deactivate previous active version
        for v in versions.iter_mut() {
            v.is_active = false;
        }
        
        // Add new version
        versions.push_back(version);
        
        // Enforce max versions limit
        while versions.len() > self.config.max_versions {
            versions.pop_front();
        }
        
        // Update active model
        {
            let mut active = self.active_model.write();
            *active = Some(Arc::new(model));
        }
        
        {
            let mut active_id = self.active_version_id.write();
            *active_id = Some(version_id.to_string());
        }
        
        info!("Loaded model version {} (total versions: {})", version_id, versions.len());
        
        Ok(())
    }
    
    /// Load model from file
    pub async fn load_from_file(
        &self,
        version_id: &str,
        file_path: &Path,
    ) -> Result<(), String> {
        info!("Loading model {} from {:?}", version_id, file_path);
        
        // Load model using Burn's file loading
        let model = AetherGraphCast::<B>::load_file(file_path)
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        let config = GraphCastConfig::default();
        
        self.load_model(version_id, config, model).await
    }
    
    /// Get read lock for inference
    pub async fn acquire_read(&self) -> ModelReadGuard<B> {
        // Wait for read permit
        let _permit = self.rw_semaphore.acquire().await
            .expect("Semaphore closed");
        
        ModelReadGuard {
            model: self.active_model.read(),
            version_id: self.active_version_id.read().clone(),
            _permit,
        }
    }
    
    /// Get active model (blocking)
    pub fn get_active_model(&self) -> Option<Arc<AetherGraphCast<B>>> {
        self.active_model.read().clone()
    }
    
    /// Get active version info
    pub fn get_active_version(&self) -> Option<ModelVersion> {
        let versions = self.versions.read();
        let active_id = self.active_version_id.read();
        
        active_id.as_ref()
            .and_then(|id| versions.iter().find(|v| &v.version_id == id))
            .cloned()
    }
    
    /// List all loaded versions
    pub fn list_versions(&self) -> Vec<ModelVersion> {
        self.versions.read().iter().cloned().collect()
    }
    
    /// Rollback to previous version
    pub async fn rollback(&self) -> Result<(), String> {
        let mut versions = self.versions.write();
        let mut active_id = self.active_version_id.write();
        
        if versions.len() < 2 {
            return Err("Cannot rollback: only one version loaded".to_string());
        }
        
        // Find and deactivate current
        for v in versions.iter_mut() {
            if v.is_active {
                v.is_active = false;
            }
        }
        
        // Activate second-to-last
        if let Some(prev) = versions.iter_mut().rev().nth(1) {
            prev.is_active = true;
            *active_id = Some(prev.version_id.clone());
        }
        
        // Reload model
        drop(versions);
        drop(active_id);
        
        self.reload_active_model().await
    }
    
    /// Reload the active model (after rollback or hot-reload)
    async fn reload_active_model(&self) -> Result<(), String> {
        let version_id = self.active_version_id.read().clone();
        
        if let Some(id) = version_id {
            let versions = self.versions.read();
            if let Some(version) = versions.iter().find(|v| v.version_id == id) {
                if let Some(ref path) = version.file_path {
                    drop(versions);
                    self.load_from_file(&id, Path::new(path)).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Save current model to file
    pub async fn save_active(&self, path: &Path) -> Result<(), String> {
        let model = self.get_active_model()
            .ok_or("No active model to save")?;
        
        model.save_file(path)
            .map_err(|e| format!("Failed to save model: {}", e))?;
        
        // Update version metadata
        let mut versions = self.versions.write();
        let active_id = self.active_version_id.read();
        
        if let Some(id) = active_id.as_ref() {
            if let Some(version) = versions.iter_mut().find(|v| &v.version_id == id) {
                version.file_path = Some(path.to_string_lossy().to_string());
            }
        }
        
        info!("Saved active model to {:?}", path);
        Ok(())
    }
    
    /// Update inference statistics
    pub fn record_inference(&self, latency_ms: f64) {
        let mut versions = self.versions.write();
        
        if let Some(version) = versions.iter_mut().find(|v| v.is_active) {
            let n = version.inference_count as f64;
            version.avg_inference_time_ms = 
                (version.avg_inference_time_ms * n + latency_ms) / (n + 1.0);
            version.inference_count += 1;
        }
    }
    
    /// Clear all models from memory
    pub fn clear(&self) {
        let mut active = self.active_model.write();
        *active = None;
        
        let mut versions = self.versions.write();
        versions.clear();
        
        let mut active_id = self.active_version_id.write();
        *active_id = None;
        
        info!("Cleared all models");
    }
}

/// RAII guard for model read access
pub struct ModelReadGuard<'a, B: Backend> {
    model: RwLockReadGuard<'a, Option<Arc<AetherGraphCast<B>>>>,
    version_id: RwLockReadGuard<'a, Option<String>>,
    _permit: tokio::sync::SemaphorePermit<'a>,
}

impl<'a, B: Backend> ModelReadGuard<'a, B> {
    pub fn model(&self) -> Option<&AetherGraphCast<B>> {
        self.model.as_ref().map(|m| m.as_ref())
    }
    
    pub fn version_id(&self) -> Option<&str> {
        self.version_id.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArrayBackend;
    
    type TestBackend = NdArrayBackend<f32>;
    
    #[tokio::test]
    async fn test_model_manager() {
        let manager = ModelManager::<TestBackend>::new(ModelConfig::default());
        
        // Initially no model
        assert!(manager.get_active_model().is_none());
        
        // Would need to actually load a model in real test
    }
}