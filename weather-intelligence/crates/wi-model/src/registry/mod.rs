//! # Model Registry
//!
//! Version management, hot-reload, and model selection.

use std::collections::HashMap;

use chrono::DateTime;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::traits::{DynDownscaler, DynForecaster, ModelInfo};
use wi_core::{Result, WIError};

/// Model version info
#[derive(Debug, Clone)]
pub struct ModelVersion {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub loaded_at: DateTime<chrono::Utc>,
    pub info: ModelInfo,
    pub is_active: bool,
    pub inference_count: u64,
    pub avg_latency_ms: f32,
}

/// Model registry with hot-reload support
pub struct ModelRegistry {
    forecasters: RwLock<HashMap<String, DynForecaster>>,
    downscalers: RwLock<HashMap<String, DynDownscaler>>,
    versions: RwLock<Vec<ModelVersion>>,
    active_forecaster: RwLock<Option<String>>,
    active_downscaler: RwLock<Option<String>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            forecasters: RwLock::new(HashMap::new()),
            downscalers: RwLock::new(HashMap::new()),
            versions: RwLock::new(Vec::new()),
            active_forecaster: RwLock::new(None),
            active_downscaler: RwLock::new(None),
        }
    }

    pub fn register_forecaster(&self, name: String, model: DynForecaster) -> Result<()> {
        let info = model.0.model_info();

        let mut forecasters = self.forecasters.write();
        forecasters.insert(name.clone(), model);
        drop(forecasters);

        let mut versions = self.versions.write();
        versions.push(ModelVersion {
            id: Uuid::new_v4(),
            name: name.clone(),
            version: info.version.clone(),
            loaded_at: chrono::Utc::now(),
            info,
            is_active: false,
            inference_count: 0,
            avg_latency_ms: 0.0,
        });

        Ok(())
    }

    pub fn register_downscaler(&self, name: String, model: DynDownscaler) -> Result<()> {
        let mut downscalers = self.downscalers.write();
        downscalers.insert(name.clone(), model);

        Ok(())
    }

    pub fn set_active_forecaster(&self, name: &str) -> Result<()> {
        let forecasters = self.forecasters.read();
        if !forecasters.contains_key(name) {
            return Err(WIError::model(format!("Forecaster '{}' not found", name)));
        }
        drop(forecasters);

        let mut active = self.active_forecaster.write();
        *active = Some(name.to_string());

        let mut versions = self.versions.write();
        for v in versions.iter_mut() {
            v.is_active = v.name == name;
        }

        Ok(())
    }

    pub fn get_forecaster(&self, name: &str) -> Option<DynForecaster> {
        self.forecasters.read().get(name).cloned()
    }

    pub fn get_active_forecaster(&self) -> Option<DynForecaster> {
        let name = self.active_forecaster.read().clone()?;
        self.get_forecaster(&name)
    }

    pub fn get_downscaler(&self, name: &str) -> Option<DynDownscaler> {
        self.downscalers.read().get(name).cloned()
    }

    pub fn get_active_downscaler(&self) -> Option<DynDownscaler> {
        let name = self.active_downscaler.read().clone()?;
        self.get_downscaler(&name)
    }

    pub fn active_forecaster_name(&self) -> Option<String> {
        self.active_forecaster.read().clone()
    }

    pub fn active_downscaler_name(&self) -> Option<String> {
        self.active_downscaler.read().clone()
    }

    pub fn list_versions(&self) -> Vec<ModelVersion> {
        self.versions.read().clone()
    }

    pub fn record_inference(&self, name: &str, latency_ms: f32) {
        let mut versions = self.versions.write();
        for v in versions.iter_mut() {
            if v.name == name {
                v.inference_count += 1;
                v.avg_latency_ms = (v.avg_latency_ms * (v.inference_count - 1) as f32 + latency_ms)
                    / v.inference_count as f32;
            }
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
