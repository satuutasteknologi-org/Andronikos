//! # API Cache Layer
//!
//! Redis-based caching for API responses.

use std::time::Duration;

use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};

use wi_core::Result;
use wi_core::WIError;

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub redis_url: String,
    pub default_ttl_secs: u64,
    pub max_connections: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            default_ttl_secs: 3600,
            max_connections: 10,
        }
    }
}

pub struct ApiCache {
    manager: ConnectionManager,
    default_ttl: Duration,
}

impl ApiCache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let client = Client::open(config.redis_url)
            .map_err(|e| WIError::storage_with_source("Failed to create Redis client", e.into()))?;

        let manager = ConnectionManager::new(client).await.map_err(|e| {
            WIError::storage_with_source("Failed to create connection manager", e.into())
        })?;

        Ok(Self {
            manager,
            default_ttl: Duration::from_secs(config.default_ttl_secs),
        })
    }

    pub async fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let mut conn = self.manager.clone();
        let value: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to get from Redis", e.into()))?;

        let json_string = match value {
            Some(v) => v,
            None => return Ok(None),
        };

        let parsed: T = serde_json::from_str(&json_string).map_err(|e| {
            WIError::serialization_with_source("Failed to deserialize cache", e.into())
        })?;
        Ok(Some(parsed))
    }

    pub async fn set<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl_secs: Option<u64>,
    ) -> Result<()> {
        let mut conn = self.manager.clone();
        let json = serde_json::to_string(value).map_err(|e| {
            WIError::serialization_with_source("Failed to serialize cache", e.into())
        })?;

        let ttl = ttl_secs.unwrap_or(self.default_ttl.as_secs());
        conn.set_ex::<_, _, ()>(key, json, ttl)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to set in Redis", e.into()))?;
        Ok(())
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let mut conn = self.manager.clone();
        conn.del::<_, ()>(key)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to delete from Redis", e.into()))?;
        Ok(())
    }

    pub async fn clear_pattern(&self, pattern: &str) -> Result<()> {
        let mut conn = self.manager.clone();
        let keys: Vec<String> = conn
            .keys(pattern)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to scan keys", e.into()))?;

        if !keys.is_empty() {
            conn.del::<_, ()>(&keys)
                .await
                .map_err(|e| WIError::storage_with_source("Failed to delete keys", e.into()))?;
        }
        Ok(())
    }

    // Cache key generators
    pub fn point_cache_key(lat: f64, lon: f64, lead_seconds: u32) -> String {
        format!("predict:point:{:.4}:{:.4}:{}", lat, lon, lead_seconds)
    }

    pub fn grid_cache_key(
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
        resolution: u8,
        lead_seconds: u32,
    ) -> String {
        format!(
            "predict:grid:{:.4}:{:.4}:{:.4}:{:.4}:{}:{}",
            min_lat, min_lon, max_lat, max_lon, resolution, lead_seconds
        )
    }
}
