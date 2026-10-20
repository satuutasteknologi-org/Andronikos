//! # Redis Cache
//!
//! High-performance caching for API responses and predictions.

use std::time::Duration;

use redis::{aio::ConnectionManager, AsyncCommands, Client};
use serde::{Deserialize, Serialize};

use wi_core::{Result, WIError};

#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
    pub default_ttl_secs: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            max_connections: 10,
            default_ttl_secs: 3600,
        }
    }
}

pub struct RedisCache {
    manager: ConnectionManager,
    default_ttl: Duration,
}

impl RedisCache {
    pub async fn new(config: RedisConfig) -> Result<Self> {
        let client = Client::open(config.url)
            .map_err(|e| WIError::storage_with_source("Failed to create Redis client", e.into()))?;

        let manager = ConnectionManager::new(client).await.map_err(|e| {
            WIError::storage_with_source("Failed to create connection manager", e.into())
        })?;

        Ok(Self {
            manager,
            default_ttl: Duration::from_secs(config.default_ttl_secs),
        })
    }

    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut conn = self.manager.clone();
        let value: Option<Vec<u8>> = conn
            .get(key)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to get from Redis", e.into()))?;
        Ok(value)
    }

    pub async fn set(&self, key: &str, value: &[u8], ttl_secs: Option<u64>) -> Result<()> {
        let mut conn = self.manager.clone();
        let ttl = ttl_secs.unwrap_or(self.default_ttl.as_secs());
        conn.set_ex::<_, _, ()>(key, value, ttl)
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

    pub async fn get_json<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let data_opt: Option<Vec<u8>> = self.get(key).await?;
        if let Some(data) = data_opt {
            let value: T = serde_json::from_slice(&data).map_err(|e| {
                WIError::serialization_with_source("Failed to deserialize JSON", e.into())
            })?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    pub async fn set_json<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl_secs: Option<u64>,
    ) -> Result<()> {
        let data = serde_json::to_vec(value).map_err(|e| {
            WIError::serialization_with_source("Failed to serialize JSON", e.into())
        })?;
        self.set(key, &data, ttl_secs).await
    }

    pub async fn stats(&self) -> Result<CacheStats> {
        let mut conn = self.manager.clone();
        let info: String = redis::cmd("INFO")
            .arg("memory")
            .query_async(&mut conn)
            .await
            .map_err(|e| WIError::storage_with_source("Failed to get Redis info", e.into()))?;

        // Parse memory info
        let used_memory = info
            .lines()
            .find(|l| l.starts_with("used_memory:"))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);

        Ok(CacheStats {
            hits: 0,
            misses: 0,
            size_bytes: used_memory,
            entry_count: 0,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
    pub entry_count: u64,
}

// RedisCache tidak mengimplementasi Default karena memerlukan koneksi async.
// Selalu gunakan RedisCache::new(config).await
