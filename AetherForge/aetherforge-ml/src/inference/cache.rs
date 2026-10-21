//! # Prediction Cache - Intelligent Caching for ML Predictions
//!
//! Implements a sophisticated caching layer with:
//! - LRU eviction policy
//! - Time-based expiration
//! - Spatial proximity caching
//! - TTL with staleness tolerance

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::debug;

use crate::inference::PredictionResult;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in cache
    pub max_entries: usize,
    /// Default TTL in seconds
    pub default_ttl_secs: u64,
    /// Enable spatial proximity caching
    pub spatial_caching: bool,
    /// Spatial proximity radius in km
    pub proximity_radius_km: f64,
    /// Staleness tolerance (percentage of TTL)
    pub staleness_tolerance: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000_000, // 1M entries for ~1km global grid
            default_ttl_secs: 300,  // 5 minutes
            spatial_caching: true,
            proximity_radius_km: 1.0,
            staleness_tolerance: 0.2, // 20% staleness tolerated
        }
    }
}

/// Cache entry with metadata
struct CacheEntry {
    result: PredictionResult,
    created_at: Instant,
    ttl: Duration,
    access_count: u32,
    last_access: Instant,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    fn is_stale(&self, tolerance: f32) -> bool {
        let elapsed = self.created_at.elapsed().as_secs_f32();
        let ttl = self.ttl.as_secs_f32();
        elapsed > ttl * (1.0 - tolerance)
    }

    fn staleness_factor(&self, tolerance: f32) -> f32 {
        let elapsed = self.created_at.elapsed().as_secs_f32();
        let ttl = self.ttl.as_secs_f32();
        (elapsed / ttl).min(1.0 + tolerance)
    }
}

/// Thread-safe prediction cache
pub struct PredictionCache {
    config: CacheConfig,

    // Main cache storage
    entries: RwLock<HashMap<u64, CacheEntry>>,

    // LRU order tracking
    lru_order: RwLock<VecDeque<u64>>,

    // Spatial index for proximity caching
    spatial_index: RwLock<HashMap<u64, Vec<u64>>>,

    // Statistics
    hits: RwLock<u64>,
    misses: RwLock<u64>,
    evictions: RwLock<u64>,
}

impl PredictionCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            lru_order: RwLock::new(VecDeque::new()),
            spatial_index: RwLock::new(HashMap::new()),
            hits: RwLock::new(0),
            misses: RwLock::new(0),
            evictions: RwLock::new(0),
        }
    }

    /// Get cached prediction for cell
    pub fn get(&self, cell_id: u64) -> Option<PredictionResult> {
        let entries = self.entries.read();

        if let Some(entry) = entries.get(&cell_id) {
            if !entry.is_expired() {
                // Update access metadata
                drop(entries);
                self.update_access(cell_id);

                *self.hits.write() += 1;
                debug!("Cache hit for cell {}", cell_id);
                return Some(self.entries.read().get(&cell_id)?.result.clone());
            }
        }

        *self.misses.write() += 1;
        debug!("Cache miss for cell {}", cell_id);
        None
    }

    /// Get with spatial proximity fallback
    pub fn get_with_proximity(&self, cell_id: u64, lat: f64, lon: f64) -> Option<PredictionResult> {
        // Try exact match first
        if let Some(result) = self.get(cell_id) {
            return Some(result);
        }

        if !self.config.spatial_caching {
            return None;
        }

        // Try nearby cells
        let nearby_ids = self.get_nearby_cells(cell_id, lat, lon);

        for nearby_id in nearby_ids {
            if let Some(entry) = self.get(nearby_id) {
                debug!("Proximity cache hit: {} -> {}", cell_id, nearby_id);
                return Some(entry);
            }
        }

        None
    }

    /// Store prediction in cache
    pub fn put(&self, cell_id: u64, result: PredictionResult) {
        let ttl = Duration::from_secs(self.config.default_ttl_secs);

        let entry = CacheEntry {
            result,
            created_at: Instant::now(),
            ttl,
            access_count: 1,
            last_access: Instant::now(),
        };

        // Check if we need to evict
        {
            let entries = self.entries.read();
            if entries.len() >= self.config.max_entries {
                drop(entries);
                self.evict_lru();
            }
        }

        // Insert or update
        {
            let mut entries = self.entries.write();
            entries.insert(cell_id, entry);
        }

        // Update LRU
        {
            let mut order = self.lru_order.write();
            order.retain(|&id| id != cell_id);
            order.push_back(cell_id);
        }

        debug!("Cached prediction for cell {}", cell_id);
    }

    /// Store with custom TTL
    pub fn put_with_ttl(&self, cell_id: u64, result: PredictionResult, ttl_secs: u64) {
        let entry = CacheEntry {
            result,
            created_at: Instant::now(),
            ttl: Duration::from_secs(ttl_secs),
            access_count: 1,
            last_access: Instant::now(),
        };

        let mut entries = self.entries.write();
        entries.insert(cell_id, entry);

        let mut order = self.lru_order.write();
        order.retain(|&id| id != cell_id);
        order.push_back(cell_id);
    }

    /// Remove entry from cache
    pub fn invalidate(&self, cell_id: u64) {
        let mut entries = self.entries.write();
        entries.remove(&cell_id);

        let mut order = self.lru_order.write();
        order.retain(|&id| id != cell_id);

        let mut spatial = self.spatial_index.write();
        spatial.remove(&cell_id);
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        entries.clear();

        let mut order = self.lru_order.write();
        order.clear();

        let mut spatial = self.spatial_index.write();
        spatial.clear();

        *self.hits.write() = 0;
        *self.misses.write() = 0;
        *self.evictions.write() = 0;

        debug!("Cache cleared");
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let evictions = *self.evictions.read();

        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            entries: entries.len(),
            max_entries: self.config.max_entries,
            hit_rate,
            hits,
            misses,
            evictions,
        }
    }

    /// Clean up expired entries
    pub fn cleanup(&self) -> usize {
        let mut entries = self.entries.write();
        let mut order = self.lru_order.write();

        let before = entries.len();

        entries.retain(|_, entry| !entry.is_expired());
        order.retain(|id| entries.contains_key(id));

        let evicted = before - entries.len();

        if evicted > 0 {
            debug!("Cleaned up {} expired entries", evicted);
        }

        evicted
    }

    fn update_access(&self, cell_id: u64) {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&cell_id) {
            entry.access_count += 1;
            entry.last_access = Instant::now();
        }

        let mut order = self.lru_order.write();
        order.retain(|&id| id != cell_id);
        order.push_back(cell_id);
    }

    fn evict_lru(&self) {
        let mut entries = self.entries.write();
        let mut order = self.lru_order.write();

        while entries.len() >= self.config.max_entries {
            if let Some(lru_id) = order.pop_front() {
                entries.remove(&lru_id);
                *self.evictions.write() += 1;
            } else {
                break;
            }
        }
    }

    fn get_nearby_cells(&self, cell_id: u64, lat: f64, lon: f64) -> Vec<u64> {
        let spatial = self.spatial_index.read();

        if let Some(neighbors) = spatial.get(&cell_id) {
            return neighbors.clone();
        }

        // Compute neighbors on-the-fly (simplified)
        let lat_delta = self.config.proximity_radius_km / 110.0;
        let lon_delta = self.config.proximity_radius_km / (111.0 * lat.to_radians().cos());

        let mut neighbors = Vec::new();

        // Generate neighboring cell IDs
        for dlat in [-1.0_f64, 0.0, 1.0] {
            for dlon in [-1.0_f64, 0.0, 1.0] {
                if dlat == 0.0 && dlon == 0.0 {
                    continue;
                }

                let nlat = lat + dlat * lat_delta;
                let nlon = lon + dlon * lon_delta;

                let nlat_int = ((nlat + 90.0) * 10000.0) as u64;
                let nlon_int = ((nlon + 180.0) * 10000.0) as u64;
                let nid = (nlat_int << 32) | nlon_int;

                if nid != cell_id {
                    neighbors.push(nid);
                }
            }
        }

        neighbors
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub max_entries: usize,
    pub hit_rate: f64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl Default for PredictionCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = PredictionCache::new(CacheConfig::default());

        let result = PredictionResult::new(
            12345,
            std::collections::HashMap::from([("Temperature".to_string(), 280.0)]),
        );

        cache.put(12345, result.clone());

        let cached = cache.get(12345);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().predictions.get("Temperature"), Some(&280.0));
    }

    #[test]
    fn test_cache_miss() {
        let cache = PredictionCache::new(CacheConfig::default());

        let cached = cache.get(99999);
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = PredictionCache::new(CacheConfig::default());

        cache.put(
            1,
            PredictionResult::new(1, std::collections::HashMap::new()),
        );
        cache.get(1);
        cache.get(2);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.hits >= 1);
        assert!(stats.misses >= 1);
    }
}
