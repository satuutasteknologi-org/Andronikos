//! # Data Sync Engine
//!
//! Event-driven, incremental, fault-tolerant data synchronization.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use parking_lot::RwLock;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use wi_core::WIError;
use wi_core::{Result, WeatherObservation};

use crate::adapters::create_default_registry;
use crate::checkpoint::CheckpointStore;
use crate::scheduler::{SchedulerConfig, SchedulerStats, SyncScheduler};
use crate::sources::{RegionBbox, SourceRegistry, WeatherSource};
use crate::validation::{GapFiller, WeatherValidator};
use wi_storage::StorageBackend;

/// Configuration for the sync engine
#[derive(Debug, Clone)]
pub struct SyncEngineConfig {
    pub parallel_workers: usize,
    pub batch_size: usize,
    pub timeout_secs: u64,
    pub retry_attempts: u8,
    pub retry_delay_ms: u64,
    pub checkpoint_interval: usize,
    pub enable_validation: bool,
    pub enable_gap_filling: bool,
}

impl Default for SyncEngineConfig {
    fn default() -> Self {
        Self {
            parallel_workers: 8,
            batch_size: 1000,
            timeout_secs: 300,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            checkpoint_interval: 100,
            enable_validation: true,
            enable_gap_filling: true,
        }
    }
}

/// Sync statistics
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub records_fetched: u64,
    pub records_validated: u64,
    pub records_written: u64,
    pub records_failed: u64,
    pub duration_secs: f64,
    pub records_per_second: f64,
    pub errors: Vec<String>,
}

/// Main sync engine orchestrator
pub struct SyncEngine<S: StorageBackend> {
    config: SyncEngineConfig,
    sources: Arc<SourceRegistry>,
    scheduler: Arc<SyncScheduler>,
    checkpoint_store: Arc<CheckpointStore>,
    validator: WeatherValidator,
    gap_filler: GapFiller,
    storage: Arc<S>,
    stats: RwLock<SyncStats>,
}

impl<S: StorageBackend> SyncEngine<S> {
    /// Create a new sync engine with default configuration and storage backend
    pub fn new(storage: Arc<S>) -> Self {
        Self::with_config(SyncEngineConfig::default(), storage)
    }

    /// Create with custom configuration and storage backend
    pub fn with_config(config: SyncEngineConfig, storage: Arc<S>) -> Self {
        let sources = Arc::new(create_default_registry());
        let scheduler_config = SchedulerConfig::default();
        let scheduler = Arc::new(SyncScheduler::new(scheduler_config, sources.clone()));
        let checkpoint_store = Arc::new(CheckpointStore::default());

        Self {
            config,
            sources,
            scheduler,
            checkpoint_store,
            validator: WeatherValidator::new(),
            gap_filler: GapFiller::new(),
            storage,
            stats: RwLock::new(SyncStats::default()),
        }
    }

    /// Sync data from all sources for a time range
    pub async fn sync_all(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<SyncStats> {
        let start_time = Instant::now();
        self.reset_stats();

        info!("Starting full sync from {} to {}", start, end);

        let regions = RegionBbox::global().subdivide(10.0);
        info!("Syncing {} regions", regions.len());

        let sources = self.sources.sorted_by_priority();

        for source in sources {
            for region in &regions {
                if timeout(
                    Duration::from_secs(self.config.timeout_secs),
                    self.sync_source_region(source, region, start, end),
                )
                .await
                .is_err()
                {
                    warn!("Sync timed out for {} region {:?}", source.name(), region);
                }
            }
        }

        let elapsed = start_time.elapsed();
        let mut stats = self.stats.write();
        stats.duration_secs = elapsed.as_secs_f64();
        stats.records_per_second = stats.records_written as f64 / elapsed.as_secs_f64().max(0.001);

        info!(
            "Sync completed: {} records in {:.2}s ({:.0} rec/s)",
            stats.records_written,
            elapsed.as_secs_f64(),
            stats.records_per_second
        );

        Ok(stats.clone())
    }

    /// Sync data from a specific source
    pub async fn sync_source(
        &self,
        source_name: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<SyncStats> {
        let start_time = Instant::now();
        self.reset_stats();

        let source = self
            .sources
            .all()
            .into_iter()
            .find(|s| s.name() == source_name)
            .ok_or_else(|| WIError::not_found("source", source_name))?;

        info!("Syncing {} from {} to {}", source_name, start, end);

        let regions = RegionBbox::global().subdivide(5.0);

        for region in regions {
            self.sync_source_region(source, &region, start, end).await?;
        }

        let elapsed = start_time.elapsed();
        let mut stats = self.stats.write();
        stats.duration_secs = elapsed.as_secs_f64();
        stats.records_per_second = stats.records_written as f64 / elapsed.as_secs_f64().max(0.001);

        Ok(stats.clone())
    }

    /// Sync a specific source and region
    async fn sync_source_region(
        &self,
        source: &dyn WeatherSource,
        region: &RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<()> {
        let checkpoint_key = format!("{}:{:?}", source.name(), region);

        let mut resume_from = start;
        if let Some(cp) = self
            .checkpoint_store
            .get(source.name(), &checkpoint_key)
            .await
        {
            if cp.is_valid() {
                resume_from = cp.last_sync_time;
                debug!("Resuming {} from {}", source.name(), resume_from);
            }
        }

        let stream = source.fetch_observations(*region, resume_from, end).await?;

        let mut batch: Vec<WeatherObservation> = Vec::with_capacity(self.config.batch_size);
        let mut stream = stream.enumerate();

        while let Some((idx, result)) = stream.next().await {
            match result {
                Ok(obs) => {
                    let validated = if self.config.enable_validation {
                        self.validator.validate(obs)
                    } else {
                        crate::validation::ValidationResult {
                            observation: obs,
                            modified: false,
                            issues: vec![],
                        }
                    };

                    self.stats.write().records_fetched += 1;

                    if validated.observation.quality_flag != wi_core::types::QualityFlag::Erroneous
                    {
                        self.stats.write().records_validated += 1;
                        batch.push(validated.observation);
                    } else {
                        self.stats.write().records_failed += 1;
                    }

                    if batch.len() >= self.config.batch_size {
                        self.write_batch(&batch).await?;
                        self.stats.write().records_written += batch.len() as u64;
                        batch.clear();

                        if idx % self.config.checkpoint_interval == 0 {
                            let checkpoint = crate::checkpoint::SyncCheckpoint::new(
                                source.name(),
                                &checkpoint_key,
                            )
                            .with_record(&format!("record_{}", idx), idx as u64);
                            self.checkpoint_store.update(checkpoint).await?;
                        }
                    }
                }
                Err(e) => {
                    self.stats.write().records_failed += 1;
                    self.stats.write().errors.push(e.to_string());
                    warn!("Failed to fetch observation {}: {}", idx, e);
                }
            }
        }

        if !batch.is_empty() {
            self.write_batch(&batch).await?;
            self.stats.write().records_written += batch.len() as u64;
        }

        Ok(())
    }

    /// Write batch to storage backend
    async fn write_batch(&self, batch: &[WeatherObservation]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let written = self.storage.write_observations(batch.to_vec()).await?;
        debug!("Wrote {} observations to storage", written);
        Ok(())
    }

    fn reset_stats(&self) {
        *self.stats.write() = SyncStats::default();
    }

    /// Get current statistics
    pub fn stats(&self) -> SyncStats {
        self.stats.read().clone()
    }

    /// Get scheduler stats
    pub fn scheduler_stats(&self) -> SchedulerStats {
        self.scheduler.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use wi_storage::ParquetConfig;
    use wi_storage::ParquetStorage;

    #[tokio::test]
    async fn test_engine_creation() {
        let config = ParquetConfig {
            base_path: PathBuf::from("/tmp/test_parquet"),
            ..Default::default()
        };
        let storage = Arc::new(ParquetStorage::new(config));
        let engine = SyncEngine::new(storage);
        let stats = engine.stats();
        assert_eq!(stats.records_fetched, 0);
    }
}
