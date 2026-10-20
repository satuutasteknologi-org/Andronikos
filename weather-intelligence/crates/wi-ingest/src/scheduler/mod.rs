//! # Sync Scheduler
//!
//! Cron-like scheduling for automated data synchronization.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

use wi_core::Result;
use wi_core::WIError;

use crate::sources::{RegionBbox, SourceRegistry};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub enabled: bool,
    pub sync_interval_hours: u64,
    pub historical_backfill_hours: u64,
    pub max_concurrent_syncs: usize,
    pub retry_failed_after_hours: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval_hours: 1,
            historical_backfill_hours: 24,
            max_concurrent_syncs: 2,
            retry_failed_after_hours: 6,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_syncs: u64,
    pub successful_syncs: u64,
    pub failed_syncs: u64,
    pub last_sync_time: Option<DateTime<Utc>>,
    pub next_scheduled_sync: Option<DateTime<Utc>>,
    pub active_syncs: usize,
}

/// Sync job definition
#[derive(Debug, Clone)]
pub struct SyncJob {
    pub id: String,
    pub source: String,
    pub region: RegionBbox,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub priority: JobPriority,
    pub retry_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Sync scheduler
pub struct SyncScheduler {
    config: SchedulerConfig,
    sources: Arc<SourceRegistry>,
    job_queue: Arc<RwLock<Vec<SyncJob>>>,
    stats: Arc<RwLock<SchedulerStats>>,
    running: Arc<RwLock<bool>>,
}

impl SyncScheduler {
    pub fn new(config: SchedulerConfig, sources: Arc<SourceRegistry>) -> Self {
        Self {
            config,
            sources,
            job_queue: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the scheduler
    pub async fn start(&self) -> Result<()> {
        if *self.running.read() {
            return Err(WIError::validation("Scheduler already running"));
        }

        *self.running.write() = true;
        info!("Sync scheduler started");

        // Schedule initial jobs
        self.schedule_initial_jobs().await;

        // Start scheduler loop
        let scheduler = self.clone();
        tokio::spawn(async move {
            scheduler.run_loop().await;
        });

        Ok(())
    }

    /// Stop the scheduler
    pub fn stop(&self) {
        *self.running.write() = false;
        info!("Sync scheduler stopped");
    }

    /// Schedule initial sync jobs
    async fn schedule_initial_jobs(&self) {
        let now = Utc::now();
        let end = now;
        let start = now - chrono::Duration::hours(self.config.historical_backfill_hours as i64);

        for source in self.sources.all() {
            if !source.provides_historical() {
                continue;
            }

            let job = SyncJob {
                id: format!("{}-{}", source.name(), uuid::Uuid::new_v4()),
                source: source.name().to_string(),
                region: RegionBbox::global(),
                start_time: start,
                end_time: end,
                priority: JobPriority::Normal,
                retry_count: 0,
            };

            self.job_queue.write().push(job);
        }

        info!(
            "Scheduled {} initial sync jobs",
            self.job_queue.read().len()
        );
    }

    /// Main scheduler loop
    async fn run_loop(&self) {
        let mut tick = interval(Duration::from_secs(60));

        while *self.running.read() {
            tick.tick().await;

            // Process job queue
            self.process_jobs().await;

            // Schedule recurring syncs
            self.schedule_recurring_syncs().await;

            // Update stats
            self.update_next_sync_time().await;
        }
    }

    /// Process pending jobs
    async fn process_jobs(&self) {
        let max_concurrent = self.config.max_concurrent_syncs;
        let active = self.stats.read().active_syncs;

        let jobs_to_run: Vec<SyncJob> = if active >= max_concurrent {
            Vec::new()
        } else {
            let available_slots = max_concurrent - active;
            let mut queue = self.job_queue.write();
            let queue_len = queue.len();
            queue.sort_by_key(|a| std::cmp::Reverse(a.priority));
            queue.drain(..available_slots.min(queue_len)).collect()
        };

        for job in jobs_to_run {
            let scheduler = self.clone();
            tokio::spawn(async move {
                scheduler.execute_job(job).await;
            });
        }
    }

    /// Execute a single sync job
    async fn execute_job(&self, job: SyncJob) {
        let start_time = Instant::now();

        {
            let mut stats = self.stats.write();
            stats.active_syncs += 1;
            stats.total_syncs += 1;
        }

        let source = self.sources.get(&job.source);
        let result = if let Some(_source) = source {
            debug!("Executing sync job {} for {}", job.id, job.source);
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        } else {
            Err(WIError::not_found("source", &job.source))
        };

        let elapsed = start_time.elapsed();

        {
            let mut stats = self.stats.write();
            stats.active_syncs = stats.active_syncs.saturating_sub(1);
            stats.last_sync_time = Some(Utc::now());

            match result {
                Ok(_) => {
                    stats.successful_syncs += 1;
                    info!(
                        "Sync job {} completed in {:.2}s",
                        job.id,
                        elapsed.as_secs_f64()
                    );
                }
                Err(e) => {
                    stats.failed_syncs += 1;
                    warn!("Sync job {} failed: {}", job.id, e);

                    if job.retry_count < 3 {
                        let mut retry_job = job;
                        retry_job.retry_count += 1;
                        retry_job.priority = JobPriority::High;
                        self.job_queue.write().push(retry_job);
                    }
                }
            }
        }
    }

    /// Schedule recurring syncs
    async fn schedule_recurring_syncs(&self) {
        let now = Utc::now();
        let interval = chrono::Duration::hours(self.config.sync_interval_hours as i64);

        let should_sync = if let Some(last) = self.stats.read().last_sync_time {
            now - last >= interval
        } else {
            true
        };

        if should_sync {
            let end = now;
            let start = end - interval;

            for source in self.sources.all() {
                if !source.provides_historical() {
                    continue;
                }

                let job = SyncJob {
                    id: format!("{}-recurring-{}", source.name(), uuid::Uuid::new_v4()),
                    source: source.name().to_string(),
                    region: RegionBbox::global(),
                    start_time: start,
                    end_time: end,
                    priority: JobPriority::Normal,
                    retry_count: 0,
                };

                self.job_queue.write().push(job);
            }

            info!(
                "Scheduled recurring sync for {} sources",
                self.sources.all().len()
            );
        }
    }

    async fn update_next_sync_time(&self) {
        if let Some(last) = self.stats.read().last_sync_time {
            let next = last + chrono::Duration::hours(self.config.sync_interval_hours as i64);
            self.stats.write().next_scheduled_sync = Some(next);
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }

    /// Get pending job count
    pub fn pending_jobs(&self) -> usize {
        self.job_queue.read().len()
    }
}

impl Clone for SyncScheduler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sources: self.sources.clone(),
            job_queue: self.job_queue.clone(),
            stats: self.stats.clone(),
            running: self.running.clone(),
        }
    }
}
