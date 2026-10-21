//! # Sync Scheduler
//! 
//! Cron-like scheduler for periodic data synchronization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, info};

use crate::sources::{RegionBbox, SourceRegistry};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub sync_interval_hours: u64,
    pub incremental: bool,
    pub parallel_regions: usize,
    pub backfill_days: i64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            sync_interval_hours: 6,
            incremental: true,
            parallel_regions: 8,
            backfill_days: 7,
        }
    }
}

/// Sync job definition
#[derive(Debug, Clone)]
pub struct SyncJob {
    pub id: String,
    pub sources: Vec<String>,
    pub region: RegionBbox,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub priority: u8,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job result
#[derive(Debug)]
pub struct JobResult {
    pub job_id: String,
    pub status: JobStatus,
    pub records_synced: usize,
    pub errors: Vec<String>,
    pub duration_secs: u64,
    pub completed_at: DateTime<Utc>,
}

/// Scheduler for managing sync jobs
pub struct SyncScheduler {
    config: SchedulerConfig,
    sources: Arc<SourceRegistry>,
    pending_jobs: RwLock<Vec<SyncJob>>,
    running_jobs: RwLock<HashMap<String, SyncJob>>,
    completed_jobs: RwLock<Vec<JobResult>>,
    job_tx: mpsc::Sender<SyncJob>,
    job_rx: RwLock<Option<mpsc::Receiver<SyncJob>>>,
}

impl SyncScheduler {
    pub fn new(config: SchedulerConfig, sources: Arc<SourceRegistry>) -> Self {
        let (job_tx, job_rx) = mpsc::channel(100);
        
        Self {
            config,
            sources,
            pending_jobs: RwLock::new(Vec::new()),
            running_jobs: RwLock::new(HashMap::new()),
            completed_jobs: RwLock::new(Vec::new()),
            job_tx,
            job_rx: RwLock::new(Some(job_rx)),
        }
    }
    
    /// Schedule a new sync job
    pub fn schedule(&self, job: SyncJob) {
        info!("Scheduling sync job: {} for region {:?}", job.id, job.region);
        self.pending_jobs.write().push(job);
    }
    
    /// Schedule automatic sync for all sources
    pub fn schedule_automatic(&self) {
        let sources = self.sources.sorted_by_priority();
        
        for source in sources {
            let region = RegionBbox::global();
            let end = Utc::now();
            let start = end - ChronoDuration::days(self.config.backfill_days);
            
            let job = SyncJob {
                id: format!("auto_{}_{}", source.name(), Utc::now().timestamp()),
                sources: vec![source.name().to_string()],
                region,
                start_date: start,
                end_date: end,
                priority: source.priority(),
            };
            
            self.schedule(job);
        }
    }
    
    /// Get next pending job
    pub fn next_job(&self) -> Option<SyncJob> {
        let mut pending = self.pending_jobs.write();
        if pending.is_empty() {
            return None;
        }
        
        pending.sort_by_key(|j| j.priority);
        Some(pending.remove(0))
    }
    
    /// Update job status
    pub fn mark_running(&self, job: &SyncJob) {
        let mut running = self.running_jobs.write();
        running.insert(job.id.clone(), job.clone());
    }
    
    pub fn mark_completed(&self, result: JobResult) {
        let mut running = self.running_jobs.write();
        running.remove(&result.job_id);
        
        let mut completed = self.completed_jobs.write();
        completed.push(result);
        
        if completed.len() > 1000 {
            completed.drain(0..500);
        }
    }
    
    /// Get scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            pending: self.pending_jobs.read().len(),
            running: self.running_jobs.read().len(),
            completed_total: self.completed_jobs.read().len(),
            last_sync: self.completed_jobs.read().last()
                .map(|r| r.completed_at),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub pending: usize,
    pub running: usize,
    pub completed_total: usize,
    pub last_sync: Option<DateTime<Utc>>,
}

/// Start the scheduler background loop
pub async fn run_scheduler(scheduler: Arc<SyncScheduler>) {
    info!("Starting sync scheduler");
    
    let mut interval = interval(Duration::from_secs(3600));
    
    loop {
        interval.tick().await;
        
        let stats = scheduler.stats();
        debug!(
            "Scheduler stats: pending={}, running={}, completed={}",
            stats.pending, stats.running, stats.completed_total
        );
        
        if stats.pending == 0 {
            scheduler.schedule_automatic();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scheduler_stats() {
        let sources = crate::adapters::create_default_registry();
        let scheduler = SyncScheduler::new(SchedulerConfig::default(), Arc::new(sources));
        
        let stats = scheduler.stats();
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.running, 0);
    }
}