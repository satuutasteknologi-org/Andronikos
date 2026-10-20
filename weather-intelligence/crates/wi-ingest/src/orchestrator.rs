//! # Ingest Orchestrator — Data Fusion Layer (Trip 4)
//!
//! Menyatukan BMKG, GFS, NOAA, Open-Meteo menjadi satu pipeline yang:
//! 1. **Tidak mencampur ERA5** ke dalam real-time fusion (training-only safeguard).
//! 2. **Label `informasi risiko`** dicatat pada semua output prediktif sesuai
//!    UU 31/2009 Pasal 36 ayat 1 dan PP 11/2016 Pasal 4.
//! 3. **Deduplication** antar sumber (tidak double-count Open-Meteo + GFS).
//! 4. **Fault isolation** — gagal satu sumber tidak menghentikan pipeline.
//!
//! ## Arsitektur
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │              IngestOrchestrator                │
//! │  ┌─────────┐ ┌─────────┐ ┌────────────────┐  │
//! │  │  BMKG   │ │   GFS   │ │   Open-Meteo   │  │
//! │  │ adapter │ │ adapter │ │    adapter     │  │
//! │  └────┬────┘ └────┬────┘ └───────┬────────┘  │
//! │       └───────────┴───────────────┘            │
//! │                   │                            │
//! │        ┌──────────▼──────────┐                │
//! │        │   FusionMerger      │ (dedup+median) │
//! │        └──────────┬──────────┘                │
//! │                   │                            │
//! │        ┌──────────▼──────────┐                │
//! │        │ ComplianceLog       │ (label_istilah)│
//! │        └──────────┬──────────┘                │
//! │                   │                            │
//! │        ┌──────────▼──────────┐                │
//! │        │   StorageBackend    │                │
//! │        └─────────────────────┘                │
//! └────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, WeatherObservation, WeatherVariable,
};
use wi_storage::StorageBackend;

use crate::adapters::create_registry_with_keys;
use crate::sources::RegionBbox;

// ─────────────────────────────────────────────────────────────────────────────
// Konfigurasi
// ─────────────────────────────────────────────────────────────────────────────

/// Konfigurasi IngestOrchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Token NOAA API (opsional — tanpa token: 1 req/s)
    pub noaa_token: Option<String>,
    /// API key BMKG (opsional)
    pub bmkg_key: Option<String>,
    /// API key Open-Meteo Pro (opsional)
    pub open_meteo_key: Option<String>,
    /// API key WeatherNext (opsional)
    pub weather_next_key: Option<String>,
    /// Maksimum request paralel ke semua sumber
    pub max_concurrent: usize,
    /// Timeout per sumber (detik)
    pub source_timeout_secs: u64,
    /// Batch size untuk menulis ke storage
    pub write_batch_size: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            noaa_token: None,
            bmkg_key: None,
            open_meteo_key: None,
            weather_next_key: None,
            max_concurrent: 4,
            source_timeout_secs: 120,
            write_batch_size: 500,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistik
// ─────────────────────────────────────────────────────────────────────────────

/// Statistik satu run ingest
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    pub records_fetched: u64,
    pub records_merged: u64,
    pub records_written: u64,
    pub records_skipped: u64, // ERA5 atau invalid
    pub source_errors: HashMap<String, u64>,
    pub duration_ms: u64,
}

impl std::fmt::Display for IngestStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fetched={} merged={} written={} skipped={} duration={}ms",
            self.records_fetched,
            self.records_merged,
            self.records_written,
            self.records_skipped,
            self.duration_ms,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compliance
// ─────────────────────────────────────────────────────────────────────────────

/// Label wajib pada semua output prediktif (UU 31/2009 Pasal 36 ayat 1)
pub const LABEL_INFORMASI_RISIKO: &str = "informasi risiko";

/// Sumber yang DILARANG masuk ke real-time fusion (training-only)
const BLOCKED_REALTIME: &[DataSource] = &[DataSource::ERA5];

#[inline]
fn is_realtime_eligible(source: DataSource) -> bool {
    !BLOCKED_REALTIME.contains(&source)
}

// ─────────────────────────────────────────────────────────────────────────────
// FusionMerger — deduplication + median ensemble
// ─────────────────────────────────────────────────────────────────────────────

/// Kunci deduplication: (lat_i×1000, lon_i×1000, variable, hour_bucket)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FusionKey {
    lat_i: i32,
    lon_i: i32,
    variable: WeatherVariable,
    hour: i64,
}

impl FusionKey {
    fn from_obs(obs: &WeatherObservation, var: WeatherVariable) -> Self {
        Self {
            lat_i: (obs.coord.lat * 1000.0).round() as i32,
            lon_i: (obs.coord.lon * 1000.0).round() as i32,
            variable: var,
            hour: obs.time.timestamp() / 3600,
        }
    }
}

/// Merger: kumpulkan semua nilai per key → median
struct FusionMerger {
    buckets: HashMap<FusionKey, Vec<(f64, DataSource, bool)>>,
    first_time: HashMap<FusionKey, DateTime<Utc>>,
}

impl FusionMerger {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            first_time: HashMap::new(),
        }
    }

    fn ingest(&mut self, obs: &WeatherObservation, stats: &mut IngestStats) {
        // HARD BLOCK ERA5
        if !is_realtime_eligible(obs.source) {
            stats.records_skipped += obs.variables.len() as u64;
            return;
        }
        // Overall quality check
        if obs.quality_flag == QualityFlag::Erroneous {
            stats.records_skipped += obs.variables.len() as u64;
            return;
        }

        for (var, observed) in &obs.variables {
            if !observed.is_valid || observed.value.is_nan() {
                stats.records_skipped += 1;
                continue;
            }
            let key = FusionKey::from_obs(obs, *var);
            self.buckets.entry(key.clone()).or_default().push((
                observed.value,
                obs.source,
                observed.is_valid,
            ));
            self.first_time.entry(key).or_insert(obs.time);
            stats.records_fetched += 1;
        }
    }

    fn finalize(self, stats: &mut IngestStats) -> Vec<WeatherObservation> {
        let mut result = Vec::with_capacity(self.buckets.len());

        for (key, mut values) in self.buckets {
            // Median
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let median_val = if values.len() % 2 == 1 {
                values[values.len() / 2].0
            } else {
                let mid = values.len() / 2;
                (values[mid - 1].0 + values[mid].0) / 2.0
            };

            let primary_source = prioritize_source(values.iter().map(|(_, s, _)| *s));
            let time = self.first_time.get(&key).copied().unwrap_or_else(Utc::now);

            let mut variables: HashMap<WeatherVariable, ObservedValue> = HashMap::new();
            variables.insert(key.variable, ObservedValue::new(median_val, 0.9));

            result.push(WeatherObservation {
                station_id: None,
                coord: GeoCoord::new(key.lat_i as f64 / 1000.0, key.lon_i as f64 / 1000.0),
                time,
                variables,
                quality_flag: QualityFlag::Valid,
                source: primary_source,
            });

            stats.records_merged += 1;
        }

        result
    }
}

/// Prioritas sumber (reliability_score dari wi-core, lebih tinggi = lebih baik)
fn prioritize_source(sources: impl Iterator<Item = DataSource>) -> DataSource {
    sources
        .max_by(|a, b| {
            a.reliability_score()
                .partial_cmp(&b.reliability_score())
                .unwrap()
        })
        .unwrap_or(DataSource::OpenMeteo)
}

// ─────────────────────────────────────────────────────────────────────────────
// IngestOrchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Pusat orkestrasi ingest data — entry-point tunggal untuk semua sumber
pub struct IngestOrchestrator<S: StorageBackend> {
    config: OrchestratorConfig,
    storage: Arc<S>,
    semaphore: Arc<Semaphore>,
}

impl<S: StorageBackend + 'static> IngestOrchestrator<S> {
    /// Buat orchestrator baru
    pub fn new(config: OrchestratorConfig, storage: Arc<S>) -> Self {
        let sem = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            semaphore: sem,
            config,
            storage,
        }
    }

    /// Jalankan satu siklus ingest untuk region dan rentang waktu
    ///
    /// # Jaminan
    /// - ERA5 tidak akan pernah masuk ke storage real-time
    /// - Output adalah median ensemble dari semua sumber yang tersedia
    /// - Sumber yang gagal di-log dan di-skip tanpa menghentikan pipeline
    pub async fn run_ingest(
        &self,
        region: RegionBbox,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<IngestStats> {
        let t0 = Instant::now();
        let mut stats = IngestStats::default();

        info!(
            "[{}] IngestOrchestrator: region={:?} {} → {}",
            LABEL_INFORMASI_RISIKO, region, start, end
        );

        let registry = create_registry_with_keys(
            self.config.open_meteo_key.clone(),
            self.config.noaa_token.clone(),
            self.config.bmkg_key.clone(),
            self.config.weather_next_key.clone(),
        );

        let sources = registry.all();
        info!("Active sources: {}", sources.len());

        let mut merger = FusionMerger::new();
        let timeout_dur = Duration::from_secs(self.config.source_timeout_secs);

        // Fetch semua sumber — tangani error per sumber secara terisolasi
        for source in sources {
            let name = source.name().to_string();
            let _permit = self.semaphore.acquire().await;

            let res =
                tokio::time::timeout(timeout_dur, source.fetch_observations(region, start, end))
                    .await;

            match res {
                Ok(Ok(mut stream)) => {
                    let mut count = 0u64;
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(obs) => {
                                merger.ingest(&obs, &mut stats);
                                count += 1;
                            }
                            Err(e) => {
                                warn!("Stream error from {}: {}", name, e);
                                *stats.source_errors.entry(name.clone()).or_default() += 1;
                            }
                        }
                    }
                    debug!("Source {}: {} records", name, count);
                }
                Ok(Err(e)) => {
                    warn!("Source {} failed: {}", name, e);
                    *stats.source_errors.entry(name.clone()).or_default() += 1;
                }
                Err(_) => {
                    warn!("Source {} timed out after {}s", name, timeout_dur.as_secs());
                    *stats.source_errors.entry(name.clone()).or_default() += 1;
                }
            }
        }

        // Finalize fusion
        let fused = merger.finalize(&mut stats);
        info!("Fused {} merged observations", fused.len());

        // Tulis ke storage dalam batch
        let batch_size = self.config.write_batch_size;
        for chunk in fused.chunks(batch_size) {
            match self.storage.write_observations(chunk.to_vec()).await {
                Ok(n) => stats.records_written += n as u64,
                Err(e) => {
                    error!("Storage write error: {}", e);
                    *stats
                        .source_errors
                        .entry("storage".to_string())
                        .or_default() += 1;
                }
            }
        }

        stats.duration_ms = t0.elapsed().as_millis() as u64;
        info!("IngestOrchestrator done: {}", stats);
        Ok(stats)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(
        source: DataSource,
        lat: f64,
        lon: f64,
        var: WeatherVariable,
        val: f64,
    ) -> WeatherObservation {
        let mut variables = HashMap::new();
        variables.insert(var, ObservedValue::new(val, 0.9));
        WeatherObservation {
            station_id: None,
            coord: GeoCoord::new(lat, lon),
            time: Utc::now(),
            variables,
            quality_flag: QualityFlag::Valid,
            source,
        }
    }

    #[test]
    fn test_era5_blocked() {
        assert!(
            !is_realtime_eligible(DataSource::ERA5),
            "ERA5 harus selalu diblokir dari real-time fusion"
        );
    }

    #[test]
    fn test_bmkg_gfs_allowed() {
        assert!(is_realtime_eligible(DataSource::BMKG));
        assert!(is_realtime_eligible(DataSource::GFS));
        assert!(is_realtime_eligible(DataSource::OpenMeteo));
    }

    #[test]
    fn test_fusion_merger_dedup() {
        let mut merger = FusionMerger::new();
        let mut stats = IngestStats::default();

        // Tiga sumber berbeda, lokasi sama → median
        merger.ingest(
            &make_obs(
                DataSource::BMKG,
                -6.2,
                106.8,
                WeatherVariable::Temperature,
                28.0,
            ),
            &mut stats,
        );
        merger.ingest(
            &make_obs(
                DataSource::GFS,
                -6.2,
                106.8,
                WeatherVariable::Temperature,
                30.0,
            ),
            &mut stats,
        );
        merger.ingest(
            &make_obs(
                DataSource::OpenMeteo,
                -6.2,
                106.8,
                WeatherVariable::Temperature,
                29.0,
            ),
            &mut stats,
        );

        let fused = merger.finalize(&mut stats);

        assert_eq!(
            fused.len(),
            1,
            "Tiga sumber, satu lokasi → satu hasil fused"
        );
        let temp_val = fused[0]
            .variables
            .values()
            .next()
            .map(|v| v.value)
            .unwrap_or(0.0);
        assert!(
            (temp_val - 29.0).abs() < 0.01,
            "Median dari [28,29,30] = 29, got {}",
            temp_val
        );
    }

    #[test]
    fn test_era5_skipped_by_merger() {
        let mut merger = FusionMerger::new();
        let mut stats = IngestStats::default();

        merger.ingest(
            &make_obs(
                DataSource::ERA5,
                -6.0,
                106.0,
                WeatherVariable::Temperature,
                25.0,
            ),
            &mut stats,
        );

        let fused = merger.finalize(&mut stats);
        assert_eq!(fused.len(), 0, "ERA5 harus di-skip oleh FusionMerger");
        assert_eq!(stats.records_skipped, 1);
    }

    #[test]
    fn test_erroneous_obs_skipped() {
        let mut merger = FusionMerger::new();
        let mut stats = IngestStats::default();

        let mut obs = make_obs(
            DataSource::BMKG,
            -6.0,
            106.0,
            WeatherVariable::Temperature,
            25.0,
        );
        obs.quality_flag = QualityFlag::Erroneous;
        merger.ingest(&obs, &mut stats);

        let fused = merger.finalize(&mut stats);
        assert_eq!(fused.len(), 0, "Erroneous obs harus di-skip");
    }

    #[test]
    fn test_source_priority_uses_reliability() {
        // NOAA reliability = 0.95, OpenMeteo = 0.88
        // Pilih NOAA
        let sources = vec![DataSource::OpenMeteo, DataSource::NOAA];
        let winner = prioritize_source(sources.into_iter());
        assert_eq!(winner, DataSource::NOAA);
    }

    #[test]
    fn test_default_config() {
        let cfg = OrchestratorConfig::default();
        assert_eq!(cfg.max_concurrent, 4);
        assert_eq!(cfg.write_batch_size, 500);
        assert_eq!(cfg.source_timeout_secs, 120);
    }

    #[test]
    fn test_label_constant() {
        assert_eq!(
            LABEL_INFORMASI_RISIKO, "informasi risiko",
            "Label harus persis 'informasi risiko' per UU 31/2009"
        );
    }
}
