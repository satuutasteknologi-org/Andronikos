//! # Data Splits for Verification
//!
//! Temporal and spatial cross-validation splits for rigorous verification.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

use wi_core::WeatherObservation;

/// Data splitter for verification
pub struct DataSplitter;

impl DataSplitter {
    /// Temporal split: train on earlier period, test on later
    pub fn temporal_split(
        observations: &[WeatherObservation],
        train_ratio: f64,
    ) -> (Vec<WeatherObservation>, Vec<WeatherObservation>) {
        let mut sorted = observations.to_vec();
        sorted.sort_by_key(|o| o.time);

        let split_idx = (sorted.len() as f64 * train_ratio) as usize;
        let train = sorted[..split_idx].to_vec();
        let test = sorted[split_idx..].to_vec();

        (train, test)
    }

    /// Temporal split with gap to avoid leakage
    pub fn temporal_split_with_gap(
        observations: &[WeatherObservation],
        train_ratio: f64,
        gap_hours: i64,
    ) -> (Vec<WeatherObservation>, Vec<WeatherObservation>) {
        let mut sorted = observations.to_vec();
        sorted.sort_by_key(|o| o.time);

        let split_idx = (sorted.len() as f64 * train_ratio) as usize;
        let train_end = sorted[split_idx - 1].time;
        let gap_start = train_end + Duration::hours(gap_hours);

        let train: Vec<_> = sorted
            .iter()
            .filter(|o| o.time <= train_end)
            .cloned()
            .collect();
        let test: Vec<_> = sorted
            .iter()
            .filter(|o| o.time >= gap_start)
            .cloned()
            .collect();

        (train, test)
    }

    /// Spatial split: hold out stations
    pub fn spatial_split(
        observations: &[WeatherObservation],
        holdout_fraction: f64,
    ) -> (Vec<WeatherObservation>, Vec<WeatherObservation>) {
        let mut station_groups: HashMap<String, Vec<WeatherObservation>> = HashMap::new();
        for obs in observations {
            if let Some(id) = &obs.station_id {
                station_groups
                    .entry(id.clone())
                    .or_default()
                    .push(obs.clone());
            }
        }

        let stations: Vec<_> = station_groups.keys().cloned().collect();
        let holdout_count = (stations.len() as f64 * holdout_fraction).ceil() as usize;

        let mut train = Vec::new();
        let mut test = Vec::new();

        for (i, station) in stations.iter().enumerate() {
            let group = station_groups.get(station).unwrap();
            if i < holdout_count {
                test.extend(group.clone());
            } else {
                train.extend(group.clone());
            }
        }

        (train, test)
    }

    /// Spatial split by region
    pub fn spatial_split_by_region(
        observations: &[WeatherObservation],
        test_regions: &[wi_core::types::Bbox],
    ) -> (Vec<WeatherObservation>, Vec<WeatherObservation>) {
        let mut train = Vec::new();
        let mut test = Vec::new();

        for obs in observations {
            let in_test = test_regions
                .iter()
                .any(|r| r.contains(obs.coord.lat, obs.coord.lon));
            if in_test {
                test.push(obs.clone());
            } else {
                train.push(obs.clone());
            }
        }

        (train, test)
    }

    /// K-fold cross validation
    pub fn k_fold<T: Clone>(data: &[T], k: usize) -> Vec<(Vec<T>, Vec<T>)> {
        let n = data.len();
        let fold_size = n / k;

        (0..k)
            .map(|i| {
                let start = i * fold_size;
                let end = if i == k - 1 { n } else { (i + 1) * fold_size };

                let test = data[start..end].to_vec();
                let train = data[..start]
                    .iter()
                    .chain(data[end..].iter())
                    .cloned()
                    .collect();

                (train, test)
            })
            .collect()
    }

    /// Blocked cross validation for time series
    pub fn blocked_cv<T: Clone>(data: &[T], n_blocks: usize) -> Vec<(Vec<T>, Vec<T>)> {
        let n = data.len();
        let block_size = n / n_blocks;

        (0..n_blocks)
            .map(|i| {
                let test_start = i * block_size;
                let test_end = if i == n_blocks - 1 {
                    n
                } else {
                    (i + 1) * block_size
                };

                let gap = block_size / 10;
                let train_start = test_start.saturating_sub(gap);
                let train_end = (test_end + gap).min(n);

                let test = data[test_start..test_end].to_vec();
                let train = data[..train_start]
                    .iter()
                    .chain(data[train_end..].iter())
                    .cloned()
                    .collect();

                (train, test)
            })
            .collect()
    }
}

/// Temporal split configuration
#[derive(Debug, Clone)]
pub struct TemporalSplit {
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub gap_hours: i64,
}

impl TemporalSplit {
    pub fn new(
        train_start: DateTime<Utc>,
        train_end: DateTime<Utc>,
        test_start: DateTime<Utc>,
        test_end: DateTime<Utc>,
        gap_hours: i64,
    ) -> Self {
        Self {
            train_start,
            train_end,
            test_start,
            test_end,
            gap_hours,
        }
    }
}

/// Spatial split configuration
#[derive(Debug, Clone)]
pub struct SpatialSplit {
    pub train_regions: Vec<wi_core::types::Bbox>,
    pub test_regions: Vec<wi_core::types::Bbox>,
}

impl SpatialSplit {
    pub fn new(
        train_regions: Vec<wi_core::types::Bbox>,
        test_regions: Vec<wi_core::types::Bbox>,
    ) -> Self {
        Self {
            train_regions,
            test_regions,
        }
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidation {
    pub n_folds: usize,
    pub temporal: bool,
    pub spatial: bool,
    pub gap_hours: i64,
}

impl CrossValidation {
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            temporal: true,
            spatial: false,
            gap_hours: 24,
        }
    }
}
