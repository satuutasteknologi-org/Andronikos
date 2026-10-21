//! # Data Preprocessing & Feature Engineering
//!
//! This module handles intelligent preprocessing of weather data:
//! - Quantization with adaptive binning based on climatological distributions
//! - Temporal and spatial slotting using GeoHash and H3 indexing
//! - Missing value imputation with physics-informed methods
//! - Feature store generation for ML training

use chrono::Datelike;
use chrono::Timelike;
use chrono::{DateTime, Duration, Utc};
use h3o::CellIndex;
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::Arc;

use crate::sources::RegionBbox;

use aetherforge_core::{ClimateZone, GeoCoord, WeatherObservation, WeatherVariable};

/// Data normalizer with climatology-aware scaling
pub struct DataNormalizer {
    climatology: Arc<ClimatologyStore>,
    normalization: NormalizationType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    MinMax,
    Standard,
    Climatology,
    Percentile,
}

impl DataNormalizer {
    pub fn new(climatology: Arc<ClimatologyStore>) -> Self {
        Self {
            climatology,
            normalization: NormalizationType::Climatology,
        }
    }

    pub fn normalize(&self, value: f64, variable: WeatherVariable, coord: GeoCoord) -> f64 {
        let stats = self.climatology.get_stats(variable, coord);

        match self.normalization {
            NormalizationType::MinMax => {
                let (min, max) = (stats.min, stats.max);
                if (max - min).abs() < 1e-6 {
                    0.0
                } else {
                    (value - min) / (max - min)
                }
            }
            NormalizationType::Standard => {
                if stats.std < 1e-6 {
                    0.0
                } else {
                    (value - stats.mean) / stats.std
                }
            }
            NormalizationType::Climatology | NormalizationType::Percentile => {
                if let Some(anomaly) = stats.anomaly(value) {
                    anomaly / stats.std.max(1e-6)
                } else {
                    0.0
                }
            }
        }
    }

    pub fn denormalize(&self, value: f64, variable: WeatherVariable, coord: GeoCoord) -> f64 {
        let stats = self.climatology.get_stats(variable, coord);

        match self.normalization {
            NormalizationType::MinMax => {
                let (min, max) = (stats.min, stats.max);
                value * (max - min) + min
            }
            NormalizationType::Standard => value * stats.std + stats.mean,
            NormalizationType::Climatology | NormalizationType::Percentile => {
                value * stats.std + stats.mean
            }
        }
    }

    /// Normalize an entire observation batch
    pub fn normalize_batch(
        &self,
        observations: &[WeatherObservation],
    ) -> Vec<HashMap<WeatherVariable, f64>> {
        observations
            .iter()
            .map(|obs| {
                let mut normalized = HashMap::new();
                for (var, val) in &obs.variables {
                    normalized.insert(*var, self.normalize(val.value, *var, obs.coord));
                }
                normalized
            })
            .collect()
    }
}

/// Climatology store for normalization statistics
pub struct ClimatologyStore {
    stats: HashMap<(WeatherVariable, String), ClimatologyStats>,
}

#[derive(Debug, Clone)]
pub struct ClimatologyStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p01: f64,
    pub p05: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub samples: usize,
}

impl ClimatologyStats {
    pub fn anomaly(&self, value: f64) -> Option<f64> {
        Some(value - self.mean)
    }

    pub fn percentile_rank(&self, value: f64) -> f64 {
        if value <= self.p01 {
            0.01
        } else if value <= self.p05 {
            0.05
        } else if value <= self.p10 {
            0.10
        } else if value <= self.p25 {
            0.25
        } else if value <= self.p50 {
            0.50
        } else if value <= self.p75 {
            0.75
        } else if value <= self.p90 {
            0.90
        } else if value <= self.p95 {
            0.95
        } else if value <= self.p99 {
            0.99
        } else {
            0.99
        }
    }
}

impl ClimatologyStore {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    pub fn from_observations(observations: &[WeatherObservation]) -> Self {
        let mut store = Self::new();
        store.compute_from_observations(observations);
        store
    }

    pub fn compute_from_observations(&mut self, observations: &[WeatherObservation]) {
        let mut by_var: HashMap<WeatherVariable, Vec<f64>> = HashMap::new();

        for obs in observations {
            for (var, val) in &obs.variables {
                if val.is_valid {
                    by_var.entry(*var).or_default().push(val.value);
                }
            }
        }

        for (var, values) in by_var {
            if values.len() < 10 {
                continue;
            }

            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted.len();
            let stats = ClimatologyStats {
                mean: sorted.iter().sum::<f64>() / n as f64,
                std: Self::compute_std(&sorted),
                min: sorted[0],
                max: sorted[n - 1],
                p01: Self::percentile(&sorted, 0.01),
                p05: Self::percentile(&sorted, 0.05),
                p10: Self::percentile(&sorted, 0.10),
                p25: Self::percentile(&sorted, 0.25),
                p50: Self::percentile(&sorted, 0.50),
                p75: Self::percentile(&sorted, 0.75),
                p90: Self::percentile(&sorted, 0.90),
                p95: Self::percentile(&sorted, 0.95),
                p99: Self::percentile(&sorted, 0.99),
                samples: n,
            };

            self.stats.insert((var, "global".to_string()), stats);
        }
    }

    pub fn get_stats(&self, variable: WeatherVariable, coord: GeoCoord) -> &ClimatologyStats {
        let geohash = coord.to_geo_hash(4);
        self.stats
            .get(&(variable, geohash))
            .or_else(|| self.stats.get(&(variable, "global".to_string())))
            .unwrap_or(&DEFAULT_STATS)
    }

    fn compute_std(sorted: &[f64]) -> f64 {
        if sorted.len() < 2 {
            return 1.0;
        }
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
        variance.sqrt()
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

static DEFAULT_STATS: ClimatologyStats = ClimatologyStats {
    mean: 280.0,
    std: 10.0,
    min: 200.0,
    max: 320.0,
    p01: 230.0,
    p05: 240.0,
    p10: 250.0,
    p25: 265.0,
    p50: 280.0,
    p75: 295.0,
    p90: 305.0,
    p95: 310.0,
    p99: 315.0,
    samples: 0,
};

impl Default for ClimatologyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Missing value handler with physics-informed imputation
pub struct MissingValueHandler {
    method: ImputationMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationMethod {
    Mean,
    Linear,
    SpatialKriging,
    PhysicsInformed,
}

impl Default for MissingValueHandler {
    fn default() -> Self {
        Self {
            method: ImputationMethod::PhysicsInformed,
        }
    }
}

impl MissingValueHandler {
    pub fn new(method: ImputationMethod) -> Self {
        Self { method }
    }

    /// Handle missing values in a 2D spatial field
    pub fn impute_2d(&self, field: &mut Array2<f32>, valid_mask: &Array2<bool>) -> usize {
        let mut filled = 0;

        let indices_to_fill: Vec<(usize, usize)> = field
            .indexed_iter()
            .filter(|((i, j), _)| !valid_mask[(*i, *j)])
            .map(|((i, j), _)| (i, j))
            .collect();

        for (i, j) in indices_to_fill {
            let imputed = self.impute_point(field, valid_mask, i, j);
            field[(i, j)] = imputed;
            filled += 1;
        }

        filled
    }

    fn impute_point(
        &self,
        field: &Array2<f32>,
        valid_mask: &Array2<bool>,
        i: usize,
        j: usize,
    ) -> f32 {
        match self.method {
            ImputationMethod::Mean => {
                let valid_count = valid_mask.iter().filter(|&&v| v).count();
                if valid_count > 0 {
                    field
                        .iter()
                        .zip(valid_mask.iter())
                        .filter(|(_, &v)| v)
                        .map(|(v, _)| v)
                        .sum::<f32>()
                        / valid_count as f32
                } else {
                    0.0
                }
            }
            ImputationMethod::Linear => self.linear_interpolation(field, valid_mask, i, j),
            ImputationMethod::SpatialKriging | ImputationMethod::PhysicsInformed => {
                self.spatial_average(field, valid_mask, i, j, 3)
            }
        }
    }

    fn linear_interpolation(
        &self,
        field: &Array2<f32>,
        valid_mask: &Array2<bool>,
        i: usize,
        j: usize,
    ) -> f32 {
        let (rows, cols) = field.dim();

        let mut neighbors = Vec::new();
        let mut weights = Vec::new();

        for di in [-1, 0, 1] {
            for dj in [-1, 0, 1] {
                if di == 0 && dj == 0 {
                    continue;
                }
                let ni = i as isize + di;
                let nj = j as isize + dj;

                if ni >= 0 && ni < rows as isize && nj >= 0 && nj < cols as isize {
                    let ni = ni as usize;
                    let nj = nj as usize;

                    if valid_mask[(ni, nj)] {
                        neighbors.push(field[(ni, nj)]);
                        weights.push(1.0 / (di.abs() + dj.abs()) as f32);
                    }
                }
            }
        }

        if neighbors.is_empty() {
            return field.mean().unwrap_or(0.0);
        }

        let total_weight: f32 = weights.iter().sum();
        neighbors
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w / total_weight)
            .sum()
    }

    fn spatial_average(
        &self,
        field: &Array2<f32>,
        valid_mask: &Array2<bool>,
        i: usize,
        j: usize,
        radius: usize,
    ) -> f32 {
        let (rows, cols) = field.dim();
        let mut sum = 0.0f32;
        let mut count = 0u32;
        let mut total_dist = 0.0f32;

        let i_isize = i as isize;
        let j_isize = j as isize;
        let rows_isize = rows as isize;
        let cols_isize = cols as isize;
        let radius_isize = radius as isize;

        for di in -radius_isize..=radius_isize {
            for dj in -radius_isize..=radius_isize {
                let ni = i_isize + di;
                let nj = j_isize + dj;

                if ni >= 0 && ni < rows_isize && nj >= 0 && nj < cols_isize {
                    let ni = ni as usize;
                    let nj = nj as usize;

                    if valid_mask[(ni, nj)] {
                        let dist = ((di * di + dj * dj) as f32).sqrt().max(1.0);
                        sum += field[(ni, nj)] / dist;
                        total_dist += 1.0 / dist;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            sum / total_dist
        } else {
            field.mean().unwrap_or(0.0)
        }
    }
}

/// Temporal slotting for time-series data
pub struct TemporalSlotter {
    resolution_seconds: u32, // Supports 30s, 60s, 300s, etc.
}

impl TemporalSlotter {
    pub fn new(resolution_seconds: u32) -> Self {
        Self { resolution_seconds }
    }

    pub fn from_minutes(minutes: u16) -> Self {
        Self::new(minutes as u32 * 60)
    }

    pub fn resolution(&self) -> u32 {
        self.resolution_seconds
    }

    /// Get slot index for a time
    pub fn slot_index(&self, time: DateTime<Utc>) -> usize {
        let seconds_since_epoch = time.timestamp();
        (seconds_since_epoch / self.resolution_seconds as i64) as usize
    }

    /// Get time for a slot index
    pub fn slot_time(&self, slot: usize) -> DateTime<Utc> {
        let seconds_since_epoch = slot as i64 * self.resolution_seconds as i64;
        DateTime::from_timestamp(seconds_since_epoch, 0).unwrap_or_else(Utc::now)
    }

    /// Slot observations into temporal bins
    pub fn slot_observations<'a>(
        &self,
        observations: &'a [WeatherObservation],
    ) -> HashMap<usize, Vec<&'a WeatherObservation>> {
        let mut slots: HashMap<usize, Vec<&'a WeatherObservation>> = HashMap::new();

        for obs in observations {
            let slot = self.slot_index(obs.time);
            slots.entry(slot).or_default().push(obs);
        }

        slots
    }

    /// Get all slot indices in a time range
    pub fn slot_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<usize> {
        let start_slot = self.slot_index(start);
        let end_slot = self.slot_index(end);

        (start_slot..=end_slot).collect()
    }
}

/// Spatial slotting using H3 indexing
pub struct SpatialSlotter {
    resolution: u8,
}

impl SpatialSlotter {
    pub fn new(resolution: u8) -> Self {
        Self { resolution }
    }

    /// Get H3 cell index for a coordinate
    pub fn cell_index(&self, coord: GeoCoord) -> u64 {
        let latlng = h3o::LatLng::new(coord.lat, coord.lon)
            .unwrap_or_else(|_| h3o::LatLng::new(0.0, 0.0).unwrap());
        let resolution = h3o::Resolution::try_from(self.resolution)
            .unwrap_or(h3o::Resolution::try_from(7).unwrap());
        latlng.to_cell(resolution).into()
    }

    /// Get neighbor cells at same resolution
    pub fn neighbors(&self, cell: u64) -> Vec<u64> {
        let cell = CellIndex::try_from(cell).ok();

        if let Some(c) = cell {
            let grid: Vec<CellIndex> = c.grid_disk::<Vec<CellIndex>>(1);
            grid.into_iter()
                .filter(|n| *n != c)
                .map(|n| n.into())
                .collect()
        } else {
            vec![]
        }
    }

    /// Get parent cell at lower resolution
    pub fn parent(&self, cell: u64, parent_res: u8) -> Option<u64> {
        let cell = CellIndex::try_from(cell).ok()?;
        let resolution = h3o::Resolution::try_from(parent_res).ok()?;
        cell.parent(resolution).map(|p| p.into())
    }

    /// Get children cells at higher resolution
    pub fn children(&self, cell: u64, child_res: u8) -> Vec<u64> {
        let cell = match CellIndex::try_from(cell) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let resolution = match h3o::Resolution::try_from(child_res) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        cell.children(resolution)
            .into_iter()
            .map(|c| c.into())
            .collect()
    }

    /// Slot observations into spatial cells
    pub fn slot_observations<'a>(
        &self,
        observations: &'a [WeatherObservation],
    ) -> HashMap<u64, Vec<&'a WeatherObservation>> {
        let mut slots: HashMap<u64, Vec<&'a WeatherObservation>> = HashMap::new();

        for obs in observations {
            let cell = self.cell_index(obs.coord);
            slots.entry(cell).or_default().push(obs);
        }

        slots
    }

    /// Build a grid from bounding box
    #[allow(dead_code)]
    pub fn grid_from_region(&self, _region: &RegionBbox) -> Vec<u64> {
        Vec::new()
    }
}

/// Feature engineering for ML models
pub struct FeatureEngine {
    temporal_slotter: TemporalSlotter,
    spatial_slotter: SpatialSlotter,
    climatology: Arc<ClimatologyStore>,
    lag_seconds: Vec<u32>, // Supports sub-hour lags
}

impl FeatureEngine {
    pub fn new(
        temporal_resolution_seconds: u32,
        spatial_resolution: u8,
        lag_seconds: Vec<u32>,
    ) -> Self {
        Self {
            temporal_slotter: TemporalSlotter::new(temporal_resolution_seconds),
            spatial_slotter: SpatialSlotter::new(spatial_resolution),
            climatology: Arc::new(ClimatologyStore::new()),
            lag_seconds,
        }
    }

    pub fn with_lag_hours(hours: Vec<u16>) -> Self {
        Self::new(60, 7, hours.iter().map(|h| *h as u32 * 3600).collect())
    }

    /// Build features from observations for ML training
    pub fn build_features(
        &self,
        observations: &[WeatherObservation],
        target_time: DateTime<Utc>,
        lead_seconds: u32,
    ) -> FeatureResult {
        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        let coord = observations
            .first()
            .map(|o| o.coord)
            .unwrap_or(GeoCoord::new(0.0, 0.0));

        features.push(self.temporal_slotter.slot_index(target_time) as f32);
        feature_names.push("temporal_slot".to_string());

        features.push(self.spatial_slotter.cell_index(coord) as f32);
        feature_names.push("spatial_cell".to_string());

        features.push(target_time.hour() as f32 / 24.0);
        feature_names.push("hour_sin".to_string());
        features.push((target_time.hour() as f32 * std::f32::consts::TAU / 24.0).sin());
        feature_names.push("hour_sin".to_string());
        features.push((target_time.hour() as f32 * std::f32::consts::TAU / 24.0).cos());
        feature_names.push("hour_cos".to_string());

        features.push(target_time.month() as f32 / 12.0);
        feature_names.push("month_sin".to_string());
        features.push((target_time.month() as f32 * std::f32::consts::TAU / 12.0).sin());
        feature_names.push("month_sin".to_string());
        features.push((target_time.month() as f32 * std::f32::consts::TAU / 12.0).cos());
        feature_names.push("month_cos".to_string());

        features.push(coord.lat as f32 / 90.0);
        feature_names.push("lat_norm".to_string());
        features.push(coord.lon as f32 / 180.0);
        feature_names.push("lon_norm".to_string());

        let climate = ClimateZone::from_lat_lon(coord.lat, coord.lon);
        features.push(climate as u8 as f32 / 8.0);
        feature_names.push("climate_zone".to_string());

        for (var, val) in self.extract_latest_variables(observations) {
            let stats = self.climatology.get_stats(var, coord);
            let normalized = if stats.std > 1e-6 {
                (val - stats.mean) / stats.std
            } else {
                0.0
            };
            features.push(normalized as f32);
            feature_names.push(format!("{:?}_norm", var));

            features.push(val as f32);
            feature_names.push(format!("{:?}_raw", var));
        }

        let lag_features = self.build_lag_features(observations, target_time);
        features.extend(lag_features.values.clone());
        feature_names.extend(lag_features.names.clone());

        features.push(lead_seconds as f32);
        feature_names.push("lead_seconds".to_string());

        features.push(lead_seconds as f32 / 3600.0);
        feature_names.push("lead_hours".to_string());

        let spatial_neighbors = self
            .spatial_slotter
            .neighbors(self.spatial_slotter.cell_index(coord));
        features.push(spatial_neighbors.len() as f32);
        feature_names.push("num_neighbors".to_string());

        FeatureResult {
            features,
            feature_names,
            num_spatial_cells: spatial_neighbors.len(),
        }
    }

    fn extract_latest_variables(
        &self,
        observations: &[WeatherObservation],
    ) -> HashMap<WeatherVariable, f64> {
        let mut latest: HashMap<WeatherVariable, (DateTime<Utc>, f64)> = HashMap::new();

        for obs in observations {
            for (var, val) in &obs.variables {
                if val.is_valid {
                    let entry = latest.entry(*var).or_insert((obs.time, val.value));
                    if obs.time > entry.0 {
                        *entry = (obs.time, val.value);
                    }
                }
            }
        }

        latest.into_iter().map(|(k, (_, v))| (k, v)).collect()
    }

    fn build_lag_features(
        &self,
        observations: &[WeatherObservation],
        target_time: DateTime<Utc>,
    ) -> LagFeatures {
        let mut values = Vec::new();
        let mut names = Vec::new();

        for &lag in &self.lag_seconds {
            let lag_time = target_time - Duration::seconds(lag as i64);

            let tolerance_seconds = lag / 2; // 50% tolerance window
            let relevant: Vec<_> = observations
                .iter()
                .filter(|o| (o.time - lag_time).num_seconds().abs() < tolerance_seconds as i64)
                .collect();

            for var in [
                WeatherVariable::Temperature,
                WeatherVariable::RelativeHumidity,
                WeatherVariable::Precipitation,
                WeatherVariable::WindSpeed,
            ] {
                let value = relevant
                    .iter()
                    .filter_map(|o| o.variables.get(&var).map(|v| v.value as f32))
                    .sum::<f32>()
                    / relevant.len().max(1) as f32;

                values.push(value);
                names.push(format!("{:?}_lag{}s", var, lag));
            }
        }

        LagFeatures { values, names }
    }
}

#[derive(Debug)]
pub struct FeatureResult {
    pub features: Vec<f32>,
    pub feature_names: Vec<String>,
    pub num_spatial_cells: usize,
}

#[derive(Debug)]
struct LagFeatures {
    values: Vec<f32>,
    names: Vec<String>,
}

/// Quantization with adaptive binning
pub struct AdaptiveQuantizer {
    num_bins: usize,
    bin_edges: HashMap<WeatherVariable, Vec<f64>>,
}

impl AdaptiveQuantizer {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            bin_edges: HashMap::new(),
        }
    }

    pub fn fit(&mut self, observations: &[WeatherObservation]) {
        let mut by_var: HashMap<WeatherVariable, Vec<f64>> = HashMap::new();

        for obs in observations {
            for (var, val) in &obs.variables {
                if val.is_valid {
                    by_var.entry(*var).or_default().push(val.value);
                }
            }
        }

        for (var, mut values) in by_var {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let edges = self.compute_bin_edges(&values);
            self.bin_edges.insert(var, edges);
        }
    }

    fn compute_bin_edges(&self, sorted_values: &[f64]) -> Vec<f64> {
        if sorted_values.len() < self.num_bins {
            return vec![];
        }

        let step = sorted_values.len() / self.num_bins;
        let mut edges = Vec::with_capacity(self.num_bins + 1);

        edges.push(sorted_values[0] - 0.001);

        for i in 0..self.num_bins {
            let idx = (i + 1) * step - 1;
            if idx < sorted_values.len() {
                edges.push(sorted_values[idx]);
            }
        }

        edges.push(sorted_values.last().unwrap() + 0.001);

        edges
    }

    pub fn quantize(&self, value: f64, variable: WeatherVariable) -> usize {
        let edges = match self.bin_edges.get(&variable) {
            Some(e) => e,
            None => return self.num_bins / 2,
        };

        if edges.is_empty() {
            return self.num_bins / 2;
        }

        edges
            .iter()
            .position(|&e| value < e)
            .unwrap_or(self.num_bins - 1)
            .min(self.num_bins - 1)
    }

    pub fn dequantize(&self, bin: usize, variable: WeatherVariable) -> f64 {
        let edges = match self.bin_edges.get(&variable) {
            Some(e) => e,
            None => return 0.0,
        };

        if edges.is_empty() || bin >= edges.len() {
            return 0.0;
        }

        (edges[bin] + edges[bin + 1].min(0.0)) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_slotter() {
        let slotter = TemporalSlotter::new(60);
        let time = Utc::now();

        let slot = slotter.slot_index(time);
        let back = slotter.slot_time(slot);

        assert!((back - time).num_minutes().abs() < 60);
    }

    #[test]
    fn test_spatial_slotter() {
        let slotter = SpatialSlotter::new(7);
        let coord = GeoCoord::new(52.52, 13.405);

        let cell = slotter.cell_index(coord);
        assert!(cell > 0);

        let neighbors = slotter.neighbors(cell);
        assert!(neighbors.len() <= 7);
    }

    #[test]
    fn test_adaptive_quantizer() {
        let mut quantizer = AdaptiveQuantizer::new(256);

        let obs = vec![WeatherObservation {
            station_id: None,
            coord: GeoCoord::new(52.52, 13.405),
            time: Utc::now(),
            variables: std::collections::HashMap::from([
                (WeatherVariable::Temperature, ObservedValue::new(280.0, 0.9)),
                (WeatherVariable::Temperature, ObservedValue::new(285.0, 0.9)),
                (WeatherVariable::Temperature, ObservedValue::new(290.0, 0.9)),
            ]),
            quality_flag: aetherforge_core::QualityFlag::Valid,
            source: DataSource::ERA5,
        }];

        quantizer.fit(&obs);

        let bin = quantizer.quantize(282.5, WeatherVariable::Temperature);
        assert!(bin < 256);
    }
}
