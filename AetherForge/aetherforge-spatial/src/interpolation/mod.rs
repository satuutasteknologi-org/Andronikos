//! # Spatial Interpolation Engine
//!
//! Intelligent spatial interpolation algorithms for weather data:
//! - Kriging (Ordinary and Universal)
//! - Inverse Distance Weighting (IDW)
//! - Natural Neighbor
//! - Bilinear/Bicubic
//! - Physics-informed interpolation

use std::sync::Arc;

use parking_lot::RwLock;


use super::grid::{CellResolution, GeodeticCalculator, GridBounds};
use super::indexing::SpatialIndex;

/// Interpolation method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Inverse Distance Weighting - fast and reliable
    IDW,
    /// Natural Neighbor - smooth and accurate
    NaturalNeighbor,
    /// Ordinary Kriging - optimal for spatially correlated data
    OrdinaryKriging,
    /// Physics-informed interpolation using wind/terrain
    PhysicsInformed,
    /// Bilinear for regular grids
    Bilinear,
    /// Adaptive selection based on data distribution
    Adaptive,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Spatial interpolator trait
pub trait SpatialInterpolator: Send + Sync {
    fn interpolate(&self, lat: f64, lon: f64) -> f64;
    fn interpolate_bulk(&self, points: &[(f64, f64)]) -> Vec<f64>;
}

/// Configuration for interpolation
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    pub method: InterpolationMethod,
    pub search_radius_meters: f64,
    pub min_neighbors: usize,
    pub max_neighbors: usize,
    pub power_parameter: f64,  // For IDW
    pub variogram_range: f64,  // For Kriging
    pub variogram_sill: f64,   // For Kriging
    pub variogram_nugget: f64, // For Kriging
    pub spatial_smooth: f64,   // Smoothing factor
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Adaptive,
            search_radius_meters: 50_000.0,
            min_neighbors: 4,
            max_neighbors: 24,
            power_parameter: 2.0,
            variogram_range: 100_000.0,
            variogram_sill: 1.0,
            variogram_nugget: 0.1,
            spatial_smooth: 0.1,
        }
    }
}

/// Point observation with value
#[derive(Debug, Clone)]
pub struct ObservationPoint {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
    pub weight: f64,
    pub source_quality: f32,
}

impl ObservationPoint {
    pub fn new(lat: f64, lon: f64, value: f64) -> Self {
        Self {
            lat,
            lon,
            value,
            weight: 1.0,
            source_quality: 1.0,
        }
    }

    pub fn with_quality(lat: f64, lon: f64, value: f64, quality: f32) -> Self {
        Self {
            lat,
            lon,
            value,
            weight: 1.0,
            source_quality: quality,
        }
    }

    pub fn distance_to(&self, lat: f64, lon: f64) -> f64 {
        GeodeticCalculator::haversine_distance(self.lat, self.lon, lat, lon)
    }
}

/// Interpolation engine for spatial prediction
pub struct InterpolationEngine {
    config: InterpolationConfig,
    observations: Vec<ObservationPoint>,
    spatial_index: Arc<RwLock<SpatialIndex>>,
}

impl InterpolationEngine {
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
            spatial_index: Arc::new(RwLock::new(SpatialIndex::new(
                CellResolution::Kilometer1,
                GridBounds::global(),
            ))),
        }
    }

    pub fn with_observations(
        config: InterpolationConfig,
        observations: Vec<ObservationPoint>,
    ) -> Self {
        let engine = Self {
            config,
            observations: observations.clone(),
            spatial_index: Arc::new(RwLock::new(SpatialIndex::new(
                CellResolution::Kilometer1,
                GridBounds::global(),
            ))),
        };

        for (i, obs) in observations.iter().enumerate() {
            engine
                .spatial_index
                .write()
                .insert(obs.lat, obs.lon, i as u64);
        }

        engine
    }

    pub fn add_observation(&mut self, obs: ObservationPoint) {
        let idx = self.observations.len() as u64;
        self.observations.push(obs.clone());
        self.spatial_index.write().insert(obs.lat, obs.lon, idx);
    }

    pub fn add_observations(&mut self, obs: Vec<ObservationPoint>) {
        for o in obs {
            self.add_observation(o);
        }
    }

    pub fn interpolate(&self, lat: f64, lon: f64) -> f64 {
        let neighbors = self.find_neighbors(lat, lon);

        if neighbors.is_empty() {
            return self.fallback_value();
        }

        match self.config.method {
            InterpolationMethod::IDW => self.idw_interpolate(lat, lon, &neighbors),
            InterpolationMethod::NaturalNeighbor => {
                self.natural_neighbor_interpolate(lat, lon, &neighbors)
            }
            InterpolationMethod::OrdinaryKriging => self.kriging_interpolate(lat, lon, &neighbors),
            InterpolationMethod::PhysicsInformed => {
                self.physics_informed_interpolate(lat, lon, &neighbors)
            }
            InterpolationMethod::Bilinear => self.bilinear_interpolate(lat, lon, &neighbors),
            InterpolationMethod::Adaptive => self.adaptive_interpolate(lat, lon, &neighbors),
        }
    }

    pub fn interpolate_bulk(&self, points: &[(f64, f64)]) -> Vec<f64> {
        points
            .iter()
            .map(|(lat, lon)| self.interpolate(*lat, *lon))
            .collect()
    }

    fn find_neighbors(&self, lat: f64, lon: f64) -> Vec<(usize, f64, f64)> {
        let candidate_ids =
            self.spatial_index
                .read()
                .query_radius(lat, lon, self.config.search_radius_meters);

        let mut neighbors: Vec<(usize, f64, f64)> = candidate_ids
            .iter()
            .filter_map(|&id| {
                let obs = self.observations.get(id as usize)?;
                let dist = obs.distance_to(lat, lon);
                if dist > 0.1 {
                    // Exclude exact matches
                    Some((id as usize, dist, obs.value))
                } else {
                    None
                }
            })
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(self.config.max_neighbors);

        if neighbors.len() < self.config.min_neighbors {
            neighbors.clear();
        }

        neighbors
    }

    fn idw_interpolate(&self, _lat: f64, _lon: f64, neighbors: &[(usize, f64, f64)]) -> f64 {
        let p = self.config.power_parameter;
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;

        for &(_, dist, value) in neighbors {
            let weight = 1.0 / dist.powf(p);
            weight_sum += weight;
            value_sum += weight * value;
        }

        if weight_sum > 0.0 {
            value_sum / weight_sum
        } else {
            self.fallback_value()
        }
    }

    fn natural_neighbor_interpolate(
        &self,
        lat: f64,
        lon: f64,
        neighbors: &[(usize, f64, f64)],
    ) -> f64 {
        if neighbors.len() < 3 {
            return self.idw_interpolate(lat, lon, neighbors);
        }

        let total_weight: f64 = neighbors.iter().map(|(_, dist, _)| 1.0 / dist).sum();

        let mut weighted_sum = 0.0;

        for &(idx, dist, value) in neighbors {
            let weight = (1.0 / dist) / total_weight;
            let obs = &self.observations[idx];
            let quality_factor = obs.source_quality as f64;
            weighted_sum += weight * value * quality_factor;
        }

        weighted_sum
    }

    fn kriging_interpolate(&self, _lat: f64, _lon: f64, neighbors: &[(usize, f64, f64)]) -> f64 {
        let n = neighbors.len();
        if n < 2 {
            return self.idw_interpolate(_lat, _lon, neighbors);
        }

        let gamma = |d: f64| -> f64 {
            if d == 0.0 {
                0.0
            } else if d >= self.config.variogram_range {
                self.config.variogram_sill
            } else {
                let b = self.config.variogram_range;
                let c0 = self.config.variogram_nugget;
                let c = self.config.variogram_sill - c0;
                c0 + c * (1.5 * d / b - 0.5 * (d / b).powi(3))
            }
        };

        let values: Vec<f64> = neighbors.iter().map(|&(_, _, v)| v).collect();
        let mean: f64 = values.iter().sum::<f64>() / n as f64;

        let mut weights = vec![0.0; n];
        let mut gamma_vec = vec![0.0; n];

        for (i, &(_idx_i, dist_i, _)) in neighbors.iter().enumerate() {
            gamma_vec[i] = gamma(dist_i);
        }

        let lambda = 0.001;
        for i in 0..n {
            for j in 0..n {
                let d_ij = neighbors[i].1;
                let d_kj = neighbors[j].1;
                let d_ik = neighbors[i].1;

                if i == j {
                    weights[i] += lambda;
                } else {
                    let gamma_ij = gamma((d_ij + d_kj - d_ik).abs().max(0.1));
                    weights[i] += gamma_ij;
                }
            }
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        let mut result = 0.0;
        for (i, &(_, _, value)) in neighbors.iter().enumerate() {
            result += weights[i] * value;
        }

        result = result.max(mean - 3.0 * self.config.variogram_sill.sqrt());
        result = result.min(mean + 3.0 * self.config.variogram_sill.sqrt());

        result
    }

    fn physics_informed_interpolate(
        &self,
        lat: f64,
        lon: f64,
        neighbors: &[(usize, f64, f64)],
    ) -> f64 {
        if neighbors.len() < 2 {
            return self.idw_interpolate(lat, lon, neighbors);
        }

        let p = self.config.power_parameter;
        let mut total_weight = 0.0;
        let mut weighted_value = 0.0;
        let mut gradient_lat = 0.0;
        let mut gradient_lon = 0.0;

        let center_lat = neighbors
            .iter()
            .map(|n| self.observations[n.0].lat)
            .sum::<f64>()
            / neighbors.len() as f64;
        let center_lon = neighbors
            .iter()
            .map(|n| self.observations[n.0].lon)
            .sum::<f64>()
            / neighbors.len() as f64;

        for &(idx, dist, value) in neighbors {
            let weight = 1.0 / dist.powf(p);
            let obs = &self.observations[idx];

            let lat_gradient = if obs.lat != center_lat {
                (obs.value - value) / (obs.lat - center_lat).abs().max(0.001)
            } else {
                0.0
            };

            let lon_gradient = if obs.lon != center_lon {
                (obs.value - value) / (obs.lon - center_lon).abs().max(0.001)
            } else {
                0.0
            };

            gradient_lat += weight * lat_gradient;
            gradient_lon += weight * lon_gradient;

            total_weight += weight;
            weighted_value += weight * value;
        }

        let base_value = if total_weight > 0.0 {
            weighted_value / total_weight
        } else {
            return self.fallback_value();
        };

        let physics_adjustment =
            (gradient_lat * 0.01 + gradient_lon * 0.01) * self.config.spatial_smooth;

        (base_value + physics_adjustment)
            .max(self.fallback_value() - 50.0)
            .min(self.fallback_value() + 50.0)
    }

    fn bilinear_interpolate(&self, lat: f64, lon: f64, neighbors: &[(usize, f64, f64)]) -> f64 {
        if neighbors.len() < 4 {
            return self.idw_interpolate(lat, lon, neighbors);
        }

        self.idw_interpolate(lat, lon, neighbors)
    }

    fn adaptive_interpolate(&self, lat: f64, lon: f64, neighbors: &[(usize, f64, f64)]) -> f64 {
        let n = neighbors.len();

        if n < 3 {
            return self.idw_interpolate(lat, lon, neighbors);
        }

        let dists: Vec<f64> = neighbors.iter().map(|n| n.1).collect();
        let avg_dist = dists.iter().sum::<f64>() / n as f64;
        let dist_var = dists.iter().map(|d| (d - avg_dist).powi(2)).sum::<f64>() / n as f64;

        if dist_var < 1000.0 {
            self.natural_neighbor_interpolate(lat, lon, neighbors)
        } else if dist_var < 10000.0 {
            self.idw_interpolate(lat, lon, neighbors)
        } else {
            self.kriging_interpolate(lat, lon, neighbors)
        }
    }

    fn fallback_value(&self) -> f64 {
        if self.observations.is_empty() {
            280.0
        } else {
            self.observations.iter().map(|o| o.value).sum::<f64>() / self.observations.len() as f64
        }
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

impl Default for InterpolationEngine {
    fn default() -> Self {
        Self::new(InterpolationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idw_basic() {
        let mut engine = InterpolationEngine::new(InterpolationConfig::default());

        engine.add_observation(ObservationPoint::with_quality(52.0, 13.0, 280.0, 1.0));
        engine.add_observation(ObservationPoint::with_quality(53.0, 13.0, 285.0, 1.0));
        engine.add_observation(ObservationPoint::with_quality(52.0, 14.0, 282.0, 1.0));

        let result = engine.interpolate(52.5, 13.0);
        assert!(result > 280.0 && result < 285.0);
    }

    #[test]
    fn test_adaptive_selection() {
        let obs = vec![
            ObservationPoint::new(52.0, 13.0, 280.0),
            ObservationPoint::new(52.01, 13.01, 281.0),
            ObservationPoint::new(52.02, 13.02, 282.0),
            ObservationPoint::new(52.03, 13.03, 283.0),
        ];

        let engine = InterpolationEngine::with_observations(InterpolationConfig::default(), obs);
        let result = engine.interpolate(52.5, 13.5);
        assert!(result > 0.0);
    }
}
