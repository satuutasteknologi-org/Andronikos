//! # Weather Verification
//!
//! Comprehensive forecast verification metrics:
//! - Deterministic: MAE, RMSE, Bias
//! - Probabilistic: Brier Score, Reliability Diagram, CRPS
//! - Ensemble: Spread-Skill, Rank Histogram
//! - Categorical: CSI, ETS, HSS, POD, FAR (WMO standard)
//! - Spatial/Temporal splits for rigorous validation

#![forbid(unsafe_code)]

pub mod baselines;
pub mod csi;
pub mod dashboard;
pub mod metrics;
pub mod splits;

pub use baselines::{BaselineComparer, BaselineForecast, ClimatologyBaseline, PersistenceBaseline};
pub use csi::{CategoricalScores, ContingencyTable, CsiScorer, RainfallCsiReport};
pub use dashboard::{DashboardConfig, VerificationDashboard};
pub use metrics::{
    DeterministicMetrics, EnsembleMetrics, MetricCalculator, ProbabilisticMetrics,
    VerificationMetrics,
};
pub use splits::{CrossValidation, DataSplitter, SpatialSplit, TemporalSplit};

#[cfg(test)]
mod tests {
    #[test]
    fn test_verification() {
        // Placeholder — detailed tests ada di masing-masing modul
    }
}
