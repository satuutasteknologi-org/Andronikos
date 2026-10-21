//! # Verification Metrics
//!
//! Implementation of standard meteorological verification metrics.

use std::collections::HashMap;

use ndarray::Array1;
use wi_core::types::WeatherVariable;

/// Complete verification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMetrics {
    pub deterministic: DeterministicMetrics,
    pub probabilistic: ProbabilisticMetrics,
    pub ensemble: Option<EnsembleMetrics>,
    pub sample_count: usize,
    pub period_start: chrono::DateTime<chrono::Utc>,
    pub period_end: chrono::DateTime<chrono::Utc>,
    pub variable: WeatherVariable,
}

impl VerificationMetrics {
    pub fn new(variable: WeatherVariable) -> Self {
        Self {
            deterministic: DeterministicMetrics::default(),
            probabilistic: ProbabilisticMetrics::default(),
            ensemble: None,
            sample_count: 0,
            period_start: chrono::Utc::now(),
            period_end: chrono::Utc::now(),
            variable,
        }
    }
}

/// Deterministic forecast metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeterministicMetrics {
    pub mae: f32,
    pub rmse: f32,
    pub bias: f32,
    pub correlation: f32,
    pub mae_by_lead: HashMap<u32, f32>,
    pub rmse_by_lead: HashMap<u32, f32>,
    pub bias_by_lead: HashMap<u32, f32>,
}

/// Probabilistic forecast metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbabilisticMetrics {
    pub brier_score: f32,
    pub brier_score_decomposition: BrierDecomposition,
    pub reliability: Vec<(f32, f32)>, // (predicted_prob, observed_freq)
    pub resolution: f32,
    pub uncertainty: f32,
    pub roc_auc: f32,
    pub reliability_by_lead: HashMap<u32, Vec<(f32, f32)>>,
}

/// Brier score decomposition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrierDecomposition {
    pub reliability: f32,
    pub resolution: f32,
    pub uncertainty: f32,
}

/// Ensemble metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnsembleMetrics {
    pub crps: f32,
    pub spread: f32,
    pub spread_skill_ratio: f32,
    pub rank_histogram: Vec<usize>,
    pub crps_by_lead: HashMap<u32, f32>,
}

/// Metric calculator
pub struct MetricCalculator;

impl MetricCalculator {
    /// Calculate deterministic metrics
    pub fn deterministic(forecasts: &[f32], observations: &[f32]) -> DeterministicMetrics {
        assert_eq!(forecasts.len(), observations.len());
        let n = forecasts.len() as f32;

        let mut mae = 0.0;
        let mut mse = 0.0;
        let mut bias = 0.0;
        let mut sum_f = 0.0;
        let mut sum_o = 0.0;
        let mut sum_fo = 0.0;
        let mut sum_f2 = 0.0;
        let mut sum_o2 = 0.0;

        for (&f, &o) in forecasts.iter().zip(observations.iter()) {
            let diff = f - o;
            mae += diff.abs();
            mse += diff * diff;
            bias += diff;

            sum_f += f;
            sum_o += o;
            sum_fo += f * o;
            sum_f2 += f * f;
            sum_o2 += o * o;
        }

        let mean_f = sum_f / n;
        let mean_o = sum_o / n;

        let correlation = if n > 1.0 {
            let cov = sum_fo / n - mean_f * mean_o;
            let std_f = (sum_f2 / n - mean_f * mean_f).sqrt();
            let std_o = (sum_o2 / n - mean_o * mean_o).sqrt();
            if std_f > 0.0 && std_o > 0.0 {
                cov / (std_f * std_o)
            } else {
                0.0
            }
        } else {
            0.0
        };

        DeterministicMetrics {
            mae: mae / n,
            rmse: (mse / n).sqrt(),
            bias: bias / n,
            correlation,
            ..Default::default()
        }
    }

    /// Calculate Brier score for binary events
    pub fn brier_score(probabilities: &[f32], outcomes: &[bool]) -> (f32, BrierDecomposition) {
        assert_eq!(probabilities.len(), outcomes.len());
        let n = probabilities.len() as f32;

        let mut bs = 0.0;
        for (&p, &o) in probabilities.iter().zip(outcomes.iter()) {
            let obs = if o { 1.0 } else { 0.0 };
            bs += (p - obs).powi(2);
        }
        bs /= n;

        // Decomposition (simplified)
        let climatology = outcomes.iter().filter(|&&o| o).count() as f32 / n;
        let uncertainty = climatology * (1.0 - climatology);

        // Bin probabilities for reliability
        let mut bins: Vec<Vec<(f32, bool)>> = vec![Vec::new(); 10];
        for (&p, &o) in probabilities.iter().zip(outcomes.iter()) {
            let bin = (p * 10.0).floor() as usize;
            bins[bin.min(9)].push((p, o));
        }

        let mut reliability = 0.0;
        let mut resolution = 0.0;

        for bin in bins {
            if !bin.is_empty() {
                let bin_n = bin.len() as f32;
                let mean_p = bin.iter().map(|(p, _)| p).sum::<f32>() / bin_n;
                let mean_o = bin
                    .iter()
                    .map(|(_, o)| if *o { 1.0 } else { 0.0 })
                    .sum::<f32>()
                    / bin_n;
                reliability += bin_n * (mean_p - mean_o).powi(2);
                resolution += bin_n * (mean_o - climatology).powi(2);
            }
        }
        reliability /= n;
        resolution /= n;

        let decomposition = BrierDecomposition {
            reliability,
            resolution,
            uncertainty,
        };

        (bs, decomposition)
    }

    /// Calculate reliability diagram data
    pub fn reliability_diagram(
        probabilities: &[f32],
        outcomes: &[bool],
        n_bins: usize,
    ) -> Vec<(f32, f32)> {
        assert_eq!(probabilities.len(), outcomes.len());

        let mut bins: Vec<Vec<(f32, bool)>> = vec![Vec::new(); n_bins];
        for (&p, &o) in probabilities.iter().zip(outcomes.iter()) {
            let bin = ((p * n_bins as f32).floor() as usize).min(n_bins - 1);
            bins[bin].push((p, o));
        }

        bins.into_iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let mean_p = b.iter().map(|(p, _)| p).sum::<f32>() / b.len() as f32;
                let mean_o = b
                    .iter()
                    .map(|(_, o)| if *o { 1.0 } else { 0.0 })
                    .sum::<f32>()
                    / b.len() as f32;
                (mean_p, mean_o)
            })
            .collect()
    }

    /// Calculate CRPS for ensemble forecasts
    pub fn crps(ensemble: &[Array1<f32>], observation: f32) -> f32 {
        // CRPS = E|X - y| - 0.5 * E|X - X'|
        // where X, X' are independent ensemble members
        let n = ensemble.len();
        if n == 0 {
            return 0.0;
        }

        // First term: mean absolute error of ensemble mean
        let ensemble_mean = ensemble
            .iter()
            .map(|e| e.mean().unwrap_or(0.0))
            .sum::<f32>()
            / n as f32;
        let term1 = (ensemble_mean - observation).abs();

        // Second term: mean pairwise difference
        let mut term2 = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let diff = (&ensemble[i] - &ensemble[j])
                    .mapv(|x| x.abs())
                    .mean()
                    .unwrap_or(0.0);
                term2 += diff;
            }
        }
        term2 = if n > 1 {
            term2 / (n * (n - 1) / 2) as f32
        } else {
            0.0
        };

        term1 - 0.5 * term2
    }

    /// Calculate rank histogram
    pub fn rank_histogram(ensemble: &[Array1<f32>], observation: f32) -> Vec<usize> {
        let n = ensemble.len();
        let mut ranks = vec![0; n + 1];

        // Sort ensemble
        let mut sorted: Vec<f32> = ensemble.iter().map(|e| e.mean().unwrap_or(0.0)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find rank of observation
        let rank = sorted.iter().position(|&x| x >= observation).unwrap_or(n);
        ranks[rank] += 1;

        ranks
    }

    /// Calculate ROC AUC
    pub fn roc_auc(probabilities: &[f32], outcomes: &[bool]) -> f32 {
        let mut pairs: Vec<(f32, bool)> = probabilities
            .iter()
            .zip(outcomes.iter())
            .map(|(&p, &o)| (p, o))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Descending probability

        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_p = outcomes.iter().filter(|&&o| o).count() as f32;
        let total_n = outcomes.iter().filter(|&&o| !o).count() as f32;

        if total_p == 0.0 || total_n == 0.0 {
            return 0.5;
        }

        let mut auc = 0.0;
        let mut prev_fp = 0.0;
        let mut prev_tp = 0.0;

        for (_, outcome) in pairs {
            if outcome {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            if fp != prev_fp {
                auc += (tp + prev_tp) * (fp - prev_fp) / 2.0;
                prev_fp = fp;
                prev_tp = tp;
            }
        }

        auc / (total_p * total_n)
    }
}

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_metrics() {
        let forecasts = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let observations = vec![11.0, 11.0, 12.0, 12.0, 10.0];

        let metrics = MetricCalculator::deterministic(&forecasts, &observations);

        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.correlation.abs() <= 1.0);
    }

    #[test]
    fn test_brier_score() {
        let probabilities = vec![0.8, 0.3, 0.9, 0.1, 0.6];
        let outcomes = vec![true, false, true, false, true];

        let (bs, decomp) = MetricCalculator::brier_score(&probabilities, &outcomes);

        assert!((0.0..=1.0).contains(&bs));
        assert!(decomp.uncertainty >= 0.0);
    }

    #[test]
    fn test_reliability_diagram() {
        let probabilities = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let outcomes = vec![false, false, true, true, true, true];

        let diagram = MetricCalculator::reliability_diagram(&probabilities, &outcomes, 5);
        assert!(!diagram.is_empty());
    }
}
