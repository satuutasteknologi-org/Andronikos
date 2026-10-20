//! # CSI Scorer — Categorical & Combined Verification (Trip 4)
//!
//! Mengimplementasikan metrik verifikasi kategorikal standar WMO:
//! - **CSI** (Critical Success Index) — kritis untuk presipitasi
//! - **POD** (Probability of Detection)
//! - **FAR** (False Alarm Rate / False Alarm Ratio)
//! - **ETS** (Equitable Threat Score) — CSI yang dikoreksi terhadap random hits
//! - **HSS** (Heidke Skill Score) — skill relatif terhadap random
//! - **GSS** (Gilbert Skill Score) = ETS
//!
//! ## Referensi
//! - WMO No. 49, Technical Regulations (2019), Appendix II
//! - Jolliffe & Stephenson (2012): Forecast Verification: A Practitioner's Guide
//!
//! ## Penggunaan
//! ```rust
//! use wi_verify::csi::{ContingencyTable, CsiScorer};
//!
//! let table = ContingencyTable { hits: 80, misses: 20, false_alarms: 15, correct_negatives: 300 };
//! let scores = CsiScorer::compute(&table);
//! println!("CSI={:.3} ETS={:.3} HSS={:.3}", scores.csi, scores.ets, scores.hss);
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Contingency Table
// ─────────────────────────────────────────────────────────────────────────────

/// Tabel kontingensi 2×2 untuk verifikasi biner (ya/tidak)
///
/// ```text
///                    Observed
///                 Yes        No
/// Forecast  Yes | hits  | false_alarms |
///           No  | misses | correct_neg  |
/// ```
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ContingencyTable {
    /// A: forecast=YES, observed=YES
    pub hits: u64,
    /// C: forecast=NO, observed=YES
    pub misses: u64,
    /// B: forecast=YES, observed=NO
    pub false_alarms: u64,
    /// D: forecast=NO, observed=NO
    pub correct_negatives: u64,
}

impl ContingencyTable {
    pub fn total(&self) -> u64 {
        self.hits + self.misses + self.false_alarms + self.correct_negatives
    }

    /// Hitung dari pasangan (forecast_exceeded, observed_exceeded) dengan threshold
    pub fn from_pairs(forecasts: &[f32], observations: &[f32], threshold: f32) -> Self {
        assert_eq!(forecasts.len(), observations.len());
        let mut table = Self::default();
        for (&f, &o) in forecasts.iter().zip(observations.iter()) {
            let fc_yes = f >= threshold;
            let ob_yes = o >= threshold;
            match (fc_yes, ob_yes) {
                (true, true) => table.hits += 1,
                (true, false) => table.false_alarms += 1,
                (false, true) => table.misses += 1,
                (false, false) => table.correct_negatives += 1,
            }
        }
        table
    }

    /// Merge tabel lain ke dalam self (akumulasi)
    pub fn merge(&mut self, other: &Self) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.false_alarms += other.false_alarms;
        self.correct_negatives += other.correct_negatives;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CategoricalScores
// ─────────────────────────────────────────────────────────────────────────────

/// Semua skor kategorikal dari satu contingency table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalScores {
    /// Threshold yang digunakan (mis. 1 mm/h untuk hujan)
    pub threshold: f32,
    /// Critical Success Index = hits / (hits + misses + false_alarms)
    /// Range: [0,1], perfect=1
    pub csi: f32,
    /// Probability of Detection = hits / (hits + misses)
    /// Range: [0,1], perfect=1
    pub pod: f32,
    /// False Alarm Rate = false_alarms / (false_alarms + correct_negatives)
    /// Range: [0,1], perfect=0
    pub far: f32,
    /// False Alarm Ratio (berbeda dari FAR) = false_alarms / (hits + false_alarms)
    pub false_alarm_ratio: f32,
    /// Equitable Threat Score (= Gilbert Skill Score)
    /// Range: [-1/3, 1], perfect=1, random=0
    pub ets: f32,
    /// Heidke Skill Score
    /// Range: [-inf, 1], perfect=1, random=0
    pub hss: f32,
    /// Bias (frequency bias) = (hits + false_alarms) / (hits + misses)
    /// Perfect=1, >1 over-forecast, <1 under-forecast
    pub frequency_bias: f32,
    /// Total sample count
    pub sample_count: u64,
}

impl CategoricalScores {
    /// Return true jika sample cukup untuk skor yang valid
    pub fn is_reliable(&self) -> bool {
        self.sample_count >= 30
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CsiScorer
// ─────────────────────────────────────────────────────────────────────────────

/// Kalkulator skor kategorikal dari ContingencyTable
pub struct CsiScorer;

impl CsiScorer {
    /// Hitung semua skor kategorikal dari contingency table dan threshold
    pub fn compute_with_threshold(table: &ContingencyTable, threshold: f32) -> CategoricalScores {
        let h = table.hits as f32;
        let m = table.misses as f32;
        let fa = table.false_alarms as f32;
        let cn = table.correct_negatives as f32;
        let n = h + m + fa + cn;

        // CSI = H / (H + M + FA)
        let csi = if h + m + fa > 0.0 {
            h / (h + m + fa)
        } else {
            f32::NAN
        };

        // POD = H / (H + M)
        let pod = if h + m > 0.0 { h / (h + m) } else { f32::NAN };

        // FAR (rate) = FA / (FA + CN)
        let far = if fa + cn > 0.0 {
            fa / (fa + cn)
        } else {
            f32::NAN
        };

        // False Alarm Ratio = FA / (H + FA)
        let false_alarm_ratio = if h + fa > 0.0 {
            fa / (h + fa)
        } else {
            f32::NAN
        };

        // Frequency Bias = (H + FA) / (H + M)
        let frequency_bias = if h + m > 0.0 {
            (h + fa) / (h + m)
        } else {
            f32::NAN
        };

        // Hits_random = (H+FA)(H+M) / N
        let hits_random = if n > 0.0 { (h + fa) * (h + m) / n } else { 0.0 };

        // ETS = (H - hits_random) / (H + M + FA - hits_random)
        let ets = {
            let denom = h + m + fa - hits_random;
            if denom.abs() > 1e-6 {
                (h - hits_random) / denom
            } else {
                f32::NAN
            }
        };

        // HSS = 2*(H*CN - M*FA) / ((H+M)*(M+CN) + (H+FA)*(FA+CN))
        let hss = {
            let num = 2.0 * (h * cn - m * fa);
            let denom = (h + m) * (m + cn) + (h + fa) * (fa + cn);
            if denom.abs() > 1e-6 {
                num / denom
            } else {
                f32::NAN
            }
        };

        CategoricalScores {
            threshold,
            csi,
            pod,
            far,
            false_alarm_ratio,
            ets,
            hss,
            frequency_bias,
            sample_count: table.total(),
        }
    }

    /// Hitung dari pasangan nilai dan threshold langsung
    pub fn from_pairs(
        forecasts: &[f32],
        observations: &[f32],
        threshold: f32,
    ) -> CategoricalScores {
        let table = ContingencyTable::from_pairs(forecasts, observations, threshold);
        Self::compute_with_threshold(&table, threshold)
    }

    /// Hitung CSI untuk berbagai threshold sekaligus (berguna untuk kurva ROC)
    pub fn multi_threshold(
        forecasts: &[f32],
        observations: &[f32],
        thresholds: &[f32],
    ) -> Vec<CategoricalScores> {
        thresholds
            .iter()
            .map(|&t| Self::from_pairs(forecasts, observations, t))
            .collect()
    }

    /// Hitung CSI standar hujan Indonesia (WMO threshold)
    /// - Trace: ≥ 0.1 mm/h
    /// - Light:  ≥ 1.0 mm/h
    /// - Moderate: ≥ 5.0 mm/h
    /// - Heavy:  ≥ 10.0 mm/h
    pub fn rainfall_csi(forecasts: &[f32], observations: &[f32]) -> RainfallCsiReport {
        RainfallCsiReport {
            trace: Self::from_pairs(forecasts, observations, 0.1),
            light: Self::from_pairs(forecasts, observations, 1.0),
            moderate: Self::from_pairs(forecasts, observations, 5.0),
            heavy: Self::from_pairs(forecasts, observations, 10.0),
        }
    }
}

/// Laporan CSI curah hujan multi-threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RainfallCsiReport {
    pub trace: CategoricalScores,
    pub light: CategoricalScores,
    pub moderate: CategoricalScores,
    pub heavy: CategoricalScores,
}

impl RainfallCsiReport {
    /// Return true jika semua threshold mencapai target minimum CSI
    pub fn meets_minimum_skill(&self, min_csi: f32) -> bool {
        [&self.trace, &self.light, &self.moderate, &self.heavy]
            .iter()
            .filter(|s| s.is_reliable())
            .all(|s| !s.csi.is_nan() && s.csi >= min_csi)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_table() -> ContingencyTable {
        ContingencyTable {
            hits: 100,
            misses: 0,
            false_alarms: 0,
            correct_negatives: 200,
        }
    }

    fn random_table() -> ContingencyTable {
        ContingencyTable {
            hits: 50,
            misses: 50,
            false_alarms: 50,
            correct_negatives: 150,
        }
    }

    #[test]
    fn test_perfect_csi() {
        let scores = CsiScorer::compute_with_threshold(&perfect_table(), 1.0);
        assert!((scores.csi - 1.0).abs() < 1e-4, "Perfect CSI should be 1.0");
        assert!((scores.pod - 1.0).abs() < 1e-4, "Perfect POD should be 1.0");
        assert!(scores.far.abs() < 1e-4, "Perfect FAR should be 0.0");
        assert!((scores.ets - 1.0).abs() < 1e-3, "Perfect ETS should be 1.0");
    }

    #[test]
    fn test_csi_range() {
        let scores = CsiScorer::compute_with_threshold(&random_table(), 1.0);
        assert!(
            scores.csi >= 0.0 && scores.csi <= 1.0,
            "CSI must be in [0,1]"
        );
        assert!(scores.pod >= 0.0 && scores.pod <= 1.0);
        assert!(scores.far >= 0.0 && scores.far <= 1.0);
    }

    #[test]
    fn test_from_pairs() {
        let fc = vec![0.0, 2.0, 8.0, 0.5, 12.0];
        let ob = vec![0.0, 1.5, 7.0, 0.0, 3.0];
        let scores = CsiScorer::from_pairs(&fc, &ob, 1.0);
        assert!(scores.sample_count == 5);
        assert!(!scores.csi.is_nan());
    }

    #[test]
    fn test_contingency_from_pairs() {
        let fc = vec![5.0, 0.5, 8.0, 0.1];
        let ob = vec![4.0, 0.2, 0.5, 3.0];
        let table = ContingencyTable::from_pairs(&fc, &ob, 1.0);
        // fc[0]=5≥1, ob[0]=4≥1 → hit
        // fc[1]=0.5<1, ob[1]=0.2<1 → correct_neg
        // fc[2]=8≥1, ob[2]=0.5<1 → false_alarm
        // fc[3]=0.1<1, ob[3]=3≥1 → miss
        assert_eq!(table.hits, 1);
        assert_eq!(table.misses, 1);
        assert_eq!(table.false_alarms, 1);
        assert_eq!(table.correct_negatives, 1);
    }

    #[test]
    fn test_rainfall_csi_report() {
        let fc = vec![0.5, 2.0, 6.0, 11.0];
        let ob = vec![0.2, 1.5, 5.5, 10.5];
        let report = CsiScorer::rainfall_csi(&fc, &ob);
        // trace: 0.5≥0.1 yes, 0.2≥0.1 yes → hit. Etc.
        assert!(report.trace.sample_count == 4);
        assert!(!report.light.csi.is_nan());
    }

    #[test]
    fn test_contingency_merge() {
        let mut t1 = ContingencyTable {
            hits: 10,
            misses: 5,
            false_alarms: 3,
            correct_negatives: 20,
        };
        let t2 = ContingencyTable {
            hits: 5,
            misses: 2,
            false_alarms: 1,
            correct_negatives: 10,
        };
        t1.merge(&t2);
        assert_eq!(t1.hits, 15);
        assert_eq!(t1.total(), 56);
    }

    #[test]
    fn test_multi_threshold() {
        let fc = vec![1.0, 5.0, 0.0, 10.0];
        let ob = vec![0.5, 4.0, 0.0, 12.0];
        let scores = CsiScorer::multi_threshold(&fc, &ob, &[0.1, 1.0, 5.0]);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_frequency_bias_overforecast() {
        // More false alarms → bias > 1
        let table = ContingencyTable {
            hits: 50,
            misses: 10,
            false_alarms: 60,
            correct_negatives: 100,
        };
        let scores = CsiScorer::compute_with_threshold(&table, 1.0);
        assert!(
            scores.frequency_bias > 1.0,
            "Should be over-forecast (bias > 1)"
        );
    }
}
