//! # Fusion Layer
//!
//! Penggabungan ensembel dari banyak sumber data dengan mencegah double-counting.
//! - SourceRegistry mencegah sumber dependensi (hgefs vs gefs+aigefs, firms vs sipongi).
//! - Open-Meteo dan ERA5 diekspor dari fusion real-time.
//! - Bobot ensembel: Σw = 1, w ≥ 0.
//! - Probabilitas ensembel: p = (1/M) · Σ(1[x_m > T]) — hazard pada setiap anggota, baru proporsi.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wi_core::error::WIError;
use wi_core::types::{
    DependencyGroup, FusionInput, SourceId, SourceMetadata, SourceRegistry, SourceRole,
    WeatherVariable,
};

/// Hasil fusisi ensembel per variabel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedVariable {
    pub variable: WeatherVariable,
    /// Nilai weighted mean dari semua anggota
    pub weighted_mean: f32,
    /// Probabilitas melewati ambang T (jika variabel memiliki ambang)
    pub exceedance_probability: Option<f32>,
    /// Ambang yang digunakan
    pub threshold: Option<f32>,
    /// Jumlah anggota ensemble yang digunakan
    pub member_count: usize,
    /// Sumber yang berkontribusi
    pub source_ids: Vec<SourceId>,
    /// Bias telah diterapkan?
    pub bias_applied: bool,
}

/// Hasil fusisi lengkap untuk semua variabel pada satu titik/grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionOutput {
    pub variables: HashMap<WeatherVariable, FusedVariable>,
    pub total_members: usize,
    pub sources_used: Vec<SourceId>,
    pub model_version: String,
    pub computed_at: DateTime<Utc>,
    /// Kontrak output Andronikos
    pub unit_id: Option<String>,
    pub jenis_unit: Option<String>,
    pub modul: Option<String>,
    pub terbit: Option<DateTime<Utc>>,
    pub berlaku_dari: Option<DateTime<Utc>>,
    pub berlaku_sampai: Option<DateTime<Utc>>,
    pub lead_seconds: u32,
    pub probabilitas: Option<f32>,
    pub tingkat_siaga: Option<String>,
    pub anggota: usize,
    pub versi_model: String,
    pub koreksi_bias: bool,
    pub sumber_id: Vec<String>,
    pub atribusi: Vec<String>,
    pub label_istilah: String,
    pub klaim: Vec<String>,
}

impl FusionOutput {
    pub fn new(model_version: String) -> Self {
        Self {
            variables: HashMap::new(),
            total_members: 0,
            sources_used: Vec::new(),
            model_version: model_version.clone(),
            computed_at: Utc::now(),
            unit_id: None,
            jenis_unit: None,
            modul: None,
            terbit: None,
            berlaku_dari: None,
            berlaku_sampai: None,
            lead_seconds: 0,
            probabilitas: None,
            tingkat_siaga: None,
            anggota: 0,
            versi_model: model_version,
            koreksi_bias: false,
            sumber_id: Vec::new(),
            atribusi: Vec::new(),
            label_istilah: "analisis pendukung".to_string(),
            klaim: Vec::new(),
        }
    }

    pub fn insert(&mut self, variable: WeatherVariable, fused: FusedVariable) {
        self.variables.insert(variable, fused);
    }

    pub fn get(&self, variable: &WeatherVariable) -> Option<&FusedVariable> {
        self.variables.get(variable)
    }

    /// Set semua field kontrak output sekaligus.
    #[allow(clippy::too_many_arguments)]
    pub fn set_contract(
        &mut self,
        unit_id: Option<String>,
        jenis_unit: Option<String>,
        modul: Option<String>,
        terbit: Option<DateTime<Utc>>,
        berlaku_dari: Option<DateTime<Utc>>,
        berlaku_sampai: Option<DateTime<Utc>>,
        lead_seconds: u32,
        probabilitas: Option<f32>,
        tingkat_siaga: Option<String>,
        anggota: usize,
        koreksi_bias: bool,
        sumber_id: Vec<String>,
        atribusi: Vec<String>,
        label_istilah: String,
        klaim: Vec<String>,
    ) {
        self.unit_id = unit_id;
        self.jenis_unit = jenis_unit;
        self.modul = modul;
        self.terbit = terbit;
        self.berlaku_dari = berlaku_dari;
        self.berlaku_sampai = berlaku_sampai;
        self.lead_seconds = lead_seconds;
        self.probabilitas = probabilitas;
        self.tingkat_siaga = tingkat_siaga;
        self.anggota = anggota;
        self.koreksi_bias = koreksi_bias;
        self.sumber_id = sumber_id;
        self.atribusi = atribusi;
        self.label_istilah = label_istilah;
        self.klaim = klaim;
    }
}

/// Engine fusisi ensembel.
#[derive(Debug, Clone)]
pub struct FusionEngine {
    registry: SourceRegistry,
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            registry: SourceRegistry::new(),
        }
    }

    /// Daftarkan semua sumber standar sesuai registri_variabel_v2_5.yaml.
    pub fn register_standard_sources(&mut self) -> Result<(), WIError> {
        // NOAA GFS — BasePhysics, Independent
        self.registry.register(SourceMetadata {
            id: SourceId::MetGfs,
            role: SourceRole::BasePhysics,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.93,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA NOMADS",
        });

        // NOAA GEFS — BaseEnsemble, NoaaGefsFamily
        self.registry.register(SourceMetadata {
            id: SourceId::MetGefs,
            role: SourceRole::BaseEnsemble,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA NOMADS",
        });

        // NOAA AIGFS — AiMember, NoaaGefsFamily (turunan GFS, tapi dalam keluarga GEFS untuk dependensi)
        self.registry.register(SourceMetadata {
            id: SourceId::MetAigfs,
            role: SourceRole::AiMember,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.85,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: Some(120),
            attribution_text: "NOAA NOMADS",
        });

        // NOAA AIGEFS — AiMember, NoaaGefsFamily
        self.registry.register(SourceMetadata {
            id: SourceId::MetAigefs,
            role: SourceRole::AiMember,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA NOMADS",
        });

        // NOAA HGEFS — Excluded, NoaaGefsFamily (gabungan, tidak independen)
        self.registry.register(SourceMetadata {
            id: SourceId::MetHgefs,
            role: SourceRole::Excluded,
            dependency_group: DependencyGroup::NoaaGefsFamily,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "NOAA NOMADS",
        });

        // BMKG prakiraan terbuka — LocalMember, Independent
        self.registry.register(SourceMetadata {
            id: SourceId::MetBmkgFc,
            role: SourceRole::LocalMember,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.95,
            requires_written_permission: true,
            commercial_allowed: false,
            rate_limit_per_min: Some(60),
            attribution_text: "BMKG (atribusi wajib, UU 31/2009)",
        });

        // BMKG nowcast — ComparatorOnly, Independent (bukan input model)
        self.registry.register(SourceMetadata {
            id: SourceId::MetBmkgNowcast,
            role: SourceRole::ComparatorOnly,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.95,
            requires_written_permission: true,
            commercial_allowed: false,
            rate_limit_per_min: Some(60),
            attribution_text: "BMKG CAP",
        });

        // GraphCast — Excluded, GraphCastNonCommercial (CC BY-NC-SA)
        self.registry.register(SourceMetadata {
            id: SourceId::MetGraphcast,
            role: SourceRole::Excluded,
            dependency_group: DependencyGroup::GraphCastNonCommercial,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "DeepMind (CC BY-NC-SA 4.0, tidak komersial)",
        });

        // Open-Meteo — ComparatorOnly, OpenMeteoAggregator (diekspor dari fusion)
        self.registry.register(SourceMetadata {
            id: SourceId::MetOpenMeteo,
            role: SourceRole::ComparatorOnly,
            dependency_group: DependencyGroup::OpenMeteoAggregator,
            reliability_score: 0.88,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "Open-Meteo CC BY 4.0",
        });

        // ERA5 — TrainingOnly, Era5Reanalysis (bukan real-time input)
        self.registry.register(SourceMetadata {
            id: SourceId::RefEra5,
            role: SourceRole::TrainingOnly,
            dependency_group: DependencyGroup::Era5Reanalysis,
            reliability_score: 0.98,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "Copernicus/ECMWF ERA5",
        });

        // Observasi BMKG stasiun — ObservationOnly, Independent
        self.registry.register(SourceMetadata {
            id: SourceId::ObsBmkgStn,
            role: SourceRole::ObservationOnly,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.95,
            requires_written_permission: true,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "BMKG pengamatan stasiun",
        });

        // NOAA ISD — ObservationOnly, Independent
        self.registry.register(SourceMetadata {
            id: SourceId::ObsIsd,
            role: SourceRole::ObservationOnly,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.90,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NOAA ISD",
        });

        // FIRMS — ModuleSpecific, FireHotspotFamily
        self.registry.register(SourceMetadata {
            id: SourceId::FireFirms,
            role: SourceRole::ModuleSpecific,
            dependency_group: DependencyGroup::FireHotspotFamily,
            reliability_score: 0.92,
            requires_written_permission: false,
            commercial_allowed: true,
            rate_limit_per_min: None,
            attribution_text: "NASA FIRMS (acknowledgment wajib)",
        });

        // SiPongi+ — ModuleSpecific, FireHotspotFamily (sensor sama dengan FIRMS)
        self.registry.register(SourceMetadata {
            id: SourceId::FireSipongi,
            role: SourceRole::ModuleSpecific,
            dependency_group: DependencyGroup::FireHotspotFamily,
            reliability_score: 0.85,
            requires_written_permission: true,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "Kementerian Kehutanan SiPongi+",
        });

        // BRIN hotspot — ModuleSpecific, Independent
        self.registry.register(SourceMetadata {
            id: SourceId::FireBrin,
            role: SourceRole::ModuleSpecific,
            dependency_group: DependencyGroup::Independent,
            reliability_score: 0.85,
            requires_written_permission: false,
            commercial_allowed: false,
            rate_limit_per_min: None,
            attribution_text: "BRIN hotspot",
        });

        Ok(())
    }

    /// Pilih sumber untuk fusion. Mengembalikan error jika double-counting terdeteksi.
    pub fn select_sources(&mut self, source_ids: &[SourceId]) -> Result<Vec<SourceId>, WIError> {
        let mut selected = Vec::new();
        for &id in source_ids {
            self.registry.select(id)?;
            selected.push(id);
        }
        Ok(selected)
    }

    /// Fuse anggota ensemble menjadi output tunggal per variabel.
    /// - Weighted mean: Σ(w_i · x_i)
    /// - Exceedance probability: p = (1/M) · Σ(1[x_m > T])
    pub fn fuse_ensemble(
        &self,
        input: &FusionInput,
        variable: WeatherVariable,
        values_by_member: &[(f32, bool)], // (nilai, melewati_ambang)
        threshold: Option<f32>,
    ) -> Result<FusedVariable, WIError> {
        if input.members.is_empty() {
            return Err(WIError::validation(
                "Tidak ada anggota ensemble untuk difuse",
            ));
        }

        if input.members.len() != values_by_member.len() {
            return Err(WIError::validation(format!(
                "Jumlah anggota ({}) tidak cocok dengan jumlah nilai ({})",
                input.members.len(),
                values_by_member.len()
            )));
        }

        // Weighted mean
        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;
        for (i, member) in input.members.iter().enumerate() {
            if i < values_by_member.len() {
                let (val, _) = values_by_member[i];
                weighted_sum += member.weight * val;
                total_weight += member.weight;
            }
        }

        let weighted_mean = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Exceedance probability: p = (1/M) · Σ(1[x_m > T])
        let exceedance_probability = threshold.map(|_t| {
            let count_exceed = values_by_member
                .iter()
                .filter(|&(_, exceeds)| *exceeds)
                .count();
            count_exceed as f32 / values_by_member.len() as f32
        });

        let member_count = input.members.len();
        let source_ids: Vec<SourceId> = input.members.iter().map(|m| m.source_id).collect();
        let bias_applied = input.members.iter().all(|m| m.bias_applied);

        Ok(FusedVariable {
            variable,
            weighted_mean,
            exceedance_probability,
            threshold,
            member_count,
            source_ids,
            bias_applied,
        })
    }

    /// Hitung probabilitas ensembel dari banyak anggota dengan threshold.
    /// p = (1/M) * Σ(1[x_m > T])
    /// Hazard function dijalankan pada setiap anggota, baru proporsi dihitung.
    pub fn ensemble_exceedance_probability(values: &[f32], threshold: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let count_exceed = values.iter().filter(|&&v| v > threshold).count();
        count_exceed as f32 / values.len() as f32
    }

    /// Validasi bahwa total weight = 1.0 (dengan toleransi).
    pub fn validate_weights(input: &FusionInput) -> Result<(), WIError> {
        if (input.total_weight - 1.0).abs() > 1e-4 {
            return Err(WIError::validation(format!(
                "Total weight ensemble harus 1.0, got {:.6}",
                input.total_weight
            )));
        }
        for m in &input.members {
            if m.weight < 0.0 || m.weight > 1.0 {
                return Err(WIError::validation(format!(
                    "Weight anggota {} di luar rentang [0,1]: {}",
                    m.source_id.as_str(),
                    m.weight
                )));
            }
        }
        Ok(())
    }

    pub fn registry(&self) -> &SourceRegistry {
        &self.registry
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wi_core::types::EnsembleMember;

    #[test]
    fn test_register_standard_sources() {
        let mut engine = FusionEngine::new();
        engine.register_standard_sources().unwrap();

        // Open-Meteo dan HGEFS tidak boleh dipilih untuk fusion
        assert!(engine.registry.select(SourceId::MetOpenMeteo).is_err());
        assert!(engine.registry.select(SourceId::MetHgefs).is_err());
        // ERA5 tidak boleh real-time input
        assert!(engine.registry.select(SourceId::RefEra5).is_err());
        // GFS dan GEFS boleh (group berbeda)
        assert!(engine.registry.select(SourceId::MetGfs).is_ok());
        assert!(engine.registry.select(SourceId::MetGefs).is_ok());
        // AIGEFS tidak boleh (NoaaGefsFamily sama dengan GEFS)
        assert!(engine.registry.select(SourceId::MetAigefs).is_err());
        // FIRMS boleh, SiPongi tidak (FireHotspotFamily sama)
        assert!(engine.registry.select(SourceId::FireFirms).is_ok());
        assert!(engine.registry.select(SourceId::FireSipongi).is_err());
    }

    #[test]
    fn test_fuse_ensemble_weighted_mean() {
        let mut input = FusionInput::new();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGfs,
                member_index: 0,
                weight: 0.6,
                version: "gfs-1.0".to_string(),
                bias_applied: true,
            })
            .unwrap();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGefs,
                member_index: 0,
                weight: 0.4,
                version: "gefs-1.0".to_string(),
                bias_applied: true,
            })
            .unwrap();
        input.normalize_weights().unwrap();

        let engine = FusionEngine::new();
        // GFS: 30, GEFS: 32 → weighted = 0.6*30 + 0.4*32 = 30.8
        let values = vec![(30.0, true), (32.0, false)];
        let fused = engine
            .fuse_ensemble(&input, WeatherVariable::Temperature, &values, Some(31.0))
            .unwrap();

        assert!((fused.weighted_mean - 30.8).abs() < 1e-6);
        // p = 1/2 * (1[30>31] + 1[32>31]) = 1/2 * (0 + 1) = 0.5
        assert_eq!(fused.exceedance_probability, Some(0.5));
        assert_eq!(fused.member_count, 2);
        assert!(fused.bias_applied);
    }

    #[test]
    fn test_ensemble_exceedance_probability() {
        // 3 dari 5 anggota melewati ambang 20
        let values = vec![15.0, 25.0, 18.0, 22.0, 19.0];
        let p = FusionEngine::ensemble_exceedance_probability(&values, 20.0);
        assert!((p - 0.4).abs() < 1e-6);

        // Kosong → 0
        let p = FusionEngine::ensemble_exceedance_probability(&[], 20.0);
        assert_eq!(p, 0.0);

        // Semua melewati
        let values = vec![25.0, 30.0, 35.0];
        let p = FusionEngine::ensemble_exceedance_probability(&values, 20.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_fusion_output_creation() {
        let output = FusionOutput::new("fusion-v1".to_string());
        assert_eq!(output.model_version, "fusion-v1");
        assert_eq!(output.total_members, 0);
        assert!(output.variables.is_empty());
    }

    #[test]
    fn test_validate_weights() {
        let mut input = FusionInput::new();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGfs,
                member_index: 0,
                weight: 0.5,
                version: "gfs".to_string(),
                bias_applied: false,
            })
            .unwrap();
        input
            .add_member(EnsembleMember {
                source_id: SourceId::MetGefs,
                member_index: 0,
                weight: 0.5,
                version: "gefs".to_string(),
                bias_applied: false,
            })
            .unwrap();
        input.normalize_weights().unwrap();
        FusionEngine::validate_weights(&input).unwrap();

        // Total weight != 1 (bypass add_member yang mencegah > 1.0)
        let mut bad = FusionInput::new();
        bad.members.push(EnsembleMember {
            source_id: SourceId::MetGfs,
            member_index: 0,
            weight: 0.8,
            version: "gfs".to_string(),
            bias_applied: false,
        });
        bad.members.push(EnsembleMember {
            source_id: SourceId::MetGefs,
            member_index: 0,
            weight: 0.5,
            version: "gefs".to_string(),
            bias_applied: false,
        });
        bad.total_weight = 1.3;
        let err = FusionEngine::validate_weights(&bad);
        assert!(err.is_err());
    }

    #[test]
    fn test_source_metadata_commercial() {
        let mut engine = FusionEngine::new();
        engine.register_standard_sources().unwrap();

        // GraphCast tidak komersial
        let gc = engine.registry.metadata(SourceId::MetGraphcast).unwrap();
        assert!(!gc.commercial_allowed);
        assert_eq!(gc.role, SourceRole::Excluded);

        // GFS komersial
        let gfs = engine.registry.metadata(SourceId::MetGfs).unwrap();
        assert!(gfs.commercial_allowed);
    }
}
