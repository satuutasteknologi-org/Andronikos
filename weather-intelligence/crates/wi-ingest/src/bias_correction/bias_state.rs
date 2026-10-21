//! Bias state per variabel × lead × siklus UTC × stasiun × musim.
//!
//! Tidak ada interior mutability — semua operasi melalui immutable reference.
//! Caller bertanggung jawab atas sinkronisasi (misalnya Arc<tokio::sync::RwLock>).

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wi_core::error::WIError;
use wi_core::types::WeatherVariable;

/// Kunci unik untuk state koreksi bias per grup.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BiasKey {
    pub variable: WeatherVariable,
    pub lead_seconds: u32,
    /// Siklus UTC: 0, 6, 12, atau 18
    pub cycle_hour: u8,
    pub station_id: String,
    /// Musim (opsional, hanya jika data cukup): 0=DJF, 1=MAM, 2=JJA, 3=SON
    pub season: Option<u8>,
}

impl BiasKey {
    pub fn from_utc(
        time: DateTime<Utc>,
        station_id: &str,
        variable: WeatherVariable,
        lead_seconds: u32,
    ) -> Self {
        let hour = time.hour() as u8;
        let cycle_hour = (hour / 6) * 6; // 0, 6, 12, 18
        let month = time.month();
        let season = if (3..=5).contains(&month) {
            1u8 // MAM
        } else if (6..=8).contains(&month) {
            2u8 // JJA
        } else if (9..=11).contains(&month) {
            3u8 // SON
        } else {
            0u8 // DJF
        };
        Self {
            variable,
            lead_seconds,
            cycle_hour,
            station_id: station_id.to_string(),
            season: Some(season),
        }
    }

    pub fn without_season(
        station_id: &str,
        variable: WeatherVariable,
        lead_seconds: u32,
        cycle_hour: u8,
    ) -> Self {
        Self {
            variable,
            lead_seconds,
            cycle_hour,
            station_id: station_id.to_string(),
            season: None,
        }
    }
}

/// Nilai alpha dan panjang jendela — BELUM DIPUTUSKAN, kode harus mengembalikan error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BiasUpdateConfig {
    pub alpha: Option<f32>,
    pub window_length_hours: Option<u32>,
}

impl BiasUpdateConfig {
    pub fn new() -> Self {
        Self {
            alpha: None,
            window_length_hours: None,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) -> Result<(), WIError> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(WIError::validation(format!(
                "Alpha harus dalam rentang (0, 1], got {}",
                alpha
            )));
        }
        self.alpha = Some(alpha);
        Ok(())
    }

    pub fn set_window_length(&mut self, hours: u32) -> Result<(), WIError> {
        if hours == 0 {
            return Err(WIError::validation("Panjang jendela harus > 0 jam"));
        }
        self.window_length_hours = Some(hours);
        Ok(())
    }

    pub fn require_alpha(&self) -> Result<f32, WIError> {
        self.alpha.ok_or_else(|| {
            WIError::belum_diputuskan(
                "alpha",
                "nilai alpha belum ditetapkan lewat validasi historis",
            )
        })
    }

    pub fn require_window_length(&self) -> Result<u32, WIError> {
        self.window_length_hours.ok_or_else(|| {
            WIError::belum_diputuskan(
                "panjang_jendela",
                "panjang jendela koreksi bias belum ditetapkan lewat validasi historis",
            )
        })
    }
}

/// State bias per kunci: nilai b dan jumlah sampel.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BiasState {
    pub b: f32,
    pub sample_count: u64,
    pub last_updated: Option<DateTime<Utc>>,
}

impl BiasState {
    pub fn new() -> Self {
        Self {
            b: 0.0,
            sample_count: 0,
            last_updated: None,
        }
    }

    /// Perbarui bias dengan rata-rata bergerak eksponensial:
    /// b_baru = (1 - alpha) * b_lama + alpha * (prakiraan - pengamatan)
    pub fn update(&mut self, forecast_minus_observation: f32, alpha: f32, now: DateTime<Utc>) {
        self.b = (1.0 - alpha) * self.b + alpha * forecast_minus_observation;
        self.sample_count += 1;
        self.last_updated = Some(now);
    }
}

/// Koleksi state bias, diindeks oleh BiasKey.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BiasStore {
    states: HashMap<BiasKey, BiasState>,
}

impl BiasStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &BiasKey) -> Option<&BiasState> {
        self.states.get(key)
    }

    pub fn get_or_default(&mut self, key: BiasKey) -> &mut BiasState {
        self.states.entry(key).or_default()
    }

    pub fn remove(&mut self, key: &BiasKey) -> Option<BiasState> {
        self.states.remove(key)
    }

    pub fn reset_all(&mut self) {
        self.states.clear();
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, BiasKey, BiasState> {
        self.states.iter()
    }
}
