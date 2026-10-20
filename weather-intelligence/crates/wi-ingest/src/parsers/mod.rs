//! # Format Parsers — GRIB2 & NetCDF
//!
//! Pure-Rust parser untuk format meteorologi biner.
//!
//! ## GRIB2 (GFS, GEFS, HGEFS)
//! Crate: `grib = "0.19"` (pure Rust)
//! - `grib::from_bytes()` → Grib2
//! - `submessage.indicator().discipline` → u8
//! - `submessage.prod_def().parameter_number()` → Option<(u8, u8)> (category, param)
//! - `Grib2SubmessageDecoder::from(msg)?.dispatch()` → Iterator<f32>
//! - `submessage.latlons()` → Iterator<(lat, lon)>
//!
//! ## NetCDF (ERA5 — TRAINING ONLY)
//! Crate: `netcdf-reader = "0.9"` (pure Rust)
//! - `netcdf_reader::NcFile::from_bytes()` → NcFile
//! - `.read_variable::<f64>(name)` → ArrayD<f64>
//!
//! ## Anti-Halusinasi
//! ERA5 adalah training only — fungsi `parse_era5_netcdf` mengembalikan
//! `WeatherObservation` dengan `source = DataSource::ERA5`.
//! Caller bertanggung jawab memastikan tidak masuk fusion real-time.

use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;
use tracing::{debug, warn};

use wi_core::{
    DataSource, GeoCoord, ObservedValue, QualityFlag, Result, VerticalLevel, WIError,
    WeatherForecast, WeatherObservation, WeatherVariable,
};

// ─────────────────────────────────────────────────────────────────────────────
// GRIB2 Parameter Table
// ─────────────────────────────────────────────────────────────────────────────

/// Key GRIB2: (discipline, category, parameter_number)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Grib2Param {
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
}

impl Grib2Param {
    pub const fn new(discipline: u8, category: u8, number: u8) -> Self {
        Self {
            discipline,
            category,
            number,
        }
    }
}

/// Mapping GRIB2 (discipline, category, number) → (WeatherVariable, VerticalLevel)
///
/// Referensi: WMO GRIB2 Code Table 4.2
/// <https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-0.shtml>
pub fn grib2_param_table() -> HashMap<Grib2Param, (WeatherVariable, VerticalLevel)> {
    let surf = VerticalLevel::Surface;

    // helper macro
    macro_rules! p {
        ($d:expr, $c:expr, $n:expr, $v:expr) => {
            (Grib2Param::new($d, $c, $n), ($v, surf))
        };
    }

    HashMap::from([
        // Suhu — Discipline 0, Category 0
        p!(0, 0, 0, WeatherVariable::Temperature), // TMP (K) — default 2m
        p!(0, 0, 6, WeatherVariable::DewPoint),    // DPT (K)
        // Kelembaban — Discipline 0, Category 1
        p!(0, 1, 1, WeatherVariable::RelativeHumidity), // RH (%)
        p!(0, 1, 8, WeatherVariable::Precipitation),    // PRATE (kg/m²/s)
        p!(0, 1, 52, WeatherVariable::Precipitation),   // PWAT (kg/m²)
        // Angin — Discipline 0, Category 2
        p!(0, 2, 2, WeatherVariable::WindU),         // UGRD (m/s)
        p!(0, 2, 3, WeatherVariable::WindV),         // VGRD (m/s)
        p!(0, 2, 1, WeatherVariable::WindSpeed),     // WIND (m/s)
        p!(0, 2, 0, WeatherVariable::WindDirection), // WDIR (deg)
        // Tekanan — Discipline 0, Category 3
        p!(0, 3, 0, WeatherVariable::Pressure), // PRES (Pa)
        p!(0, 3, 1, WeatherVariable::SurfacePressure), // PRMSL (Pa)
        // Awan — Discipline 0, Category 6
        p!(0, 6, 1, WeatherVariable::CloudCover), // TCDC (%)
        // Spesifik humidity — Discipline 0, Category 1
        p!(0, 1, 0, WeatherVariable::SpecificHumidity), // SPFH (kg/kg)
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Konversi GRIB2
// ─────────────────────────────────────────────────────────────────────────────

/// Konversi nilai GRIB2 ke unit standar Andronikos
pub fn convert_grib2_value(var: WeatherVariable, raw: f64) -> f64 {
    match var {
        // Kelvin → Celsius
        WeatherVariable::Temperature | WeatherVariable::DewPoint => raw - 273.15,
        // Pa → hPa
        WeatherVariable::Pressure | WeatherVariable::SurfacePressure => raw / 100.0,
        // kg/m²/s → mm/jam (PRATE)
        WeatherVariable::Precipitation => raw * 3600.0,
        // 0-1 → %
        WeatherVariable::CloudCover => raw * 100.0,
        // Lainnya: sudah dalam unit yang benar
        _ => raw,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GRIB2 Parser — menggunakan grib 0.19 API yang benar
// ─────────────────────────────────────────────────────────────────────────────

/// Parse bytes GRIB2 → WeatherForecast pada titik lat/lon terdekat
///
/// Menggunakan grib 0.19 API:
/// ```text
/// grib::from_bytes(bytes)          → Grib2
/// grib2.iter()                     → (index, SubMessage)
/// submessage.indicator().discipline → u8
/// submessage.prod_def().parameter_number() → Option<(u8, u8)>
/// Grib2SubmessageDecoder::from(msg)?.dispatch() → f32 iterator
/// submessage.latlons()             → (lat, lon) iterator
/// ```
pub fn parse_grib2_bytes(
    bytes: &[u8],
    target_lat: f64,
    target_lon: f64,
    lead_seconds: u32,
    init_time: DateTime<Utc>,
    valid_time: DateTime<Utc>,
    model_id: &str,
) -> Result<WeatherForecast> {
    use grib::{Grib2SubmessageDecoder, LatLons};

    let param_table = grib2_param_table();

    // Parse GRIB2 stream dari bytes
    let grib2 = grib::from_bytes(bytes.to_vec())
        .map_err(|e| WIError::data_ingestion(format!("Gagal parse GRIB2: {e}")))?;

    let mut variables: HashMap<(WeatherVariable, VerticalLevel), f64> = HashMap::new();

    // Iterasi semua submessage
    for (_index, submessage) in grib2.iter() {
        let discipline = submessage.indicator().discipline;

        // Dapatkan (category, parameter_number) dari product definition
        let Some(param_num) = submessage.prod_def().parameter_number() else {
            debug!("GRIB2: submessage tanpa parameter number, skip");
            continue;
        };

        // category dari product_definition_template (generic fallback = 0)
        let category = 0u8; // TODO: ekstrak dari template jika diperlukan
        let key = Grib2Param::new(discipline, category, param_num);
        let Some(&(weather_var, level)) = param_table.get(&key) else {
            debug!(
                "GRIB2 skip ({},{},{}): tidak dalam tabel",
                discipline, category, param_num
            );
            continue;
        };

        // Kumpulkan lat/lon SEBELUM decoder mengambil ownership submessage
        let latlons_vec: Vec<(f64, f64)> = match submessage.latlons() {
            Ok(it) => it.collect(),
            Err(e) => {
                warn!(
                    "GRIB2: gagal baca latlons ({},{},{}): {e}",
                    discipline, category, param_num
                );
                continue;
            }
        };

        // Decode nilai grid — mengonsumsi submessage
        let decoder = match Grib2SubmessageDecoder::from(submessage) {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "GRIB2: gagal buat decoder ({},{},{}): {e}",
                    discipline, category, param_num
                );
                continue;
            }
        };

        let values_iter = match decoder.dispatch() {
            Ok(it) => it,
            Err(e) => {
                warn!(
                    "GRIB2: gagal dispatch ({},{},{}): {e}",
                    discipline, category, param_num
                );
                continue;
            }
        };

        // Nearest-neighbor search: iterate bersama latlons_vec + values
        let mut best_val: Option<f64> = None;
        let mut best_dist = f64::MAX;

        for ((lat, lon), val) in latlons_vec.into_iter().zip(values_iter) {
            if val.is_nan() {
                continue; // missing value
            }
            let dlat = lat - target_lat;
            let dlon = lon - target_lon;
            let dist = dlat * dlat + dlon * dlon;
            if dist < best_dist {
                best_dist = dist;
                best_val = Some(val as f64);
            }
        }

        let Some(raw) = best_val else {
            warn!(
                "GRIB2: tidak ada nilai valid untuk ({},{},{})",
                discipline, category, param_num
            );
            continue;
        };

        let converted = convert_grib2_value(weather_var, raw);
        variables.insert((weather_var, level), converted);
    }

    if variables.is_empty() {
        return Err(WIError::data_ingestion(
            "GRIB2: tidak ada variabel yang berhasil di-decode. Cek parameter table.",
        ));
    }

    Ok(WeatherForecast {
        coord: GeoCoord::new(target_lat, target_lon),
        init_time,
        valid_time,
        lead_seconds,
        variables,
        ensemble_members: None,
        model_id: model_id.to_string(),
        probability: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ERA5 Variable Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Mapping nama variabel ERA5 (short name) → WeatherVariable
///
/// ERA5 short names: <https://codes.ecmwf.int/grib/param-db/>
pub fn era5_var_mapping() -> HashMap<&'static str, WeatherVariable> {
    HashMap::from([
        ("t2m", WeatherVariable::Temperature),    // 2m temperature (K)
        ("d2m", WeatherVariable::DewPoint),       // 2m dewpoint (K)
        ("r", WeatherVariable::RelativeHumidity), // relative humidity (%)
        ("tp", WeatherVariable::Precipitation),   // total precipitation (m)
        ("u10", WeatherVariable::WindU),          // 10m U-wind (m/s)
        ("v10", WeatherVariable::WindV),          // 10m V-wind (m/s)
        ("sp", WeatherVariable::SurfacePressure), // surface pressure (Pa)
        ("tcc", WeatherVariable::CloudCover),     // total cloud cover (0-1)
        ("q", WeatherVariable::SpecificHumidity), // specific humidity (kg/kg)
        // Long names juga didukung
        ("2m_temperature", WeatherVariable::Temperature),
        ("2m_dewpoint_temperature", WeatherVariable::DewPoint),
        ("total_precipitation", WeatherVariable::Precipitation),
        ("10m_u_component_of_wind", WeatherVariable::WindU),
        ("10m_v_component_of_wind", WeatherVariable::WindV),
        ("surface_pressure", WeatherVariable::SurfacePressure),
        ("total_cloud_cover", WeatherVariable::CloudCover),
    ])
}

/// Konversi nilai ERA5 ke unit standar Andronikos
pub fn convert_era5_value(var: WeatherVariable, raw: f64) -> f64 {
    match var {
        WeatherVariable::Temperature | WeatherVariable::DewPoint => raw - 273.15, // K → °C
        WeatherVariable::Precipitation => raw * 1000.0,                           // m → mm
        WeatherVariable::SurfacePressure | WeatherVariable::Pressure => raw / 100.0, // Pa → hPa
        WeatherVariable::CloudCover => raw * 100.0,                               // 0-1 → %
        _ => raw,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NetCDF Parser — menggunakan netcdf-reader 0.9 API yang benar
// ─────────────────────────────────────────────────────────────────────────────

/// Parse file NetCDF ERA5 → Vec<WeatherObservation>
///
/// ⚠️ ERA5 adalah TRAINING ONLY — tidak boleh masuk fusion real-time.
/// Output digunakan untuk training GraphCast/Downscaler dan bias correction.
///
/// Format ERA5: NetCDF-4, dimensi (time, latitude, longitude)
/// API: `netcdf_reader::NcFile::from_bytes()` → `NcFile`
/// Read: `.read_variable::<f64>(name)` → `ArrayD<f64>`
pub fn parse_era5_netcdf(
    bytes: &[u8],
    target_lat: f64,
    target_lon: f64,
) -> Result<Vec<WeatherObservation>> {
    use netcdf_reader::NcFile;

    let var_map = era5_var_mapping();

    let nc = NcFile::from_bytes(bytes)
        .map_err(|e| WIError::data_ingestion(format!("Gagal parse NetCDF ERA5: {e}")))?;

    // Baca koordinat latitude
    let lats = read_nc_f64_1d(&nc, &["latitude", "lat"])
        .ok_or_else(|| WIError::data_ingestion("ERA5: variabel latitude/lat tidak ditemukan"))?;

    // Baca koordinat longitude
    let lons = read_nc_f64_1d(&nc, &["longitude", "lon"])
        .ok_or_else(|| WIError::data_ingestion("ERA5: variabel longitude/lon tidak ditemukan"))?;

    // Baca time dimension (hours since 1900-01-01)
    let times = read_nc_f64_1d(&nc, &["time"]).unwrap_or_else(|| vec![0.0]);

    // Nearest-neighbor index
    let lat_idx = nearest_index(&lats, target_lat);
    let lon_idx = nearest_index(&lons, target_lon);
    let n_times = times.len();
    let n_lats = lats.len();
    let n_lons = lons.len();

    let mut observations = Vec::with_capacity(n_times);

    for (t_idx, &t_val) in times.iter().enumerate() {
        let valid_time = era5_time_to_utc(t_val as i64);
        let mut obs_vars: HashMap<WeatherVariable, ObservedValue> = HashMap::new();

        for (&var_name, &weather_var) in &var_map {
            // Read variabel sebagai array multidimensi
            let arr = match nc.read_variable::<f64>(var_name) {
                Ok(a) => a,
                Err(_) => continue,
            };

            // Hitung linear index untuk (time, lat, lon)
            let flat_idx = t_idx * n_lats * n_lons + lat_idx * n_lons + lon_idx;
            let raw = match arr.as_slice() {
                Some(s) if flat_idx < s.len() => s[flat_idx],
                _ => {
                    debug!(
                        "ERA5: indeks {} di luar batas untuk var {var_name}",
                        flat_idx
                    );
                    continue;
                }
            };

            if raw.abs() > 9e20 || raw.is_nan() {
                continue; // missing value
            }

            let converted = convert_era5_value(weather_var, raw);
            obs_vars.insert(weather_var, ObservedValue::new(converted, 0.98));
        }

        if obs_vars.is_empty() {
            continue;
        }

        observations.push(WeatherObservation {
            station_id: None,
            coord: GeoCoord::new(target_lat, target_lon),
            time: valid_time,
            variables: obs_vars,
            quality_flag: QualityFlag::Valid,
            source: DataSource::ERA5,
        });
    }

    if observations.is_empty() {
        warn!(
            "ERA5 NetCDF: tidak ada observasi valid di lat={:.3} lon={:.3}",
            target_lat, target_lon
        );
    }

    Ok(observations)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Baca variabel 1D sebagai Vec<f64> dari NetCDF
fn read_nc_f64_1d(nc: &netcdf_reader::NcFile, names: &[&str]) -> Option<Vec<f64>> {
    for name in names {
        if let Ok(arr) = nc.read_variable::<f64>(name) {
            let vals: Vec<f64> = arr.iter().copied().collect();
            if !vals.is_empty() {
                return Some(vals);
            }
        }
    }
    None
}

/// Nearest-neighbor index dalam slice koordinat 1D
pub fn nearest_index(coords: &[f64], target: f64) -> usize {
    coords
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - target).abs().partial_cmp(&(*b - target).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Konversi ERA5 time (hours since 1900-01-01 00:00 UTC) → DateTime<Utc>
///
/// ERA5 epoch: 1900-01-01T00:00:00Z
/// Selisih ke Unix epoch (1970-01-01): 25567 hari = 613_608 jam
pub fn era5_time_to_utc(hours_since_1900: i64) -> DateTime<Utc> {
    const HOURS_ERA5_TO_UNIX: i64 = 613_608;
    let unix_secs = (hours_since_1900 - HOURS_ERA5_TO_UNIX) * 3600;
    Utc.timestamp_opt(unix_secs, 0)
        .single()
        .unwrap_or_else(Utc::now)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (tidak butuh file eksternal)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GRIB2 param table ──────────────────────────────────────────────────

    #[test]
    fn test_grib2_param_table_populated() {
        let t = grib2_param_table();
        assert!(
            t.len() >= 12,
            "tabel param harus punya >= 12 entri, dapat {}",
            t.len()
        );
    }

    #[test]
    fn test_grib2_temperature_in_table() {
        let t = grib2_param_table();
        let key = Grib2Param::new(0, 0, 0);
        assert_eq!(
            t.get(&key).map(|(v, _)| *v),
            Some(WeatherVariable::Temperature)
        );
    }

    #[test]
    fn test_grib2_wind_components_in_table() {
        let t = grib2_param_table();
        assert!(t.contains_key(&Grib2Param::new(0, 2, 2)), "WindU harus ada");
        assert!(t.contains_key(&Grib2Param::new(0, 2, 3)), "WindV harus ada");
    }

    #[test]
    fn test_grib2_precipitation_in_table() {
        let t = grib2_param_table();
        assert!(t.contains_key(&Grib2Param::new(0, 1, 8)), "PRATE harus ada");
    }

    // ── GRIB2 unit konversi ────────────────────────────────────────────────

    #[test]
    fn test_convert_kelvin_to_celsius() {
        let c = convert_grib2_value(WeatherVariable::Temperature, 300.0);
        assert!((c - 26.85).abs() < 0.01, "K→°C salah: {c}");
    }

    #[test]
    fn test_convert_pa_to_hpa() {
        let h = convert_grib2_value(WeatherVariable::Pressure, 101325.0);
        assert!((h - 1013.25).abs() < 0.01, "Pa→hPa salah: {h}");
    }

    #[test]
    fn test_convert_prate_to_mm_per_hour() {
        // 1 kg/m²/s = 3600 mm/jam
        let mm_h = convert_grib2_value(WeatherVariable::Precipitation, 1.0);
        assert!((mm_h - 3600.0).abs() < 0.1, "PRATE→mm/jam salah: {mm_h}");
    }

    #[test]
    fn test_convert_cloud_fraction_to_pct() {
        let pct = convert_grib2_value(WeatherVariable::CloudCover, 0.75);
        assert!((pct - 75.0).abs() < 0.01, "cloud fraction→% salah: {pct}");
    }

    #[test]
    fn test_convert_wind_passthrough() {
        let v = convert_grib2_value(WeatherVariable::WindSpeed, 12.5);
        assert!(
            (v - 12.5).abs() < 1e-9,
            "wind speed tidak boleh dikonversi: {v}"
        );
    }

    // ── Parse error pada bytes kosong ──────────────────────────────────────

    #[test]
    fn test_parse_grib2_empty_returns_error() {
        let now = Utc::now();
        let result = parse_grib2_bytes(&[], -6.2, 106.8, 3600, now, now, "gfs");
        assert!(result.is_err(), "bytes kosong harus error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("GRIB2"),
            "pesan error harus menyebut GRIB2: {msg}"
        );
    }

    // ── ERA5 variable mapping ──────────────────────────────────────────────

    #[test]
    fn test_era5_var_mapping_populated() {
        let m = era5_var_mapping();
        assert!(
            m.len() >= 8,
            "ERA5 mapping harus >= 8 entri, dapat {}",
            m.len()
        );
    }

    #[test]
    fn test_era5_short_names_present() {
        let m = era5_var_mapping();
        assert!(m.contains_key("t2m"), "t2m harus ada");
        assert!(m.contains_key("u10"), "u10 harus ada");
        assert!(m.contains_key("v10"), "v10 harus ada");
        assert!(m.contains_key("tp"), "tp harus ada");
    }

    // ── ERA5 unit konversi ─────────────────────────────────────────────────

    #[test]
    fn test_convert_era5_kelvin() {
        let c = convert_era5_value(WeatherVariable::Temperature, 300.15);
        assert!((c - 27.0).abs() < 0.01, "{c}");
    }

    #[test]
    fn test_convert_era5_precip_m_to_mm() {
        let mm = convert_era5_value(WeatherVariable::Precipitation, 0.005);
        assert!((mm - 5.0).abs() < 0.01, "{mm}");
    }

    #[test]
    fn test_convert_era5_surface_pressure() {
        let hpa = convert_era5_value(WeatherVariable::SurfacePressure, 101325.0);
        assert!((hpa - 1013.25).abs() < 0.01, "{hpa}");
    }

    #[test]
    fn test_convert_era5_cloud_cover() {
        let pct = convert_era5_value(WeatherVariable::CloudCover, 0.5);
        assert!((pct - 50.0).abs() < 0.01, "{pct}");
    }

    // ── Parse ERA5 error pada bytes kosong ────────────────────────────────

    #[test]
    fn test_parse_era5_empty_returns_error() {
        let result = parse_era5_netcdf(&[], -6.2, 106.8);
        assert!(result.is_err(), "bytes kosong harus error");
    }

    // ── Nearest index ─────────────────────────────────────────────────────

    #[test]
    fn test_nearest_index_basic() {
        let coords = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        assert_eq!(nearest_index(&coords, -6.0), 1); // terdekat ke -5.0
        assert_eq!(nearest_index(&coords, 4.9), 3); // terdekat ke 5.0
        assert_eq!(nearest_index(&coords, 11.0), 4); // clamp kanan
        assert_eq!(nearest_index(&coords, -11.0), 0); // clamp kiri
    }

    #[test]
    fn test_nearest_index_single_element() {
        assert_eq!(nearest_index(&[42.0], 0.0), 0);
    }

    // ── ERA5 epoch konversi ───────────────────────────────────────────────

    #[test]
    fn test_era5_epoch_to_unix_zero() {
        // 613_608 jam sejak 1900 = Unix epoch (1970-01-01)
        let dt = era5_time_to_utc(613_608);
        assert_eq!(
            dt.timestamp(),
            0,
            "epoch Unix harus timestamp 0, dapat {}",
            dt.timestamp()
        );
    }

    #[test]
    fn test_era5_time_positive_offset() {
        // 613_608 + 24 = 1 hari setelah Unix epoch
        let dt = era5_time_to_utc(613_608 + 24);
        assert_eq!(
            dt.timestamp(),
            86400,
            "harus 86400 s, dapat {}",
            dt.timestamp()
        );
    }

    #[test]
    fn test_era5_time_year_2024() {
        // 2024-01-01 00:00 UTC ≈ 1_088_976 jam dari 1900 (dari kalender)
        let dt = era5_time_to_utc(1_088_976);
        assert_eq!(
            dt.format("%Y").to_string(),
            "2024",
            "harus 2024, dapat {}",
            dt.format("%Y-%m-%d")
        );
    }
}
