//! # Benchmark: GraphCast Inference Throughput
//!
//! Mengukur waktu inferensi GraphCast pada berbagai skala cell.
//! Target: < 60 detik untuk 38.6 juta cell (seluruh Indonesia 0.5km).
//!
//! ```bash
//! # Jalankan benchmark:
//! cargo bench -p wi-model --bench inference_bench
//!
//! # Dengan GPU WGPU:
//! cargo bench -p wi-model --bench inference_bench --features wgpu
//! ```
//!
//! Output HTML di `target/criterion/` — buka `report/index.html`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use wi_core::{Bbox, WeatherVariable};
use wi_model::burn::{GraphCastConfig, GraphCastForecaster};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn default_forecaster() -> GraphCastForecaster {
    GraphCastForecaster::with_best_device(GraphCastConfig::default())
        .expect("GraphCastForecaster init gagal")
}

/// Buat Bbox untuk ≈ N cell (persegi di center Jakarta, 0.0045° ≈ 500m)
fn bbox_for_n_cells(n: usize) -> Bbox {
    let side = (n as f64).sqrt() * 0.0045;
    Bbox {
        min_lat: -6.2 - side / 2.0,
        max_lat: -6.2 + side / 2.0,
        min_lon: 106.8 - side / 2.0,
        max_lon: 106.8 + side / 2.0,
    }
}

fn surface_variables() -> Vec<WeatherVariable> {
    vec![
        WeatherVariable::Temperature,
        WeatherVariable::RelativeHumidity,
        WeatherVariable::Precipitation,
        WeatherVariable::WindSpeed,
        WeatherVariable::Pressure,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 1: predict_cells_sync — single tile, berbagai ukuran
//
// Mengukur throughput murni forward() tanpa tiling overhead.
// Baseline paling penting untuk membandingkan CPU vs GPU.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_single_tile(c: &mut Criterion) {
    let forecaster = default_forecaster();
    let vars = surface_variables();

    let mut group = c.benchmark_group("single_tile_cells");
    group.measurement_time(std::time::Duration::from_secs(30));

    for &n_cells in &[100usize, 1_000, 5_000, 10_000] {
        let bbox = bbox_for_n_cells(n_cells);
        group.throughput(Throughput::Elements(n_cells as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_cells_sync", n_cells),
            &n_cells,
            |b, &limit| {
                b.iter(|| {
                    forecaster
                        .predict_cells_sync(&bbox, 3600, &vars, Some(limit))
                        .expect("predict_cells_sync gagal")
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 2: build_mesh_edges — overhead konstruksi edge graph
//
// Jika > 100ms untuk 50K cell → wajib pre-compute & cache.
// Cache sudah ada: GraphCastForecaster::edge_cache (RwLock<HashMap>)
// ─────────────────────────────────────────────────────────────────────────────
fn bench_mesh_edges(c: &mut Criterion) {
    let forecaster = default_forecaster();
    let mut group = c.benchmark_group("mesh_edges_build");

    for &n in &[1_000usize, 10_000, 50_000] {
        let width = (n as f64).sqrt().ceil() as usize;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("4connectivity", n), &n, |b, &n_cells| {
            // Bypass cache — ukur biaya build langsung
            b.iter(|| GraphCastForecaster::build_mesh_edges(n_cells, width, forecaster.device()));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 3: edge_cache hit rate — cold vs warm
//
// Membuktikan bahwa cache amortize biaya build edges secara signifikan.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_edge_cache(c: &mut Criterion) {
    let forecaster = default_forecaster();
    let vars = surface_variables();
    let n = 10_000;
    let bbox = bbox_for_n_cells(n);

    let mut group = c.benchmark_group("edge_cache");

    // Warm: panggil sekali dulu agar cache terisi
    let _ = forecaster.predict_cells_sync(&bbox, 3600, &vars, Some(n));

    group.bench_function("warm_cache_10k_cells", |b| {
        b.iter(|| {
            forecaster
                .predict_cells_sync(&bbox, 3600, &vars, Some(n))
                .expect("predict gagal")
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_tile,
    bench_mesh_edges,
    bench_edge_cache
);
criterion_main!(benches);
