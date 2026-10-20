# Andronikos

Weather Machine Learning for Nusantara

[![CI - Build & Test](https://img.shields.io/github/actions/workflow/status/satuutasteknologi-org/Andronikos/ci.yml?branch=main&label=CI%20Build%20%26%20Test)](https://github.com/satuutasteknologi-org/Andronikos/actions/workflows/ci.yml)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 📋 Deskripsi

Proyek Machine Learning untuk prediksi cuaca dan kebakaran lahan gambut di Nusantara, berbasis Rust dan Python.

## 🏗️ Struktur Proyek

| Folder | Deskripsi |
|--------|-----------|
| `AetherForge/` | Core engine Rust (API, CLI, Core, Data, ML, Spatial, UI) |
| `andronikos/` | Frontend Leptos (Rust WASM) |
| `tera-porte/` | Project legacy/archived |
| `training/` | Model training Python (GraphCast, Downscaler) |
| `weather-intelligence/` | Crates Rust untuk ingest, model, storage, verify |

## 🚀 CI / Build Status

Setiap push ke `main` atau `deploy` akan otomatis menjalankan:
- **Rust**: `cargo check --all-targets` dan `cargo test`
- **Python**: `py_compile` dan install dependencies

Lihat detail di: [GitHub Actions](https://github.com/satuutasteknologi-org/Andronikos/actions)

## 📦 Cara Kompilasi Lokal

```bash
# Rust (per project)
cd AetherForge && cargo build
cd andronikos && cargo build
cd weather-intelligence && cargo build

# Python
cd training && pip install -r requirements.txt
```

## 📄 Lisensi

Apache 2.0
