# Andronikos

Weather Machine Learning for Nusantara

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 📋 Deskripsi

Proyek Machine Learning untuk prediksi cuaca dan kebakaran lahan gambut di Nusantara, berbasis Rust dan Python.

## 🏗️ Struktur Proyek

| Folder | Deskripsi | Deploy ke GitHub |
|--------|-----------|------------------|
| `andronikos/` | Frontend Leptos (Rust WASM) | ✅ |
| `training/` | Model training Python (GraphCast, Downscaler) | ✅ |
| `weather-intelligence/` | Crates Rust untuk ingest, model, storage, verify | ✅ |
| `AetherForge/` | Core engine Rust (API, CLI, Core, Data, ML, Spatial, UI) | ❌ (local only) |
| `tera-porte/` | Project legacy/archived | ❌ (local only) |
| `andronikos.satuutas.co.id/` | Server config (nginx SSL) | ❌ (local only) |

## 🚀 CI / Build Status

**Saat ini tidak ada CI/CD otomatis di GitHub** (`.github/workflows/` telah dihapus).

Build & test dilakukan **manual di local**:
```bash
# Rust (per project)
cd andronikos && cargo check --all-targets && cargo test
cd weather-intelligence && cargo check --all-targets && cargo test

# Python
cd training && python -m py_compile **/*.py && pip install -r requirements.txt

# AetherForge (local only - tidak di-deploy ke GitHub)
cd AetherForge && cargo check --all-targets && cargo test
```

Deploy ke GitHub hanya source code via manual script:
```bash
./deploy-source.sh
```

## 📦 Cara Kompilasi Lokal

```bash
# Rust (project yang di-deploy ke GitHub)
cd andronikos && cargo build
cd weather-intelligence && cargo build

# Rust (local only - AetherForge tidak di-deploy)
cd AetherForge && cargo build

# Python
cd training && pip install -r requirements.txt
```

## 📄 Lisensi

Apache 2.0
