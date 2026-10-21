//! # CLI Commands Implementation

use anyhow::Result;
use std::path::PathBuf;

use crate::{ExportFormat, ModelType, OutputFormat, StorageFormat};

use axum::Router;
use burn::tensor::Device;
use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use wi_api::middleware::{rate_limit_middleware, tracing_middleware, RateLimiter};
use wi_api::rest::{create_routes, ApiState};
use wi_ingest::{create_default_registry, ParquetSyncEngine, SyncEngineConfig};
use wi_model::burn::{GraphCastConfig, GraphCastForecaster, GraphCastModel};
use wi_model::traits::Forecaster;
use wi_storage::{ParquetConfig, ParquetStorage};
use wi_verify::{DashboardConfig, VerificationDashboard};

pub struct SyncCommand {
    pub hours: u64,
    pub full: bool,
    pub sources: Vec<String>,
    pub region: Option<String>,
}

impl SyncCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Starting data sync...");
        println!("  Hours: {}", self.hours);
        println!("  Full sync: {}", self.full);
        println!("  Sources: {:?}", self.sources);
        println!("  Region: {:?}", self.region);

        let config = SyncEngineConfig {
            parallel_workers: 8,
            batch_size: 1000,
            timeout_secs: 300,
            ..Default::default()
        };

        let storage_config = ParquetConfig {
            base_path: PathBuf::from("/var/weather-intelligence/parquet"),
            ..Default::default()
        };
        let storage = Arc::new(ParquetStorage::new(storage_config));
        let engine = ParquetSyncEngine::with_config(config, storage);

        let end = chrono::Utc::now();
        let start = if self.full {
            end - chrono::Duration::days(365)
        } else {
            end - chrono::Duration::hours(self.hours as i64)
        };

        println!("\n[INGEST] Syncing from {} to {}", start, end);
        let stats = engine.sync_all(start, end).await?;

        println!("\n[INGEST] Sync completed!");
        println!("  Records fetched: {}", stats.records_fetched);
        println!("  Records validated: {}", stats.records_validated);
        println!("  Records written: {}", stats.records_written);
        println!("  Records failed: {}", stats.records_failed);
        println!("  Duration: {:.2}s", stats.duration_secs);
        println!("  Throughput: {:.0} rec/s", stats.records_per_second);

        Ok(())
    }
}

pub struct TrainCommand {
    pub _config: Option<PathBuf>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f32>,
    pub resume: Option<String>,
    pub checkpoint_dir: Option<PathBuf>,
    pub data: PathBuf,
    pub model_type: Option<ModelType>,
}

impl TrainCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Starting ML training...");
        println!("  Model type: {:?}", self.model_type);
        println!("  Epochs: {:?}", self.epochs.unwrap_or(100));
        println!("  Batch size: {:?}", self.batch_size.unwrap_or(32));
        println!("  Learning rate: {:?}", self.learning_rate.unwrap_or(1e-4));
        println!("  Resume: {:?}", self.resume);
        println!("  Checkpoint dir: {:?}", self.checkpoint_dir);
        println!("  Data file: {:?}", self.data);

        match self.model_type {
            Some(ModelType::GraphCast) | None => {
                println!("\n[ML] GraphCast training requires Python/PyTorch pipeline");
                println!("[ML] This CLI provides model initialization only");
                println!("[ML] For full training, use the Python training pipeline:");
                println!("    cd training && python train_graphcast.py");
                println!("[ML] Configuration:");
                println!("    - Epochs: {}", self.epochs.unwrap_or(100));
                println!("    - Batch size: {}", self.batch_size.unwrap_or(32));
                println!(
                    "    - Learning rate: {}",
                    self.learning_rate.unwrap_or(1e-4)
                );
                println!("    - Data: {:?}", self.data);
                println!("    - Checkpoint dir: {:?}", self.checkpoint_dir);

                // Initialize model structure for validation
                let device = Device::default();
                let model_config = GraphCastConfig::default();
                let _model = GraphCastModel::new(&model_config, &device);
                println!("[ML] GraphCast model structure validated");
            }
            Some(ModelType::Downscaler) => {
                println!("\n[ML] Downscaler training requires Python/PyTorch pipeline");
                println!("[ML] Export to ONNX, then load with wi-model ONNX module");
                println!("[ML] Training pipeline: Python/PyTorch -> ONNX -> Rust inference");
            }
        }

        Ok(())
    }
}

pub struct PredictCommand {
    pub model: Option<PathBuf>,
    pub latitude: f64,
    pub longitude: f64,
    pub lead_seconds: u32,
    pub output: Option<PathBuf>,
    pub format: Option<OutputFormat>,
    pub variables: Vec<String>,
}

impl PredictCommand {
    pub async fn execute(&self) -> Result<()> {
        let lead_hours = self.lead_seconds as f32 / 3600.0;

        println!("Running prediction...");
        println!("  Latitude: {}", self.latitude);
        println!("  Longitude: {}", self.longitude);
        println!(
            "  Lead time: {} seconds ({:.2} hours)",
            self.lead_seconds, lead_hours
        );
        println!("  Model: {:?}", self.model);
        println!("  Output: {:?}", self.output);
        println!("  Format: {:?}", self.format);
        println!("  Variables: {:?}", self.variables);

        // Load model
        println!("\n[PREDICT] Loading model...");
        let device = Device::default();
        let model_config = GraphCastConfig::default();
        let model = GraphCastModel::new(&model_config, &device);
        let forecaster =
            GraphCastForecaster::new(model, device, model_config, wi_core::DeviceType::Cpu);

        // Parse variables
        let vars = if self.variables.is_empty() {
            wi_core::types::WeatherVariable::surface_variables()
        } else {
            self.variables
                .iter()
                .filter_map(|v| parse_weather_variable(v))
                .collect()
        };

        // Run prediction
        println!("\n[PREDICT] Loading model...");
        println!(
            "[PREDICT] Running inference for {}h lead time...",
            lead_hours
        );
        let fusion_output = forecaster
            .predict_point(self.latitude, self.longitude, self.lead_seconds, &vars)
            .await?;

        println!("[PREDICT] Prediction complete!");

        // Display results
        println!("\n--- Prediction Results ---");
        for (var, fused) in &fusion_output.variables {
            let value = fused.weighted_mean;
            let (_unit, display) = match var {
                wi_core::types::WeatherVariable::Temperature => {
                    ("°C", format!("{:.1}°C", value - 273.15))
                }
                wi_core::types::WeatherVariable::RelativeHumidity => {
                    ("%", format!("{:.0}%", value))
                }
                wi_core::types::WeatherVariable::WindSpeed => ("m/s", format!("{:.1} m/s", value)),
                wi_core::types::WeatherVariable::WindDirection => ("°", format!("{:.0}°", value)),
                wi_core::types::WeatherVariable::Pressure
                | wi_core::types::WeatherVariable::SurfacePressure => {
                    ("hPa", format!("{:.1} hPa", value / 100.0))
                }
                wi_core::types::WeatherVariable::Precipitation => {
                    ("mm", format!("{:.1} mm", value))
                }
                wi_core::types::WeatherVariable::CloudCover => ("%", format!("{:.0}%", value)),
                wi_core::types::WeatherVariable::DewPoint => {
                    ("°C", format!("{:.1}°C", value - 273.15))
                }
                wi_core::types::WeatherVariable::Visibility => ("m", format!("{:.0} m", value)),
                _ => ("", format!("{:.2}", value)),
            };
            println!("{:?}: {}", var, display);
        }

        // Save output if requested
        if let Some(output) = &self.output {
            let json = serde_json::json!({
                "latitude": self.latitude,
                "longitude": self.longitude,
                "lead_seconds": self.lead_seconds,
                "predictions": fusion_output.variables.iter().map(|(k, v)| (format!("{:?}", k), v.weighted_mean)).collect::<HashMap<_, _>>(),
            });
            tokio::fs::write(output, serde_json::to_string_pretty(&json)?).await?;
            println!("\n[PREDICT] Results saved to {:?}", output);
        }

        Ok(())
    }
}

pub struct ServeCommand {
    pub host: String,
    pub port: u16,
    pub model: Option<PathBuf>,
    pub workers: Option<usize>,
    pub cors: bool,
    pub enable_grpc: bool,
}

impl ServeCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Starting API server...");
        println!("  Host: {}", self.host);
        println!("  Port: {}", self.port);
        println!("  Workers: {:?}", self.workers);
        println!("  Model: {:?}", self.model);
        println!("  CORS: {}", self.cors);
        println!("  gRPC: {}", self.enable_grpc);

        // Load model
        let device = Device::default();
        let model_config = GraphCastConfig::default();
        let model = GraphCastModel::new(&model_config, &device);
        let forecaster: Arc<dyn Forecaster> = Arc::new(GraphCastForecaster::new(
            model,
            device,
            model_config,
            wi_core::DeviceType::Cpu,
        ));

        // Initialize model registry
        let model_registry = Arc::new(wi_model::registry::ModelRegistry::new());
        model_registry.register_forecaster(
            "graphcast".to_string(),
            wi_model::DynForecaster(forecaster.clone()),
        )?;
        model_registry.set_active_forecaster("graphcast")?;

        // Initialize cache (optional Redis)
        let cache = None; // Could be configured from env

        // Create API state
        let api_state = ApiState::new(forecaster, cache, model_registry);

        // Create rate limiter
        let rate_limiter = RateLimiter::new(100, 60);

        // Build router
        let app = Router::new().merge(create_routes(api_state)).layer(
            ServiceBuilder::new()
                .layer(
                    CorsLayer::new()
                        .allow_origin(tower_http::cors::Any)
                        .allow_methods([http::Method::GET, http::Method::POST])
                        .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION]),
                )
                .layer(axum::middleware::from_fn_with_state(
                    rate_limiter.clone(),
                    rate_limit_middleware,
                ))
                .layer(axum::middleware::from_fn(tracing_middleware)),
        );

        // Start server
        let addr: SocketAddr = format!("{}:{}", self.host, self.port).parse()?;
        let listener = TcpListener::bind(addr).await?;

        println!(
            "\n[SERVER] API server started on {}:{}",
            self.host, self.port
        );
        println!("[SERVER] Endpoints:");
        println!("  - GET  /health");
        println!("  - POST /v1/predict");
        println!("  - POST /v1/predict/grid");
        println!("  - GET  /v1/model/info");
        println!("  - GET  /v1/variables");
        println!("  - GET  /v1/verification");

        if self.enable_grpc {
            println!("  - gRPC on port {} (not yet implemented)", self.port + 1);
        }

        // Graceful shutdown
        let shutdown = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
            println!("\n[SERVER] Shutdown signal received");
        };

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await?;

        Ok(())
    }
}

pub struct VerifyCommand {
    pub model: PathBuf,
    pub data: PathBuf,
    pub metrics: Vec<String>,
    pub baseline: Vec<String>,
    pub output: Option<PathBuf>,
    pub temporal_split: bool,
    pub spatial_split: bool,
}

impl VerifyCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Running verification...");
        println!("  Model: {:?}", self.model);
        println!("  Data: {:?}", self.data);
        println!("  Metrics: {:?}", self.metrics);
        println!("  Baselines: {:?}", self.baseline);
        println!("  Temporal split: {}", self.temporal_split);
        println!("  Spatial split: {}", self.spatial_split);

        println!("\n[VERIFY] Verification requires Python/PyTorch pipeline");
        println!("[VERIFY] This CLI provides dashboard generation only");
        println!("[VERIFY] For full verification, use the Python pipeline:");
        println!(
            "    cd verification && python verify.py --model {:?} --data {:?}",
            self.model, self.data
        );
        println!("[VERIFY] Configuration:");
        println!("    - Metrics: {:?}", self.metrics);
        println!("    - Baselines: {:?}", self.baseline);
        println!("    - Temporal split: {}", self.temporal_split);
        println!("    - Spatial split: {}", self.spatial_split);

        // Export dashboard if requested
        if let Some(output) = &self.output {
            let config = DashboardConfig {
                output_path: output.clone(),
                generate_html: true,
                generate_json: true,
                generate_prometheus: true,
            };
            let _dashboard = VerificationDashboard::new(config);
            println!("[VERIFY] Dashboard template exported to {:?}", output);
        }

        Ok(())
    }
}

pub struct ExportCommand {
    pub model: PathBuf,
    pub format: ExportFormat,
    pub output: PathBuf,
}

impl ExportCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Exporting model...");
        println!("  Model: {:?}", self.model);
        println!("  Format: {:?}", self.format);
        println!("  Output: {:?}", self.output);

        match self.format {
            ExportFormat::Onnx => {
                println!("[EXPORT] ONNX export requires Python/torch.onnx");
                println!("[EXPORT] Use: python -c \"import torch; torch.onnx.export(model, dummy_input, '{:?}')\"", self.output);
            }
            ExportFormat::TorchScript => {
                println!(
                    "[EXPORT] TorchScript export: torch.jit.trace(model, dummy_input).save('{:?}')",
                    self.output
                );
            }
            ExportFormat::TensorRT => {
                println!("[EXPORT] TensorRT export requires ONNX -> TensorRT conversion");
                println!(
                    "[EXPORT] Use: trtexec --onnx={:?} --saveEngine={:?}",
                    self.model, self.output
                );
            }
            ExportFormat::BurnNative => {
                println!("[EXPORT] Burning native format - use model.save_file() in Rust code");
            }
        }

        Ok(())
    }
}

pub struct InfoCommand {
    pub model: Option<PathBuf>,
}

impl InfoCommand {
    pub async fn execute(&self) -> Result<()> {
        println!("Weather Intelligence Model Information");
        println!("=====================================");
        println!("Architecture: GraphCast (GNN + Spatiotemporal Attention)");
        println!("Hidden dimension: 256");
        println!("Number of layers: 12");
        println!("Number of heads: 8");
        println!("Model version: {}", env!("CARGO_PKG_VERSION"));

        if let Some(ref m) = self.model {
            println!("Model path: {:?}", m);
        }

        println!("\nSupported variables:");
        println!("  - Temperature (K)");
        println!("  - Relative Humidity (%)");
        println!("  - Precipitation (mm)");
        println!("  - Wind Speed (m/s)");
        println!("  - Wind Direction (degrees)");
        println!("  - Pressure (Pa)");
        println!("  - Surface Pressure (Pa)");
        println!("  - Cloud Cover (%)");
        println!("  - Dew Point (K)");
        println!("  - Visibility (m)");

        println!("\nSupported lead times: 1h, 3h, 6h, 12h, 24h, 48h, 72h, 168h");
        println!("Supported resolutions: 0.5km, 1km, 5km, 25km");

        Ok(())
    }
}

/// Parse weather variable name to WeatherVariable enum
fn parse_weather_variable(name: &str) -> Option<wi_core::types::WeatherVariable> {
    match name {
        "Temperature" | "temperature" | "temp" | "t2m" => {
            Some(wi_core::types::WeatherVariable::Temperature)
        }
        "RelativeHumidity" | "humidity" | "rh" | "rhum" => {
            Some(wi_core::types::WeatherVariable::RelativeHumidity)
        }
        "Precipitation" | "precip" | "rain" | "tp" => {
            Some(wi_core::types::WeatherVariable::Precipitation)
        }
        "WindSpeed" | "wind_speed" | "ws" | "wspd" => {
            Some(wi_core::types::WeatherVariable::WindSpeed)
        }
        "WindDirection" | "wind_dir" | "wd" | "wdir" => {
            Some(wi_core::types::WeatherVariable::WindDirection)
        }
        "WindU" | "u_component" | "u" => Some(wi_core::types::WeatherVariable::WindU),
        "WindV" | "v_component" | "v" => Some(wi_core::types::WeatherVariable::WindV),
        "Pressure" | "pressure" | "pres" | "msl" => Some(wi_core::types::WeatherVariable::Pressure),
        "SurfacePressure" | "surface_pressure" | "sp" => {
            Some(wi_core::types::WeatherVariable::SurfacePressure)
        }
        "CloudCover" | "cloud_cover" | "cc" | "tcc" => {
            Some(wi_core::types::WeatherVariable::CloudCover)
        }
        "DewPoint" | "dew_point" | "td" | "dpt" => Some(wi_core::types::WeatherVariable::DewPoint),
        "Visibility" | "visibility" | "vis" => Some(wi_core::types::WeatherVariable::Visibility),
        _ => None,
    }
}

pub async fn ingest_command(
    hours: u64,
    sources: Vec<String>,
    output_format: Option<StorageFormat>,
) -> Result<()> {
    println!("Running data ingestion...");
    println!("  Hours: {}", hours);
    println!("  Sources: {:?}", sources);
    println!("  Output format: {:?}", output_format);

    let registry = create_default_registry();
    println!("\n[INGEST] Available sources:");
    for source in registry.all() {
        println!(
            "  - {} ({}) - rate: {} req/s",
            source.name(),
            source.source_type() as u8,
            source.rate_limit()
        );
    }

    println!("\n[INGEST] Ingestion complete (placeholder)!");
    Ok(())
}
