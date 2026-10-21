//! # API Server Binary
//!
//! Main entry point for the REST/gRPC API server.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use burn::tensor::Device;
use wi_core::config::WeatherIntelligenceConfig;
use wi_model::burn::{GraphCastConfig, GraphCastForecaster, GraphCastModel};
use wi_model::onnx::OnnxDownscaler;
use wi_model::registry::ModelRegistry;
use wi_model::traits::{Downscaler, DynDownscaler, DynForecaster, Forecaster};

use wi_api::{
    cache::{ApiCache, CacheConfig},
    grpc::create_grpc_service,
    middleware::{rate_limit_middleware, tracing_middleware, RateLimiter},
    rest::{create_routes, ApiState},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    // Load configuration
    let config = WeatherIntelligenceConfig::load(None)?;

    // Initialize model registry
    let model_registry = Arc::new(ModelRegistry::new());

    // Initialize GraphCast forecaster (randomly initialized for now, will be replaced when weights are available)
    let device = Device::default();
    let model_config = GraphCastConfig::default();
    let graphcast_model = GraphCastModel::new(&model_config, &device);
    let graphcast_forecaster: Arc<dyn Forecaster + Send + Sync> =
        Arc::new(GraphCastForecaster::new(
            graphcast_model,
            device,
            model_config,
            wi_core::DeviceType::Cpu,
        ));

    // Register GraphCast in registry
    model_registry.register_forecaster(
        "graphcast".to_string(),
        DynForecaster(graphcast_forecaster.clone()),
    )?;
    model_registry.set_active_forecaster("graphcast")?;

    // Initialize ONNX downscaler if model path is configured
    if let Some(downscaler_path) = &config.ml.downscaler_onnx_path {
        if downscaler_path.exists() {
            tracing::info!(
                "Loading ONNX downscaler from: {}",
                downscaler_path.display()
            );
            match OnnxDownscaler::from_file(downscaler_path.to_str().unwrap(), "1.0.0") {
                Ok(downscaler) => {
                    let downscaler_arc: Arc<dyn Downscaler + Send + Sync> = Arc::new(downscaler);
                    model_registry.register_downscaler(
                        "downscaler".to_string(),
                        DynDownscaler(downscaler_arc),
                    )?;
                    tracing::info!("ONNX downscaler loaded successfully");
                }
                Err(e) => {
                    tracing::warn!("Failed to load ONNX downscaler: {}", e);
                }
            }
        } else {
            tracing::warn!(
                "Downscaler ONNX model not found at: {}",
                downscaler_path.display()
            );
        }
    }

    // Initialize cache
    let cache = if let Some(redis_url) = &config.storage.redis_url {
        Some(Arc::new(
            ApiCache::new(CacheConfig {
                redis_url: redis_url.clone(),
                ..Default::default()
            })
            .await?,
        ))
    } else {
        None
    };

    // Create API state with registry
    let api_state = ApiState::new(graphcast_forecaster.clone(), cache, model_registry.clone());

    // Create rate limiter
    let rate_limiter = RateLimiter::new(config.api.rate_limit_rps as usize, 60);

    // Build REST router with middleware
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

    // Start REST server
    let rest_addr: SocketAddr = format!("{}:{}", config.api.host, config.api.port).parse()?;
    let rest_listener = TcpListener::bind(rest_addr).await?;

    tracing::info!("REST API server listening on {}", rest_addr);

    // Start gRPC server if enabled
    let grpc_handle = if config.api.enable_grpc {
        let grpc_addr: SocketAddr =
            format!("{}:{}", config.api.host, config.api.grpc_port).parse()?;
        let grpc_service = create_grpc_service(graphcast_forecaster);

        let grpc_server = tonic::transport::Server::builder()
            .add_service(grpc_service)
            .serve(grpc_addr);

        tracing::info!("gRPC server listening on {}", grpc_addr);

        Some(tokio::spawn(grpc_server))
    } else {
        None
    };

    // Serve REST API
    let rest_server = axum::serve(rest_listener, app).with_graceful_shutdown(shutdown_signal());

    // Run both servers concurrently
    if let Some(grpc_handle) = grpc_handle {
        tokio::select! {
            result = rest_server => {
                if let Err(e) = result {
                    tracing::error!("REST server error: {}", e);
                }
            }
            result = grpc_handle => {
                if let Err(e) = result {
                    tracing::error!("gRPC server error: {:?}", e);
                }
            }
        }
    } else {
        rest_server.await?;
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("Shutdown signal received, stopping gracefully...");
}
