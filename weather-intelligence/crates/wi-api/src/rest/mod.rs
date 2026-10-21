//! # REST API Handlers
//!
//! HTTP handlers for weather prediction endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::cache::ApiCache;
use wi_core::types::{Bbox, GridPredictRequest, WeatherVariable};
use wi_model::registry::ModelRegistry;
use wi_model::traits::{
    DownscalerInput, Forecaster, QuantilePrediction as ModelQuantilePrediction,
};

#[derive(Clone)]
pub struct ApiState {
    pub forecaster: Arc<dyn Forecaster + Send + Sync>,
    pub model_registry: Arc<ModelRegistry>,
    pub model_loaded: bool,
    pub cache: Option<Arc<ApiCache>>,
}

impl ApiState {
    pub fn new(
        forecaster: Arc<dyn Forecaster + Send + Sync>,
        cache: Option<Arc<ApiCache>>,
        model_registry: Arc<ModelRegistry>,
    ) -> Self {
        Self {
            forecaster,
            model_registry,
            model_loaded: true,
            cache,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_seconds: u32,
    pub variables: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub valid_time: DateTime<Utc>,
    pub lead_seconds: u32,
    pub lead_hours: f32,
    pub predictions: Vec<VariablePrediction>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VariablePrediction {
    pub variable: String,
    pub value: f32,
    pub unit: String,
    pub confidence: f32,
    pub quantiles: Option<ModelQuantilePrediction>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridPredictionRequest {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
    pub resolution: u8,
    pub lead_seconds: u32,
    pub variables: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridPredictionResponse {
    pub grid: Vec<GridCellResponse>,
    pub resolution: u8,
    pub lead_seconds: u32,
    pub lead_hours: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridCellResponse {
    pub lat: f64,
    pub lon: f64,
    pub cell_id: u64,
    pub predictions: Vec<(String, f32)>,
    pub quantiles: Option<HashMap<String, RestQuantilePrediction>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RestQuantilePrediction {
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
    pub prob_rain: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub num_parameters: usize,
    pub supported_variables: Vec<String>,
    pub supported_lead_times: Vec<u32>,
    pub supported_resolutions: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownscaleRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_seconds: u32,
    pub coarse_temp: f32,
    pub coarse_humidity: f32,
    pub coarse_precip: f32,
    pub coarse_wind_u: f32,
    pub coarse_wind_v: f32,
    pub coarse_pressure: f32,
    pub elevation: f32,
    pub land_mask: f32,
    pub month: u8,
    pub hour: u8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownscaleResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_seconds: u32,
    pub lead_hours: f32,
    pub valid_time: DateTime<Utc>,
    pub temperature: ModelQuantilePrediction,
    pub humidity: ModelQuantilePrediction,
    pub precipitation: ModelQuantilePrediction,
    pub wind_u: ModelQuantilePrediction,
    pub wind_v: ModelQuantilePrediction,
    pub pressure: ModelQuantilePrediction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub timestamp: DateTime<Utc>,
    pub uptime_seconds: u64,
}

static START_TIME: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

pub fn create_routes(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/predict", post(predict_handler))
        .route("/v1/predict/grid", post(grid_predict_handler))
        .route("/v1/downscale", post(downscale_handler))
        .route("/v1/model/info", get(model_info_handler))
        .route("/v1/variables", get(variables_handler))
        .route("/v1/verification", get(verification_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<ApiState>) -> impl IntoResponse {
    let response = HealthResponse {
        status: if state.model_loaded {
            "healthy"
        } else {
            "degraded"
        }
        .to_string(),
        model_loaded: state.model_loaded,
        timestamp: Utc::now(),
        uptime_seconds: START_TIME.elapsed().as_secs(),
    };

    let status = if state.model_loaded {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(response))
}

async fn predict_handler(
    State(state): State<ApiState>,
    Json(request): Json<PredictionRequest>,
) -> Response {
    if !state.model_loaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Model not loaded"})),
        )
            .into_response();
    }

    // Generate cache key
    let cache_key = state.cache.as_ref().map(|_| {
        ApiCache::point_cache_key(request.latitude, request.longitude, request.lead_seconds)
    });

    // Try cache first
    if let (Some(cache), Some(ref key)) = (state.cache.as_ref(), cache_key.as_ref()) {
        if let Ok(Some(cached)) = cache.get::<PredictionResponse>(key).await {
            info!("Cache hit for point prediction: {}", key);
            return Json(cached).into_response();
        }
    }

    info!(
        "Prediction request: lat={}, lon={}, lead={}s",
        request.latitude, request.longitude, request.lead_seconds
    );

    let variables = request.variables.unwrap_or_else(|| {
        WeatherVariable::surface_variables()
            .into_iter()
            .map(|v| format!("{:?}", v))
            .collect()
    });

    let weather_vars: Vec<WeatherVariable> = variables
        .iter()
        .filter_map(|v| match v.as_str() {
            "Temperature" => Some(WeatherVariable::Temperature),
            "RelativeHumidity" => Some(WeatherVariable::RelativeHumidity),
            "Precipitation" => Some(WeatherVariable::Precipitation),
            "WindSpeed" => Some(WeatherVariable::WindSpeed),
            "WindDirection" => Some(WeatherVariable::WindDirection),
            "Pressure" => Some(WeatherVariable::Pressure),
            "SurfacePressure" => Some(WeatherVariable::SurfacePressure),
            "CloudCover" => Some(WeatherVariable::CloudCover),
            "DewPoint" => Some(WeatherVariable::DewPoint),
            "Visibility" => Some(WeatherVariable::Visibility),
            _ => None,
        })
        .collect();

    let fusion_output = match state
        .forecaster
        .predict_point(
            request.latitude,
            request.longitude,
            request.lead_seconds,
            &weather_vars,
        )
        .await
    {
        Ok(output) => output,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let predictions: Vec<VariablePrediction> = fusion_output
        .variables
        .into_iter()
        .map(|(variable, fused)| {
            let unit = match variable {
                WeatherVariable::Temperature => "K",
                WeatherVariable::RelativeHumidity => "%",
                WeatherVariable::Precipitation => "mm",
                WeatherVariable::WindSpeed => "m/s",
                WeatherVariable::WindDirection => "degrees",
                WeatherVariable::Pressure | WeatherVariable::SurfacePressure => "Pa",
                WeatherVariable::CloudCover => "%",
                WeatherVariable::DewPoint => "K",
                WeatherVariable::Visibility => "m",
                _ => "",
            };
            VariablePrediction {
                variable: format!("{:?}", variable),
                value: fused.weighted_mean,
                unit: unit.to_string(),
                confidence: fused.exceedance_probability.unwrap_or(0.5),
                quantiles: None,
            }
        })
        .collect();

    let response = PredictionResponse {
        latitude: request.latitude,
        longitude: request.longitude,
        valid_time: Utc::now() + chrono::Duration::seconds(request.lead_seconds as i64),
        lead_seconds: request.lead_seconds,
        lead_hours: request.lead_seconds as f32 / 3600.0,
        predictions,
    };

    // Cache the response
    if let (Some(ref cache), Some(key)) = (&state.cache, &cache_key) {
        let ttl = Duration::from_secs(3600); // 1 hour
        if let Err(e) = cache.set(key, &response, Some(ttl.as_secs())).await {
            warn!("Failed to cache prediction: {}", e);
        }
    }

    Json(response).into_response()
}

async fn downscale_handler(
    State(state): State<ApiState>,
    Json(request): Json<DownscaleRequest>,
) -> Response {
    if !state.model_loaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Model not loaded"})),
        )
            .into_response();
    }

    // Get downscaler from registry
    let downscaler = match state.model_registry.get_active_downscaler() {
        Some(d) => d,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Downscaler not available"})),
            )
                .into_response();
        }
    };

    info!(
        "Downscale request: lat={}, lon={}, lead={}s",
        request.latitude, request.longitude, request.lead_seconds
    );

    let input = DownscalerInput {
        lat: request.latitude,
        lon: request.longitude,
        lead_seconds: request.lead_seconds,
        coarse_temp: request.coarse_temp,
        coarse_humidity: request.coarse_humidity,
        coarse_precip: request.coarse_precip,
        coarse_wind_u: request.coarse_wind_u,
        coarse_wind_v: request.coarse_wind_v,
        coarse_pressure: request.coarse_pressure,
        elevation: request.elevation,
        land_mask: request.land_mask,
        month: request.month,
        hour: request.hour,
    };

    let output = match downscaler.0.downscale(input).await {
        Ok(out) => out,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let response = DownscaleResponse {
        latitude: request.latitude,
        longitude: request.longitude,
        lead_seconds: request.lead_seconds,
        lead_hours: request.lead_seconds as f32 / 3600.0,
        valid_time: Utc::now() + chrono::Duration::seconds(request.lead_seconds as i64),
        temperature: ModelQuantilePrediction {
            p10: output.temp.p10,
            p50: output.temp.p50,
            p90: output.temp.p90,
            prob_rain: output.temp.prob_rain,
        },
        humidity: ModelQuantilePrediction {
            p10: output.humidity.p10,
            p50: output.humidity.p50,
            p90: output.humidity.p90,
            prob_rain: output.humidity.prob_rain,
        },
        precipitation: ModelQuantilePrediction {
            p10: output.precip.p10,
            p50: output.precip.p50,
            p90: output.precip.p90,
            prob_rain: output.precip.prob_rain,
        },
        wind_u: ModelQuantilePrediction {
            p10: output.wind_u.p10,
            p50: output.wind_u.p50,
            p90: output.wind_u.p90,
            prob_rain: output.wind_u.prob_rain,
        },
        wind_v: ModelQuantilePrediction {
            p10: output.wind_v.p10,
            p50: output.wind_v.p50,
            p90: output.wind_v.p90,
            prob_rain: output.wind_v.prob_rain,
        },
        pressure: ModelQuantilePrediction {
            p10: output.pressure.p10,
            p50: output.pressure.p50,
            p90: output.pressure.p90,
            prob_rain: output.pressure.prob_rain,
        },
    };

    Json(response).into_response()
}

async fn grid_predict_handler(
    State(state): State<ApiState>,
    Json(request): Json<GridPredictionRequest>,
) -> Response {
    if !state.model_loaded {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "Model not loaded"})),
        )
            .into_response();
    }

    // Generate cache key
    let cache_key = state.cache.as_ref().map(|_| {
        ApiCache::grid_cache_key(
            request.min_lat,
            request.min_lon,
            request.max_lat,
            request.max_lon,
            request.resolution,
            request.lead_seconds,
        )
    });

    // Try cache first
    if let (Some(cache), Some(ref key)) = (state.cache.as_ref(), cache_key.as_ref()) {
        if let Ok(Some(cached)) = cache.get::<GridPredictionResponse>(key).await {
            info!("Cache hit for grid prediction: {}", key);
            return Json(cached).into_response();
        }
    }

    info!(
        "Grid prediction: {}x{} from ({},{}) to ({},{})",
        request.resolution,
        request.resolution,
        request.min_lat,
        request.min_lon,
        request.max_lat,
        request.max_lon
    );

    let variables = request.variables.unwrap_or_else(|| {
        WeatherVariable::surface_variables()
            .into_iter()
            .map(|v| format!("{:?}", v))
            .collect()
    });

    let weather_vars: Vec<WeatherVariable> = variables
        .iter()
        .filter_map(|v| match v.as_str() {
            "Temperature" => Some(WeatherVariable::Temperature),
            "RelativeHumidity" => Some(WeatherVariable::RelativeHumidity),
            "Precipitation" => Some(WeatherVariable::Precipitation),
            "WindSpeed" => Some(WeatherVariable::WindSpeed),
            "WindDirection" => Some(WeatherVariable::WindDirection),
            "Pressure" => Some(WeatherVariable::Pressure),
            "SurfacePressure" => Some(WeatherVariable::SurfacePressure),
            "CloudCover" => Some(WeatherVariable::CloudCover),
            "DewPoint" => Some(WeatherVariable::DewPoint),
            "Visibility" => Some(WeatherVariable::Visibility),
            _ => None,
        })
        .collect();

    let grid_request = GridPredictRequest::new(
        Bbox::new(
            request.min_lon,
            request.min_lat,
            request.max_lon,
            request.max_lat,
        ),
        request.resolution,
        request.lead_seconds,
        weather_vars,
    );

    let grid_predictions = match state.forecaster.predict_grid(grid_request).await {
        Ok(preds) => preds,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let grid: Vec<GridCellResponse> = grid_predictions
        .into_iter()
        .map(|p| GridCellResponse {
            lat: p.lat,
            lon: p.lon,
            cell_id: p.cell_id,
            predictions: p
                .predictions
                .into_iter()
                .map(|(v, val)| (format!("{:?}", v), val))
                .collect(),
            quantiles: p.quantiles.map(|q| {
                q.into_iter()
                    .map(|(v, qv)| {
                        (
                            format!("{:?}", v),
                            RestQuantilePrediction {
                                p10: qv.p10,
                                p50: qv.p50,
                                p90: qv.p90,
                                prob_rain: qv.prob_rain,
                            },
                        )
                    })
                    .collect()
            }),
        })
        .collect();

    let response = GridPredictionResponse {
        grid,
        resolution: request.resolution,
        lead_seconds: request.lead_seconds,
        lead_hours: request.lead_seconds as f32 / 3600.0,
    };

    // Cache the response
    if let (Some(ref cache), Some(key)) = (&state.cache, &cache_key) {
        let ttl = Duration::from_secs(1800); // 30 minutes for grid
        if let Err(e) = cache.set(key, &response, Some(ttl.as_secs())).await {
            warn!("Failed to cache grid prediction: {}", e);
        }
    }

    Json(response).into_response()
}

async fn model_info_handler(State(state): State<ApiState>) -> Json<ModelInfoResponse> {
    // Get active forecaster info
    let active_forecaster_info = state.forecaster.model_info();

    // Get all registered models from registry
    let versions = state.model_registry.list_versions();
    let active_forecaster_name = state.model_registry.active_forecaster_name();
    let active_downscaler_name = state.model_registry.active_downscaler_name();

    let mut model_list = Vec::new();
    for v in versions {
        model_list.push(serde_json::json!({
            "name": v.name,
            "version": v.version,
            "architecture": v.info.architecture,
            "num_parameters": v.info.num_params,
            "is_active_forecaster": Some(v.name.clone()) == active_forecaster_name,
            "is_active_downscaler": Some(v.name.clone()) == active_downscaler_name,
            "inference_count": v.inference_count,
            "avg_latency_ms": v.avg_latency_ms,
        }));
    }

    // Build response with active model details
    let response = ModelInfoResponse {
        name: active_forecaster_info.name,
        version: active_forecaster_info.version,
        architecture: active_forecaster_info.architecture,
        num_parameters: active_forecaster_info.num_params,
        supported_variables: active_forecaster_info
            .input_vars
            .iter()
            .chain(active_forecaster_info.output_vars.iter())
            .map(|v| format!("{:?}", v))
            .collect(),
        supported_lead_times: active_forecaster_info.supported_lead_times,
        supported_resolutions: active_forecaster_info.supported_resolutions,
    };

    Json(response)
}

async fn variables_handler() -> Json<Vec<serde_json::Value>> {
    let variables: Vec<_> = WeatherVariable::all()
        .into_iter()
        .map(|v| {
            serde_json::json!({
                "name": format!("{:?}", v),
                "unit": v.units(),
                "display_name": v.display_name(),
            })
        })
        .collect();

    Json(variables)
}

async fn verification_handler(State(_state): State<ApiState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "message": "Verification endpoint not yet implemented"
    }))
}
