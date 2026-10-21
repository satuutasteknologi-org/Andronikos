//! # REST API Handlers
//! 
//! HTTP handlers for weather prediction endpoints.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::info;

use aetherforge_core::{AetherError, GeoCoord, WeatherVariable};
use aetherforge_ml::models::AetherPredictor;

#[derive(Clone)]
pub struct ApiState {
    pub model: Arc<dyn AetherPredictor>,
    pub model_loaded: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_seconds: u32,  // Supports 30s, 60s, 300s, 3600s, etc.
    pub variables: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub valid_time: DateTime<Utc>,
    pub lead_seconds: u32,
    pub lead_hours: f32,  // Convenience field
    pub predictions: Vec<VariablePrediction>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VariablePrediction {
    pub variable: String,
    pub value: f32,
    pub unit: String,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridPredictionRequest {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
    pub resolution: u8,
    pub lead_seconds: u32,  // Supports sub-hourly granularity
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridPredictionResponse {
    pub grid: Vec<GridCell>,
    pub resolution: u8,
    pub lead_seconds: u32,
    pub lead_hours: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GridCell {
    pub lat: f64,
    pub lon: f64,
    pub geohash: String,
    pub predictions: Vec<(String, f32)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub hidden_dim: usize,
    pub num_parameters: usize,
    pub supported_variables: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub timestamp: DateTime<Utc>,
}

pub fn create_routes(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .route("/predict/grid", post(grid_predict_handler))
        .route("/model/info", get(model_info_handler))
        .route("/variables", get(variables_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<ApiState>) -> impl IntoResponse {
    let response = HealthResponse {
        status: if state.model_loaded { "healthy" } else { "degraded" }.to_string(),
        model_loaded: state.model_loaded,
        timestamp: Utc::now(),
    };
    
    if state.model_loaded {
        (StatusCode::OK, Json(response))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response))
    }
}

async fn predict_handler(
    State(state): State<ApiState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, ApiError> {
    if !state.model_loaded {
        return Err(AetherError::api("Model not loaded").into());
    }
    
    info!("Prediction request: lat={}, lon={}, lead={}s", 
          request.latitude, request.longitude, request.lead_seconds);
    
    let predictions_map = state.model.predict(request.latitude, request.longitude, request.lead_seconds);
    
    let predictions: Vec<VariablePrediction> = predictions_map
        .into_iter()
        .map(|(variable, value)| {
            let unit = match variable.as_str() {
                "Temperature" => "K",
                "RelativeHumidity" => "%",
                "WindSpeed" => "m/s",
                "Precipitation" => "mm/h",
                "Pressure" => "hPa",
                _ => "",
            };
            VariablePrediction {
                variable,
                value,
                unit: unit.to_string(),
                confidence: 0.9,
            }
        })
        .collect();
    
    let response = PredictionResponse {
        latitude: request.latitude,
        longitude: request.longitude,
        valid_time: Utc::now(),
        lead_seconds: request.lead_seconds,
        lead_hours: request.lead_seconds as f32 / 3600.0,
        predictions,
    };
    
    Ok(Json(response))
}

async fn grid_predict_handler(
    State(state): State<ApiState>,
    Json(request): Json<GridPredictionRequest>,
) -> Result<Json<GridPredictionResponse>, ApiError> {
    if !state.model_loaded {
        return Err(AetherError::api("Model not loaded").into());
    }
    
    info!("Grid prediction: {}x{} from ({},{}) to ({},{})",
          request.resolution, request.resolution,
          request.min_lat, request.min_lon, request.max_lat, request.max_lon);
    
    let grid_predictions = state.model.predict_grid(
        request.min_lat,
        request.min_lon,
        request.max_lat,
        request.max_lon,
        request.resolution,
        request.lead_seconds,
    );
    
    let grid: Vec<GridCell> = grid_predictions
        .into_iter()
        .map(|p| {
            let coord = GeoCoord::new(p.lat, p.lon);
            GridCell {
                lat: p.lat,
                lon: p.lon,
                geohash: coord.to_geo_hash(6),
                predictions: p.predictions.into_iter().collect(),
            }
        })
        .collect();
    
    let response = GridPredictionResponse {
        grid,
        resolution: request.resolution,
        lead_seconds: request.lead_seconds,
        lead_hours: request.lead_seconds as f32 / 3600.0,
    };
    
    Ok(Json(response))
}

async fn model_info_handler(State(state): State<ApiState>) -> Json<ModelInfo> {
    let info = ModelInfo {
        name: "AetherForge".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        architecture: "GraphCast-v2".to_string(),
        hidden_dim: 256,
        num_parameters: if state.model_loaded { state.model.num_params() } else { 0 },
        supported_variables: vec![
            "temperature".to_string(),
            "relative_humidity".to_string(),
            "precipitation".to_string(),
            "wind_speed".to_string(),
            "pressure".to_string(),
        ],
    };
    
    Json(info)
}

async fn variables_handler() -> Json<Vec<serde_json::Value>> {
    let variables: Vec<_> = WeatherVariable::all()
        .into_iter()
        .map(|v| {
            serde_json::json!({
                "name": format!("{:?}", v),
                "unit": v.units()
            })
        })
        .collect();
    
    Json(variables)
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

#[derive(Debug)]
pub struct ApiError(pub AetherError);

impl From<AetherError> for ApiError {
    fn from(err: AetherError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error) = match &self.0 {
            AetherError::NotFound { .. } => (StatusCode::NOT_FOUND, self.0.to_string()),
            AetherError::Validation { .. } => (StatusCode::BAD_REQUEST, self.0.to_string()),
            AetherError::Api { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };
        
        let body = Json(ErrorResponse {
            error,
            code: format!("{:?}", self.0),
        });
        
        (status, body).into_response()
    }
}