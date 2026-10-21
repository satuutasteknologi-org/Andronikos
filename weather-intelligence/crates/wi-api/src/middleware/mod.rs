//! # API Middleware
//!
//! Rate limiting, authentication, request validation, tracing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, Algorithm, DecodingKey, TokenData, Validation};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::{info, warn};

use wi_core::{types::WeatherVariable, WIError};

/// Rate limiter state
#[derive(Clone)]
pub struct RateLimiter {
    requests: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_requests,
            window: Duration::from_secs(window_secs),
        }
    }

    pub async fn check(&self, key: &str) -> Result<(), WIError> {
        let now = Instant::now();
        let mut requests = self.requests.write();

        let entry = requests.entry(key.to_string()).or_default();

        // Remove old entries
        entry.retain(|&t| now.duration_since(t) < self.window);

        if entry.len() >= self.max_requests {
            return Err(WIError::rate_limit(format!(
                "Rate limit exceeded for {}",
                key
            )));
        }

        entry.push(now);
        Ok(())
    }
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(rate_limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP or API key
    let key = request
        .headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-api-key"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if rate_limiter.check(&key).await.is_err() {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    Ok(next.run(request).await)
}

/// Request tracing middleware
pub async fn tracing_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = Instant::now();

    let response = next.run(request).await;

    let latency = start.elapsed();
    let status = response.status();

    info!(
        method = %method,
        uri = %uri,
        status = %status,
        latency_ms = latency.as_millis(),
        "Request completed"
    );

    response
}

/// JWT Claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sub: String,         // Subject (user ID)
    pub exp: usize,          // Expiration time (Unix timestamp)
    pub iat: usize,          // Issued at (Unix timestamp)
    pub scopes: Vec<String>, // Permissions/scopes
    pub iss: String,         // Issuer
    pub aud: String,         // Audience
}

/// JWT Authentication configuration
#[derive(Clone)]
pub struct AuthConfig {
    pub public_key_pem: String,
    pub issuer: String,
    pub audience: String,
    pub required_scopes: Vec<String>,
}

impl AuthConfig {
    pub fn new(public_key_pem: String) -> Self {
        Self {
            public_key_pem,
            issuer: "weather-intelligence".to_string(),
            audience: "weather-api".to_string(),
            required_scopes: vec!["weather:read".to_string()],
        }
    }

    pub fn decoding_key(&self) -> Result<DecodingKey, WIError> {
        DecodingKey::from_rsa_pem(self.public_key_pem.as_bytes())
            .map_err(|e| WIError::config(format!("Invalid public key: {}", e)))
    }
}

/// JWT Authentication state
#[derive(Clone)]
pub struct AuthState {
    config: AuthConfig,
    decoding_key: DecodingKey,
}

impl AuthState {
    pub fn new(config: AuthConfig) -> Result<Self, WIError> {
        let decoding_key = config.decoding_key()?;
        Ok(Self {
            config,
            decoding_key,
        })
    }

    /// Validate JWT token
    pub fn validate_token(&self, token: &str) -> Result<TokenData<JwtClaims>, WIError> {
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_audience(&[&self.config.audience]);
        validation.validate_exp = true;

        decode::<JwtClaims>(token, &self.decoding_key, &validation)
            .map_err(|e| WIError::api(format!("Invalid token: {}", e)))
    }

    /// Check if token has required scopes
    pub fn check_scopes(&self, claims: &JwtClaims) -> bool {
        self.config
            .required_scopes
            .iter()
            .all(|scope| claims.scopes.contains(scope))
    }
}

/// Authentication middleware - validates JWT RS256 token
pub async fn auth_middleware(
    State(auth_state): State<Arc<AuthState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract Authorization header
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing Authorization header");
            StatusCode::UNAUTHORIZED
        })?;

    // Parse Bearer token
    let token = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
        warn!("Invalid Authorization header format");
        StatusCode::UNAUTHORIZED
    })?;

    // Validate token
    let token_data = auth_state.validate_token(token).map_err(|e| {
        warn!("Token validation failed: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    // Check scopes
    if !auth_state.check_scopes(&token_data.claims) {
        warn!("Insufficient scopes: {:?}", token_data.claims.scopes);
        return Err(StatusCode::FORBIDDEN);
    }

    // Add claims to request extensions for downstream use
    request.extensions_mut().insert(token_data.claims);

    Ok(next.run(request).await)
}

/// Optional authentication - allows unauthenticated but adds claims if present
pub async fn optional_auth_middleware(
    State(auth_state): State<Arc<AuthState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(auth_header) = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            if let Ok(token_data) = auth_state.validate_token(token) {
                if auth_state.check_scopes(&token_data.claims) {
                    request.extensions_mut().insert(token_data.claims);
                }
            }
        }
    }

    Ok(next.run(request).await)
}

/// Request validation configuration
#[derive(Clone)]
pub struct ValidationConfig {
    pub max_lead_hours: u32,
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
    pub allowed_variables: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_lead_hours: 168, // 7 days
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
            allowed_variables: WeatherVariable::all()
                .into_iter()
                .map(|v| format!("{:?}", v))
                .collect(),
        }
    }
}

/// Request validation middleware
pub async fn validation_middleware(
    State(config): State<ValidationConfig>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Only validate JSON bodies for POST /v1/predict and /v1/predict/grid
    let path = request.uri().path().to_string();
    let method = request.method();

    if method == axum::http::Method::POST && (path == "/v1/predict" || path == "/v1/predict/grid") {
        // Buffer the body for validation
        let (parts, body) = request.into_parts();
        let body_bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        // Validate JSON structure
        let json_value: serde_json::Value = serde_json::from_slice(&body_bytes).map_err(|_| {
            warn!("Invalid JSON body");
            StatusCode::BAD_REQUEST
        })?;

        // Validate based on endpoint
        if path == "/v1/predict" {
            validate_point_request(&json_value, &config)?;
        } else if path == "/v1/predict/grid" {
            validate_grid_request(&json_value, &config)?;
        }

        // Reconstruct request with validated body
        request = Request::from_parts(parts, Body::from(body_bytes));
    }

    Ok(next.run(request).await)
}

/// Validate point prediction request
fn validate_point_request(
    value: &serde_json::Value,
    config: &ValidationConfig,
) -> Result<(), StatusCode> {
    let obj = value.as_object().ok_or_else(|| {
        warn!("Request body must be an object");
        StatusCode::BAD_REQUEST
    })?;

    // latitude
    let lat = obj
        .get("latitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            warn!("Missing or invalid latitude");
            StatusCode::BAD_REQUEST
        })?;
    if lat < config.min_lat || lat > config.max_lat {
        warn!("Latitude out of bounds: {}", lat);
        return Err(StatusCode::BAD_REQUEST);
    }

    // longitude
    let lon = obj
        .get("longitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            warn!("Missing or invalid longitude");
            StatusCode::BAD_REQUEST
        })?;
    if lon < config.min_lon || lon > config.max_lon {
        warn!("Longitude out of bounds: {}", lon);
        return Err(StatusCode::BAD_REQUEST);
    }

    // lead_seconds
    let lead_seconds = obj
        .get("lead_seconds")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            warn!("Missing or invalid lead_seconds");
            StatusCode::BAD_REQUEST
        })?;
    if lead_seconds > (config.max_lead_hours as u64 * 3600) {
        warn!("Lead time exceeds maximum: {}s", lead_seconds);
        return Err(StatusCode::BAD_REQUEST);
    }

    // variables (optional)
    if let Some(vars) = obj.get("variables") {
        if let Some(arr) = vars.as_array() {
            for var in arr {
                if let Some(s) = var.as_str() {
                    if !config.allowed_variables.contains(&s.to_string()) {
                        warn!("Invalid variable: {}", s);
                        return Err(StatusCode::BAD_REQUEST);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Validate grid prediction request
fn validate_grid_request(
    value: &serde_json::Value,
    config: &ValidationConfig,
) -> Result<(), StatusCode> {
    let obj = value.as_object().ok_or_else(|| {
        warn!("Request body must be an object");
        StatusCode::BAD_REQUEST
    })?;

    let min_lat = obj.get("min_lat").and_then(|v| v.as_f64()).ok_or_else(|| {
        warn!("Missing min_lat");
        StatusCode::BAD_REQUEST
    })?;
    let min_lon = obj.get("min_lon").and_then(|v| v.as_f64()).ok_or_else(|| {
        warn!("Missing min_lon");
        StatusCode::BAD_REQUEST
    })?;
    let max_lat = obj.get("max_lat").and_then(|v| v.as_f64()).ok_or_else(|| {
        warn!("Missing max_lat");
        StatusCode::BAD_REQUEST
    })?;
    let max_lon = obj.get("max_lon").and_then(|v| v.as_f64()).ok_or_else(|| {
        warn!("Missing max_lon");
        StatusCode::BAD_REQUEST
    })?;

    if min_lat >= max_lat || min_lon >= max_lon {
        warn!("Invalid bounding box: min must be less than max");
        return Err(StatusCode::BAD_REQUEST);
    }

    if min_lat < config.min_lat
        || max_lat > config.max_lat
        || min_lon < config.min_lon
        || max_lon > config.max_lon
    {
        warn!("Bounding box out of bounds");
        return Err(StatusCode::BAD_REQUEST);
    }

    // resolution
    let resolution = obj.get("resolution").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    if resolution > 10 {
        warn!("Resolution too high: {}", resolution);
        return Err(StatusCode::BAD_REQUEST);
    }

    // lead_seconds
    let lead_seconds = obj
        .get("lead_seconds")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            warn!("Missing or invalid lead_seconds");
            StatusCode::BAD_REQUEST
        })?;
    if lead_seconds > (config.max_lead_hours as u64 * 3600) {
        warn!("Lead time exceeds maximum: {}s", lead_seconds);
        return Err(StatusCode::BAD_REQUEST);
    }

    // variables (optional)
    if let Some(vars) = obj.get("variables") {
        if let Some(arr) = vars.as_array() {
            for var in arr {
                if let Some(s) = var.as_str() {
                    if !config.allowed_variables.contains(&s.to_string()) {
                        warn!("Invalid variable: {}", s);
                        return Err(StatusCode::BAD_REQUEST);
                    }
                }
            }
        }
    }

    Ok(())
}

/// CORS configuration
pub fn cors_layer() -> tower_http::cors::CorsLayer {
    tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([http::Method::GET, http::Method::POST])
        .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
        .allow_credentials(false)
}
