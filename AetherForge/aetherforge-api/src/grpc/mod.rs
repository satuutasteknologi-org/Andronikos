//! # gRPC Service Definitions
//! 
//! Protocol Buffers and gRPC service implementation for weather prediction.
//! 
//! Generated from aetherforge.proto using tonic-build.

use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use aetherforge_core::{GeoCoord, WeatherVariable};
use aetherforge_ml::models::AetherPredictor;

#[derive(Debug, Clone)]
pub struct WeatherService {
    predictor: Arc<dyn AetherPredictor>,
}

impl WeatherService {
    pub fn new(predictor: Arc<dyn AetherPredictor>) -> Self {
        Self { predictor }
    }
}

impl WeatherService {
    pub async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        
        info!("gRPC Predict: lat={}, lon={}, lead_hours={}", req.latitude, req.longitude, req.lead_hours);
        
        let predictions_map = self.predictor.predict(req.latitude, req.longitude, req.lead_hours * 3600);
        
        let predictions: Vec<Prediction> = predictions_map
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
                Prediction {
                    variable,
                    value,
                    unit: unit.to_string(),
                    confidence: 0.9,
                }
            })
            .collect();
        
        let response = PredictResponse {
            latitude: req.latitude,
            longitude: req.longitude,
            valid_time: Some(Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
            predictions,
        };
        
        Ok(Response::new(response))
    }
    
    pub async fn predict_grid(
        &self,
        request: Request<GridPredictRequest>,
    ) -> Result<Response<GridPredictResponse>, Status> {
        let req = request.into_inner();
        
        info!("gRPC Grid Predict: {}x{} cells, lead_hours={}", req.resolution, req.resolution, req.lead_hours);
        
        let grid_predictions = self.predictor.predict_grid(
            req.min_lat,
            req.min_lon,
            req.max_lat,
            req.max_lon,
            req.resolution,
            req.lead_hours * 3600,
        );
        
        let cells: Vec<GridCell> = grid_predictions
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
        
        let response = GridPredictResponse {
            cells,
            resolution: req.resolution,
            lead_hours: req.lead_hours,
        };
        
        Ok(Response::new(response))
    }
    
    pub async fn stream_predict(
        &self,
        request: Request<StreamPredictRequest>,
    ) -> Result<Response<tonic::Streaming<PredictResponse>>, Status> {
        let req = request.into_inner();
        
        let lead_seconds = req.lead_hours * 3600;
        let interval = req.interval_seconds.unwrap_or(3600) as u64;
        let max_predictions = req.max_predictions.unwrap_or(168) as usize;
        
        let predictor = self.predictor.clone();
        let lat = req.latitude;
        let lon = req.longitude;
        
        let stream = async_stream::stream! {
            for i in 0..max_predictions {
                let current_lead = lead_seconds + (i as u32 * interval as u32);
                let predictions_map = predictor.predict(lat, lon, current_lead);
                
                let predictions: Vec<Prediction> = predictions_map
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
                        Prediction {
                            variable,
                            value,
                            unit: unit.to_string(),
                            confidence: 0.9,
                        }
                    })
                    .collect();
                
                yield Ok(PredictResponse {
                    latitude: lat,
                    longitude: lon,
                    valid_time: Some(Timestamp {
                        seconds: (chrono::Utc::now() + chrono::Duration::seconds(current_lead as i64)).timestamp(),
                        nanos: 0,
                    }),
                    predictions,
                });
                
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        };
        
        Ok(Response::new(Box::pin(stream)))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timestamp {
    pub seconds: i64,
    pub nanos: i32,
}

impl Timestamp {
    pub fn now() -> Self {
        let now = chrono::Utc::now();
        Self {
            seconds: now.timestamp(),
            nanos: now.timestamp_subsec_nanos() as i32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_hours: u32,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PredictResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub valid_time: Option<Timestamp>,
    pub predictions: Vec<Prediction>,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub variable: String,
    pub value: f32,
    pub unit: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct GridPredictRequest {
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
    pub resolution: u8,
    pub lead_hours: u32,
}

#[derive(Debug, Clone)]
pub struct GridPredictResponse {
    pub cells: Vec<GridCell>,
    pub resolution: u8,
    pub lead_hours: u32,
}

#[derive(Debug, Clone)]
pub struct GridCell {
    pub lat: f64,
    pub lon: f64,
    pub geohash: String,
    pub predictions: Vec<Prediction>,
}

#[derive(Debug, Clone)]
pub struct StreamPredictRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub lead_hours: u32,
    pub interval_seconds: Option<u32>,
    pub max_predictions: Option<u32>,
}

// Manual protobuf definitions for tonic-build compatibility
mod proto {
    tonic::include_proto!("aetherforge");
}

pub use proto::aetherforge_weather_service_server::{AetherforgeWeatherService, AetherforgeWeatherServiceServer};

impl AetherforgeWeatherService for WeatherService {
    async fn predict(
        &self,
        request: Request<proto::PredictRequest>,
    ) -> Result<Response<proto::PredictResponse>, Status> {
        let req = request.into_inner();
        let predictions_map = self.predictor.predict(req.latitude, req.longitude, req.lead_hours * 3600);
        
        let predictions: Vec<proto::Prediction> = predictions_map
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
                proto::Prediction {
                    variable,
                    value,
                    unit: unit.to_string(),
                    confidence: 0.9,
                }
            })
            .collect();
        
        Ok(Response::new(proto::PredictResponse {
            latitude: req.latitude,
            longitude: req.longitude,
            valid_time: Some(proto::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
            predictions,
        }))
    }
    
    async fn predict_grid(
        &self,
        request: Request<proto::GridPredictRequest>,
    ) -> Result<Response<proto::GridPredictResponse>, Status> {
        let req = request.into_inner();
        let grid_predictions = self.predictor.predict_grid(
            req.min_lat,
            req.min_lon,
            req.max_lat,
            req.max_lon,
            req.resolution,
            req.lead_hours * 3600,
        );
        
        let cells: Vec<proto::GridCell> = grid_predictions
            .into_iter()
            .map(|p| {
                let coord = GeoCoord::new(p.lat, p.lon);
                proto::GridCell {
                    lat: p.lat,
                    lon: p.lon,
                    geohash: coord.to_geo_hash(6),
                    predictions: p.predictions.into_iter().collect(),
                }
            })
            .collect();
        
        Ok(Response::new(proto::GridPredictResponse {
            cells,
            resolution: req.resolution,
            lead_hours: req.lead_hours,
        }))
    }
    
    type StreamPredictStream = tonic::Streaming<proto::PredictResponse>;
    
    async fn stream_predict(
        &self,
        request: Request<proto::StreamPredictRequest>,
    ) -> Result<Response<Self::StreamPredictStream>, Status> {
        let req = request.into_inner();
        let lead_seconds = req.lead_hours * 3600;
        let interval = req.interval_seconds.unwrap_or(3600) as u64;
        let max_predictions = req.max_predictions.unwrap_or(168) as usize;
        
        let predictor = self.predictor.clone();
        let lat = req.latitude;
        let lon = req.longitude;
        
        let stream = async_stream::stream! {
            for i in 0..max_predictions {
                let current_lead = lead_seconds + (i as u32 * interval as u32);
                let predictions_map = predictor.predict(lat, lon, current_lead);
                
                let predictions: Vec<proto::Prediction> = predictions_map
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
                        proto::Prediction {
                            variable,
                            value,
                            unit: unit.to_string(),
                            confidence: 0.9,
                        }
                    })
                    .collect();
                
                yield Ok(proto::PredictResponse {
                    latitude: lat,
                    longitude: lon,
                    valid_time: Some(proto::Timestamp {
                        seconds: (chrono::Utc::now() + chrono::Duration::seconds(current_lead as i64)).timestamp(),
                        nanos: 0,
                    }),
                    predictions,
                });
                
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        };
        
        Ok(Response::new(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aetherforge_ml::models::DefaultPredictor;
    
    #[tokio::test]
    async fn test_grpc_service() {
        let predictor = Arc::new(DefaultPredictor::new());
        let service = WeatherService::new(predictor);
        
        let request = Request::new(PredictRequest {
            latitude: 52.52,
            longitude: 13.405,
            lead_hours: 24,
            variables: vec!["temperature".to_string()],
        });
        
        let response = service.predict(request).await.unwrap();
        assert_eq!(response.get_ref().latitude, 52.52);
    }
}
