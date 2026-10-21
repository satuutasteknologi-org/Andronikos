//! # gRPC Service Definitions
//!
//! Protocol Buffers and gRPC service implementation for weather prediction.

use std::pin::Pin;
use std::sync::Arc;
use tonic::codegen::tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::info;

use prost_types::Timestamp;
use wi_core::types::{Bbox, GridPredictRequest as CoreGridPredictRequest, WeatherVariable};
use wi_model::traits::Forecaster;

#[allow(clippy::result_large_err)]
pub mod weather {
    tonic::include_proto!("weather.v1");
}

use weather::weather_service_server::{
    WeatherService as WeatherServiceTrait, WeatherServiceServer,
};
use weather::{
    CellPrediction, GridCell, GridPredictRequest, GridPredictResponse, HealthRequest,
    HealthResponse, ModelInfoRequest, ModelInfoResponse, PredictRequest, PredictResponse,
    Prediction, StreamPredictRequest,
};

fn timestamp_now() -> Timestamp {
    let now = chrono::Utc::now();
    Timestamp {
        seconds: now.timestamp(),
        nanos: now.timestamp_subsec_nanos() as i32,
    }
}

#[derive(Debug, Clone)]
pub struct WeatherService {
    forecaster: Arc<dyn Forecaster + Send + Sync>,
}

impl WeatherService {
    pub fn new(forecaster: Arc<dyn Forecaster + Send + Sync>) -> Self {
        Self { forecaster }
    }
}

type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl WeatherServiceTrait for WeatherService {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();

        info!(
            "gRPC Predict: lat={}, lon={}, lead_seconds={}",
            req.latitude, req.longitude, req.lead_seconds
        );

        let weather_vars: Vec<WeatherVariable> = req
            .variables
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

        let fusion_output = self
            .forecaster
            .predict_point(req.latitude, req.longitude, req.lead_seconds, &weather_vars)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let predictions: Vec<Prediction> = fusion_output
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
                Prediction {
                    variable: format!("{:?}", variable),
                    value: fused.weighted_mean,
                    unit: unit.to_string(),
                    confidence: fused.exceedance_probability.unwrap_or(0.5),
                    quantiles: None,
                }
            })
            .collect();

        let response = PredictResponse {
            latitude: req.latitude,
            longitude: req.longitude,
            lead_seconds: req.lead_seconds,
            lead_hours: (req.lead_seconds as f32 / 3600.0) as u32,
            valid_time: Some(timestamp_now()),
            predictions,
            model_version: "GraphCast-0.1.0".to_string(),
        };

        Ok(Response::new(response))
    }

    async fn predict_grid(
        &self,
        request: Request<GridPredictRequest>,
    ) -> Result<Response<GridPredictResponse>, Status> {
        let req = request.into_inner();

        info!(
            "gRPC Grid Predict: {}x{} cells, lead_seconds={}",
            req.resolution, req.resolution, req.lead_seconds
        );

        let weather_vars: Vec<WeatherVariable> = WeatherVariable::surface_variables();

        let core_request = CoreGridPredictRequest::new(
            Bbox::new(req.min_lon, req.min_lat, req.max_lon, req.max_lat),
            req.resolution.try_into().unwrap_or(1),
            req.lead_seconds,
            weather_vars,
        );

        let grid_predictions = self
            .forecaster
            .predict_grid(core_request)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let cells: Vec<GridCell> = grid_predictions
            .into_iter()
            .map(|p| GridCell {
                lat: p.lat,
                lon: p.lon,
                cell_id: p.cell_id,
                predictions: p
                    .predictions
                    .into_iter()
                    .map(|(v, val)| CellPrediction {
                        variable: format!("{:?}", v),
                        value: val,
                    })
                    .collect(),
                quantiles: None,
                confidence: 0.5,
            })
            .collect();

        let response = GridPredictResponse {
            cells,
            resolution: req.resolution,
            lead_seconds: req.lead_seconds,
            lead_hours: (req.lead_seconds as f32 / 3600.0) as u32,
            model_version: "GraphCast-0.1.0".to_string(),
            valid_time: Some(timestamp_now()),
        };

        Ok(Response::new(response))
    }

    type StreamPredictStream = BoxStream<PredictResponse>;

    async fn stream_predict(
        &self,
        request: Request<StreamPredictRequest>,
    ) -> Result<Response<Self::StreamPredictStream>, Status> {
        let req = request.into_inner();
        let forecaster = self.forecaster.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            let mut lead = req.lead_seconds;
            let max_lead = req.lead_seconds + req.interval_seconds * req.max_predictions;

            while lead <= max_lead {
                let weather_vars = WeatherVariable::surface_variables();
                match forecaster
                    .predict_point(req.latitude, req.longitude, lead, &weather_vars)
                    .await
                {
                    Ok(fusion_output) => {
                        let predictions: Vec<Prediction> = fusion_output
                            .variables
                            .into_iter()
                            .map(|(variable, fused)| {
                                let unit = match variable {
                                    WeatherVariable::Temperature => "K",
                                    WeatherVariable::RelativeHumidity => "%",
                                    WeatherVariable::Precipitation => "mm",
                                    WeatherVariable::WindSpeed => "m/s",
                                    WeatherVariable::WindDirection => "degrees",
                                    WeatherVariable::Pressure
                                    | WeatherVariable::SurfacePressure => "Pa",
                                    WeatherVariable::CloudCover => "%",
                                    WeatherVariable::DewPoint => "K",
                                    WeatherVariable::Visibility => "m",
                                    _ => "",
                                };
                                Prediction {
                                    variable: format!("{:?}", variable),
                                    value: fused.weighted_mean,
                                    unit: unit.to_string(),
                                    confidence: fused.exceedance_probability.unwrap_or(0.5),
                                    quantiles: None,
                                }
                            })
                            .collect();

                        let response = PredictResponse {
                            latitude: req.latitude,
                            longitude: req.longitude,
                            lead_seconds: lead,
                            lead_hours: (lead as f32 / 3600.0) as u32,
                            valid_time: Some(timestamp_now()),
                            predictions,
                            model_version: "GraphCast-0.1.0".to_string(),
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                        break;
                    }
                }
                lead += req.interval_seconds;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        let info = self.forecaster.model_info();
        let response = ModelInfoResponse {
            name: info.name,
            version: info.version,
            architecture: info.architecture,
            num_parameters: info.num_params as u64,
            supported_variables: info.input_vars.iter().map(|v| format!("{:?}", v)).collect(),
            supported_lead_times: info.supported_lead_times,
            supported_resolutions: info.supported_resolutions,
        };
        Ok(Response::new(response))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let response = HealthResponse {
            status: "healthy".to_string(),
            model_loaded: true,
            timestamp: Some(timestamp_now()),
            uptime_seconds: 0,
        };
        Ok(Response::new(response))
    }
}

/// Create gRPC service
pub fn create_grpc_service(
    forecaster: Arc<dyn Forecaster + Send + Sync>,
) -> WeatherServiceServer<WeatherService> {
    WeatherServiceServer::new(WeatherService::new(forecaster))
}
