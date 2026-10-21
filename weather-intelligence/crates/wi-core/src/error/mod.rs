//! # Error Handling
//!
//! Unified error types for the weather intelligence engine.

use thiserror::Error;

/// Main error type for the weather intelligence engine
#[derive(Error, Debug)]
pub enum WIError {
    #[error("Data ingestion failed: {message}")]
    DataIngestion {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Validation failed: {message}")]
    Validation { message: String },

    #[error("Storage operation failed: {message}")]
    Storage {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },

    #[error("Inference failed: {message}")]
    Inference {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Network request failed: {message}")]
    Network {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    #[error("Configuration error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Not found: {resource} with id {id}")]
    NotFound { resource: String, id: String },

    #[error("Internal error: {message}")]
    Internal {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Spatial error: {message}")]
    Spatial { message: String },

    #[error("Model error: {message}")]
    Model {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Verification error: {message}")]
    Verification { message: String },

    #[error("API error: {message}")]
    Api { message: String },

    #[error("Nilai {nama} belum diputuskan dan tidak boleh diisi tebakan: {alasan}")]
    BelumDiputuskan { nama: String, alasan: String },
}

impl WIError {
    pub fn data_ingestion(msg: impl Into<String>) -> Self {
        Self::DataIngestion {
            message: msg.into(),
            source: None,
        }
    }

    pub fn data_ingestion_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::DataIngestion {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation {
            message: msg.into(),
        }
    }

    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage {
            message: msg.into(),
            source: None,
        }
    }

    pub fn storage_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Storage {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn model_not_found(id: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model_id: id.into(),
        }
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference {
            message: msg.into(),
            source: None,
        }
    }

    pub fn inference_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Inference {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network {
            message: msg.into(),
            source: None,
        }
    }

    pub fn network_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Network {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn rate_limit(msg: impl Into<String>) -> Self {
        Self::RateLimit {
            message: msg.into(),
        }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config {
            message: msg.into(),
            source: None,
        }
    }

    pub fn config_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Config {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn not_found(resource: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
            id: id.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal {
            message: msg.into(),
            source: None,
        }
    }

    pub fn internal_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Internal {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization {
            message: msg.into(),
            source: None,
        }
    }

    pub fn serialization_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Serialization {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn spatial(msg: impl Into<String>) -> Self {
        Self::Spatial {
            message: msg.into(),
        }
    }

    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model {
            message: msg.into(),
            source: None,
        }
    }

    pub fn model_with_source(msg: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Model {
            message: msg.into(),
            source: Some(source),
        }
    }

    pub fn verification(msg: impl Into<String>) -> Self {
        Self::Verification {
            message: msg.into(),
        }
    }

    pub fn api(msg: impl Into<String>) -> Self {
        Self::Api {
            message: msg.into(),
        }
    }

    pub fn belum_diputuskan(nama: impl Into<String>, alasan: impl Into<String>) -> Self {
        Self::BelumDiputuskan {
            nama: nama.into(),
            alasan: alasan.into(),
        }
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, WIError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_creation() {
        let err = WIError::data_ingestion("test error");
        assert!(err.to_string().contains("Data ingestion failed"));
    }

    #[test]
    fn test_error_chain() {
        let source = anyhow::anyhow!("source error");
        let err = WIError::inference_with_source("inference failed", source);
        assert!(err.to_string().contains("Inference failed"));
        assert!(err.source().is_some());
    }
}
