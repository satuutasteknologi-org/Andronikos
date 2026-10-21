//! # Unified Error Handling for AetherForge
//!
//! This module provides a comprehensive error handling system using `anyhow` and `thiserror`
//! with proper error categorization, context propagation, and recovery suggestions.

use std::io;
use std::sync::Arc;
use thiserror::Error;

/// Result type alias using AetherForge error
pub type Result<T> = std::result::Result<T, AetherError>;

/// Main error enum for AetherForge
#[derive(Error, Debug)]
pub enum AetherError {
    #[error("Data ingestion error: {context}")]
    DataIngestion {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Data preprocessing error: {context}")]
    Preprocessing {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("ML model error: {context}")]
    Model {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Training error: {context}")]
    Training {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Storage error: {context}")]
    Storage {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Configuration error: {context}")]
    Config {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("API error: {context}")]
    Api {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Geospatial error: {context}")]
    Geospatial {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Network error: {context}")]
    Network {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Serialization error: {context}")]
    Serialization {
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: String,
    },

    #[error("Validation error: {context}")]
    Validation { context: String },

    #[error("Not found: {resource} ({id})")]
    NotFound { resource: String, id: String },

    #[error("Rate limit exceeded: {source_name}")]
    RateLimitExceeded {
        source_name: String,
        retry_after_secs: u64,
    },

    #[error("Authentication failed: {context}")]
    Authentication { context: String },

    #[error("Timeout after {duration_secs}s: {context}")]
    Timeout { duration_secs: u64, context: String },

    #[error("Internal error: {context}")]
    Internal { context: String },
}

impl AetherError {
    pub fn data_ingestion<S: Into<String>>(context: S) -> Self {
        Self::DataIngestion {
            source: None,
            context: context.into(),
        }
    }

    pub fn preprocessing<S: Into<String>>(context: S) -> Self {
        Self::Preprocessing {
            source: None,
            context: context.into(),
        }
    }

    pub fn model<S: Into<String>>(context: S) -> Self {
        Self::Model {
            source: None,
            context: context.into(),
        }
    }

    pub fn training<S: Into<String>>(context: S) -> Self {
        Self::Training {
            source: None,
            context: context.into(),
        }
    }

    pub fn storage<S: Into<String>>(context: S) -> Self {
        Self::Storage {
            source: None,
            context: context.into(),
        }
    }

    pub fn config<S: Into<String>>(context: S) -> Self {
        Self::Config {
            source: None,
            context: context.into(),
        }
    }

    pub fn api<S: Into<String>>(context: S) -> Self {
        Self::Api {
            source: None,
            context: context.into(),
        }
    }

    pub fn geospatial<S: Into<String>>(context: S) -> Self {
        Self::Geospatial {
            source: None,
            context: context.into(),
        }
    }

    pub fn serialization<S: Into<String>>(context: S) -> Self {
        Self::Serialization {
            source: None,
            context: context.into(),
        }
    }

    pub fn network<S: Into<String>>(context: S) -> Self {
        Self::Network {
            source: None,
            context: context.into(),
        }
    }

    pub fn not_found(resource: &str, id: &str) -> Self {
        Self::NotFound {
            resource: resource.into(),
            id: id.into(),
        }
    }

    pub fn rate_limit(source_name: &str, retry_after_secs: u64) -> Self {
        Self::RateLimitExceeded {
            source_name: source_name.into(),
            retry_after_secs,
        }
    }

    pub fn timeout(duration_secs: u64, context: &str) -> Self {
        Self::Timeout {
            duration_secs,
            context: context.into(),
        }
    }

    pub fn internal<S: Into<String>>(context: S) -> Self {
        Self::Internal {
            context: context.into(),
        }
    }

    pub fn with_source<E: std::error::Error + Send + Sync + 'static>(mut self, source: E) -> Self {
        let source = Some(Box::new(source) as Box<_>);
        match &mut self {
            Self::DataIngestion { source: s, .. } => *s = source,
            Self::Preprocessing { source: s, .. } => *s = source,
            Self::Model { source: s, .. } => *s = source,
            Self::Training { source: s, .. } => *s = source,
            Self::Storage { source: s, .. } => *s = source,
            Self::Config { source: s, .. } => *s = source,
            Self::Api { source: s, .. } => *s = source,
            Self::Geospatial { source: s, .. } => *s = source,
            Self::Network { source: s, .. } => *s = source,
            _ => {}
        }
        self
    }

    pub fn context(&self) -> String {
        self.to_string()
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network { .. } | Self::RateLimitExceeded { .. } | Self::Timeout { .. }
        )
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::DataIngestion { .. } => 1,
            Self::Preprocessing { .. } => 2,
            Self::Model { .. } => 3,
            Self::Training { .. } => 4,
            Self::Storage { .. } => 5,
            Self::Config { .. } => 6,
            Self::Api { .. } => 7,
            Self::Geospatial { .. } => 8,
            Self::Network { .. } => 9,
            Self::Serialization { .. } => 10,
            Self::Validation { .. } => 11,
            Self::NotFound { .. } => 12,
            Self::RateLimitExceeded { .. } => 13,
            Self::Authentication { .. } => 14,
            Self::Timeout { .. } => 15,
            Self::Internal { .. } => 99,
        }
    }
}

impl From<io::Error> for AetherError {
    fn from(e: io::Error) -> Self {
        Self::storage(e.to_string()).with_source(e)
    }
}

impl From<serde_json::Error> for AetherError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization {
            source: None,
            context: format!("JSON: {}", e),
        }
        .with_source(e)
    }
}

impl From<toml::de::Error> for AetherError {
    fn from(e: toml::de::Error) -> Self {
        Self::Config {
            source: None,
            context: format!("TOML: {}", e),
        }
        .with_source(e)
    }
}

impl From<reqwest::Error> for AetherError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            Self::timeout(30, "HTTP request").with_source(e)
        } else if e.is_connect() {
            Self::network("Connection failed").with_source(e)
        } else if e.is_request() {
            Self::network("Request error").with_source(e)
        } else {
            Self::network("HTTP error").with_source(e)
        }
    }
}

/// Wrapper for errors that need to be shared across threads
pub type SharedError = Arc<AetherError>;

pub fn shared<E: Into<AetherError>>(e: E) -> SharedError {
    Arc::new(e.into())
}

impl AetherError {
    pub fn into_tracing_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("error_type", format!("{:?}", self)),
            ("error_msg", self.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = AetherError::data_ingestion("Failed to fetch data");
        assert!(err.to_string().contains("Data ingestion"));
    }

    #[test]
    fn test_error_with_source() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = AetherError::storage("Config file").with_source(io_err);
        assert!(err.to_string().contains("Config file"));
    }

    #[test]
    fn test_retryable() {
        assert!(AetherError::network("test").is_retryable());
        assert!(AetherError::rate_limit("test", 60).is_retryable());
        assert!(!AetherError::validation("test").is_retryable());
    }
}
