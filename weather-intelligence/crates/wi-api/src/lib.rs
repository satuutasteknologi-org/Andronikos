//! # Weather Intelligence API Server
//!
//! REST and gRPC API for weather prediction inference.

#![forbid(unsafe_code)]
#![recursion_limit = "512"]

pub mod cache;
pub mod grpc;
pub mod middleware;
pub mod rest;

#[cfg(test)]
mod tests {
    #[test]
    fn test_api() {
        // TODO: Add tests
    }
}
