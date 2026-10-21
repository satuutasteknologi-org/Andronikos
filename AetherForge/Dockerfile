# AetherForge Weather Intelligence Engine - Production Dockerfile
# Multi-stage build for minimal production image

# =============================================================================
# Stage 1: Build Environment (Rust + System Dependencies)
# =============================================================================
FROM rust:1.75-slim-bookworm AS builder

# Install system dependencies for Burn, Arrow, Parquet, and native TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libclang-dev \
    cmake \
    build-essential \
    libsasl2-dev \
    libzstd-dev \
    liblz4-dev \
    libbrotli-dev \
    libsnappy-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace Cargo.toml and Cargo.lock first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY aetherforge-core/Cargo.toml aetherforge-core/
COPY aetherforge-data/Cargo.toml aetherforge-data/
COPY aetherforge-ml/Cargo.toml aetherforge-ml/
COPY aetherforge-spatial/Cargo.toml aetherforge-spatial/
COPY aetherforge-api/Cargo.toml aetherforge-api/
COPY aetherforge-cli/Cargo.toml aetherforge-cli/
COPY aetherforge-ui/Cargo.toml aetherforge-ui/

# Create dummy source files to cache dependencies
RUN mkdir -p aetherforge-core/src aetherforge-data/src aetherforge-ml/src \
    aetherforge-spatial/src aetherforge-api/src aetherforge-cli/src aetherforge-ui/src && \
    echo "// dummy" > aetherforge-core/src/lib.rs && \
    echo "// dummy" > aetherforge-data/src/lib.rs && \
    echo "// dummy" > aetherforge-ml/src/lib.rs && \
    echo "// dummy" > aetherforge-spatial/src/lib.rs && \
    echo "// dummy" > aetherforge-api/src/lib.rs && \
    echo "// dummy" > aetherforge-cli/src/main.rs && \
    echo "// dummy" > aetherforge-ui/src/main.rs

# Pre-build dependencies (this layer will be cached)
RUN cargo build --release --workspace 2>/dev/null || true

# Now copy actual source code
COPY . .

# Build the actual project
RUN cargo build --release --workspace --bin aetherforge-serve --bin aetherforge-cli

# =============================================================================
# Stage 2: Runtime Environment (Minimal)
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libzstd1 \
    liblz4-1 \
    libbrotli1 \
    libsnappy1 \
    libprotobuf32 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash aetherforge

# Create directories
RUN mkdir -p /var/aetherforge/{data,models,logs,checkpoints} && \
    chown -R aetherforge:aetherforge /var/aetherforge

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/target/release/aetherforge-serve /app/
COPY --from=builder /app/target/release/aetherforge-cli /app/

# Copy config template
COPY config.example.toml /app/config.toml

# Switch to non-root user
USER aetherforge

# Environment variables
ENV AETHERFORGE_CONFIG=/app/config.toml \
    AETHERFORGE_DATA_DIR=/var/aetherforge/data \
    RUST_LOG=info \
    NOAA_API_TOKEN="" \
    GRAPHCAST_WEIGHTS=/var/aetherforge/models/graphcast_burn.safetensors

# Expose ports
EXPOSE 8085 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8085/health || exit 1

# Entry point
ENTRYPOINT ["/app/aetherforge-serve"]
