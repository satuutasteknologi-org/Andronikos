-- Initial schema for weather_intelligence database
-- Requires PostGIS extension

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Stations table with PostGIS geography
CREATE TABLE IF NOT EXISTS stations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    elevation DOUBLE PRECISION,
    country_code VARCHAR(10),
    timezone VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_stations_country ON stations (country_code);

-- Observations table with PostGIS geography
CREATE TABLE IF NOT EXISTS observations (
    id BIGSERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    altitude DOUBLE PRECISION,
    station_id VARCHAR(50) REFERENCES stations(id) ON DELETE SET NULL,
    source INTEGER NOT NULL, -- DataSource enum as integer
    quality INTEGER NOT NULL DEFAULT 0, -- QualityFlag enum as integer
    variables JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_observations_time ON observations (time DESC);
CREATE INDEX IF NOT EXISTS idx_observations_geom ON observations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_observations_station ON observations (station_id);
CREATE INDEX IF NOT EXISTS idx_observations_source ON observations (source);
CREATE INDEX IF NOT EXISTS idx_observations_variables ON observations USING GIN (variables);

-- Composite unique constraint for deduplication
CREATE UNIQUE INDEX IF NOT EXISTS uq_observations_dedup 
    ON observations (time, lat, lon, station_id);

-- Forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    init_time TIMESTAMPTZ NOT NULL,
    valid_time TIMESTAMPTZ NOT NULL,
    lead_seconds INTEGER NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    variables JSONB NOT NULL DEFAULT '{}',
    ensemble_members INTEGER,
    probability JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_model_time ON forecasts (model_id, init_time);
CREATE INDEX IF NOT EXISTS idx_forecasts_valid_time ON forecasts (valid_time);
CREATE INDEX IF NOT EXISTS idx_forecasts_geom ON forecasts USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_forecasts_variables ON forecasts USING GIN (variables);

-- Verification results table
CREATE TABLE IF NOT EXISTS verification_results (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    baseline_id VARCHAR(100),
    metric_name VARCHAR(50) NOT NULL,
    variable VARCHAR(50) NOT NULL,
    lead_hours INTEGER NOT NULL,
    region VARCHAR(50),
    value DOUBLE PRECISION NOT NULL,
    sample_size INTEGER,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_verification_model ON verification_results (model_id);
CREATE INDEX IF NOT EXISTS idx_verification_metric ON verification_results (metric_name, variable, lead_hours);

-- Model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    architecture VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL, -- 'burn', 'onnx', 'pytorch'
    model_path VARCHAR(500) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'archived', 'deprecated'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry (status);

-- Data sync checkpoints
CREATE TABLE IF NOT EXISTS sync_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    region_key VARCHAR(100) NOT NULL,
    last_sync_time TIMESTAMPTZ NOT NULL,
    last_record_hash VARCHAR(64),
    records_synced BIGINT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'ok', -- 'ok', 'error', 'partial'
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, region_key)
);

-- API keys for authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    scopes TEXT[] DEFAULT ARRAY['weather:read'],
    rate_limit_rps INTEGER DEFAULT 100,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys (expires_at);

-- Trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers
DROP TRIGGER IF EXISTS update_stations_updated_at ON stations;
CREATE TRIGGER update_stations_updated_at
    BEFORE UPDATE ON stations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_model_registry_updated_at ON model_registry;
CREATE TRIGGER update_model_registry_updated_at
    BEFORE UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_sync_checkpoints_updated_at ON sync_checkpoints;
CREATE TRIGGER update_sync_checkpoints_updated_at
    BEFORE UPDATE ON sync_checkpoints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();