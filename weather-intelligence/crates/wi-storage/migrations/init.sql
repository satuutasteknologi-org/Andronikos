-- Weather Intelligence Database Schema
-- PostgreSQL + PostGIS

CREATE EXTENSION IF NOT EXISTS postgis;

-- Stations table
CREATE TABLE IF NOT EXISTS stations (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    geom GEOGRAPHY(POINT, 4326) NOT NULL,
    elevation_m REAL,
    country_code VARCHAR(2),
    timezone VARCHAR(64),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);

-- Observations table
CREATE TABLE IF NOT EXISTS observations (
    id BIGSERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    altitude REAL,
    station_id VARCHAR(64),
    source INTEGER NOT NULL,
    quality INTEGER NOT NULL,
    variables JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_observations_time ON observations (time);
CREATE INDEX IF NOT EXISTS idx_observations_station ON observations (station_id);
CREATE INDEX IF NOT EXISTS idx_observations_geom ON observations USING GIST (ST_MakePoint(lon, lat)::geography);

-- Forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    init_time TIMESTAMPTZ NOT NULL,
    valid_time TIMESTAMPTZ NOT NULL,
    lead_seconds INTEGER NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    variables JSONB NOT NULL,
    ensemble_members INTEGER,
    probability JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_model ON forecasts (model_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_valid_time ON forecasts (valid_time);
CREATE INDEX IF NOT EXISTS idx_forecasts_geom ON forecasts USING GIST (ST_MakePoint(lon, lat)::geography);

-- Verification results table
CREATE TABLE IF NOT EXISTS verification_results (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    variable INTEGER NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    sample_count INTEGER NOT NULL,
    mae REAL,
    rmse REAL,
    bias REAL,
    correlation REAL,
    brier_score REAL,
    crps REAL,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_verification_model ON verification_results (model_id);
CREATE INDEX IF NOT EXISTS idx_verification_period ON verification_results (period_start, period_end);
