-- =============================================================================
-- ML-Weather: PostGIS Migration Schema
-- Trip 4 — wi-storage/migrations
-- =============================================================================
-- Persyaratan: PostgreSQL 14+ dengan ekstensi PostGIS 3.x
-- Jalankan: psql -U <user> -d <db> -f 001_initial_postgis_schema.sql
-- =============================================================================

-- Aktifkan ekstensi
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE; -- opsional, untuk time-series
-- =============================================================================
-- TABEL: stations
-- Menyimpan metadata stasiun pengamatan (BMKG, NOAA, dsb.)
-- =============================================================================
CREATE TABLE IF NOT EXISTS stations (
    id             TEXT        PRIMARY KEY,
    name           TEXT        NOT NULL,
    source         TEXT        NOT NULL,          -- 'BMKG' | 'NOAA' | 'WMO'
    lat            DOUBLE PRECISION NOT NULL,
    lon            DOUBLE PRECISION NOT NULL,
    elevation_m    DOUBLE PRECISION,
    country_code   CHAR(2),
    timezone       TEXT        NOT NULL DEFAULT 'UTC',
    metadata       JSONB       NOT NULL DEFAULT '{}',
    geom           GEOGRAPHY(Point, 4326) GENERATED ALWAYS AS
                       (ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) STORED,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indeks spasial pada stasiun
CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_stations_source ON stations (source);

-- =============================================================================
-- TABEL: observations
-- Pengamatan cuaca historis — dipartisi per bulan untuk performa
-- =============================================================================
CREATE TABLE IF NOT EXISTS observations (
    id             BIGSERIAL,
    station_id     TEXT        REFERENCES stations(id) ON DELETE SET NULL,
    time           TIMESTAMPTZ NOT NULL,
    lat            DOUBLE PRECISION NOT NULL,
    lon            DOUBLE PRECISION NOT NULL,
    source         TEXT        NOT NULL,          -- 'BMKG' | 'GFS' | 'ERA5' (training only)
    variable       TEXT        NOT NULL,          -- 'Temperature' | 'Precipitation' | ...
    value          DOUBLE PRECISION NOT NULL,
    quality_flag   TEXT        NOT NULL DEFAULT 'Valid',
    unit           TEXT        NOT NULL DEFAULT '',
    geom           GEOGRAPHY(Point, 4326) GENERATED ALWAYS AS
                       (ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) STORED,
    PRIMARY KEY (id, time)
) PARTITION BY RANGE (time);

-- Partisi per tahun (auto-generate setiap tahun dengan stored procedure di bawah)
CREATE TABLE IF NOT EXISTS observations_2024
    PARTITION OF observations FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE IF NOT EXISTS observations_2025
    PARTITION OF observations FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS observations_2026
    PARTITION OF observations FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Indeks observasi
CREATE INDEX IF NOT EXISTS idx_obs_time       ON observations USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_obs_geom       ON observations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_obs_station    ON observations (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_obs_variable   ON observations (variable, time DESC);
CREATE INDEX IF NOT EXISTS idx_obs_source     ON observations (source);

-- =============================================================================
-- TABEL: forecasts
-- Prediksi model cuaca (GraphCast, ONNX Downscaler)
-- label_istilah = 'informasi risiko' sesuai UU 31/2009 Pasal 36 dan PP 11/2016
-- =============================================================================
CREATE TABLE IF NOT EXISTS forecasts (
    id                  BIGSERIAL PRIMARY KEY,
    model_id            TEXT        NOT NULL,
    init_time           TIMESTAMPTZ NOT NULL,
    valid_time          TIMESTAMPTZ NOT NULL,
    lead_seconds        INTEGER     NOT NULL CHECK (lead_seconds > 0),
    lat                 DOUBLE PRECISION NOT NULL,
    lon                 DOUBLE PRECISION NOT NULL,
    variables           JSONB       NOT NULL DEFAULT '{}',
    ensemble_members    SMALLINT,
    probability         JSONB,
    -- Label kepatuhan regulasi (WAJIB: UU 31/2009 Pasal 36 ayat 1)
    label_istilah       TEXT        NOT NULL DEFAULT 'informasi risiko',
    atribusi            TEXT[]      NOT NULL DEFAULT ARRAY['GFS/0.25°'],
    sumber_id           TEXT[]      NOT NULL DEFAULT ARRAY['graphcast'],
    geom                GEOGRAPHY(Point, 4326) GENERATED ALWAYS AS
                            (ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) STORED,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partisi forecasts per model + bulan
CREATE TABLE IF NOT EXISTS forecasts_2025
    PARTITION OF forecasts FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')
    PARTITION BY RANGE (init_time);

-- Indeks forecasts
CREATE INDEX IF NOT EXISTS idx_fc_model_time  ON forecasts (model_id, init_time DESC);
CREATE INDEX IF NOT EXISTS idx_fc_geom        ON forecasts USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_fc_valid       ON forecasts (valid_time DESC);
CREATE INDEX IF NOT EXISTS idx_fc_lead        ON forecasts (lead_seconds);
-- Indeks GiST untuk bbox query cepat
CREATE INDEX IF NOT EXISTS idx_fc_geom_brin   ON forecasts USING BRIN (init_time);

-- Constraint: jangan simpan data sumber ERA5 di tabel forecasts (training-only)
CREATE OR REPLACE RULE no_era5_in_forecasts AS
    ON INSERT TO forecasts
    WHERE 'ERA5' = ANY(NEW.sumber_id)
    DO INSTEAD NOTHING;

-- =============================================================================
-- TABEL: grid_predictions
-- Prediksi per sel grid (0.5 km resolusi) — tabel lebar untuk batch read
-- =============================================================================
CREATE TABLE IF NOT EXISTS grid_predictions (
    id              BIGSERIAL PRIMARY KEY,
    model_id        TEXT        NOT NULL,
    init_time       TIMESTAMPTZ NOT NULL,
    lead_seconds    INTEGER     NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    cell_id         TEXT,
    resolution_deg  DOUBLE PRECISION NOT NULL DEFAULT 0.0045,  -- ~500m
    -- Variabel cuaca (kolom terpisah untuk query efisien)
    temp_p10        REAL,
    temp_p50        REAL,
    temp_p90        REAL,
    humidity_p50    REAL,
    precip_p50      REAL,
    precip_prob     REAL,
    wind_speed_p50  REAL,
    wind_dir_p50    REAL,
    pressure_p50    REAL,
    -- Metadata compliance
    label_istilah   TEXT        NOT NULL DEFAULT 'informasi risiko',
    geom            GEOGRAPHY(Point, 4326) GENERATED ALWAYS AS
                        (ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_id, init_time, lead_seconds, lat, lon)
);

-- Indeks grid_predictions — kritis untuk query bbox + lead time
CREATE INDEX IF NOT EXISTS idx_gp_geom    ON grid_predictions USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_gp_model   ON grid_predictions (model_id, init_time DESC);
CREATE INDEX IF NOT EXISTS idx_gp_lead    ON grid_predictions (lead_seconds);
CREATE INDEX IF NOT EXISTS idx_gp_time    ON grid_predictions USING BRIN (init_time);

-- =============================================================================
-- TABEL: verification_scores
-- Skor verifikasi model (MAE, RMSE, CSI, Brier) per periode dan variabel
-- =============================================================================
CREATE TABLE IF NOT EXISTS verification_scores (
    id              BIGSERIAL PRIMARY KEY,
    model_id        TEXT        NOT NULL,
    variable        TEXT        NOT NULL,
    lead_seconds    INTEGER     NOT NULL,
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    sample_count    INTEGER     NOT NULL DEFAULT 0,
    -- Deterministic metrics
    mae             REAL,
    rmse            REAL,
    bias            REAL,
    correlation     REAL,
    -- Probabilistic metrics
    brier_score     REAL,
    roc_auc         REAL,
    crps            REAL,
    -- Categorical metrics
    csi             REAL,   -- Critical Success Index (WMO standard)
    far             REAL,   -- False Alarm Rate
    pod             REAL,   -- Probability of Detection
    -- Metadata
    threshold_used  REAL,
    region          TEXT    DEFAULT 'indonesia',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vs_model   ON verification_scores (model_id, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_vs_var     ON verification_scores (variable, lead_seconds);

-- =============================================================================
-- FUNGSI: bbox_query
-- Query cepat untuk seluruh sel dalam bounding box + lead_seconds
-- =============================================================================
CREATE OR REPLACE FUNCTION get_grid_predictions(
    p_model_id    TEXT,
    p_min_lat     DOUBLE PRECISION,
    p_max_lat     DOUBLE PRECISION,
    p_min_lon     DOUBLE PRECISION,
    p_max_lon     DOUBLE PRECISION,
    p_lead_secs   INTEGER,
    p_limit       INTEGER DEFAULT 50000
)
RETURNS SETOF grid_predictions AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM grid_predictions
    WHERE model_id = p_model_id
      AND lead_seconds = p_lead_secs
      AND ST_Within(
          geom::geometry,
          ST_MakeEnvelope(p_min_lon, p_min_lat, p_max_lon, p_max_lat, 4326)
      )
    ORDER BY lat, lon
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FUNGSI: create_observation_partition
-- Auto-generate partisi tahunan
-- =============================================================================
CREATE OR REPLACE FUNCTION create_observation_partition(p_year INTEGER)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date     TEXT;
    end_date       TEXT;
BEGIN
    partition_name := 'observations_' || p_year;
    start_date     := p_year || '-01-01';
    end_date       := (p_year + 1) || '-01-01';

    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF observations FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Auto-create partition 2027
SELECT create_observation_partition(2027);

-- =============================================================================
-- VIEW: latest_observations_by_station
-- Pengamatan terbaru per stasiun untuk monitoring real-time
-- =============================================================================
CREATE OR REPLACE VIEW latest_observations_by_station AS
SELECT DISTINCT ON (station_id, variable)
    station_id, variable, value, quality_flag, source, time, lat, lon
FROM observations
ORDER BY station_id, variable, time DESC;

-- =============================================================================
-- VIEW: model_performance_summary
-- Ringkasan performa model (digunakan oleh verification API endpoint)
-- =============================================================================
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    model_id,
    variable,
    lead_seconds / 3600 AS lead_hours,
    AVG(mae)         AS avg_mae,
    AVG(rmse)        AS avg_rmse,
    AVG(bias)        AS avg_bias,
    AVG(correlation) AS avg_correlation,
    AVG(brier_score) AS avg_brier,
    AVG(crps)        AS avg_crps,
    AVG(csi)         AS avg_csi,
    MAX(period_end)  AS latest_verification,
    SUM(sample_count) AS total_samples
FROM verification_scores
GROUP BY model_id, variable, lead_seconds
ORDER BY model_id, variable, lead_seconds;

-- =============================================================================
-- GRANT: role read-only untuk API server
-- =============================================================================
-- CREATE ROLE wi_api_ro LOGIN PASSWORD 'GANTI_INI';
-- GRANT CONNECT ON DATABASE weather_intel TO wi_api_ro;
-- GRANT USAGE ON SCHEMA public TO wi_api_ro;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO wi_api_ro;

-- =============================================================================
-- GRANT: role read-write untuk ingest service
-- =============================================================================
-- CREATE ROLE wi_ingest_rw LOGIN PASSWORD 'GANTI_INI';
-- GRANT CONNECT ON DATABASE weather_intel TO wi_ingest_rw;
-- GRANT USAGE ON SCHEMA public TO wi_ingest_rw;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO wi_ingest_rw;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO wi_ingest_rw;

-- =============================================================================
-- SELESAI — verifikasi dengan:
-- SELECT PostGIS_Version();
-- \dt
-- \dv
-- =============================================================================
