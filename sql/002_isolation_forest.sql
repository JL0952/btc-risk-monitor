CREATE TABLE model_runs (
    run_id UUID PRIMARY KEY,
    run_identity TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL CHECK (interval IN ('5m', '1h')),
    training_start TIMESTAMPTZ NOT NULL,
    training_end TIMESTAMPTZ NOT NULL,
    scoring_start TIMESTAMPTZ NOT NULL,
    scoring_end TIMESTAMPTZ NOT NULL,
    feature_set JSONB NOT NULL,
    model_parameters JSONB NOT NULL,
    training_rows INTEGER NOT NULL CHECK (training_rows > 0),
    scoring_rows INTEGER NOT NULL CHECK (scoring_rows > 0),
    data_hash TEXT NOT NULL,
    code_version TEXT NOT NULL,
    library_versions JSONB NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CHECK (training_start < training_end AND training_end <= scoring_start AND scoring_start < scoring_end),
    CHECK (completed_at >= created_at AND scoring_end <= completed_at),
    UNIQUE (run_id, symbol, interval)
);

CREATE TABLE isolation_forest_scores (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    run_id UUID NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL CHECK (
        anomaly_score > '-Infinity'::float8 AND anomaly_score < 'Infinity'::float8
    ),
    anomaly_flag BOOLEAN NOT NULL,
    feature_values JSONB NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (run_id, timestamp),
    FOREIGN KEY (run_id, symbol, interval) REFERENCES model_runs (run_id, symbol, interval),
    FOREIGN KEY (symbol, interval, timestamp) REFERENCES market_bars (symbol, interval, timestamp)
);
CREATE INDEX isolation_scores_latest ON isolation_forest_scores (symbol, interval, timestamp DESC);
