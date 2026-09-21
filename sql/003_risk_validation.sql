CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE research.validation_runs (
    validation_run_id UUID PRIMARY KEY,
    validation_identity TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    evaluation_start TIMESTAMPTZ NOT NULL,
    evaluation_end TIMESTAMPTZ NOT NULL CHECK (evaluation_end > evaluation_start),
    if_run_ids UUID[] NOT NULL,
    label_definitions JSONB NOT NULL,
    analysis_config JSONB NOT NULL,
    code_version TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    library_versions JSONB NOT NULL,
    summary JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE research.risk_validation (
    validation_run_id UUID NOT NULL REFERENCES research.validation_runs(validation_run_id),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    if_run_id UUID NOT NULL,
    zscore_alert BOOLEAN NOT NULL,
    isolation_alert BOOLEAN NOT NULL,
    signal_group TEXT NOT NULL CHECK (signal_group IN ('A','B','C','D')),
    horizon_minutes INTEGER NOT NULL CHECK (horizon_minutes IN (30,60)),
    label_status TEXT NOT NULL CHECK (label_status IN ('available','incomplete_horizon','gap')),
    future_realized_volatility DOUBLE PRECISION,
    future_maximum_adverse_move DOUBLE PRECISION,
    future_absolute_return DOUBLE PRECISION,
    high_risk_flag BOOLEAN,
    PRIMARY KEY (validation_run_id,timestamp,horizon_minutes),
    FOREIGN KEY (symbol,interval,timestamp) REFERENCES realtime_signals(symbol,interval,timestamp),
    FOREIGN KEY (if_run_id,timestamp) REFERENCES isolation_forest_scores(run_id,timestamp),
    CHECK ((label_status = 'available' AND future_realized_volatility IS NOT NULL
            AND future_maximum_adverse_move IS NOT NULL AND future_absolute_return IS NOT NULL)
        OR (label_status <> 'available' AND future_realized_volatility IS NULL
            AND future_maximum_adverse_move IS NULL AND future_absolute_return IS NULL)),
    CHECK (future_realized_volatility >= 0 AND future_realized_volatility < 'Infinity'::float8),
    CHECK (future_maximum_adverse_move >= 0 AND future_maximum_adverse_move < 1),
    CHECK (future_absolute_return >= 0 AND future_absolute_return < 'Infinity'::float8)
);
