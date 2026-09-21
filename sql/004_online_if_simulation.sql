-- Historical "as-if live" IF publication and scores.  These rows deliberately
-- remain separate from the retrospective daily batch tables in 002.
CREATE TABLE online_if_model_publications (
    publication_id UUID PRIMARY KEY,
    publication_identity TEXT NOT NULL UNIQUE,
    source_run_id UUID NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL CHECK (interval IN ('5m', '1h')),
    training_start TIMESTAMPTZ NOT NULL,
    training_end TIMESTAMPTZ NOT NULL,
    scoring_start TIMESTAMPTZ NOT NULL,
    scoring_end TIMESTAMPTZ NOT NULL,
    simulated_model_available_at TIMESTAMPTZ NOT NULL,
    availability_policy TEXT NOT NULL,
    feature_set JSONB NOT NULL,
    model_parameters JSONB NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    code_version TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (source_run_id, symbol, interval)
        REFERENCES model_runs (run_id, symbol, interval),
    CHECK (training_start < training_end AND training_end <= scoring_start AND scoring_start < scoring_end),
    CHECK (simulated_model_available_at >= scoring_start AND simulated_model_available_at < scoring_end)
);
CREATE INDEX online_if_publications_period
    ON online_if_model_publications (symbol, interval, scoring_start);

CREATE TABLE online_isolation_forest_scores (
    publication_id UUID NOT NULL REFERENCES online_if_model_publications(publication_id),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL CHECK (
        anomaly_score > '-Infinity'::float8 AND anomaly_score < 'Infinity'::float8
    ),
    anomaly_flag BOOLEAN NOT NULL,
    feature_values JSONB NOT NULL,
    bar_closed_at TIMESTAMPTZ NOT NULL,
    simulated_available_at TIMESTAMPTZ NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (publication_id, timestamp),
    FOREIGN KEY (symbol, interval, timestamp)
        REFERENCES market_bars (symbol, interval, timestamp),
    CHECK (bar_closed_at > timestamp),
    CHECK (simulated_available_at >= bar_closed_at)
);
CREATE INDEX online_if_scores_latest
    ON online_isolation_forest_scores (symbol, interval, timestamp DESC);
