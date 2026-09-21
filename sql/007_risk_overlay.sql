CREATE TABLE research.overlay_runs (
    overlay_run_id UUID PRIMARY KEY,
    overlay_identity TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    evaluation_start TIMESTAMPTZ NOT NULL,
    evaluation_end TIMESTAMPTZ NOT NULL CHECK (evaluation_end > evaluation_start),
    return_observations INTEGER NOT NULL CHECK (return_observations > 0),
    actionable_publication_ids UUID[] NOT NULL,
    strategy_definitions JSONB NOT NULL,
    code_version TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    library_versions JSONB NOT NULL,
    summary JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE research.overlay_results (
    overlay_run_id UUID NOT NULL REFERENCES research.overlay_runs(overlay_run_id),
    strategy_name TEXT NOT NULL CHECK (strategy_name IN ('buy_hold','z_overlay','if_overlay','union_overlay')),
    metrics JSONB NOT NULL,
    PRIMARY KEY (overlay_run_id, strategy_name)
);
