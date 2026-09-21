CREATE TABLE if_training_attempts (
    attempt_id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL CHECK (interval IN ('5m','1h')),
    scoring_day DATE NOT NULL,
    training_started_at TIMESTAMPTZ NOT NULL,
    training_completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running','failed','published')),
    publication_id UUID REFERENCES online_if_model_publications(publication_id),
    error_message TEXT,
    UNIQUE(symbol,interval,scoring_day)
);

CREATE TABLE active_if_publications (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    publication_id UUID NOT NULL REFERENCES online_if_model_publications(publication_id),
    activated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(symbol,interval)
);
