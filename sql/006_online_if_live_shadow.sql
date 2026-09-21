-- Generalize the 004 historical-simulation names before adding actual live
-- shadow publications. Existing historical rows retain their simulated policy.
ALTER TABLE online_if_model_publications
    RENAME COLUMN simulated_model_available_at TO model_available_at;
ALTER TABLE online_isolation_forest_scores
    RENAME COLUMN simulated_available_at TO score_available_at;
ALTER TABLE online_if_model_publications
    ALTER COLUMN source_run_id DROP NOT NULL,
    ADD COLUMN publication_mode TEXT NOT NULL DEFAULT 'historical_simulation'
        CHECK (publication_mode IN ('historical_simulation','live_shadow'));
