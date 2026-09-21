-- Historical as-if-live parity and actual shadow-mode audit.
SET TIME ZONE 'UTC';

SELECT p.publication_mode, count(*) AS publications, count(o.timestamp) AS scores,
       count(*) FILTER (WHERE o.anomaly_flag) AS anomalies
FROM online_if_model_publications p
LEFT JOIN online_isolation_forest_scores o USING (publication_id)
GROUP BY p.publication_mode ORDER BY p.publication_mode;

SELECT count(*) AS common_scores,
       count(*) FILTER (WHERE o.anomaly_score <> r.anomaly_score) AS score_mismatches,
       count(*) FILTER (WHERE o.anomaly_flag IS DISTINCT FROM r.anomaly_flag) AS flag_mismatches
FROM online_isolation_forest_scores o
JOIN online_if_model_publications p ON p.publication_id=o.publication_id
JOIN isolation_forest_scores r ON r.run_id=p.source_run_id AND r.timestamp=o.timestamp
WHERE p.publication_mode='historical_simulation';

SELECT p.publication_id,p.model_available_at,o.timestamp,o.anomaly_score,o.anomaly_flag,o.score_available_at
FROM online_if_model_publications p JOIN online_isolation_forest_scores o USING (publication_id)
WHERE p.publication_mode='live_shadow' ORDER BY o.timestamp DESC LIMIT 10;
