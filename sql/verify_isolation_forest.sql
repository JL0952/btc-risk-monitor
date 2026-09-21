-- Current baseline runs only. If adding configurations later, filter explicit run_ids.
SET TIME ZONE 'UTC';

SELECT count(*) AS model_runs, min(training_start) AS earliest_training_start,
       min(scoring_start) AS first_scoring_day, max(scoring_end) AS final_scoring_end,
       min(training_rows) AS min_training_rows, max(training_rows) AS max_training_rows
FROM model_runs WHERE symbol='BTCUSDT' AND interval='5m' AND model_version='1.0';

SELECT m.scoring_start::date AS utc_day, m.training_rows,
       count(*) AS scored, count(*) FILTER (WHERE s.anomaly_flag) AS anomalies
FROM model_runs m JOIN isolation_forest_scores s USING (run_id, symbol, interval)
WHERE m.symbol='BTCUSDT' AND m.interval='5m' AND m.model_version='1.0'
GROUP BY m.scoring_start, m.training_rows ORDER BY utc_day;

SELECT count(*) AS total_scores, count(*) FILTER (WHERE s.anomaly_flag) AS anomalies,
       round(100.0 * count(*) FILTER (WHERE s.anomaly_flag) / count(*), 6) AS anomaly_rate_percent
FROM isolation_forest_scores s JOIN model_runs m USING (run_id, symbol, interval)
WHERE m.symbol='BTCUSDT' AND m.interval='5m' AND m.model_version='1.0';

SELECT s.timestamp, b.close,
       (s.feature_values->>'log_return')::double precision AS log_return,
       (s.feature_values->>'rolling_volatility')::double precision AS rolling_volatility,
       (s.feature_values->>'volume_zscore')::double precision AS volume_zscore,
       (s.feature_values->>'high_low_range')::double precision AS high_low_range,
       (s.feature_values->>'volume_change')::double precision AS volume_change,
       s.anomaly_score, s.anomaly_flag, z.alert_flag AS z_alert
FROM isolation_forest_scores s
JOIN model_runs m USING (run_id, symbol, interval)
JOIN market_bars b USING (symbol, interval, timestamp)
LEFT JOIN realtime_signals z USING (symbol, interval, timestamp)
WHERE m.symbol='BTCUSDT' AND m.interval='5m' AND m.model_version='1.0'
ORDER BY s.anomaly_score DESC, s.timestamp LIMIT 10;

-- Only contemporaneous joint counts; NULL Z is not classified as normal.
WITH counts AS (
    SELECT z.alert_flag AS z_alert, s.anomaly_flag AS if_anomaly, count(*) AS n
    FROM isolation_forest_scores s
    JOIN model_runs m USING (run_id, symbol, interval)
    LEFT JOIN realtime_signals z USING (symbol, interval, timestamp)
    WHERE m.symbol='BTCUSDT' AND m.interval='5m' AND m.model_version='1.0'
    GROUP BY z.alert_flag, s.anomaly_flag
)
SELECT z.flag AS z_alert, i.flag AS if_anomaly, coalesce(c.n,0) AS observations
FROM (VALUES(false),(true)) z(flag) CROSS JOIN (VALUES(false),(true)) i(flag)
LEFT JOIN counts c ON c.z_alert=z.flag AND c.if_anomaly=i.flag
ORDER BY z.flag, i.flag;

SELECT count(*) AS missing_or_unavailable_z
FROM isolation_forest_scores s
JOIN model_runs m USING (run_id,symbol,interval)
LEFT JOIN realtime_signals z USING (symbol,interval,timestamp)
WHERE m.symbol='BTCUSDT' AND m.interval='5m' AND m.model_version='1.0' AND z.alert_flag IS NULL;

SELECT count(*) AS invalid_period_associations
FROM isolation_forest_scores s JOIN model_runs m USING (run_id,symbol,interval)
WHERE s.timestamp < m.scoring_start OR s.timestamp >= m.scoring_end OR m.training_end > m.scoring_start;
