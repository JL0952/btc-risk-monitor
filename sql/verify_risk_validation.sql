-- psql: -v validation_run_id=<UUID>. Explicit run selection prevents mixed versions.
SET TIME ZONE 'UTC';
SELECT validation_run_id,evaluation_start,evaluation_end,created_at,code_version,
       cardinality(if_run_ids) AS if_runs,summary->'if_available_at_range' AS actual_if_availability
FROM research.validation_runs WHERE validation_run_id=:'validation_run_id'::uuid;

SELECT horizon_minutes,label_status,signal_group,count(*) AS observations,
       avg(future_realized_volatility)*100 AS mean_rv_percent,
       percentile_cont(.5) WITHIN GROUP (ORDER BY future_realized_volatility)*100 AS median_rv_percent,
       percentile_cont(.9) WITHIN GROUP (ORDER BY future_realized_volatility)*100 AS p90_rv_percent,
       percentile_cont(.95) WITHIN GROUP (ORDER BY future_realized_volatility)*100 AS p95_rv_percent
FROM research.risk_validation WHERE validation_run_id=:'validation_run_id'::uuid
GROUP BY horizon_minutes,label_status,signal_group ORDER BY horizon_minutes,signal_group;

WITH groups AS (
 SELECT signal_group,count(*) AS n,count(*) FILTER (WHERE high_risk_flag) AS events,
        avg(high_risk_flag::integer) AS event_probability
 FROM research.risk_validation
 WHERE validation_run_id=:'validation_run_id'::uuid AND horizon_minutes=60 AND label_status='available'
 GROUP BY signal_group
)
SELECT *,event_probability / NULLIF((SELECT event_probability FROM groups WHERE signal_group='A'),0) AS enrichment FROM groups ORDER BY signal_group;

SELECT count(*) FILTER (WHERE v.zscore_alert IS DISTINCT FROM z.alert_flag
                           OR v.isolation_alert IS DISTINCT FROM i.anomaly_flag
                           OR v.symbol<>i.symbol OR v.interval<>i.interval) AS mismatched_signals,
       count(*) AS validation_records
FROM research.risk_validation v
JOIN realtime_signals z USING (symbol,interval,timestamp)
JOIN isolation_forest_scores i ON i.run_id=v.if_run_id AND i.timestamp=v.timestamp
WHERE v.validation_run_id=:'validation_run_id'::uuid;
