-- Historical replay verification only. No forward-looking outcomes.
SET TIME ZONE 'UTC';

SELECT s.status, count(*) AS observations
FROM realtime_signals s JOIN market_bars b USING (symbol, interval, timestamp)
WHERE b.source = 'binance' AND s.symbol = 'BTCUSDT' AND s.interval = '5m'
GROUP BY s.status ORDER BY s.status;

SELECT s.timestamp, b.close, s.log_return, s.rolling_median, s.rolling_mad,
       s.robust_zscore, s.alert_flag
FROM realtime_signals s JOIN market_bars b USING (symbol, interval, timestamp)
WHERE b.source = 'binance' AND s.symbol = 'BTCUSDT' AND s.interval = '5m'
  AND s.robust_zscore IS NOT NULL
ORDER BY abs(s.robust_zscore) DESC, s.timestamp LIMIT 10;

SELECT count(*) AS nonfinite_values
FROM realtime_signals s,
LATERAL (VALUES (s.log_return), (s.rolling_median), (s.rolling_mad), (s.robust_zscore)) v(value)
WHERE value IS NOT NULL AND NOT (value > '-Infinity'::float8 AND value < 'Infinity'::float8);

SELECT symbol, interval, timestamp, count(*) FROM realtime_signals
GROUP BY symbol, interval, timestamp HAVING count(*) > 1;

-- Contemporaneous sanity checks only: these are not future risk validation.
SELECT count(*) FILTER (WHERE s.log_return >= 0.005) AS positive_moves_ge_0_5pct_log,
       count(*) FILTER (WHERE s.log_return >= 0.005 AND s.robust_zscore > 0) AS those_with_positive_z,
       count(*) FILTER (WHERE s.log_return <= -0.005) AS negative_moves_ge_0_5pct_log,
       count(*) FILTER (WHERE s.log_return <= -0.005 AND s.robust_zscore < 0) AS those_with_negative_z,
       count(*) FILTER (WHERE abs(s.log_return) <= 0.0005) AS small_moves_le_0_05pct_log,
       count(*) FILTER (WHERE abs(s.log_return) <= 0.0005 AND NOT s.alert_flag) AS small_moves_without_alert
FROM realtime_signals s JOIN market_bars b USING (symbol, interval, timestamp)
WHERE b.source = 'binance' AND s.symbol = 'BTCUSDT' AND s.interval = '5m'
  AND s.robust_zscore IS NOT NULL;
