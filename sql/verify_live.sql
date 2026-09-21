SET TIME ZONE 'UTC';

SELECT count(*) AS binance_bars, min(timestamp), max(timestamp)
FROM market_bars WHERE symbol='BTCUSDT' AND interval='5m' AND source='binance';

SELECT b.timestamp, b.close, s.log_return, s.rolling_median, s.rolling_mad,
       s.robust_zscore, s.alert_flag, s.status, b.ingested_at, s.calculated_at
FROM market_bars b LEFT JOIN realtime_signals s USING (symbol, interval, timestamp)
WHERE b.symbol='BTCUSDT' AND b.interval='5m' AND b.source='binance'
ORDER BY b.timestamp DESC LIMIT 10;

SELECT count(*) AS missing_signals
FROM market_bars b LEFT JOIN realtime_signals s USING (symbol, interval, timestamp)
WHERE b.symbol='BTCUSDT' AND b.interval='5m' AND b.source='binance' AND s.timestamp IS NULL;

SELECT s.status, count(*) FROM realtime_signals s JOIN market_bars b USING (symbol, interval, timestamp)
WHERE b.symbol='BTCUSDT' AND b.interval='5m' AND b.source='binance'
  AND b.timestamp >= '2026-09-13T00:00:00Z'
GROUP BY s.status;
