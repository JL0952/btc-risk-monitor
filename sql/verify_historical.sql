-- Manual audit: psql -f this file. All real-data queries exclude Foundation demo.
SET TIME ZONE 'UTC';

SELECT count(*) AS all_btc_rows,
       count(*) FILTER (WHERE source = 'binance') AS binance_rows,
       count(*) FILTER (WHERE source <> 'binance') AS other_source_rows
FROM market_bars WHERE symbol = 'BTCUSDT' AND interval = '5m';

SELECT count(*) AS bars, min(timestamp) AS first_open, max(timestamp) AS last_open
FROM market_bars WHERE symbol = 'BTCUSDT' AND interval = '5m' AND source = 'binance';

SELECT timestamp, open, high, low, close, volume, source
FROM market_bars WHERE symbol = 'BTCUSDT' AND interval = '5m' AND source = 'binance'
ORDER BY timestamp DESC LIMIT 10;

SELECT symbol, interval, timestamp, count(*)
FROM market_bars GROUP BY symbol, interval, timestamp HAVING count(*) > 1;

SELECT (timestamp AT TIME ZONE 'UTC')::date AS utc_day, count(*) AS bars
FROM market_bars WHERE symbol = 'BTCUSDT' AND interval = '5m' AND source = 'binance'
GROUP BY utc_day ORDER BY utc_day;
