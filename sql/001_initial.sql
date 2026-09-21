-- TIMESTAMPTZ stores instants; connections and the server display UTC.
-- timestamp is the candle OPEN time. No market data is fetched in Foundation.
CREATE TABLE market_bars (
    timestamp TIMESTAMPTZ NOT NULL CHECK (isfinite(timestamp)),
    symbol TEXT NOT NULL CHECK (length(trim(symbol)) > 0),
    interval TEXT NOT NULL CHECK (interval IN ('5m', '1h')),
    open NUMERIC(30,12) NOT NULL,
    high NUMERIC(30,12) NOT NULL,
    low NUMERIC(30,12) NOT NULL,
    close NUMERIC(30,12) NOT NULL,
    volume NUMERIC(30,12) NOT NULL,
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, interval, timestamp),
    CONSTRAINT market_bars_positive_prices CHECK (
        open > 0 AND open < 'Infinity'::numeric AND
        high > 0 AND high < 'Infinity'::numeric AND
        low > 0 AND low < 'Infinity'::numeric AND
        close > 0 AND close < 'Infinity'::numeric
    ),
    CONSTRAINT market_bars_nonnegative_volume CHECK (
        volume >= 0 AND volume < 'Infinity'::numeric
    ),
    CONSTRAINT market_bars_ohlc_order CHECK (
        low <= open AND open <= high AND low <= close AND close <= high
    )
);

CREATE TABLE realtime_signals (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    log_return DOUBLE PRECISION,
    rolling_median DOUBLE PRECISION,
    rolling_mad DOUBLE PRECISION,
    robust_zscore DOUBLE PRECISION,
    alert_flag BOOLEAN,
    status TEXT NOT NULL CHECK (length(trim(status)) > 0),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, interval, timestamp),
    FOREIGN KEY (symbol, interval, timestamp)
        REFERENCES market_bars (symbol, interval, timestamp),
    CHECK (log_return > '-Infinity'::float8 AND log_return < 'Infinity'::float8),
    CHECK (rolling_median > '-Infinity'::float8 AND rolling_median < 'Infinity'::float8),
    CHECK (rolling_mad >= 0 AND rolling_mad < 'Infinity'::float8),
    CHECK (robust_zscore > '-Infinity'::float8 AND robust_zscore < 'Infinity'::float8)
);
