"""Minimal parameterized bar storage; no ingestion or detector implementation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import logging
import time

import psycopg

from btc_risk.database.connection import connect


@dataclass(frozen=True)
class MarketBar:
    timestamp: datetime
    symbol: str
    interval: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: str


def utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc)


def insert_bar(conn: psycopg.Connection, bar: MarketBar) -> bool:
    """Return True for an insertion, False for identical overlap; caller commits.

    Conflicting OHLCV for the same key raises rather than silently rewriting data.
    A differing source alone is permitted; preserve the first source/ingestion time.
    """
    timestamp = utc(bar.timestamp)
    values = (timestamp, bar.symbol, bar.interval, bar.open, bar.high, bar.low,
              bar.close, bar.volume, bar.source)
    row = conn.execute("""
        INSERT INTO market_bars
            (timestamp, symbol, interval, open, high, low, close, volume, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, timestamp) DO NOTHING
        RETURNING timestamp
    """, values).fetchone()
    if row is not None:
        return True
    stored = get_bar(conn, bar.symbol, bar.interval, timestamp)
    if stored is None or any(
        stored[field] != getattr(bar, field)
        for field in ("open", "high", "low", "close", "volume")
    ):
        raise ValueError("Existing bar has different OHLCV; refusing silent replacement")
    return False


def get_bar(conn: psycopg.Connection, symbol: str, interval: str, timestamp: datetime):
    return conn.execute("""
        SELECT timestamp, symbol, interval, open, high, low, close, volume,
               source, ingested_at
        FROM market_bars
        WHERE symbol = %s AND interval = %s AND timestamp = %s
    """, (symbol, interval, utc(timestamp))).fetchone()


def inspect_range(conn, symbol, interval, start, end, step):
    """Check every expected timestamp, including missing leading/trailing bars.

    Only Binance rows count as market-data coverage; synthetic Foundation examples
    cannot conceal a missing real observation.
    """
    result = conn.execute("""
        SELECT count(*) AS stored_rows, min(timestamp) AS first_stored_timestamp,
               max(timestamp) AS last_stored_timestamp
        FROM market_bars WHERE symbol = %s AND interval = %s AND source = 'binance'
            AND timestamp >= %s AND timestamp < %s
    """, (symbol, interval, utc(start), utc(end))).fetchone()
    gaps = conn.execute("""
        WITH missing AS (
            SELECT expected.timestamp
            FROM generate_series(%s::timestamptz, %s::timestamptz - %s::interval,
                                 %s::interval) AS expected(timestamp)
            WHERE NOT EXISTS (
                SELECT 1 FROM market_bars b
                WHERE b.symbol = %s AND b.interval = %s AND b.source = 'binance'
                  AND b.timestamp = expected.timestamp
            )
        )
        SELECT (SELECT count(*) FROM missing) AS gap_count,
               ARRAY(SELECT timestamp FROM missing ORDER BY timestamp LIMIT 10) AS missing_timestamps
    """, (utc(start), utc(end), step, step, symbol, interval)).fetchone()
    return {**result, **gaps}


def demo() -> None:
    """Persist one explicitly synthetic, fixed-key BTC bar for manual inspection."""
    bar = MarketBar(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="BTCUSDT", interval="5m", open=Decimal("90000"),
        high=Decimal("90100"), low=Decimal("89900"), close=Decimal("90050"),
        volume=Decimal("12.5"), source="synthetic-foundation-demo",
    )
    with connect() as conn:
        inserted = insert_bar(conn, bar)
        saved = get_bar(conn, bar.symbol, bar.interval, bar.timestamp)
    logging.getLogger(__name__).info("Demo committed: inserted=%s bar=%s", inserted, saved)


def insert_signal(conn, signal):
    """Exact deterministic-result comparison; calculated_at is first computation time."""
    from dataclasses import asdict

    values = asdict(signal)
    row = conn.execute("""
        INSERT INTO realtime_signals
            (timestamp, symbol, interval, log_return, rolling_median, rolling_mad,
             robust_zscore, alert_flag, status)
        VALUES (%(timestamp)s, %(symbol)s, %(interval)s, %(log_return)s,
                %(rolling_median)s, %(rolling_mad)s, %(robust_zscore)s, %(alert_flag)s, %(status)s)
        ON CONFLICT (symbol, interval, timestamp) DO NOTHING
        RETURNING timestamp
    """, values).fetchone()
    if row is not None:
        return True
    stored = conn.execute("""
        SELECT timestamp, symbol, interval, log_return, rolling_median, rolling_mad,
               robust_zscore, alert_flag, status
        FROM realtime_signals
        WHERE symbol = %(symbol)s AND interval = %(interval)s AND timestamp = %(timestamp)s
    """, values).fetchone()
    if stored != values:
        raise ValueError(f"Signal inconsistency at {signal.symbol}/{signal.interval}/{signal.timestamp}: "
                         "existing result differs; check algorithm, config, data, and replay origin")
    return False


def signal_summary(conn, symbol, interval, start, end):
    return conn.execute("""
        SELECT count(*) AS total_market_bars_processed,
               count(s.log_return) AS valid_returns,
               count(*) FILTER (WHERE s.status = 'no_previous_close') AS no_previous_close,
               count(*) FILTER (WHERE s.status = 'warmup') AS warmup_observations,
               count(*) FILTER (WHERE s.status = 'gap') AS gap_observations,
               count(*) FILTER (WHERE s.status = 'scale_floored') AS scale_floored_observations,
               count(s.robust_zscore) AS scored_observations,
               count(*) FILTER (WHERE s.alert_flag) AS alert_count,
               count(*) FILTER (WHERE s.alert_flag AND s.robust_zscore > 0) AS positive_alerts,
               count(*) FILTER (WHERE s.alert_flag AND s.robust_zscore < 0) AS negative_alerts,
               100.0 * count(*) FILTER (WHERE s.alert_flag) /
                   NULLIF(count(s.robust_zscore), 0) AS alert_rate_percent,
               min(s.robust_zscore) AS minimum_zscore,
               max(s.robust_zscore) AS maximum_zscore,
               min(s.timestamp) FILTER (WHERE s.robust_zscore IS NOT NULL) AS first_valid_zscore_timestamp,
               max(s.timestamp) FILTER (WHERE s.robust_zscore IS NOT NULL) AS last_valid_zscore_timestamp
        FROM realtime_signals s JOIN market_bars b USING (symbol, interval, timestamp)
        WHERE b.source = 'binance' AND s.symbol = %s AND s.interval = %s
          AND s.timestamp >= %s AND s.timestamp < %s
    """, (symbol, interval, utc(start), utc(end))).fetchone()


if __name__ == "__main__":
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    demo()
