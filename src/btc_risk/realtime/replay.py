"""Chronological PostgreSQL replay; no exchange requests or future outcomes."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import time

from btc_risk.config import DetectorConfig
from btc_risk.database.connection import connect
from btc_risk.database.repository import insert_signal, signal_summary, utc
from btc_risk.realtime.robust_zscore import EPOCH, Observation, RobustZScore

logger = logging.getLogger(__name__)


def replay(conn, symbol, interval, start, end, config=None, fetch_size=1000):
    """Caller commits the whole replay or rolls it back; never retain failed state.

    Cold-start at the first Binance bar within [start,end). Reproducible reruns
    use the same start/origin. Changing the origin can legitimately conflict.
    """
    config = config or DetectorConfig.from_env()
    detector = RobustZScore(symbol, interval, config)
    start, end = utc(start), utc(end)
    if start >= end or any((stamp - EPOCH) % detector.step for stamp in (start, end)):
        raise ValueError("Require interval-aligned start < end")
    if fetch_size < 1:
        raise ValueError("fetch_size must be positive")
    inserted = duplicates = processed = 0
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"realtime:{symbol}:{interval}",))

    def save(signal):
        nonlocal inserted, duplicates
        if insert_signal(conn, signal):
            inserted += 1
        else:
            duplicates += 1

    # A server-side cursor fetches at most fetch_size rows; SQL owns ordering.
    with conn.cursor(name="historical_replay") as cursor:
        cursor.itersize = fetch_size
        cursor.execute("""
            SELECT timestamp, symbol, interval, close
            FROM market_bars
            WHERE symbol = %s AND interval = %s AND source = 'binance'
              AND timestamp >= %s AND timestamp < %s
            ORDER BY timestamp ASC
        """, (symbol, interval, start, end))
        for row in cursor:
            detector.process(Observation(**row), save=save)
            processed += 1
            if processed % 5000 == 0:
                logger.info("Replay processed=%s; transaction not yet committed", processed)
    if processed == 0:
        raise ValueError("No Binance bars in requested range")
    summary = signal_summary(conn, symbol, interval, start, end)
    if summary["total_market_bars_processed"] != processed:
        raise RuntimeError("Stored signal count differs from processed bar count")
    summary.update(symbol=symbol, interval=interval, start=start, end=end,
                   config=asdict(config), inserted=inserted, duplicates_skipped=duplicates,
                   expected_bars=(end - start) // detector.step,
                   missing_bars=(end - start) // detector.step - processed)
    return summary


def parse_time(value):
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use timezone-aware ISO time, e.g. 2026-06-15T00:00:00Z") from exc


def main():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m", choices=("5m", "1h"))
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    args = parser.parse_args()
    try:
        with connect() as conn:
            summary = replay(conn, args.symbol, args.interval, args.start, args.end)
        # The context manager has now committed, so the report represents durable results.
        logger.info("Replay committed\n%s", json.dumps(summary, default=str, indent=2))
        return 2 if summary["missing_bars"] else 0
    except Exception:
        logger.exception("Replay failed; transaction rolled back; existing signals not overwritten")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
