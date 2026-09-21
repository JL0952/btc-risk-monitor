"""CLI: fetch closed historical candles, commit per page, verify SQL coverage."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import re
import time

from btc_risk.config import IngestionConfig
from btc_risk.database.connection import connect
from btc_risk.database.repository import insert_bar, inspect_range, utc
from btc_risk.ingestion.binance_rest import BinanceREST, DataQualityError, EPOCH, INTERVALS, parse_kline, validate_range

logger = logging.getLogger(__name__)


def backfill(client, symbol, interval, start, end, as_of, connection_factory=connect, check_only=False):
    start, end = validate_range(start, end, interval)
    if not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
        raise ValueError("symbol must be an uppercase exchange symbol")
    step = INTERVALS[interval]
    cutoff = EPOCH + ((utc(as_of) - EPOCH) // step) * step
    effective_end = min(end, cutoff)
    if start >= effective_end:
        raise ValueError("Requested range contains no completed candles")
    summary = dict(symbol=symbol, interval=interval, requested_start=start, requested_end=end,
                   effective_end=effective_end, closed_as_of=as_of, api_bars_received=0,
                   new_rows_inserted=0, duplicate_rows_skipped=0, invalid_rows=0,
                   unfinished_rows_skipped=0, pages_committed=0, mode="check" if check_only else "download")
    try:
        if not check_only:
            for page_start, page_end, rows in client.pages(symbol, interval, start, effective_end):
                summary["api_bars_received"] += len(rows)
                bars = []
                for row in rows:
                    try:
                        bar = parse_kline(row, symbol, interval, as_of)
                        if bar is None:
                            summary["unfinished_rows_skipped"] += 1
                            continue
                        if not page_start <= bar.timestamp < page_end:
                            raise DataQualityError(f"Response outside requested page: {bar.timestamp}")
                        bars.append(bar)
                    except DataQualityError:
                        summary["invalid_rows"] += 1
                        logger.exception("Invalid kline page=%s row=%r", page_start, row)
                        raise
                inserted = duplicates = 0
                with connection_factory() as conn:
                    for bar in sorted(bars, key=lambda item: item.timestamp):
                        try:
                            if insert_bar(conn, bar):
                                inserted += 1
                            else:
                                duplicates += 1
                        except ValueError:
                            logger.exception("OHLCV conflict symbol=%s interval=%s timestamp=%s; page rolled back",
                                             symbol, interval, bar.timestamp)
                            raise
                summary["new_rows_inserted"] += inserted
                summary["duplicate_rows_skipped"] += duplicates
                summary["pages_committed"] += 1
                logger.info("Page committed [%s, %s): received=%s inserted=%s duplicates=%s",
                            page_start, page_end, len(rows), inserted, duplicates)
        summary["status"] = "complete"
    except Exception:
        summary["status"] = "failed"
        logger.exception("Backfill failed; earlier committed pages remain; safe to rerun")
        raise
    finally:
        try:
            with connection_factory() as conn:
                summary.update(inspect_range(conn, symbol, interval, start, effective_end, step))
        except Exception:
            logger.exception("SQL coverage check failed")
            summary["status"] = "failed"
            raise
        finally:
            if summary.get("gap_count", 0) and summary.get("status") == "complete":
                summary["status"] = "incomplete"
            logger.info("Backfill summary\n%s", json.dumps(summary, default=str, indent=2))
    return summary


def parse_time(value):
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use timezone-aware ISO time, e.g. 2026-09-01T00:00:00Z") from exc


def main():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    config = IngestionConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=config.symbol)
    parser.add_argument("--interval", default=config.interval, choices=INTERVALS)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--days", type=int, help="Previous N complete UTC days; optionally ending at --end")
    choice.add_argument("--start", type=parse_time)
    parser.add_argument("--end", type=parse_time, help="Exclusive boundary; required with --start")
    parser.add_argument("--check-only", action="store_true", help="SQL gap check only, no Binance requests/writes")
    args = parser.parse_args()
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive")
    client = BinanceREST(config)
    try:
        now = datetime.now(timezone.utc) if args.check_only else client.server_time()
        end = args.end or now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = args.start or end - timedelta(days=args.days)
        result = backfill(client, args.symbol, args.interval, start, end, now, check_only=args.check_only)
        return 0 if result["status"] == "complete" else 2
    except Exception:
        logger.exception("Command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
