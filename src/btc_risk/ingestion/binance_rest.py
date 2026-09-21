"""Binance public REST adapter with bounded pages and strict closed-bar parsing."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from btc_risk.config import IngestionConfig
from btc_risk.database.repository import MarketBar, utc

logger = logging.getLogger(__name__)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
INTERVALS = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1)}


class DataQualityError(ValueError):
    pass


def milliseconds(value: datetime) -> int:
    return (utc(value) - EPOCH) // timedelta(milliseconds=1)


def from_milliseconds(value: int) -> datetime:
    if type(value) is not int:
        raise DataQualityError("Expected integer millisecond timestamp")
    return EPOCH + timedelta(milliseconds=value)


def validate_range(start, end, interval):
    if interval not in INTERVALS:
        raise ValueError("Supported intervals: 5m, 1h")
    step = INTERVALS[interval]
    start, end = utc(start), utc(end)
    if start >= end or any((value - EPOCH) % step for value in (start, end)):
        raise ValueError("Require start < end and interval-aligned UTC boundaries")
    return start, end


def parse_kline(row, symbol, interval, closed_as_of):
    """Return a validated bar, or None for a candle not closed at the fixed cutoff."""
    try:
        if not isinstance(row, list) or len(row) != 12:
            raise DataQualityError("Kline must be a 12-element JSON array")
        timestamp = from_milliseconds(row[0])
        step = INTERVALS[interval]
        if (timestamp - EPOCH) % step:
            raise DataQualityError(f"Unaligned opening time: {timestamp.isoformat()}")
        if type(row[6]) is not int or row[6] != milliseconds(timestamp + step) - 1:
            raise DataQualityError("Unexpected candle close timestamp")
        values = [Decimal(str(value)) for value in row[1:6]]
        if any(not value.is_finite() for value in values):
            raise DataQualityError("Nonfinite OHLCV")
        if any(value <= 0 for value in values[:4]) or values[4] < 0:
            raise DataQualityError("Prices must be positive; volume must be nonnegative")
        opening, high, low, close, volume = values
        if not low <= opening <= high or not low <= close <= high:
            raise DataQualityError("Invalid OHLC high/low ordering")
        # Reject values that NUMERIC(30,12) would round or overflow.
        if any(value >= Decimal("1e18") or value != value.quantize(Decimal("1e-12")) for value in values):
            raise DataQualityError("OHLCV exceeds database precision")
        if timestamp + step > utc(closed_as_of):
            return None
        return MarketBar(timestamp, symbol, interval, opening, high, low, close, volume, "binance")
    except (ValueError, TypeError, InvalidOperation, OverflowError, KeyError) as exc:
        if isinstance(exc, DataQualityError):
            raise
        raise DataQualityError(f"Malformed kline: {exc}") from exc


class BinanceREST:
    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig.from_env()

    def request(self, path, params=None):
        url = self.config.base_url + path
        if params:
            url += "?" + urlencode(params)
        for attempt in range(self.config.attempts):
            try:
                request = Request(url, headers={"User-Agent": "btc-risk-monitor/0.2", "Accept": "application/json"})
                with urlopen(request, timeout=self.config.timeout) as response:
                    payload = json.load(response)
                if isinstance(payload, dict) and "code" in payload:
                    raise RuntimeError(f"Binance API error: {payload}")
                return payload
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                logger.error("Binance HTTP %s endpoint=%s body=%s", exc.code, url, body)
                if exc.code == 418 or (exc.code < 500 and exc.code != 429):
                    raise RuntimeError(f"Binance HTTP {exc.code}: {body}; no alternate source used") from exc
                if attempt + 1 == self.config.attempts:
                    raise RuntimeError(f"Binance HTTP retries exhausted: {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                if delay > 30:
                    raise RuntimeError(f"Binance rate limit: retry after {delay}s; rerun later") from exc
                time.sleep(delay)
            except (URLError, TimeoutError, ConnectionError) as exc:
                logger.warning("Binance connection attempt %s failed: %s", attempt + 1, exc)
                if attempt + 1 == self.config.attempts:
                    raise RuntimeError("Binance unreachable; no alternate source used") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable request state")

    def server_time(self):
        payload = self.request("/api/v3/time")
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise DataQualityError("Malformed Binance server-time response")
        return from_milliseconds(payload["serverTime"])

    def pages(self, symbol, interval, start, end):
        start, end = validate_range(start, end, interval)
        step = INTERVALS[interval]
        cursor = start
        while cursor < end:
            # Request at most page_size EXPECTED timestamps. An empty/short page
            # is a possible gap, not a reason to skip the rest of the requested range.
            page_end = min(cursor + step * self.config.page_size, end)
            rows = self.request("/api/v3/klines", {
                "symbol": symbol, "interval": interval, "startTime": milliseconds(cursor),
                "endTime": milliseconds(page_end) - 1, "limit": self.config.page_size,
                "timeZone": "0",
            })
            if not isinstance(rows, list) or len(rows) > self.config.page_size:
                raise DataQualityError("Malformed/oversized Binance kline response")
            yield cursor, page_end, rows
            cursor = page_end
