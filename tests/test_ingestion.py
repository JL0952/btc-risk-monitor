from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from btc_risk.config import IngestionConfig
from btc_risk.database.repository import get_bar, inspect_range
from btc_risk.ingestion.backfill import backfill
from btc_risk.ingestion.binance_rest import (
    BinanceREST, DataQualityError, INTERVALS, milliseconds, parse_kline,
)

STEP = INTERVALS["5m"]


def raw(timestamp, **changes):
    row = [milliseconds(timestamp), "90000.00", "90100.00", "89900.00", "90050.00",
           "12.50000000", milliseconds(timestamp + STEP) - 1, "1125625", 20, "6", "540000", "0"]
    for index, value in changes.items():
        row[int(index)] = value
    return row


def test_parse_utc_decimal_and_unfinished(bar):
    row = raw(bar.timestamp)
    parsed = parse_kline(row, "BTCUSDT", "5m", bar.timestamp + STEP)
    assert parsed.timestamp == bar.timestamp
    assert parsed.timestamp.utcoffset() == timedelta(0)
    assert parsed.close == Decimal("90050")
    assert parsed.source == "binance"
    assert parse_kline(row, "BTCUSDT", "5m", bar.timestamp + STEP - timedelta(milliseconds=1)) is None


@pytest.mark.parametrize("index,value", [(1, "0"), (2, "-1"), (5, "-2"),
    (3, "90060"), (4, "90200"), (1, "NaN"), (5, "Infinity"), (1, "90000.0000000000001")])
def test_invalid_ohlcv_rejected(bar, index, value):
    with pytest.raises(DataQualityError):
        parse_kline(raw(bar.timestamp, **{str(index): value}), "BTCUSDT", "5m", bar.timestamp + STEP)


def test_unaligned_timestamp_rejected(bar):
    with pytest.raises(DataQualityError, match="Unaligned"):
        parse_kline(raw(bar.timestamp + timedelta(seconds=17)), "BTCUSDT", "5m", bar.timestamp + STEP * 2)


def test_pagination_inclusive_end_and_short_pages(bar):
    client = BinanceREST(IngestionConfig(page_size=2))
    client.request = Mock(side_effect=[[raw(bar.timestamp)], [], [raw(bar.timestamp + STEP * 4)]])
    pages = list(client.pages("BTCUSDT", "5m", bar.timestamp, bar.timestamp + STEP * 5))
    assert len(pages) == 3
    params = [call.args[1] for call in client.request.call_args_list]
    assert [p["startTime"] for p in params] == [milliseconds(bar.timestamp + STEP * i) for i in (0, 2, 4)]
    assert [p["endTime"] for p in params] == [milliseconds(bar.timestamp + STEP * i) - 1 for i in (2, 4, 5)]


@pytest.fixture
def pipeline(conn, bar):
    # A rollback-only outer transaction isolates real PostgreSQL integration tests.
    conn.execute("SELECT 1")

    @contextmanager
    def factory():
        with conn.transaction():
            yield conn

    symbol = bar.symbol.replace("_", "").upper()[:30]
    client = BinanceREST(IngestionConfig(page_size=2))

    def run(rows, end=None, as_of=None):
        client.request = Mock(return_value=rows)
        return backfill(client, symbol, "5m", bar.timestamp, end or bar.timestamp + STEP * 2,
                        as_of or bar.timestamp + STEP * 2, connection_factory=factory)

    return run, symbol, client


def test_backfill_rerun_and_gap_boundaries(conn, bar, pipeline):
    run, symbol, _ = pipeline
    rows = [raw(bar.timestamp + STEP), raw(bar.timestamp)]  # sort before persistence
    first = run(rows)
    second = run(rows)
    assert first["new_rows_inserted"] == 2
    assert second["new_rows_inserted"] == 0
    assert second["duplicate_rows_skipped"] == 2
    assert second["stored_rows"] == 2
    assert second["gap_count"] == 0
    conn.execute("DELETE FROM market_bars WHERE symbol = %s AND timestamp = %s", (symbol, bar.timestamp))
    gap = inspect_range(conn, symbol, "5m", bar.timestamp, bar.timestamp + STEP * 3, STEP)
    assert gap["gap_count"] == 2  # leading deleted bar and trailing absent bar
    assert gap["missing_timestamps"] == [bar.timestamp, bar.timestamp + STEP * 2]


def test_unfinished_not_written(conn, bar, pipeline):
    run, symbol, _ = pipeline
    result = run([raw(bar.timestamp), raw(bar.timestamp + STEP)], as_of=bar.timestamp + STEP * 1.5)
    assert result["unfinished_rows_skipped"] == 1
    assert result["stored_rows"] == 1
    assert get_bar(conn, symbol, "5m", bar.timestamp + STEP) is None


def test_conflict_rolls_back_page(conn, bar, pipeline):
    run, symbol, _ = pipeline
    run([raw(bar.timestamp + STEP)])
    with pytest.raises(ValueError, match="different OHLCV"):
        run([raw(bar.timestamp), raw(bar.timestamp + STEP, **{"4": "90060"})])
    assert get_bar(conn, symbol, "5m", bar.timestamp) is None
    assert get_bar(conn, symbol, "5m", bar.timestamp + STEP)["close"] == Decimal("90050")


def test_invalid_page_writes_nothing(conn, bar, pipeline):
    run, symbol, _ = pipeline
    with pytest.raises(DataQualityError):
        run([raw(bar.timestamp), raw(bar.timestamp + STEP, **{"5": "-1"})])
    assert get_bar(conn, symbol, "5m", bar.timestamp) is None


def test_repeated_old_page_rejected(bar, pipeline):
    run, _, client = pipeline
    with pytest.raises(DataQualityError, match="outside requested page"):
        run([raw(bar.timestamp)], end=bar.timestamp + STEP * 3, as_of=bar.timestamp + STEP * 3)
    assert client.request.call_count == 2


def test_http_region_error_no_fallback(monkeypatch):
    from io import BytesIO
    request = Mock(side_effect=HTTPError("url", 451, "Restricted", {}, BytesIO(b"restricted location")))
    monkeypatch.setattr("btc_risk.ingestion.binance_rest.urlopen", request)
    with pytest.raises(RuntimeError, match="451"):
        BinanceREST().server_time()
    assert request.call_count == 1
