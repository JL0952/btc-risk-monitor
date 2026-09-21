from dataclasses import replace
from datetime import timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from btc_risk.database.connection import check_connection
from btc_risk.database.repository import get_bar, insert_bar


def test_connection_and_utc(conn):
    assert check_connection()
    assert conn.execute("SHOW timezone").fetchone()["TimeZone"] == "UTC"


def test_duplicate_is_idempotent(conn, bar):
    assert insert_bar(conn, bar)
    original = get_bar(conn, bar.symbol, bar.interval, bar.timestamp)
    assert not insert_bar(conn, bar)
    assert get_bar(conn, bar.symbol, bar.interval, bar.timestamp) == original
    assert conn.execute("SELECT count(*) AS n FROM market_bars WHERE symbol = %s", (bar.symbol,)).fetchone()["n"] == 1
    # The database itself also enforces uniqueness, even without repository logic.
    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction():
            conn.execute("INSERT INTO market_bars SELECT * FROM market_bars WHERE symbol = %s", (bar.symbol,))


def test_conflicting_duplicate_rejected(conn, bar):
    insert_bar(conn, bar)
    with pytest.raises(ValueError, match="different OHLCV"):
        insert_bar(conn, replace(bar, close=Decimal("90060")))


@pytest.mark.parametrize("field,value", [
    ("open", "0"), ("high", "-1"), ("low", "0"), ("close", "-1"),
    ("volume", "-0.1"), ("open", "NaN"), ("volume", "NaN"),
    ("open", "90200"), ("close", "89800"),
])
def test_database_constraints(conn, bar, field, value):
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_bar(conn, replace(bar, **{field: Decimal(value)}))


def test_timezone_normalizes_same_instant(conn, bar):
    offset_bar = replace(bar, timestamp=bar.timestamp.astimezone(timezone(timedelta(hours=-5))))
    assert insert_bar(conn, offset_bar)
    saved = get_bar(conn, bar.symbol, bar.interval, bar.timestamp)
    assert saved["timestamp"] == bar.timestamp
    assert saved["timestamp"].utcoffset() == timedelta(0)
    assert not insert_bar(conn, bar)


def test_naive_timestamp_rejected(conn, bar):
    with pytest.raises(ValueError, match="timezone"):
        insert_bar(conn, replace(bar, timestamp=bar.timestamp.replace(tzinfo=None)))


def test_signal_requires_bar(conn, bar):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            conn.execute("""
                INSERT INTO realtime_signals (timestamp, symbol, interval, status)
                VALUES (%s, %s, %s, %s)
            """, (bar.timestamp, bar.symbol, bar.interval, "not_calculated"))
    insert_bar(conn, bar)
    conn.execute("""
        INSERT INTO realtime_signals (timestamp, symbol, interval, status)
        VALUES (%s, %s, %s, %s)
    """, (bar.timestamp, bar.symbol, bar.interval, "not_calculated"))
    row = conn.execute("SELECT robust_zscore, alert_flag FROM realtime_signals WHERE symbol = %s", (bar.symbol,)).fetchone()
    assert row == {"robust_zscore": None, "alert_flag": None}
