from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from decimal import Decimal
import json
from threading import Event
from unittest.mock import Mock

import psycopg
import pytest

from btc_risk.config import CollectorConfig, DetectorConfig, IngestionConfig
from btc_risk.database.repository import get_bar, insert_bar
from btc_risk.ingestion.binance_rest import BinanceREST, DataQualityError, milliseconds
from btc_risk.ingestion.binance_ws import ClosedBarBuffer, parse_message
from btc_risk.ingestion.collector import CollectorEngine, run_loop
from btc_risk.realtime.replay import replay

STEP = timedelta(minutes=5)


def payload(bar, closed=True):
    return json.dumps({"e": "kline", "E": milliseconds(bar.timestamp + STEP), "s": bar.symbol,
        "k": {"s": bar.symbol, "i": bar.interval, "x": closed, "t": milliseconds(bar.timestamp),
              "T": milliseconds(bar.timestamp + STEP) - 1, "o": str(bar.open), "h": str(bar.high),
              "l": str(bar.low), "c": str(bar.close), "v": str(bar.volume), "q": "0",
              "n": 1, "V": "0", "Q": "0", "B": "0"}})


def rest_row(bar):
    k = json.loads(payload(bar))["k"]
    return [k[key] for key in ("t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B")]


def test_open_ignored_and_closed_normalized(bar):
    assert parse_message(payload(bar, False), bar.symbol, "5m") is None
    result = parse_message(payload(bar), bar.symbol, "5m")
    assert result == replace(bar, source="binance")
    assert result.timestamp.utcoffset() == timedelta(0)


@pytest.mark.parametrize("message", ["bad JSON", "[]", '{"e":"kline"}', '{"e":"trade"}'])
def test_malformed_rejected(message):
    with pytest.raises(DataQualityError):
        parse_message(message, "BTCUSDT", "5m")


def test_wrong_closed_flag_and_bad_ohlcv(bar):
    for key, value in (("x", "true"), ("v", "-1"), ("c", "NaN")):
        message = json.loads(payload(bar))
        message["k"][key] = value
        with pytest.raises(DataQualityError):
            parse_message(json.dumps(message), bar.symbol, "5m")


@pytest.fixture
def setup(conn, bar):
    conn.execute("SELECT 1")

    @contextmanager
    def factory():
        with conn.transaction():
            yield conn

    bars = []
    for i in range(12):
        price = Decimal("90000") + i * i
        bars.append(replace(bar, timestamp=bar.timestamp + STEP * i, open=price, high=price,
                            low=price, close=price, source="binance"))
    rest = BinanceREST(IngestionConfig(page_size=2))
    rest.server_time = Mock(return_value=bar.timestamp + STEP * 12)

    def request(_, params):
        return [rest_row(item) for item in bars if params["startTime"] <= milliseconds(item.timestamp) <= params["endTime"]][::-1]

    rest.request = Mock(side_effect=request)
    config = DetectorConfig(window=3)
    engine = CollectorEngine(bar.symbol, "5m", rest, config, factory)
    return engine, bars, factory


def test_replay_live_recovery_exact_consistency_and_restart(conn, setup):
    engine, bars, factory = setup
    for bar in bars:
        insert_bar(conn, bar)
    replay(conn, engine.symbol, "5m", bars[0].timestamp, bars[-1].timestamp + STEP, engine.config)
    columns = "timestamp, symbol, interval, log_return, rolling_median, rolling_mad, robust_zscore, alert_flag, status"
    expected = conn.execute(f"SELECT {columns} FROM realtime_signals WHERE symbol=%s ORDER BY timestamp", (engine.symbol,)).fetchall()
    # Delete target signals only in this rollback-only synthetic test transaction.
    conn.execute("DELETE FROM realtime_signals WHERE symbol=%s AND timestamp >= %s", (engine.symbol, bars[8].timestamp))
    engine.rest.server_time.return_value = bars[10].timestamp  # closed through index9
    engine.restore_before(bars[8].timestamp)
    engine.recover(bars[8].timestamp, bars[10].timestamp, bars[10].timestamp)
    actual_live = engine.on_live_bar(parse_message(payload(bars[10]), engine.symbol, "5m"))
    assert asdict(actual_live) == expected[10]
    restarted = CollectorEngine(engine.symbol, "5m", engine.rest, engine.config, factory)
    engine.rest.server_time.return_value = bars[11].timestamp + STEP
    restarted.reconcile()
    assert len(restarted.detector.returns) == 3
    assert restarted.detector.previous.timestamp == bars[11].timestamp
    assert conn.execute(f"SELECT {columns} FROM realtime_signals WHERE symbol=%s ORDER BY timestamp", (engine.symbol,)).fetchall() == expected


def test_duplicate_and_conflict(conn, setup):
    engine, bars, _ = setup
    engine.apply(bars[0], "live")
    assert engine.on_live_bar(bars[0]) is None
    with pytest.raises(ValueError, match="different OHLCV"):
        engine.on_live_bar(replace(bars[0], volume=Decimal("99")))
    assert conn.execute("SELECT count(*) AS n FROM realtime_signals WHERE symbol=%s", (engine.symbol,)).fetchone()["n"] == 1


def test_gap_calls_existing_rest_and_preserves_chronological_state(setup):
    engine, bars, _ = setup
    engine.apply(bars[0], "live")
    engine.on_live_bar(bars[5])
    assert engine.recovery_count == 4
    assert engine.rest.request.call_count == 2
    assert engine.detector.previous.timestamp == bars[5].timestamp
    assert len(engine.detector.returns) == 3


def test_unresolved_rest_gap_refuses_live_bar(conn, setup):
    engine, bars, _ = setup
    engine.apply(bars[0], "live")
    engine.rest.request.return_value = []
    engine.rest.request.side_effect = None
    with pytest.raises(DataQualityError, match="incomplete"):
        engine.on_live_bar(bars[3])
    assert get_bar(conn, engine.symbol, "5m", bars[3].timestamp) is None
    assert engine.detector.previous.timestamp == bars[0].timestamp


def test_startup_repairs_internal_gap_and_missing_signal(conn, setup):
    engine, bars, _ = setup
    for i in (0, 1, 3, 4):
        insert_bar(conn, bars[i])
    engine.rest.server_time.return_value = bars[5].timestamp
    engine.reconcile()
    assert engine.detector.previous.timestamp == bars[4].timestamp
    assert conn.execute("SELECT count(*) AS n FROM realtime_signals WHERE symbol=%s", (engine.symbol,)).fetchone()["n"] == 5


def test_commit_failure_keeps_authoritative_state_and_rolls_back(conn, setup):
    engine, bars, factory = setup
    engine.apply(bars[0], "live")

    @contextmanager
    def failing_commit():
        with conn.transaction():
            yield conn
            raise psycopg.OperationalError("simulated COMMIT failure")

    engine.connect = failing_commit
    with pytest.raises(psycopg.OperationalError):
        engine.apply(bars[1], "live")
    assert engine.detector.previous.timestamp == bars[0].timestamp
    assert len(engine.detector.returns) == 0
    assert get_bar(conn, engine.symbol, "5m", bars[1].timestamp) is None
    engine.connect = factory
    assert engine.apply(bars[1], "live").status == "warmup"


def test_lost_commit_ack_reconciles_from_database(conn, setup):
    engine, bars, factory = setup
    engine.apply(bars[0], "live")

    @contextmanager
    def lost_ack():
        with conn.transaction():
            yield conn
        raise psycopg.OperationalError("commit succeeded but acknowledgment lost")

    engine.connect = lost_ack
    with pytest.raises(psycopg.OperationalError):
        engine.apply(bars[1], "live")
    assert engine.detector.previous.timestamp == bars[0].timestamp
    assert get_bar(conn, engine.symbol, "5m", bars[1].timestamp) is not None
    engine.connect = factory
    engine.rest.server_time.return_value = bars[2].timestamp
    engine.reconcile()
    assert engine.detector.previous.timestamp == bars[1].timestamp
    assert engine.on_live_bar(bars[1]) is None


def test_handoff_order_and_overflow(bar):
    buffer = ClosedBarBuffer(2)
    later = replace(bar, timestamp=bar.timestamp + STEP)
    buffer.add(later)
    buffer.add(bar)
    assert buffer.drain() == [bar, later]
    buffer.add(bar)
    buffer.add(later)
    with pytest.raises(ConnectionError, match="overflow"):
        buffer.add(bar)
    with pytest.raises(ConnectionError):
        buffer.drain()


def test_disconnect_reconnect_reconciles_and_handoff(monkeypatch, bar):
    stop = Event()
    engine = Mock(symbol=bar.symbol, interval="5m")
    events = []

    @contextmanager
    def lease():
        yield

    engine.lease = lease
    engine.reconcile.side_effect = lambda _: events.append("reconcile")
    engine.on_live_bar.side_effect = lambda _: (events.append("live"), stop.set())
    attempts = []

    @contextmanager
    def stream(*_):
        attempts.append(1)
        events.append("connected")
        buffer = ClosedBarBuffer(2)
        if len(attempts) == 1:
            buffer.error = ConnectionError("simulated disconnect")
        else:
            buffer.add(bar)  # candle arrived during REST handoff
        yield buffer

    run_loop(engine, CollectorConfig(backoff_max=0.001), stop, stream)
    assert events == ["connected", "reconcile", "connected", "reconcile", "live"]
