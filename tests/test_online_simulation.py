from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from btc_risk.online.simulate import MODEL_AVAILABLE_OFFSET, online_feature, simulate_day
from btc_risk.online.shadow import LiveIFShadow
from btc_risk.batch.features import FEATURES
from btc_risk.batch.run_isolation_forest import run_day
from btc_risk.config import BatchConfig


@pytest.fixture
def config(tmp_path):
    return BatchConfig(training_days=1, volatility_window=3, volume_window=4,
                       n_estimators=10, models_dir=Path(tmp_path))


@pytest.fixture
def data():
    index = pd.date_range("2026-07-01", periods=30, freq="5min", tz="UTC")
    close = 100 + np.sin(np.arange(30))
    return pd.DataFrame(dict(open=close, high=close + 2, low=close - 2, close=close,
                             volume=np.arange(30) + 1.0), index=index)


@pytest.fixture
def market(conn, config):
    score_day = date(2026, 7, 1)
    start = pd.Timestamp(score_day, tz="UTC").to_pydatetime() - timedelta(days=1, minutes=20)
    symbol = "ONLINE" + uuid.uuid4().hex[:16].upper()
    rows = []
    for i in range(580):
        close = 90000 + 50 * np.sin(i / 5)
        rows.append((start + timedelta(minutes=5 * i), symbol, "5m", close, close + 10,
                     close - 10, close, 10 + (i % 20), "binance"))
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO market_bars
            (timestamp,symbol,interval,open,high,low,close,volume,source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
    return symbol, score_day, {"scoring_start": pd.Timestamp(score_day, tz="UTC").to_pydatetime()}


def test_online_feature_is_causal_as_of_window(data, config):
    bars = data.reset_index(names="timestamp").to_dict("records")
    baseline = online_feature(bars[:5], config, "5m")
    changed = [dict(row) for row in bars]
    changed[5]["close"] *= 5
    changed[5]["high"] *= 5
    changed[5]["low"] *= 5
    changed[5]["volume"] *= 5
    assert online_feature(changed[:5], config, "5m") == baseline


def test_strict_publication_gate_and_offline_parity(conn, market, config):
    symbol, day, bounds = market
    retrospective = run_day(conn, day, symbol, config=config)
    first = simulate_day(conn, day, symbol, config=config)
    assert first["actionable_scores"] == 286
    assert first["excluded_by_availability"] == 2
    assert first["model_available_at"] == bounds["scoring_start"] + MODEL_AVAILABLE_OFFSET
    assert first["first_eligible_timestamp"] == bounds["scoring_start"] + timedelta(minutes=10)
    rows = conn.execute("""
        SELECT o.timestamp,o.anomaly_score,o.anomaly_flag,o.feature_values,o.bar_closed_at,o.score_available_at,
               r.anomaly_score AS retrospective_score,r.anomaly_flag AS retrospective_flag,r.feature_values AS retrospective_features
        FROM online_isolation_forest_scores o
        JOIN online_if_model_publications p ON p.publication_id=o.publication_id
        JOIN isolation_forest_scores r ON r.run_id=p.source_run_id AND r.timestamp=o.timestamp
        WHERE p.publication_id=%s ORDER BY o.timestamp
    """, (first["publication_id"],)).fetchall()
    assert len(rows) == 286
    for row in rows:
        assert row["bar_closed_at"] > first["model_available_at"]
        assert row["score_available_at"] == row["bar_closed_at"]
        assert row["anomaly_flag"] == row["retrospective_flag"]
        assert row["anomaly_score"] == row["retrospective_score"]
        np.testing.assert_allclose([row["feature_values"][name] for name in FEATURES],
                                   [row["retrospective_features"][name] for name in FEATURES], rtol=0, atol=1e-10)
    second = simulate_day(conn, day, symbol, config=config)
    assert second["reused"] and second["publication_id"] == first["publication_id"]
    assert retrospective["run_id"] == first["source_run_id"]


def test_live_shadow_publishes_prior_only_model_and_scores_after_publication(conn, market, config):
    symbol, day, bounds = market
    available = bounds["scoring_start"] + timedelta(hours=12)
    shadow, reused = LiveIFShadow.publish(conn, symbol, "5m", now=available, config=config)
    assert not reused and shadow.model_available_at == available
    publication = conn.execute("SELECT source_run_id,publication_mode,training_end,model_available_at FROM online_if_model_publications WHERE publication_id=%s", (shadow.publication_id,)).fetchone()
    assert publication["source_run_id"] is None
    assert publication["publication_mode"] == "live_shadow"
    assert publication["training_end"] == bounds["scoring_start"]
    early = SimpleNamespace(timestamp=bounds["scoring_start"], symbol=symbol, interval="5m")
    assert shadow.score(conn, early) is None
    row = conn.execute("SELECT timestamp FROM market_bars WHERE symbol=%s AND timestamp=%s", (symbol, available)).fetchone()
    assert row is not None
    result = shadow.score(conn, SimpleNamespace(timestamp=available, symbol=symbol, interval="5m"))
    assert result is not None and result["score_available_at"] == available + timedelta(minutes=5)
    assert shadow.score(conn, SimpleNamespace(timestamp=available, symbol=symbol, interval="5m")) is None
