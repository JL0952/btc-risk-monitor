from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
import psycopg
import pytest

from btc_risk.batch.features import FEATURES, build_features
from btc_risk.batch.run_isolation_forest import fit_score, load_features, periods, run_day
from btc_risk.config import BatchConfig


@pytest.fixture
def config(tmp_path):
    return BatchConfig(training_days=1, volatility_window=3, volume_window=4,
                       n_estimators=10, models_dir=tmp_path)


@pytest.fixture
def data():
    index = pd.date_range("2026-07-01", periods=30, freq="5min", tz="UTC")
    close = 100 + np.sin(np.arange(30))
    return pd.DataFrame(dict(open=close, high=close + 2, low=close - 2, close=close,
                             volume=np.arange(30) + 1.0), index=index)


def test_features_sorted_and_fixed_order(data, config):
    expected = build_features(data, config, pd.Timedelta(minutes=5))
    actual = build_features(data.sample(frac=1, random_state=42), config, pd.Timedelta(minutes=5))
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
    assert list(actual.columns) == FEATURES


def test_future_price_volume_mutation_preserves_past_features(data, config):
    baseline = build_features(data, config, pd.Timedelta(minutes=5))
    changed = data.copy()
    changed.iloc[15:] *= 3
    result = build_features(changed, config, pd.Timedelta(minutes=5))
    pd.testing.assert_frame_equal(baseline.iloc[:15], result.iloc[:15], check_exact=True)
    assert result.loc[data.index[15], "log_return"] != baseline.loc[data.index[15], "log_return"]


def test_volatility_is_trailing_current_closed_returns(data, config):
    features = build_features(data, config, pd.Timedelta(minutes=5))
    for i in (4, 10):
        returns = np.log(data.close.iloc[i-2:i+1].to_numpy() / data.close.iloc[i-3:i].to_numpy())
        assert features.rolling_volatility.iloc[i] == pytest.approx(np.sqrt(np.sum(returns ** 2)))


def test_volume_baseline_excludes_current(data, config):
    data.loc[data.index[4], "volume"] = 1000
    features = build_features(data, config, pd.Timedelta(minutes=5))
    expected = (1000 - np.mean([1, 2, 3, 4])) / np.std([1, 2, 3, 4], ddof=0)
    assert features.volume_zscore.iloc[4] == pytest.approx(expected)
    data.volume = 2.0
    assert np.isfinite(build_features(data, config, pd.Timedelta(minutes=5)).iloc[4:].to_numpy()).all()


@pytest.mark.parametrize("invalid", ["gap", "duplicate"])
def test_features_reject_broken_time_grid(data, config, invalid):
    broken = data.drop(data.index[10]) if invalid == "gap" else pd.concat([data, data.iloc[[10]]])
    with pytest.raises(ValueError, match="Duplicate or missing"):
        build_features(broken, config, pd.Timedelta(minutes=5))


def test_score_direction_order_and_no_scoring_training_leakage():
    rng = np.random.default_rng(42)
    training = pd.DataFrame(rng.normal(size=(300, 5)), columns=FEATURES)
    scoring = pd.DataFrame([np.zeros(5), np.full(5, 1000)], columns=FEATURES)
    params = dict(n_estimators=50, random_state=42, contamination=0.01, n_jobs=1)
    model, scores, flags = fit_score(training[FEATURES[::-1]], scoring[FEATURES[::-1]], params)
    assert list(model.feature_names_in_) == FEATURES
    np.testing.assert_array_equal(scores, -model.score_samples(scoring))
    assert scores[1] > scores[0] and flags[1]
    changed = scoring * 100
    second_model, _, _ = fit_score(training, changed, params)
    assert model.offset_ == second_model.offset_
    for first_tree, second_tree in zip(model.estimators_, second_model.estimators_):
        np.testing.assert_array_equal(first_tree.tree_.threshold, second_tree.tree_.threshold)


@pytest.fixture
def market(conn, config):
    score_date = date(2026, 7, 1)
    bounds = periods(score_date, config)
    symbol = "IFTEST" + uuid.uuid4().hex[:16].upper()
    start = bounds["training_start"] - timedelta(minutes=20)
    count = 580  # 4 warmup + 288 train + 288 score
    rows = []
    for i in range(count):
        close = 90000 + 50 * np.sin(i / 5)
        rows.append((start + timedelta(minutes=5*i), symbol, "5m", close, close+10,
                     close-10, close, 10 + (i % 20), "binance"))
    with conn.cursor() as cursor:
        cursor.executemany("""
            INSERT INTO market_bars (timestamp,symbol,interval,open,high,low,close,volume,source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)
    return symbol, score_date, bounds


def test_training_before_scoring_and_no_warmup_loss(conn, market, config):
    symbol, _, bounds = market
    training, scoring, _, warmup = load_features(conn, symbol, "5m", bounds, config)
    assert len(training) == len(scoring) == 288
    assert warmup == 4
    assert training.index.max() < bounds["scoring_start"] == scoring.index.min()
    assert not set(training.index) & set(scoring.index)


def test_artifact_linkage_and_idempotent_rerun(conn, market, config):
    symbol, day, _ = market
    first = run_day(conn, day, symbol, config=config)
    saved = conn.execute("SELECT * FROM model_runs WHERE run_id=%s", (uuid.UUID(first["run_id"]),)).fetchone()
    artifact = joblib.load(first["artifact_path"])
    assert artifact["feature_order"] == FEATURES
    assert artifact["metadata"]["training_end"] <= artifact["metadata"]["scoring_start"]
    assert artifact["estimator"].n_estimators == config.n_estimators
    second = run_day(conn, day, symbol, config=config)
    assert second["skipped"] and second["run_id"] == first["run_id"]
    assert conn.execute("SELECT * FROM model_runs WHERE run_id=%s", (uuid.UUID(first["run_id"]),)).fetchone() == saved
    assert conn.execute("SELECT count(*) AS n FROM isolation_forest_scores WHERE run_id=%s",
                        (uuid.UUID(first["run_id"]),)).fetchone()["n"] == 288
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            conn.execute("UPDATE isolation_forest_scores SET run_id=%s WHERE run_id=%s",
                         (uuid.uuid4(), uuid.UUID(first["run_id"])))


@pytest.mark.parametrize("part", ["training", "scoring", "warmup"])
def test_missing_data_refuses_training(conn, market, config, part):
    symbol, day, bounds = market
    stamp = {"training": bounds["training_start"], "scoring": bounds["scoring_start"],
             "warmup": bounds["training_start"] - timedelta(minutes=5)}[part]
    conn.execute("DELETE FROM market_bars WHERE symbol=%s AND timestamp=%s", (symbol, stamp))
    with pytest.raises(ValueError, match="Incomplete"):
        run_day(conn, day, symbol, config=config)
    assert not list(config.models_dir.glob("*.joblib"))


def test_current_incomplete_day_rejected(conn, market, config):
    symbol, day, bounds = market
    with pytest.raises(ValueError, match="not yet complete"):
        run_day(conn, day, symbol, config=config, now=bounds["scoring_start"] + timedelta(hours=12))


def test_changed_data_or_stored_results_raise_inconsistency(conn, market, config):
    symbol, day, bounds = market
    result = run_day(conn, day, symbol, config=config)
    with conn.transaction(force_rollback=True):
        conn.execute("UPDATE market_bars SET volume=volume+1 WHERE symbol=%s AND timestamp=%s", (symbol, bounds["training_start"]))
        with pytest.raises(ValueError, match="different data"):
            run_day(conn, day, symbol, config=config)
    conn.execute("UPDATE isolation_forest_scores SET anomaly_score=anomaly_score+0.1 WHERE run_id=%s", (uuid.UUID(result["run_id"]),))
    with pytest.raises(ValueError, match="different scores"):
        run_day(conn, day, symbol, config=config)


def test_artifact_tampering_rejected(conn, market, config):
    symbol, day, _ = market
    result = run_day(conn, day, symbol, config=config)
    Path(result["artifact_path"]).write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        run_day(conn, day, symbol, config=config)


def test_database_rollback_leaves_reusable_artifact(conn, market, config):
    symbol, day, _ = market
    with conn.transaction(force_rollback=True):
        first = run_day(conn, day, symbol, config=config)
    assert Path(first["artifact_path"]).is_file()
    assert conn.execute("SELECT run_id FROM model_runs WHERE run_id=%s", (uuid.UUID(first["run_id"]),)).fetchone() is None
    second = run_day(conn, day, symbol, config=config)
    assert not second["skipped"] and second["run_id"] == first["run_id"]
