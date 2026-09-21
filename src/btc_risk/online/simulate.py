"""Historical as-if-live scoring using frozen daily IF artifacts.

This module never writes the retrospective ``isolation_forest_scores`` table.
``model_available_at`` is an operational convention, not a claim
that the archived model artifact existed at that historical instant.
"""

import argparse
from collections import deque
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from btc_risk.batch.features import FEATURES, build_features
from btc_risk.config import BatchConfig
from btc_risk.database.connection import connect

POLICY = "daily_publish_after_001"
MODEL_AVAILABLE_OFFSET = timedelta(minutes=10)
MODEL_VERSION = "1.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def step_for(interval):
    return timedelta(minutes={"5m": 5, "1h": 60}[interval])


def _frame(window):
    data = pd.DataFrame(window).set_index("timestamp")
    data.index = pd.DatetimeIndex(data.index, tz="UTC") if data.index.tz is None else data.index.tz_convert("UTC")
    return data[["open", "high", "low", "close", "volume"]]


def online_feature(window, config, interval):
    """Use the batch implementation verbatim on the as-of 289-bar buffer."""
    features = build_features(_frame(window), config, pd.Timedelta(step_for(interval)))
    value = features.iloc[-1]
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        raise ValueError("Online feature is not ready or finite")
    return {name: float(value[name]) for name in FEATURES}


def source_run(conn, symbol, interval, score_day):
    start = datetime.combine(score_day, datetime.min.time(), tzinfo=timezone.utc)
    row = conn.execute("""
        SELECT * FROM model_runs
        WHERE symbol=%s AND interval=%s AND model_version=%s
          AND scoring_start=%s AND scoring_end=%s
    """, (symbol, interval, MODEL_VERSION, start, start + timedelta(days=1))).fetchone()
    if row is None:
        raise ValueError(f"No frozen IF model run for {symbol}/{interval}/{score_day}")
    if row["training_end"] > start:
        raise ValueError("Frozen model training reaches into its scoring day")
    return row


def load_verified_artifact(run, config):
    path = config.models_dir.resolve() / run["artifact_path"]
    if not path.is_file():
        raise ValueError(f"Missing source model artifact: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != run["artifact_sha256"]:
        raise ValueError(f"Source artifact checksum mismatch: {path}")
    payload = joblib.load(path)
    if payload.get("feature_order") != FEATURES or str(payload.get("metadata", {}).get("run_id")) != str(run["run_id"]):
        raise ValueError("Source artifact metadata/feature order mismatch")
    return path, payload["estimator"]


def _day_rows(conn, symbol, interval, start, config):
    step = step_for(interval)
    lookback = max(config.volatility_window, config.volume_window)
    query_start = start - lookback * step
    end = start + timedelta(days=1)
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM market_bars WHERE symbol=%s AND interval=%s AND source='binance'
          AND timestamp >= %s AND timestamp < %s ORDER BY timestamp
    """, (symbol, interval, query_start, end)).fetchall()
    expected = [query_start + i * step for i in range(lookback + int(timedelta(days=1) // step))]
    if [r["timestamp"] for r in rows] != expected:
        raise ValueError(f"Incomplete market history for online simulation day {start.date()}")
    return rows[:lookback], rows[lookback:]


def _same_record(stored, calculated):
    return (stored["timestamp"] == calculated["timestamp"] and stored["symbol"] == calculated["symbol"]
            and stored["interval"] == calculated["interval"] and stored["anomaly_flag"] == calculated["anomaly_flag"]
            and stored["feature_values"] == calculated["feature_values"]
            and stored["bar_closed_at"] == calculated["bar_closed_at"]
            and stored["score_available_at"] == calculated["score_available_at"]
            and np.isclose(stored["anomaly_score"], calculated["anomaly_score"], rtol=0, atol=1e-15))


def _feature_parity(actual, expected):
    """Pandas rolling reductions can differ at the last binary digit by prefix length.

    The online calculation deliberately receives only the current 289-bar as-of
    buffer, whereas batch receives a longer frame.  This is numerical roundoff,
    not a different feature definition; score parity remains checked separately.
    """
    return all(np.isclose(actual[name], expected[name], rtol=0, atol=1e-10) for name in FEATURES)


def simulate_day(conn, score_day, symbol="BTCUSDT", interval="5m", config=None):
    """Score only bars whose close is strictly after the fixed publication time."""
    config = config or BatchConfig.from_env()
    start = datetime.combine(score_day, datetime.min.time(), tzinfo=timezone.utc)
    end, step = start + timedelta(days=1), step_for(interval)
    run = source_run(conn, symbol, interval, score_day)
    path, model = load_verified_artifact(run, config)
    available_at = start + MODEL_AVAILABLE_OFFSET
    identity = digest(dict(source_run_id=str(run["run_id"]), policy=POLICY,
                           available_at=available_at, artifact_sha256=run["artifact_sha256"]))
    publication_id = uuid.uuid5(uuid.NAMESPACE_URL, "btc-risk/online-if/" + identity)
    history, day = _day_rows(conn, symbol, interval, start, config)
    window = deque(history, maxlen=max(config.volatility_window, config.volume_window) + 1)
    records = []
    for bar in day:
        window.append(bar)
        closed_at = bar["timestamp"] + step
        # Strict > is intentional: 00:05 close at 00:10 is NOT actionable.
        if closed_at <= available_at:
            continue
        values = online_feature(list(window), config, interval)
        matrix = pd.DataFrame([values], columns=FEATURES)
        score = float(-model.score_samples(matrix)[0])
        flag = bool(model.predict(matrix)[0] == -1)
        records.append(dict(timestamp=bar["timestamp"], symbol=symbol, interval=interval,
                            anomaly_score=score, anomaly_flag=flag, feature_values=values,
                            bar_closed_at=closed_at, score_available_at=closed_at))
    expected_count = int(timedelta(days=1) // step) - 2
    if len(records) != expected_count:
        raise RuntimeError(f"Expected {expected_count} eligible bars, got {len(records)}")
    # The archived retrospective score is the parity oracle, not a source written by this module.
    old = conn.execute("""
        SELECT timestamp, anomaly_score, anomaly_flag, feature_values
        FROM isolation_forest_scores WHERE run_id=%s ORDER BY timestamp
    """, (run["run_id"],)).fetchall()
    old_by_time = {r["timestamp"]: r for r in old}
    for record in records:
        prior = old_by_time.get(record["timestamp"])
        if prior is None or not _feature_parity(record["feature_values"], prior["feature_values"]) or prior["anomaly_flag"] != record["anomaly_flag"] or not np.isclose(prior["anomaly_score"], record["anomaly_score"], rtol=0, atol=1e-15):
            raise RuntimeError(f"Online/offline parity mismatch at {record['timestamp']}")
    publication = dict(publication_id=publication_id, publication_identity=identity,
                       source_run_id=run["run_id"], symbol=symbol, interval=interval,
                       training_start=run["training_start"], training_end=run["training_end"],
                       scoring_start=start, scoring_end=end, model_available_at=available_at,
                       availability_policy=POLICY, feature_set=run["feature_set"],
                       model_parameters=run["model_parameters"], artifact_path=run["artifact_path"],
                       artifact_sha256=run["artifact_sha256"], code_version=run["code_version"], data_hash=run["data_hash"])
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (identity,))
    existing = conn.execute("SELECT * FROM online_if_model_publications WHERE publication_identity=%s", (identity,)).fetchone()
    if existing:
        for key, value in publication.items():
            if key != "publication_id" and existing[key] != value:
                raise RuntimeError("Existing online publication differs from deterministic simulation")
        stored = conn.execute("""SELECT timestamp,symbol,interval,anomaly_score,anomaly_flag,feature_values,
                               bar_closed_at,score_available_at FROM online_isolation_forest_scores
                               WHERE publication_id=%s ORDER BY timestamp""", (publication_id,)).fetchall()
        if len(stored) != len(records) or any(not _same_record(a, b) for a, b in zip(stored, records)):
            raise RuntimeError("Existing online scores differ from deterministic simulation")
        reused = True
    else:
        conn.execute("""INSERT INTO online_if_model_publications
            (publication_id,publication_identity,source_run_id,symbol,interval,training_start,training_end,
             scoring_start,scoring_end,model_available_at,availability_policy,feature_set,
             model_parameters,artifact_path,artifact_sha256,code_version,data_hash)
             VALUES (%(publication_id)s,%(publication_identity)s,%(source_run_id)s,%(symbol)s,%(interval)s,
             %(training_start)s,%(training_end)s,%(scoring_start)s,%(scoring_end)s,%(model_available_at)s,
             %(availability_policy)s,%(feature_set)s,%(model_parameters)s,%(artifact_path)s,%(artifact_sha256)s,
             %(code_version)s,%(data_hash)s)""", {**publication, "feature_set": Jsonb(publication["feature_set"]), "model_parameters": Jsonb(publication["model_parameters"])})
        with conn.cursor() as cursor:
            cursor.executemany("""INSERT INTO online_isolation_forest_scores
                (publication_id,timestamp,symbol,interval,anomaly_score,anomaly_flag,feature_values,
                 bar_closed_at,score_available_at)
                VALUES (%(publication_id)s,%(timestamp)s,%(symbol)s,%(interval)s,%(anomaly_score)s,
                %(anomaly_flag)s,%(feature_values)s,%(bar_closed_at)s,%(score_available_at)s)""",
                [{**r, "publication_id": publication_id, "feature_values": Jsonb(r["feature_values"])} for r in records])
        reused = False
    return dict(publication_id=str(publication_id), source_run_id=str(run["run_id"]), score_day=str(score_day),
                model_available_at=available_at, first_eligible_timestamp=records[0]["timestamp"],
                excluded_by_availability=2, actionable_scores=len(records), anomalies=sum(r["anomaly_flag"] for r in records),
                artifact_path=str(path), reused=reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat, help="Inclusive UTC score date")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m", choices=("5m", "1h"))
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("end-date must not precede start-date")
    results = []
    config = BatchConfig.from_env()
    for offset in range((args.end_date - args.start_date).days + 1):
        with connect() as conn:
            results.append(simulate_day(conn, args.start_date + timedelta(days=offset), args.symbol, args.interval, config))
    print(json.dumps(results, default=str, indent=2))


if __name__ == "__main__":
    main()
