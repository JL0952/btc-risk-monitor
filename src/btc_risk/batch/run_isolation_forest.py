"""Manual, leakage-safe daily Isolation Forest runs and inclusive-date walk-forward."""

import argparse
from datetime import date, datetime, time as day_time, timedelta, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import tempfile
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb
from sklearn.ensemble import IsolationForest

from btc_risk import config as config_module
from btc_risk.batch import features as feature_module
from btc_risk.batch.features import FEATURES, build_features
from btc_risk.config import BatchConfig
from btc_risk.database.connection import connect

logger = logging.getLogger(__name__)
MODEL_VERSION = "1.0"


def periods(score_date, config):
    scoring_start = datetime.combine(score_date, day_time(), timezone.utc)
    return dict(training_start=scoring_start - timedelta(days=config.training_days),
                training_end=scoring_start, scoring_start=scoring_start,
                scoring_end=scoring_start + timedelta(days=1))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def load_features(conn, symbol, interval, bounds, config):
    step = timedelta(minutes={"5m": 5, "1h": 60}[interval])
    warmup = max(config.volatility_window, config.volume_window)
    query_start = bounds["training_start"] - warmup * step
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume FROM market_bars
        WHERE symbol=%s AND interval=%s AND source='binance'
          AND timestamp >= %s AND timestamp < %s ORDER BY timestamp
    """, (symbol, interval, query_start, bounds["scoring_end"])).fetchall()
    if not rows:
        raise ValueError("No Binance bars in required training/scoring range")
    data = pd.DataFrame(rows).set_index("timestamp")
    data.index = pd.DatetimeIndex(data.index)
    expected = pd.date_range(query_start, bounds["scoring_end"], freq=step, inclusive="left")
    if not data.index.equals(expected):
        missing = expected.difference(data.index)
        raise ValueError(f"Incomplete training/scoring/warmup data: missing={len(missing)} first={list(missing[:5])}")
    features = build_features(data, config, pd.Timedelta(step))
    # Only the known INITIAL feature warmup is discarded, outside the training period.
    usable = features.iloc[warmup:]
    training = usable.loc[(usable.index >= bounds["training_start"]) & (usable.index < bounds["training_end"])]
    scoring = usable.loc[(usable.index >= bounds["scoring_start"]) & (usable.index < bounds["scoring_end"])]
    if len(training) != config.training_days * (timedelta(days=1) // step) or len(scoring) != timedelta(days=1) // step:
        raise ValueError("Incomplete feature rows; refusing partial day")
    if training.index.max() >= scoring.index.min():
        raise ValueError("Training must strictly precede scoring observations")
    return training, scoring, digest(rows), warmup


def fit_score(training, scoring, parameters):
    model = IsolationForest(**parameters)
    model.fit(training.loc[:, FEATURES])
    scores = -model.score_samples(scoring.loc[:, FEATURES])
    flags = model.predict(scoring.loc[:, FEATURES]) == -1
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite Isolation Forest scores")
    return model, scores, flags


def persist_artifact(path, payload, scoring, expected_scores):
    """Atomically publish before DB commit. A crash can leave a reusable orphan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = joblib.load(path)  # Only load artifacts created locally by this project.
        if existing["metadata"] != payload["metadata"] or not np.array_equal(
            -existing["estimator"].score_samples(scoring.loc[:, FEATURES]), expected_scores
        ):
            raise ValueError("Existing orphan artifact differs; refusing overwrite")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".if-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                joblib.dump(payload, stream, compress=3)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    loaded = joblib.load(path)
    if loaded["feature_order"] != FEATURES or not np.array_equal(
        -loaded["estimator"].score_samples(scoring.loc[:, FEATURES]), expected_scores
    ):
        raise ValueError("Artifact reload verification failed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_day(conn, score_date, symbol="BTCUSDT", interval="5m", config=None, now=None):
    config = config or BatchConfig.from_env()
    bounds = periods(score_date, config)
    created_at = datetime.now(timezone.utc)
    if bounds["scoring_end"] > (now or created_at):
        raise ValueError("Scoring UTC day is not yet complete")
    parameters = dict(n_estimators=config.n_estimators, contamination=config.contamination,
                      random_state=config.random_state, max_samples="auto", max_features=1.0,
                      bootstrap=False, n_jobs=1)
    feature_set = dict(order=FEATURES, volatility_window=config.volatility_window,
                       volume_window=config.volume_window, volume_scale_floor=config.volume_scale_floor,
                       volatility_definition="sqrt(sum trailing r^2), including current closed bar",
                       volume_definition="prior-window mean/population std, excludes current")
    identity = digest(dict(symbol=symbol, interval=interval, model_version=MODEL_VERSION,
                           **bounds, feature_set=feature_set, model_parameters=parameters))
    run_id = uuid.uuid5(uuid.NAMESPACE_URL, "btc-risk/isolation-forest/" + identity)
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (identity,))
    previous = conn.execute("SELECT * FROM model_runs WHERE run_identity=%s", (identity,)).fetchone()
    training, scoring, data_hash, warmup = load_features(conn, symbol, interval, bounds, config)
    code_version = hashlib.sha256(b"".join(Path(p).read_bytes() for p in
                                 (__file__, feature_module.__file__, config_module.__file__))).hexdigest()
    libraries = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "scipy")}
    libraries["python"] = platform.python_version()
    metadata = dict(run_id=str(run_id), identity=identity, symbol=symbol, interval=interval,
                    model_version=MODEL_VERSION, **bounds, feature_set=feature_set, model_parameters=parameters,
                    data_hash=data_hash, code_version=code_version, library_versions=libraries)
    if previous and any(previous[key] != metadata[key] for key in (
        "data_hash", "code_version", "library_versions", "symbol", "interval", "model_version",
        "training_start", "training_end", "scoring_start", "scoring_end", "feature_set", "model_parameters"
    )):
        raise ValueError("Existing run identity has different data/code/libraries; refusing overwrite")
    model, scores, flags = fit_score(training, scoring, parameters)
    records = [dict(timestamp=stamp.to_pydatetime(), symbol=symbol, interval=interval, run_id=run_id,
                    anomaly_score=float(score), anomaly_flag=bool(flag),
                    feature_values={name: float(values[name]) for name in FEATURES})
               for (stamp, values), score, flag in zip(scoring.iterrows(), scores, flags)]
    artifact_name = f"isolation_forest_{run_id}.joblib"
    path = config.models_dir.resolve() / artifact_name
    if previous:
        if previous["artifact_path"] != artifact_name or previous["training_rows"] != len(training) or previous["scoring_rows"] != len(scoring):
            raise ValueError("Existing run metadata differs; refusing overwrite")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != previous["artifact_sha256"]:
            raise ValueError("Existing run artifact missing or checksum mismatch")
        stored = conn.execute("""
            SELECT timestamp, symbol, interval, run_id, anomaly_score, anomaly_flag, feature_values
            FROM isolation_forest_scores WHERE run_id=%s ORDER BY timestamp
        """, (run_id,)).fetchall()
        if stored != records:
            raise ValueError("Same run identity recomputed different scores/features; refusing overwrite")
    payload = dict(estimator=model, feature_order=FEATURES, metadata=metadata)
    artifact_hash = persist_artifact(path, payload, scoring, scores)
    if not previous:
        conn.execute("""
            INSERT INTO model_runs
                (run_id, run_identity, model_name, model_version, symbol, interval,
                 training_start, training_end, scoring_start, scoring_end, feature_set, model_parameters,
                 training_rows, scoring_rows, data_hash, code_version, library_versions,
                 artifact_path, artifact_sha256, created_at, completed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,clock_timestamp())
        """, (run_id, identity, "IsolationForest", MODEL_VERSION, symbol, interval,
                bounds["training_start"], bounds["training_end"], bounds["scoring_start"], bounds["scoring_end"],
                Jsonb(feature_set), Jsonb(parameters), len(training), len(scoring), data_hash, code_version,
                Jsonb(libraries), artifact_name, artifact_hash, created_at))
        with conn.cursor() as cursor:
            cursor.executemany("""
                INSERT INTO isolation_forest_scores
                    (timestamp, symbol, interval, run_id, anomaly_score, anomaly_flag, feature_values)
                VALUES (%(timestamp)s, %(symbol)s, %(interval)s, %(run_id)s, %(anomaly_score)s,
                        %(anomaly_flag)s, %(feature_values)s)
            """, [{**record, "feature_values": Jsonb(record["feature_values"])} for record in records])
    summary = conn.execute("""
        SELECT count(*) AS scoring_rows, count(*) FILTER (WHERE anomaly_flag) AS anomaly_count,
               100.0 * count(*) FILTER (WHERE anomaly_flag) / count(*) AS anomaly_rate_percent,
               min(anomaly_score) AS minimum_anomaly_score, max(anomaly_score) AS maximum_anomaly_score
        FROM isolation_forest_scores WHERE run_id=%s
    """, (run_id,)).fetchone()
    top = conn.execute("""
        SELECT s.timestamp, b.close, s.feature_values, s.anomaly_score, s.anomaly_flag
        FROM isolation_forest_scores s JOIN market_bars b USING (symbol, interval, timestamp)
        WHERE s.run_id=%s ORDER BY s.anomaly_score DESC, s.timestamp LIMIT 10
    """, (run_id,)).fetchall()
    return dict(run_id=str(run_id), **bounds, training_rows=len(training), warmup_rows_discarded=warmup,
                feature_names=FEATURES, parameters=parameters, **summary, top_10=top,
                artifact_path=str(path), skipped=bool(previous))


def main():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--score-date", type=date.fromisoformat)
    selection.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat, help="Inclusive last scoring date for walk-forward")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="5m", choices=("5m", "1h"))
    args = parser.parse_args()
    if (args.start_date and not args.end_date) or (args.score_date and args.end_date):
        parser.error("Use --score-date OR --start-date with --end-date")
    start, end = args.score_date or args.start_date, args.score_date or args.end_date
    if end < start:
        parser.error("end-date must not precede start-date")
    try:
        config = BatchConfig.from_env()
        for i in range((end - start).days + 1):
            score_date = start + timedelta(days=i)
            logger.info("Daily batch started score_date=%s", score_date)
            with connect() as conn:
                result = run_day(conn, score_date, args.symbol, args.interval, config)
            logger.info("Daily batch committed\n%s", json.dumps(result, default=str, indent=2))
        return 0
    except Exception:
        logger.exception("Daily batch failed; SQL transaction rolled back; inspect possible orphan artifact before retry")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
