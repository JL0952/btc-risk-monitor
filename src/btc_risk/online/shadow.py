"""Frozen daily IF inference for collector shadow mode; never emits an alert."""
from datetime import datetime, time as day_time, timedelta, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb
from sklearn.ensemble import IsolationForest

from btc_risk.batch.features import FEATURES, build_features
from btc_risk.config import BatchConfig
from btc_risk.online.simulate import MODEL_AVAILABLE_OFFSET, digest, online_feature, step_for

LIVE_POLICY = "live_shadow_daily_publish_after_001"


def _params(config):
    return dict(n_estimators=config.n_estimators, contamination=config.contamination,
                random_state=config.random_state, max_samples="auto", max_features=1.0,
                bootstrap=False, n_jobs=1)


def _feature_set(config):
    return dict(order=FEATURES, volatility_window=config.volatility_window,
                volume_window=config.volume_window, volume_scale_floor=config.volume_scale_floor,
                volatility_definition="sqrt(sum trailing r^2), including current closed bar",
                volume_definition="prior-window mean/population std, excludes current")


def _training(conn, symbol, interval, day, config):
    start = datetime.combine(day, day_time(), timezone.utc)
    step = step_for(interval)
    warmup = max(config.volatility_window, config.volume_window)
    training_start = start - timedelta(days=config.training_days)
    query_start = training_start - warmup * step
    rows = conn.execute("""SELECT timestamp,open,high,low,close,volume FROM market_bars
        WHERE symbol=%s AND interval=%s AND source='binance' AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp""", (symbol, interval, query_start, start)).fetchall()
    expected = [query_start + i * step for i in range(warmup + config.training_days * int(timedelta(days=1) // step))]
    if [r["timestamp"] for r in rows] != expected:
        raise ValueError("Cannot publish shadow IF with incomplete prior training history")
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = frame.index.tz_convert("UTC")
    features = build_features(frame, config, pd.Timedelta(step))
    return features.iloc[warmup:], rows, training_start, start


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".online-if-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name); joblib.dump(payload, stream, compress=3); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists(): temp.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LiveIFShadow:
    def __init__(self, publication_id, model, config, symbol, interval, available_at, scoring_start, scoring_end):
        self.publication_id, self.model, self.config = publication_id, model, config
        self.symbol, self.interval, self.model_available_at, self.step = symbol, interval, available_at, step_for(interval)
        self.scoring_start, self.scoring_end = scoring_start, scoring_end

    @classmethod
    def load_active(cls, conn, symbol, interval, config=None):
        config = config or BatchConfig.from_env()
        row = conn.execute("""SELECT p.* FROM active_if_publications a
            JOIN online_if_model_publications p ON p.publication_id=a.publication_id
            WHERE a.symbol=%s AND a.interval=%s""", (symbol, interval)).fetchone()
        if row is None: return None
        path = config.models_dir.resolve() / row["artifact_path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError("Active IF artifact missing or checksum mismatch")
        payload = joblib.load(path)
        if payload.get("feature_order") != FEATURES: raise ValueError("Active IF feature order mismatch")
        return cls(row["publication_id"], payload["estimator"], config, symbol, interval,
                   row["model_available_at"], row["scoring_start"], row["scoring_end"])

    @classmethod
    def publish(cls, conn, symbol, interval, now=None, config=None):
        config = config or BatchConfig.from_env()
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        training, rows, training_start, scoring_start = _training(conn, symbol, interval, now.date(), config)
        parameters, feature_set, data_hash = _params(config), _feature_set(config), digest(rows)
        code_version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        identity = digest(dict(mode="live_shadow", symbol=symbol, interval=interval, training_start=training_start,
                               training_end=scoring_start, parameters=parameters, feature_set=feature_set,
                               data_hash=data_hash, code_version=code_version))
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (identity,))
        existing = conn.execute("SELECT * FROM online_if_model_publications WHERE publication_identity=%s", (identity,)).fetchone()
        if existing:
            path = config.models_dir.resolve() / existing["artifact_path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != existing["artifact_sha256"]:
                raise ValueError("Existing shadow artifact missing or checksum mismatch")
            payload = joblib.load(path)
            if payload.get("feature_order") != FEATURES: raise ValueError("Existing shadow feature order mismatch")
            return cls(existing["publication_id"], payload["estimator"], config, symbol, interval,
                       existing["model_available_at"], existing["scoring_start"], existing["scoring_end"]), True
        publication_id = uuid.uuid5(uuid.NAMESPACE_URL, "btc-risk/live-if-shadow/" + identity)
        model = IsolationForest(**parameters).fit(training.loc[:, FEATURES])
        available_at = max(now, scoring_start + MODEL_AVAILABLE_OFFSET)
        path = config.models_dir.resolve() / f"online_if_shadow_{publication_id}.joblib"
        artifact_hash = _save(path, dict(estimator=model, feature_order=FEATURES, metadata={"publication_id":str(publication_id),"identity":identity}))
        conn.execute("""INSERT INTO online_if_model_publications
          (publication_id,publication_identity,source_run_id,symbol,interval,training_start,training_end,scoring_start,
           scoring_end,model_available_at,availability_policy,publication_mode,feature_set,model_parameters,artifact_path,
           artifact_sha256,code_version,data_hash)
          VALUES (%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,'live_shadow',%s,%s,%s,%s,%s,%s)""",
          (publication_id,identity,symbol,interval,training_start,scoring_start,scoring_start,scoring_start+timedelta(days=1),
           available_at,LIVE_POLICY,Jsonb(feature_set),Jsonb(parameters),path.name,artifact_hash,code_version,data_hash))
        return cls(publication_id, model, config, symbol, interval, available_at,
                   scoring_start, scoring_start + timedelta(days=1)), False

    def score(self, conn, bar):
        closed_at = bar.timestamp + self.step
        if (closed_at <= self.model_available_at or not self.scoring_start <= bar.timestamp < self.scoring_end):
            return None
        lookback = max(self.config.volatility_window, self.config.volume_window)
        rows = list(reversed(conn.execute("""SELECT timestamp,open,high,low,close,volume FROM market_bars
          WHERE symbol=%s AND interval=%s AND source='binance' AND timestamp <= %s ORDER BY timestamp DESC LIMIT %s""",
          (self.symbol,self.interval,bar.timestamp,lookback+1)).fetchall()))
        expected = [bar.timestamp-self.step*(lookback-i) for i in range(lookback+1)]
        if [r["timestamp"] for r in rows] != expected: raise ValueError("Cannot score shadow IF across a market-data gap")
        values = online_feature(rows, self.config, self.interval)
        frame = pd.DataFrame([values], columns=FEATURES)
        record = dict(publication_id=self.publication_id,timestamp=bar.timestamp,symbol=self.symbol,interval=self.interval,
          anomaly_score=float(-self.model.score_samples(frame)[0]),anomaly_flag=bool(self.model.predict(frame)[0]==-1),
          feature_values=values,bar_closed_at=closed_at,score_available_at=closed_at)
        old = conn.execute("SELECT anomaly_score,anomaly_flag,feature_values FROM online_isolation_forest_scores WHERE publication_id=%s AND timestamp=%s",(self.publication_id,bar.timestamp)).fetchone()
        if old:
            if old["anomaly_flag"] != record["anomaly_flag"] or old["feature_values"] != values or not np.isclose(old["anomaly_score"],record["anomaly_score"],rtol=0,atol=1e-15): raise RuntimeError("Existing shadow score differs")
            return None
        conn.execute("""INSERT INTO online_isolation_forest_scores
          (publication_id,timestamp,symbol,interval,anomaly_score,anomaly_flag,feature_values,bar_closed_at,score_available_at)
          VALUES (%(publication_id)s,%(timestamp)s,%(symbol)s,%(interval)s,%(anomaly_score)s,%(anomaly_flag)s,%(feature_values)s,%(bar_closed_at)s,%(score_available_at)s)""",{**record,"feature_values":Jsonb(values)})
        return record
