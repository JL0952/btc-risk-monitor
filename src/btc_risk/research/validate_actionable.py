"""Offline risk validation for historical actionable-IF simulation scores only."""

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from psycopg.types.json import Jsonb

from btc_risk.database.connection import connect
from .analysis import ANALYSIS_CONFIG, analyze, plots
from .labels import LABEL_DEFINITIONS, classify, future_labels


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def validate_actionable(conn, start, end, symbol="BTCUSDT"):
    """Validate only scores that passed the strict online availability gate."""
    if start.tzinfo is None or end.tzinfo is None or start >= end or start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0):
        raise ValueError("Require increasing UTC boundaries")
    rows = conn.execute("""
        SELECT o.timestamp,o.symbol,o.interval,o.publication_id,o.anomaly_flag AS isolation_alert,
               o.anomaly_score,o.score_available_at,p.source_run_id,p.model_available_at,
               z.alert_flag AS zscore_alert,z.status AS z_status,z.robust_zscore,b.close
        FROM online_isolation_forest_scores o
        JOIN online_if_model_publications p ON p.publication_id=o.publication_id
        JOIN market_bars b ON (b.symbol,b.interval,b.timestamp)=(o.symbol,o.interval,o.timestamp)
        LEFT JOIN realtime_signals z ON (z.symbol,z.interval,z.timestamp)=(o.symbol,o.interval,o.timestamp)
        WHERE o.symbol=%s AND o.interval='5m' AND o.timestamp >= %s AND o.timestamp < %s
        ORDER BY o.timestamp
    """, (symbol, start, end)).fetchall()
    if len({r["timestamp"] for r in rows}) != len(rows):
        raise ValueError("Multiple actionable publications score one timestamp")
    usable = [r for r in rows if r["z_status"] in ("scored", "scale_floored") and r["zscore_alert"] is not None]
    if not usable:
        raise ValueError("No common scored Z/actionable IF observations")
    market = conn.execute("""SELECT timestamp,close,low FROM market_bars
        WHERE symbol=%s AND interval='5m' AND timestamp >= %s AND timestamp < %s
          AND timestamp + INTERVAL '5 minutes' <= %s ORDER BY timestamp""",
        (symbol, start, end + timedelta(hours=1), datetime.now(timezone.utc))).fetchall()
    observations = pd.DataFrame(usable)
    observations["timestamp"] = pd.to_datetime(observations.timestamp, utc=True)
    observations["signal_group"] = [classify(r["zscore_alert"], r["isolation_alert"]) for r in usable]
    bars = pd.DataFrame(market).set_index("timestamp")
    bars.index = pd.to_datetime(bars.index, utc=True)
    labels = observations[["timestamp", "signal_group", "zscore_alert", "isolation_alert", "publication_id"]].merge(
        future_labels(bars), on="timestamp", how="left", validate="one_to_many")
    if labels.label_status.isna().any():
        raise ValueError("A actionable score has no closed market bar")
    summary = analyze(observations, labels)
    summary["input_actionable_scores"] = len(rows)
    summary["excluded_unscored_z"] = len(rows) - len(usable)
    summary["availability_gate_excluded"] = int((end - start).days * 2)
    summary["market_last_timestamp"] = str(bars.index.max())
    publication_ids = sorted({str(r["publication_id"]) for r in usable})
    code_version = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    libraries = {name: version(name) for name in ("numpy", "pandas", "psycopg", "matplotlib")}
    data_hash = digest({"scores": rows, "market": market})
    identity = digest(dict(start=start, end=end, symbol=symbol, publication_ids=publication_ids,
                           labels=LABEL_DEFINITIONS, config=ANALYSIS_CONFIG, code=code_version,
                           libraries=libraries, data=data_hash))
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (identity,))
    previous = conn.execute("SELECT validation_run_id,summary FROM research.actionable_validation_runs WHERE validation_identity=%s", (identity,)).fetchone()
    if previous:
        if previous["summary"] != summary:
            raise RuntimeError("Identical actionable inputs produced inconsistent research results")
        run_id = previous["validation_run_id"]
        count = conn.execute("SELECT count(*) AS n FROM research.actionable_risk_validation WHERE validation_run_id=%s", (run_id,)).fetchone()["n"]
        if count != len(labels):
            raise RuntimeError("Existing actionable validation is incomplete")
    else:
        run_id = uuid4()
        conn.execute("""INSERT INTO research.actionable_validation_runs
            (validation_run_id,validation_identity,symbol,interval,evaluation_start,evaluation_end,publication_ids,
             label_definitions,analysis_config,code_version,data_hash,library_versions,summary)
             VALUES (%s,%s,%s,'5m',%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id,identity,symbol,start,end,publication_ids,Jsonb(LABEL_DEFINITIONS),Jsonb(ANALYSIS_CONFIG),
             code_version,data_hash,Jsonb(libraries),Jsonb(summary)))
        records=[]
        for row in labels.to_dict("records"):
            available = row["label_status"] == "available"
            values = [float(row[k]) if available else None for k in ("future_realized_volatility", "future_maximum_adverse_move", "future_absolute_return")]
            high_risk = values[0] >= summary["high_risk_threshold_1h_rv"] if available and row["horizon_minutes"] == 60 else None
            records.append((run_id,row["timestamp"],symbol,"5m",row["publication_id"],row["zscore_alert"],row["isolation_alert"],row["signal_group"],row["horizon_minutes"],row["label_status"],*values,high_risk))
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO research.actionable_risk_validation
              (validation_run_id,timestamp,symbol,interval,publication_id,zscore_alert,isolation_alert,signal_group,
               horizon_minutes,label_status,future_realized_volatility,future_maximum_adverse_move,future_absolute_return,high_risk_flag)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", records)
    return str(run_id), bool(previous), summary, observations, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="Inclusive UTC date")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    start = datetime.combine(args.start, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(args.end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    with connect() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        run_id, reused, summary, observations, labels = validate_actionable(conn, start, end, args.symbol)
    output = args.output_dir / ("actionable_risk_validation_" + run_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(dict(validation_run_id=run_id, **summary), indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summary["distributions"]).to_csv(output / "metrics.csv", index=False)
    plots(observations, labels, summary, output)
    print(json.dumps(dict(validation_run_id=run_id, reused=reused, **summary), indent=2, default=str))


if __name__ == "__main__":
    main()
