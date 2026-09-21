"""Read-only HTTP view of operational BTC risk-monitoring state."""
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable

from fastapi import FastAPI, HTTPException, Query

from btc_risk.config import ApiConfig, IngestionConfig
from btc_risk.database.connection import connect


def _utc_now():
    return datetime.now(timezone.utc)


def _freshness(timestamp, now, threshold):
    if timestamp is None:
        return {"status": "unavailable", "age_seconds": None}
    age = max(0.0, (now - timestamp).total_seconds())
    return {"status": "fresh" if age <= threshold else "stale", "age_seconds": age}


def create_app(connection_factory: Callable = connect, now_fn: Callable = _utc_now,
               config: ApiConfig | None = None, ingestion: IngestionConfig | None = None):
    config, ingestion = config or ApiConfig.from_env(), ingestion or IngestionConfig.from_env()
    app = FastAPI(title="BTC Risk Monitor API", version="0.1.0")

    @contextmanager
    def readonly():
        try:
            with connection_factory() as conn:
                conn.execute("SET TRANSACTION READ ONLY")
                yield conn
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

    def market(conn):
        return conn.execute("""SELECT timestamp, timestamp + interval '5 minutes' AS bar_closed_at,
             open,high,low,close,volume,source,ingested_at
             FROM market_bars WHERE symbol=%s AND interval=%s AND source='binance'
             ORDER BY timestamp DESC LIMIT 1""", (ingestion.symbol, ingestion.interval)).fetchone()

    def zscore(conn):
        return conn.execute("""SELECT s.timestamp, b.timestamp + interval '5 minutes' AS bar_closed_at,
             s.log_return,s.rolling_median,s.rolling_mad,s.robust_zscore,s.alert_flag,s.status,s.calculated_at
             FROM realtime_signals s JOIN market_bars b USING (symbol,interval,timestamp)
             WHERE s.symbol=%s AND s.interval=%s AND b.source='binance'
             ORDER BY s.timestamp DESC LIMIT 1""", (ingestion.symbol, ingestion.interval)).fetchone()

    def active(conn):
        return conn.execute("""SELECT p.publication_id,p.source_run_id,p.model_available_at,p.training_start,
             p.training_end,p.artifact_path,p.artifact_sha256,p.publication_mode,a.activated_at
             FROM active_if_publications a JOIN online_if_model_publications p ON p.publication_id=a.publication_id
             WHERE a.symbol=%s AND a.interval=%s""", (ingestion.symbol, ingestion.interval)).fetchone()

    def if_score(conn):
        return conn.execute("""SELECT o.timestamp,o.bar_closed_at,o.anomaly_score,o.anomaly_flag,
             o.score_available_at,o.calculated_at,p.publication_id,p.source_run_id AS model_run_id,p.model_available_at
             FROM active_if_publications a
             JOIN online_if_model_publications p ON p.publication_id=a.publication_id
             LEFT JOIN LATERAL (
               SELECT * FROM online_isolation_forest_scores x
               WHERE x.publication_id=p.publication_id ORDER BY x.timestamp DESC LIMIT 1
             ) o ON true
             WHERE a.symbol=%s AND a.interval=%s""", (ingestion.symbol, ingestion.interval)).fetchone()

    def render_market(row, now):
        freshness = _freshness(row["bar_closed_at"] if row else None, now, config.market_stale_seconds)
        return {**(dict(row) if row else {}), "market_age_seconds": freshness["age_seconds"],
                "status": freshness["status"]}

    def render_zscore(row, now):
        freshness = _freshness(row["bar_closed_at"] if row else None, now, config.zscore_stale_seconds)
        return {**(dict(row) if row else {}), "zscore_age_seconds": freshness["age_seconds"],
                "status": freshness["status"]}

    def render_if(row, publication, now):
        if publication is None:
            return {"publication_id": None, "model_run_id": None, "model_available_at": None,
                    "score_available_at": None, "if_score_age_seconds": None,
                    "if_model_age_seconds": None, "status": "unavailable", "model_status": "unavailable"}
        model_freshness = _freshness(publication["model_available_at"], now, config.if_model_stale_seconds)
        if row is None or row["timestamp"] is None:
            return {"publication_id": publication["publication_id"], "model_run_id": publication["source_run_id"],
                    "model_available_at": publication["model_available_at"], "score_available_at": None,
                    "if_score_age_seconds": None,
                    "if_model_age_seconds": model_freshness["age_seconds"], "status": "unavailable",
                    "model_status": model_freshness["status"]}
        score_freshness = _freshness(row["score_available_at"], now, config.if_score_stale_seconds)
        return {**dict(row), "if_score_age_seconds": score_freshness["age_seconds"],
                "if_model_age_seconds": model_freshness["age_seconds"], "status": score_freshness["status"],
                "model_status": model_freshness["status"]}

    @app.get("/health")
    def health():
        now = now_fn()
        with readonly() as conn:
            latest_market, publication = market(conn), active(conn)
        result = render_market(latest_market, now)
        model = _freshness(publication["model_available_at"] if publication else None, now, config.if_model_stale_seconds)
        return {"status": "ok", "database": "connected", "market": result,
                "active_if_model": {"publication": publication, "status": model["status"],
                                    "if_model_age_seconds": model["age_seconds"]}}

    @app.get("/market/latest")
    def market_latest():
        with readonly() as conn:
            row = market(conn)
        return render_market(row, now_fn())

    @app.get("/signals/zscore/latest")
    def zscore_latest():
        with readonly() as conn:
            row = zscore(conn)
        return render_zscore(row, now_fn())

    @app.get("/signals/if/latest")
    def if_latest():
        with readonly() as conn:
            publication, row = active(conn), if_score(conn)
        return render_if(row, publication, now_fn())

    @app.get("/risk-status")
    def risk_status():
        now = now_fn()
        with readonly() as conn:
            latest_market, latest_z, publication, latest_if = market(conn), zscore(conn), active(conn), if_score(conn)
        return {"market": render_market(latest_market, now), "zscore": render_zscore(latest_z, now),
                "isolation_forest": render_if(latest_if, publication, now)}

    @app.get("/signals/history")
    def history(start: datetime, end: datetime, limit: int = Query(100, ge=1, le=1000),
                cursor: datetime | None = None):
        if start.tzinfo is None or end.tzinfo is None or (cursor is not None and cursor.tzinfo is None):
            raise HTTPException(status_code=422, detail="start, end and cursor must include a timezone")
        if end <= start:
            raise HTTPException(status_code=422, detail="end must be after start")
        with readonly() as conn:
            rows = conn.execute("""SELECT s.timestamp,b.timestamp + interval '5 minutes' AS bar_closed_at,
                s.log_return,s.robust_zscore,s.alert_flag AS zscore_alert,s.status AS zscore_status,
                i.anomaly_score,i.anomaly_flag AS if_anomaly,i.publication_id,i.model_run_id,i.model_available_at,i.score_available_at
                FROM realtime_signals s JOIN market_bars b USING(symbol,interval,timestamp)
                LEFT JOIN LATERAL (
                  SELECT o.anomaly_score,o.anomaly_flag,o.publication_id,p.source_run_id AS model_run_id,p.model_available_at,o.score_available_at
                  FROM online_isolation_forest_scores o JOIN online_if_model_publications p ON p.publication_id=o.publication_id
                  WHERE o.symbol=s.symbol AND o.interval=s.interval AND o.timestamp=s.timestamp
                  ORDER BY o.calculated_at DESC LIMIT 1
                ) i ON true
                WHERE s.symbol=%s AND s.interval=%s AND b.source='binance'
                  AND s.timestamp >= %s AND s.timestamp < %s
                  AND (%s::timestamptz IS NULL OR s.timestamp > %s)
                ORDER BY s.timestamp LIMIT %s""",
                (ingestion.symbol, ingestion.interval, start, end, cursor, cursor, limit + 1)).fetchall()
        has_more = len(rows) > limit
        records = rows[:limit]
        return {"items": records, "limit": limit,
                "next_cursor": records[-1]["timestamp"] if has_more and records else None}

    return app


app = create_app()
