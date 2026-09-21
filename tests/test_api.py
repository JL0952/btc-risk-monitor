from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from fastapi.testclient import TestClient

from btc_risk.config import ApiConfig, IngestionConfig
from btc_risk.database.repository import MarketBar, insert_bar, insert_signal
from btc_risk.operations.api import create_app
from btc_risk.realtime.robust_zscore import Signal


def api(symbol, now):
    return create_app(config=ApiConfig(market_stale_seconds=60, zscore_stale_seconds=60,
                                       if_score_stale_seconds=60, if_model_stale_seconds=60),
                      ingestion=IngestionConfig(symbol=symbol), now_fn=lambda: now)


class EmptyCursor:
    def fetchone(self):
        return None


class EmptyConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_args, **_kwargs):
        return EmptyCursor()


def test_health_and_if_are_unavailable_without_active_model():
    symbol = "APINONE_" + uuid4().hex
    client = TestClient(create_app(connection_factory=EmptyConnection,
                      config=ApiConfig(), ingestion=IngestionConfig(symbol=symbol),
                      now_fn=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["market"]["status"] == "unavailable"
    response = client.get("/signals/if/latest")
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


def test_health_reports_database_failure():
    def unavailable():
        raise OSError("database down")
    client = TestClient(create_app(connection_factory=unavailable,
                                   ingestion=IngestionConfig(symbol="APIUNAVAILABLE")))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "database unavailable"


def test_api_exposes_only_read_routes():
    app = create_app(connection_factory=EmptyConnection, ingestion=IngestionConfig(symbol="APIREADONLY"))
    routes = [route for route in app.routes if getattr(route, "path", "").startswith("/") and route.path != "/openapi.json"]
    assert all(route.methods <= {"GET", "HEAD"} for route in routes)


def test_latest_history_staleness_and_pagination(conn):
    symbol = "API_" + uuid4().hex
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        for index in range(3):
            timestamp = start + timedelta(minutes=5 * index)
            bar = MarketBar(timestamp, symbol, "5m", Decimal("100"), Decimal("200"), Decimal("99"),
                            Decimal(str(100 + index)), Decimal("1"), "binance")
            insert_bar(conn, bar)
            insert_signal(conn, Signal(timestamp, symbol, "5m", 0.01, 0.0, 0.01, 1.0, False, "scored"))
        conn.commit()  # The API deliberately uses a separate read-only connection.
        client = TestClient(api(symbol, start + timedelta(hours=1)))
        market = client.get("/market/latest")
        zscore = client.get("/signals/zscore/latest")
        assert market.status_code == zscore.status_code == 200
        assert market.json()["status"] == zscore.json()["status"] == "stale"
        assert {"timestamp", "bar_closed_at", "close", "market_age_seconds", "status"} <= market.json().keys()
        assert {"timestamp", "bar_closed_at", "robust_zscore", "alert_flag", "zscore_age_seconds", "status"} <= zscore.json().keys()
        page = client.get("/signals/history", params={"start": start.isoformat(),
                          "end": (start + timedelta(minutes=15)).isoformat(), "limit": 2})
        assert page.status_code == 200
        body = page.json()
        assert len(body["items"]) == 2 and body["next_cursor"] is not None
        next_page = client.get("/signals/history", params={"start": start.isoformat(),
                          "end": (start + timedelta(minutes=15)).isoformat(), "limit": 2,
                          "cursor": body["next_cursor"]})
        assert [item["timestamp"] for item in next_page.json()["items"]] == [(start + timedelta(minutes=10)).isoformat()]
        assert client.get("/signals/history", params={"start": start.isoformat(), "end": start.isoformat()}).status_code == 422
    finally:
        conn.rollback()
        conn.execute("DELETE FROM realtime_signals WHERE symbol=%s", (symbol,))
        conn.execute("DELETE FROM market_bars WHERE symbol=%s", (symbol,))
        conn.commit()
