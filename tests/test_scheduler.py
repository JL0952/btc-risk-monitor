from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from btc_risk.operations import scheduler


class Cursor:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


def test_scheduler_activates_only_after_publication_metadata(monkeypatch):
    calls = []

    class Connection:
        def execute(self, query, params=None):
            calls.append((query, params))
            return Cursor(None if "SELECT status" in query else None)

    @contextmanager
    def factory():
        yield Connection()

    publication = uuid4()
    shadow = SimpleNamespace(publication_id=publication, model_available_at=None)
    monkeypatch.setattr(scheduler, "connect", factory)
    monkeypatch.setattr(scheduler.LiveIFShadow, "publish", lambda *_args, **_kwargs: (shadow, False))
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    assert scheduler.publish_today(now=now, symbol="SCHED", interval="5m") == "published"
    update_index = next(i for i, (query, _) in enumerate(calls) if "SET model_available_at" in query)
    active_index = next(i for i, (query, _) in enumerate(calls) if "INSERT INTO active_if_publications" in query)
    assert update_index < active_index
    assert shadow.model_available_at >= now


def test_scheduler_published_day_is_idempotent(monkeypatch):
    calls = []

    class Connection:
        def execute(self, query, params=None):
            calls.append(query)
            return Cursor({"status": "published"} if "SELECT status" in query else None)

    @contextmanager
    def factory():
        yield Connection()

    monkeypatch.setattr(scheduler, "connect", factory)
    assert scheduler.publish_today(now=datetime(2026, 7, 2, tzinfo=timezone.utc), symbol="SCHED", interval="5m") == "already_published"
    assert len(calls) == 1
