"""Real Docker integration test. Never deletes a volume or application records."""

import json
from pathlib import Path
import subprocess

import pytest

from btc_risk.database.connection import connect
from btc_risk.database.repository import get_bar, insert_bar

ROOT = Path(__file__).resolve().parents[1]


def compose(*args):
    result = subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, check=True,
        text=True, capture_output=True, timeout=120,
    )
    return result.stdout.strip()


@pytest.mark.persistence
def test_committed_bar_survives_stop_start_and_recreation(bar):
    compose("up", "-d", "--wait", "--wait-timeout", "60", "database")
    original_id = compose("ps", "-q", "database")
    with connect() as conn:
        assert insert_bar(conn, bar)
    with connect() as conn:
        original = get_bar(conn, bar.symbol, bar.interval, bar.timestamp)
        assert original is not None
    try:
        compose("stop", "database")
        assert not compose("ps", "--status", "running", "-q", "database")
        compose("start", "--wait", "--wait-timeout", "60", "database")
        assert compose("ps", "-q", "database") == original_id
        with connect() as conn:
            assert get_bar(conn, bar.symbol, bar.interval, bar.timestamp) == original
        # Stronger than a restart: replace the container, retaining the named volume.
        compose("up", "-d", "--force-recreate", "--wait", "--wait-timeout", "60", "database")
        replacement_id = compose("ps", "-q", "database")
        assert replacement_id != original_id
        with connect() as conn:
            assert get_bar(conn, bar.symbol, bar.interval, bar.timestamp) == original
        details = json.loads(subprocess.run(
            ["docker", "inspect", replacement_id], check=True,
            text=True, capture_output=True, timeout=15,
        ).stdout)[0]
        assert any(
            mount["Type"] == "volume" and mount["Name"] == "btc-risk_db_data"
            and mount["Destination"] == "/var/lib/postgresql/data"
            for mount in details["Mounts"]
        )
    finally:
        # Ensure the service is running even if an assertion failed.
        compose("up", "-d", "--wait", "--wait-timeout", "60", "database")
        with connect() as conn:
            conn.execute("DELETE FROM market_bars WHERE symbol = %s AND interval = %s AND timestamp = %s",
                         (bar.symbol, bar.interval, bar.timestamp))
