from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from btc_risk.database.connection import connect
from btc_risk.database.repository import MarketBar


def pytest_addoption(parser):
    parser.addoption("--run-persistence", action="store_true", help="Allow database container stop/start/recreation")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-persistence"):
        return
    skip = pytest.mark.skip(reason="Use --run-persistence to permit container restart")
    for item in items:
        if "persistence" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def bar():
    return MarketBar(
        datetime(2026, 1, 1, tzinfo=timezone.utc), "TESTBTC_" + uuid4().hex,
        "5m", Decimal("90000"), Decimal("90100"), Decimal("89900"),
        Decimal("90050"), Decimal("12.5"), "synthetic-test",
    )


@pytest.fixture
def conn():
    connection = connect()
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
