"""Each context-managed connection commits on success and rolls back on error."""

import psycopg
from psycopg.rows import dict_row

from btc_risk.config import DatabaseConfig


def connect(config: DatabaseConfig | None = None) -> psycopg.Connection:
    settings = config or DatabaseConfig.from_env()
    return psycopg.connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        connect_timeout=5,
        options="-c timezone=UTC",
        application_name="btc-risk-foundation",
        row_factory=dict_row,
    )


def check_connection() -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1


if __name__ == "__main__":
    import logging
    import time

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    if not check_connection():
        raise RuntimeError("SELECT 1 failed")
    logging.getLogger(__name__).info("PostgreSQL connection OK: SELECT 1 = 1")
