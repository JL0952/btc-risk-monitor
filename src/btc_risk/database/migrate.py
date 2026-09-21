"""Apply immutable, numbered SQL files atomically, recording their checksums."""

import argparse
import hashlib
import logging
from pathlib import Path
import time

from btc_risk.database.connection import connect

logger = logging.getLogger(__name__)


def migrate(directory: Path = Path("sql")) -> list[str]:
    files = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
    if not files:
        raise ValueError(f"No migrations found in {directory.resolve()}")
    versions = [path.name.split("_", 1)[0] for path in files]
    if len(versions) != len(set(versions)):
        raise ValueError("Duplicate migration version")
    applied = []
    with connect() as conn:
        # Serialize migration invocations. One transaction covers this invocation.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (81743001,))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        previous = {
            row["version"]: row["checksum"]
            for row in conn.execute("SELECT version, checksum FROM schema_migrations")
        }
        if set(previous) - {path.name for path in files}:
            raise ValueError("Database contains migrations missing from this checkout")
        for path in files:
            content = path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            if path.name in previous:
                if previous[path.name] != checksum:
                    raise ValueError(f"Applied migration was modified: {path.name}")
                continue
            # Trusted repository SQL, not interpolated application/user data.
            conn.execute(content.decode("utf-8"), prepare=False)
            conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                (path.name, checksum),
            )
            applied.append(path.name)
    logger.info("Migration committed: %s", applied or "already up to date")
    return applied


if __name__ == "__main__":
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql-dir", type=Path, default=Path("sql"))
    migrate(parser.parse_args().sql_dir)
