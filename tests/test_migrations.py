from btc_risk.database.migrate import migrate
from btc_risk.database.connection import connect
from pathlib import Path
import shutil
import pytest


def test_migration_rerun_and_checksum(tmp_path):
    assert migrate() == []  # README requires migration before running tests.
    with connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM schema_migrations").fetchone()["n"] == len(list(Path("sql").glob("[0-9][0-9][0-9]_*.sql")))
    for path in Path("sql").glob("[0-9][0-9][0-9]_*.sql"):
        shutil.copy(path, tmp_path / path.name)
    (tmp_path / "001_initial.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        migrate(tmp_path)
