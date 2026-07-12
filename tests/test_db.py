from vidcp.config import get_settings
from vidcp.db import MIGRATIONS, connect

EXPECTED_TABLES = {
    "videos",
    "stages",
    "scenes",
    "segments",
    "ocr_blocks",
    "frames",
    "fts",
    "vec",
    "schema_version",
}


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {r[0] for r in rows}


def test_connect_creates_db_file():
    conn = connect()
    try:
        assert get_settings().db_path.exists()
    finally:
        conn.close()


def test_migration_001_creates_all_tables():
    conn = connect()
    try:
        assert EXPECTED_TABLES <= _table_names(conn)
    finally:
        conn.close()


def test_schema_version_matches_migration_count():
    conn = connect()
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == len(MIGRATIONS)
    finally:
        conn.close()


def test_migrations_are_idempotent():
    connect().close()
    conn = connect()
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == len(MIGRATIONS)
        assert EXPECTED_TABLES <= _table_names(conn)
    finally:
        conn.close()


def test_foreign_keys_enabled():
    conn = connect()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_row_factory_is_row():
    conn = connect()
    try:
        row = conn.execute("SELECT 1 AS one").fetchone()
        assert row["one"] == 1
    finally:
        conn.close()


def test_fts_table_is_queryable():
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO fts(text, video_id, kind, ref_id, ts_s) VALUES (?, ?, ?, ?, ?)",
            ("hello world", "vid", "transcript", 1, 0.0),
        )
        conn.commit()
        rows = conn.execute("SELECT text FROM fts WHERE fts MATCH ?", ("hello",)).fetchall()
        assert len(rows) == 1
    finally:
        conn.close()


def test_sqlite_vec_is_loaded():
    conn = connect()
    try:
        version = conn.execute("SELECT vec_version()").fetchone()[0]
        assert isinstance(version, str)
    finally:
        conn.close()
