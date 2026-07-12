"""SQLite connection + schema migrations.

``connect()`` returns a connection configured with WAL journaling, foreign-key
enforcement, a ``Row`` row factory, and the ``sqlite-vec`` extension loaded.
Migrations are plain SQL strings applied in order and tracked in
``schema_version``. Migration 001 creates every table except the ``vec0``
virtual table (added in migration 002, Step 6) so early steps don't depend on
vector search being wired up.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vidcp.config import get_settings

MIGRATION_001 = """
CREATE TABLE videos (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  title TEXT,
  duration_s REAL, width INT, height INT, fps REAL,
  vcodec TEXT, acodec TEXT, size_bytes INT,
  has_audio INT NOT NULL DEFAULT 1,
  created_at TEXT, ingested_at TEXT NOT NULL,
  meta JSON
);

CREATE TABLE stages (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|skipped
  started_at TEXT, finished_at TEXT, error TEXT,
  config_hash TEXT,
  PRIMARY KEY (video_id, stage)
);

CREATE TABLE scenes (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx INT NOT NULL,
  start_s REAL NOT NULL, end_s REAL NOT NULL,
  keyframe_path TEXT, phash TEXT
);
CREATE INDEX idx_scenes_video ON scenes(video_id, idx);

CREATE TABLE segments (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  start_s REAL NOT NULL, end_s REAL NOT NULL,
  text TEXT NOT NULL, confidence REAL,
  words JSON
);
CREATE INDEX idx_segments_video ON segments(video_id, start_s);

CREATE TABLE ocr_blocks (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  scene_id INT REFERENCES scenes(id) ON DELETE SET NULL,
  start_s REAL NOT NULL, end_s REAL NOT NULL,
  text TEXT NOT NULL, confidence REAL, bbox JSON
);

CREATE VIRTUAL TABLE fts USING fts5(
  text, video_id UNINDEXED, kind UNINDEXED, ref_id UNINDEXED, ts_s UNINDEXED
);
"""

# Migration 002 — the frames table (Step 3). Note: the plan numbers the vec0
# table as 002 and frames as 003, but vec0 is deferred to Step 6, so frames is
# applied as 002 here. vec0 will be the next migration.
MIGRATION_002 = """
CREATE TABLE frames (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  scene_id INT REFERENCES scenes(id) ON DELETE CASCADE,
  ts_s REAL NOT NULL, path TEXT NOT NULL, phash TEXT, kept INT NOT NULL DEFAULT 1
);
CREATE INDEX idx_frames_video ON frames(video_id, ts_s);
"""

# Migration 003 — the vec0 virtual table for hybrid search (Step 6). 384-dim
# vectors match the all-MiniLM-L6-v2 embedding model. video_id/kind/ref_id/ts_s
# are metadata columns (retrievable and filterable in KNN queries).
MIGRATION_003 = """
CREATE VIRTUAL TABLE vec USING vec0(
  embedding float[384],
  video_id TEXT,
  kind TEXT,
  ref_id INTEGER,
  ts_s FLOAT
);
"""

# Applied in order; the list index (1-based) is the schema version.
MIGRATIONS: list[str] = [MIGRATION_001, MIGRATION_002, MIGRATION_003]


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    # Take the write lock up front and re-check the version inside the
    # transaction, so two brand-new processes can't both apply migration 001.
    # (executescript would auto-commit and defeat this, so run statements one by
    # one — the migration SQL is plain DDL with no embedded semicolons.)
    conn.execute("BEGIN IMMEDIATE")
    try:
        current = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
        for version, sql in enumerate(MIGRATIONS, start=1):
            if version > current:
                for statement in filter(str.strip, sql.split(";")):
                    conn.execute(statement)
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the library database, applying any pending migrations.

    ``db_path`` defaults to ``get_settings().db_path``. The parent directory is
    created if needed.
    """
    path = db_path or get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Let concurrent writers (parallel stages) wait for the write lock instead
    # of failing with "database is locked".
    conn.execute("PRAGMA busy_timeout=30000")
    _load_sqlite_vec(conn)
    _apply_migrations(conn)
    return conn
