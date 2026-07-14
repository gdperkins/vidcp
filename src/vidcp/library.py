"""Library-level helpers shared across commands."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from vidcp.errors import VidcpError


def resolve_id(conn: sqlite3.Connection, prefix: str) -> str:
    """Resolve a (possibly abbreviated) video id to a full id.

    Matches any video whose id starts with ``prefix``. Raises ``VidcpError`` if
    nothing matches or the prefix is ambiguous.
    """
    prefix = prefix.strip()
    rows = conn.execute(
        "SELECT id FROM videos WHERE substr(id, 1, ?) = ? ORDER BY id",
        (len(prefix), prefix),
    ).fetchall()
    if not rows:
        raise VidcpError(
            f"no video matches id '{prefix}'",
            hint="run `vidcp list` to see available ids",
        )
    if len(rows) > 1:
        raise VidcpError(
            f"id prefix '{prefix}' is ambiguous ({len(rows)} matches)",
            hint="use more characters to disambiguate",
        )
    return rows[0]["id"]


def artifact_counts(conn: sqlite3.Connection, video_id: str) -> dict[str, int]:
    """Row counts of each per-video artifact table (scenes/frames/segments/ocr_blocks)."""
    counts = {}
    for table in ("scenes", "frames", "segments", "ocr_blocks"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE video_id=?", (video_id,)
        ).fetchone()[0]
    return counts


def pipeline_complete(conn: sqlite3.Connection, video_id: str, stage_names: Iterable[str]) -> bool:
    """True when every named stage finished (status 'done' or 'skipped')."""
    rows = conn.execute("SELECT stage, status FROM stages WHERE video_id=?", (video_id,)).fetchall()
    status = {row["stage"]: row["status"] for row in rows}
    return all(status.get(name) in ("done", "skipped") for name in stage_names)
