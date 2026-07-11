"""Library-level helpers shared across commands."""

from __future__ import annotations

import sqlite3

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
