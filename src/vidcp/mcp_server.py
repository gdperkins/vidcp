"""MCP server exposing the vidcp library to agents.

Each tool is a thin wrapper over the same functions the CLI uses and opens its
own SQLite connection per call. This module is imported lazily by the
``vidcp mcp`` command so the ``mcp`` SDK never slows normal CLI startup.

Nothing here may write to stdout — stdout is the MCP stdio transport.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import NoReturn

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.library import artifact_counts, resolve_id
from vidcp.models import StageState, Video

_INSTRUCTIONS = (
    "Query a local vidcp video library: hybrid search over transcripts and "
    "on-screen text (OCR), transcript retrieval, scene lists, keyframe images, "
    "and ingestion of new videos. Video ids are SHA-256 hashes; any unique "
    "prefix works. ingest() returns immediately — poll get_video() until every "
    "stage is 'done' or 'skipped'."
)


@contextmanager
def _library():
    """Per-call DB connection, mirroring how CLI commands use the database."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def _fail(message: str, hint: str | None = None) -> NoReturn:
    """Raise a tool error with the same message/hint wording the CLI shows."""
    raise ToolError(f"{message} ({hint})" if hint else message)


def _resolve(conn, video_id: str) -> str:
    try:
        return resolve_id(conn, video_id)
    except VidcpError as exc:
        _fail(exc.message, exc.hint)


def _video_payload(row) -> dict:
    video = Video.from_row(row)
    data = video.model_dump(mode="json")
    data.pop("meta", None)  # verbose ffprobe blob; wasteful in agent context
    data["short_id"] = video.short_id
    return data


def list_videos() -> dict:
    """List every video in the library, newest first."""
    with _library() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC").fetchall()
    return {"videos": [_video_payload(row) for row in rows]}


def get_video(video_id: str) -> dict:
    """Get one video's metadata, artifact counts, and per-stage pipeline status.

    Poll this after ingest(): processing is finished when every stage is
    'done' or 'skipped'; a 'failed' stage carries its error message.
    """
    with _library() as conn:
        vid = _resolve(conn, video_id)
        row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        counts = artifact_counts(conn, vid)
        stage_rows = conn.execute(
            "SELECT * FROM stages WHERE video_id=? ORDER BY stage", (vid,)
        ).fetchall()
    payload = _video_payload(row)
    payload["counts"] = counts
    payload["stages"] = [StageState.from_row(r).model_dump(mode="json") for r in stage_rows]
    return payload


_TOOLS = (list_videos, get_video)


def create_server() -> FastMCP:
    """Build the vidcp MCP server with all tools registered."""
    server = FastMCP("vidcp", instructions=_INSTRUCTIONS)
    for fn in _TOOLS:
        server.tool()(fn)
    return server
