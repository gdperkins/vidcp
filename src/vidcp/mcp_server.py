"""MCP server exposing the vidcp library to agents.

Each tool is a thin wrapper over the same functions the CLI uses and opens its
own SQLite connection per call. This module is imported lazily by the
``vidcp mcp`` command so the ``mcp`` SDK never slows normal CLI startup.

Nothing here may write to stdout — stdout is the MCP stdio transport.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.exceptions import ToolError

from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.library import artifact_counts, pipeline_complete, resolve_id
from vidcp.models import StageState, Video
from vidcp.store import artifact_dir, sha256_file

_INSTRUCTIONS = (
    "Query a local vidcp video library: hybrid search over transcripts, "
    "on-screen text (OCR), and visual keyframe content (CLIP), transcript "
    "retrieval, scene lists, keyframe images, clip extraction, and ingestion "
    "of new videos. Video ids are SHA-256 hashes; any unique prefix works. "
    "ingest() returns immediately — poll get_video() until every stage is "
    "'done' or 'skipped'. The video only appears in get_video() once the "
    "background process has registered it, so a 'no video matches' error "
    "shortly after ingest() means try again in a bit, not that ingest failed. "
    "Stop polling if any stage reports 'failed'. Do not call ingest() again "
    "for the same file while a stage shows 'running'."
)

_MAX_EDGE = 1280
_JPEG_QUALITY = 85


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


def search(
    query: str, video_id: str | None = None, kind: str | None = None, limit: int = 10
) -> dict:
    """Hybrid search over transcripts, on-screen (OCR) text, and visual keyframe content.

    Returns timestamped, scored hits. kind filters to 'transcript', 'ocr', or
    'visual' (CLIP keyframe matches — their frame_path points at the matched
    image). video_id (any unique prefix) restricts to one video. Follow up with
    get_transcript() around a hit's ts_s, get_keyframe() to see the moment, or
    get_clip() to extract it as a video file.
    """
    if kind is not None and kind not in ("transcript", "ocr", "visual"):
        _fail(f"unknown kind '{kind}'", "choose one of: transcript, ocr, visual")
    from vidcp.search import search as run_search

    with _library() as conn:
        vid = _resolve(conn, video_id) if video_id else None
        hits = run_search(conn, query, video_id=vid, kind=kind, limit=limit)
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}


def _explain_missing_transcript(conn, vid: str) -> NoReturn:
    stage = conn.execute(
        "SELECT status, error FROM stages WHERE video_id=? AND stage='transcribe'", (vid,)
    ).fetchone()
    if stage is None or stage["status"] in ("pending", "running"):
        _fail("transcript not available yet", "transcription has not completed; poll get_video")
    if stage["status"] == "failed":
        _fail(
            f"transcription failed: {stage['error']}",
            f"retry with `vidcp reindex {vid[:8]} --stage transcribe`",
        )
    if stage["status"] == "skipped":
        _fail("no transcript: the video has no audio track")
    _fail("no transcript: no speech was detected")


def get_transcript(video_id: str, start_s: float | None = None, end_s: float | None = None) -> dict:
    """Get a video's transcript segments, optionally windowed to [start_s, end_s].

    A segment is included if it overlaps the window. Use a window around a
    search hit's ts_s to pull surrounding context.
    """
    with _library() as conn:
        vid = _resolve(conn, video_id)
        sql = "SELECT id, start_s, end_s, text FROM segments WHERE video_id=?"
        params: list = [vid]
        if end_s is not None:
            sql += " AND start_s < ?"
            params.append(end_s)
        if start_s is not None:
            sql += " AND end_s > ?"
            params.append(start_s)
        rows = conn.execute(sql + " ORDER BY start_s", params).fetchall()
        if not rows:
            total = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE video_id=?", (vid,)
            ).fetchone()[0]
            if total == 0:
                _explain_missing_transcript(conn, vid)
    return {"video_id": vid, "segments": [dict(row) for row in rows]}


def list_scenes(video_id: str) -> dict:
    """List detected scene boundaries (idx, start_s, end_s) for a video."""
    with _library() as conn:
        vid = _resolve(conn, video_id)
        rows = conn.execute(
            "SELECT idx, start_s, end_s FROM scenes WHERE video_id=? ORDER BY idx", (vid,)
        ).fetchall()
    return {"video_id": vid, "scenes": [dict(row) for row in rows]}


def get_keyframe(video_id: str, ts_s: float):
    """Get the stored keyframe nearest ts_s as a JPEG image (longest side <= 1280px).

    Returns a text block stating the frame's actual timestamp plus the image
    itself, so an agent can literally look at the moment behind a search hit.
    """
    # No return annotation: the mixed [str, Image] payload has no output schema.
    from PIL import Image as PILImage

    with _library() as conn:
        vid = _resolve(conn, video_id)
        row = conn.execute(
            "SELECT ts_s, path FROM frames WHERE video_id=? AND kept=1 "
            "ORDER BY ABS(ts_s - ?) LIMIT 1",
            (vid, ts_s),
        ).fetchone()
    if row is None:
        _fail("no keyframes for this video", "keyframes may still be processing; poll get_video")
    frame_path = Path(row["path"])
    if not frame_path.exists():
        _fail(f"keyframe file missing: {frame_path}")
    with PILImage.open(frame_path) as source:
        frame = source.convert("RGB")
    frame.thumbnail((_MAX_EDGE, _MAX_EDGE))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return [
        f"keyframe at {row['ts_s']:.2f}s (requested {ts_s:.2f}s)",
        Image(data=buffer.getvalue(), format="jpeg"),
    ]


# Win32 process-creation flags; subprocess only defines the named constants on
# Windows, and start_new_session is silently ignored there.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def _detach_kwargs() -> dict:
    """Popen kwargs that detach a child from this process's session/console."""
    if sys.platform == "win32":
        return {"creationflags": _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _spawn_ingest(path: Path, force: bool) -> None:
    """Launch a detached background ingest; the child owns all DB/store writes."""
    cmd = [sys.executable, "-m", "vidcp", "ingest"]
    if force:
        cmd.append("--force")
    cmd.append(str(path))
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_detach_kwargs(),
    )


def ingest(path: str) -> dict:
    """Ingest a video file into the library (asynchronous).

    Returns immediately with the video id; probing, transcription, OCR, and
    embedding continue in a background process. Poll get_video() until every
    stage is 'done' or 'skipped'. The video shows up in get_video() only once
    that background process has registered it, so a 'no video matches' error
    shortly after calling this means try again in a bit, not that ingest
    failed. Do not call this again for the same file while a stage shows
    'running'.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        _fail(f"file not found: {file}")
    from vidcp.pipeline.stages.probe import is_media_file

    if not is_media_file(file):
        _fail(f"not a recognised media file: {file}")
    video_id = sha256_file(file)
    from vidcp.pipeline import default_stages

    stage_names = [s.name for s in default_stages()]
    with _library() as conn:
        exists = conn.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is not None
        complete = exists and pipeline_complete(conn, video_id, stage_names)
    if complete:
        return {"video_id": video_id, "short_id": video_id[:8], "status": "already_ingested"}
    # A pre-existing row with an unfinished pipeline needs --force: without it
    # the CLI child would hit its already-ingested check and skip the pipeline.
    _spawn_ingest(file, force=exists)
    return {"video_id": video_id, "short_id": video_id[:8], "status": "started"}


def get_clip(video_id: str, start_s: float, end_s: float) -> dict:
    """Extract [start_s, end_s] from a video into an MP4 file and return its path.

    The clip is stream-copied (fast; cut points land on the nearest keyframes)
    and cached under the video's artifact directory — repeated calls with the
    same range return the same file. Use search()/get_transcript() to find the
    range first. The returned path is on the local filesystem.
    """
    from vidcp.clips import extract_clip

    with _library() as conn:
        vid = _resolve(conn, video_id)
    out = artifact_dir(vid) / "clips" / f"clip_{start_s:.2f}_{end_s:.2f}.mp4"
    if not out.exists():
        try:
            extract_clip(vid, start_s, end_s, out)
        except VidcpError as exc:
            _fail(exc.message, exc.hint)
    return {
        "video_id": vid,
        "path": str(out),
        "start_s": start_s,
        "end_s": end_s,
        "size_bytes": out.stat().st_size,
    }


_TOOLS = (
    search,
    list_videos,
    get_video,
    get_transcript,
    list_scenes,
    get_keyframe,
    get_clip,
    ingest,
)


def create_server() -> FastMCP:
    """Build the vidcp MCP server with all tools registered."""
    server = FastMCP("vidcp", instructions=_INSTRUCTIONS)
    for fn in _TOOLS:
        server.tool()(fn)
    return server
