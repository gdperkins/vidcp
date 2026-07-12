# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `vidcp mcp` — an MCP server over stdio exposing the video library to agents: search, transcripts, scenes, keyframe images, and async ingest.

**Architecture:** A new `src/vidcp/mcp_server.py` module builds a v1 `FastMCP` server whose 7 tools are thin wrappers over existing library functions; each tool call opens its own SQLite connection via `db.connect()`. `ingest` spawns a detached `python -m vidcp ingest` subprocess and returns immediately; agents poll `get_video` for per-stage status. Spec: `docs/superpowers/specs/2026-07-12-mcp-server-design.md`.

**Tech Stack:** Python 3.11+, official `mcp` SDK (v1 `FastMCP` API), Typer CLI, SQLite (WAL), Pillow, pytest + anyio (in-memory MCP client sessions).

## Global Constraints

- Dependency pin: `mcp>=1.10,<2` — PyPI latest is 1.28.x. **Do not use v2 APIs** (`MCPServer`, `mcp.Client`) — v2 is unreleased; v1 uses `from mcp.server.fastmcp import FastMCP, Image` and in-memory test sessions via `mcp.shared.memory.create_connected_server_and_client_session`.
- `mcp_server.py` code must NEVER write to stdout (print/rich) — stdout is the MCP stdio transport. Stderr is fine.
- `src/vidcp/mcp_server.py` is only imported inside the `vidcp mcp` CLI command body (lazy-import convention — keeps CLI startup fast). Within `mcp_server.py`, importing the `mcp` SDK at module top is fine; `vidcp.search`, `PIL`, and `vidcp.pipeline.stages.probe` are imported inside tool functions.
- Every module starts with `from __future__ import annotations`. Ruff line length 100.
- Tool failures raise `ToolError` (`mcp.server.fastmcp.exceptions`) carrying the same message + hint wording as `VidcpError` does in the CLI.
- Tools that return data return a single `dict` (never a bare list) so MCP structured output is stable. Exception: `get_keyframe` returns a mixed `[str, Image]` list and must have **no return type annotation** (an output schema can't represent an Image).
- Tests: `uv run pytest tests/test_mcp_server.py -v`. Async tests use the anyio pytest plugin (ships with the `mcp` dependency chain) — `pytestmark = pytest.mark.anyio` plus an `anyio_backend` fixture returning `"asyncio"`. None of these tests load whisper/embedding models, so no `slow` markers.
- ffmpeg/ffprobe must be on PATH (already a project requirement; `ingest` tests call `is_media_file`, which shells out to ffprobe).
- Commit messages: imperative, matching repo history (e.g. "Add MCP search tool"). NEVER add Claude attribution / Co-Authored-By lines.

---

### Task 1: Scaffolding — dependency, `__main__.py`, server factory, `list_videos`

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Create: `src/vidcp/__main__.py`
- Create: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `vidcp.db.connect()`, `vidcp.library.resolve_id`, `vidcp.models.Video`, `vidcp.errors.VidcpError`
- Produces (later tasks rely on these exact names):
  - `mcp_server.create_server() -> FastMCP` — builds the server, registering every function in the module-level `_TOOLS` tuple
  - `mcp_server._library()` — context manager yielding a `sqlite3.Connection`
  - `mcp_server._fail(message: str, hint: str | None = None) -> NoReturn` — raises `ToolError`
  - `mcp_server._resolve(conn, video_id: str) -> str` — prefix resolution, `VidcpError` → `ToolError`
  - `mcp_server._video_payload(row) -> dict` — Video row → JSON dict (drops `meta`, adds `short_id`)
  - Test helpers: `client` fixture (in-memory MCP session), `result_payload(result) -> dict`, `error_text(result) -> str`, `seed_video(video_id=..., title=..., duration_s=...) -> str`, constants `VID_A = "a" * 64`, `VID_B = "b" * 64`

- [ ] **Step 1: Add the dependency**

```bash
uv add "mcp>=1.10,<2"
```

Expected: `pyproject.toml` gains `"mcp>=1.10,<2"` under `[project].dependencies`; `uv.lock` updates.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_mcp_server.py`:

```python
"""MCP server tests: in-memory client sessions, no subprocesses or sockets.

Seeds rows directly through ``db.connect()`` (the autouse ``_isolated_home``
fixture gives every test a fresh ``VIDCP_HOME``). The vector search leg is
never exercised (the ``vec`` table stays empty), so no models are downloaded.
"""

import json
import subprocess
import sys

import pytest
from mcp.shared.memory import create_connected_server_and_client_session as client_session

from vidcp.db import connect
from vidcp.mcp_server import create_server
from vidcp.util import now_iso

pytestmark = pytest.mark.anyio

VID_A = "a" * 64
VID_B = "b" * 64


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    server = create_server()
    async with client_session(server._mcp_server) as session:
        yield session


def result_payload(result) -> dict:
    """Unwrap a successful tool result into its dict payload."""
    assert not result.isError, result.content
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


def error_text(result) -> str:
    assert result.isError, result.content
    return result.content[0].text


def seed_video(video_id=VID_A, title="talk", duration_s=60.0) -> str:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos(id, path, title, duration_s, ingested_at) VALUES (?,?,?,?,?)",
            (video_id, f"/videos/{title}.mp4", title, duration_s, now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
    return video_id


async def test_lists_expected_tools(client):
    tools = {tool.name for tool in (await client.list_tools()).tools}
    assert "list_videos" in tools


async def test_list_videos_empty_library(client):
    payload = result_payload(await client.call_tool("list_videos", {}))
    assert payload == {"videos": []}


async def test_list_videos_returns_seeded_video(client):
    seed_video(VID_A, title="talk")
    payload = result_payload(await client.call_tool("list_videos", {}))
    assert len(payload["videos"]) == 1
    video = payload["videos"][0]
    assert video["id"] == VID_A
    assert video["short_id"] == VID_A[:8]
    assert video["title"] == "talk"
    assert "meta" not in video


def test_python_dash_m_vidcp_runs():
    result = subprocess.run(
        [sys.executable, "-m", "vidcp", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "vidcp" in result.stdout
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'vidcp.mcp_server'`

- [ ] **Step 4: Create `src/vidcp/__main__.py`**

```python
"""Allow ``python -m vidcp`` (used by the MCP server to spawn background ingests)."""

from __future__ import annotations

from vidcp.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `src/vidcp/mcp_server.py`**

```python
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
from vidcp.library import resolve_id
from vidcp.models import Video

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


_TOOLS = (list_videos,)


def create_server() -> FastMCP:
    """Build the vidcp MCP server with all tools registered."""
    server = FastMCP("vidcp", instructions=_INSTRUCTIONS)
    for fn in _TOOLS:
        server.tool()(fn)
    return server
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: 4 passed

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add pyproject.toml uv.lock src/vidcp/__main__.py src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP server scaffolding with list_videos tool"
```

---

### Task 2: `get_video` tool (+ move artifact counts into `library.py`)

**Files:**
- Modify: `src/vidcp/library.py`
- Modify: `src/vidcp/cli.py` (remove `_artifact_counts` at ~line 75; update its one call site in `inspect`; update imports)
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 1 helpers (`_library`, `_resolve`, `_video_payload`, `seed_video`, `result_payload`, `error_text`), `vidcp.models.StageState`
- Produces:
  - `vidcp.library.artifact_counts(conn, video_id) -> dict[str, int]` — moved verbatim from `cli._artifact_counts`
  - MCP tool `get_video(video_id: str) -> dict` — `_video_payload` fields + `"counts"` + `"stages"` (list of StageState dumps)
  - Test helper: `seed_stage(video_id, stage, status, error=None)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
def seed_stage(video_id, stage, status, error=None):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO stages(video_id, stage, status, error) VALUES (?,?,?,?) "
            "ON CONFLICT(video_id, stage) DO UPDATE SET status=excluded.status, "
            "error=excluded.error",
            (video_id, stage, status, error),
        )
        conn.commit()
    finally:
        conn.close()


async def test_get_video_metadata_counts_and_stages(client):
    seed_video(VID_A, title="talk")
    seed_stage(VID_A, "probe", "done")
    seed_stage(VID_A, "transcribe", "failed", error="boom")
    payload = result_payload(await client.call_tool("get_video", {"video_id": VID_A[:8]}))
    assert payload["id"] == VID_A
    assert payload["title"] == "talk"
    assert payload["counts"] == {"scenes": 0, "frames": 0, "segments": 0, "ocr_blocks": 0}
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert by_stage["probe"]["status"] == "done"
    assert by_stage["transcribe"]["status"] == "failed"
    assert by_stage["transcribe"]["error"] == "boom"


async def test_get_video_unknown_id_errors(client):
    result = await client.call_tool("get_video", {"video_id": "deadbeef"})
    assert "no video matches id 'deadbeef'" in error_text(result)


async def test_get_video_ambiguous_prefix_errors(client):
    seed_video("a" + "c" * 63)
    seed_video("a" + "d" * 63)
    result = await client.call_tool("get_video", {"video_id": "a"})
    assert "ambiguous" in error_text(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k get_video`
Expected: 3 failed — `Unknown tool: get_video` (asserted via `error_text` / `result_payload` raising AssertionError on `isError`)

- [ ] **Step 3: Move `artifact_counts` to `library.py`**

Append to `src/vidcp/library.py`:

```python
def artifact_counts(conn: sqlite3.Connection, video_id: str) -> dict[str, int]:
    """Row counts of each per-video artifact table (scenes/frames/segments/ocr_blocks)."""
    counts = {}
    for table in ("scenes", "frames", "segments", "ocr_blocks"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE video_id=?", (video_id,)
        ).fetchone()[0]
    return counts
```

In `src/vidcp/cli.py`: delete the `_artifact_counts` function (~line 75), change the import line `from vidcp.library import resolve_id` to `from vidcp.library import artifact_counts, resolve_id`, and in `inspect` change `counts = _artifact_counts(conn, vid)` to `counts = artifact_counts(conn, vid)`. (`_artifact_counts` has exactly one call site; verify with `grep -n _artifact_counts src/vidcp/cli.py`.)

- [ ] **Step 4: Implement `get_video` in `mcp_server.py`**

Add `StageState` to the models import (`from vidcp.models import StageState, Video`) and `artifact_counts` to the library import (`from vidcp.library import artifact_counts, resolve_id`). Add the tool and register it:

```python
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
```

```python
_TOOLS = (list_videos, get_video)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py tests/test_cli.py -v`
Expected: all pass (test_cli.py guards the `inspect` refactor)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/library.py src/vidcp/cli.py src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP get_video tool; move artifact_counts into library"
```

---

### Task 3: `search` tool

**Files:**
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `vidcp.search.search` (lazy import — pulls the embeddings stack), Task 1/2 helpers
- Produces:
  - MCP tool `search(query: str, video_id: str | None = None, kind: str | None = None, limit: int = 10) -> dict` returning `{"hits": [Hit dumps]}`
  - Test helper: `seed_segment(video_id, start_s, end_s, text) -> int` (inserts `segments` + matching `fts` row, returns segment id)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
def seed_segment(video_id, start_s, end_s, text) -> int:
    conn = connect()
    try:
        cur = conn.execute(
            "INSERT INTO segments(video_id, start_s, end_s, text) VALUES (?,?,?,?)",
            (video_id, start_s, end_s, text),
        )
        conn.execute(
            "INSERT INTO fts(text, video_id, kind, ref_id, ts_s) VALUES (?,?,?,?,?)",
            (text, video_id, "transcript", cur.lastrowid, start_s),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


async def test_search_returns_fts_hit(client):
    seed_video(VID_A)
    seed_segment(VID_A, 5.0, 9.0, "neural networks are discussed here")
    payload = result_payload(await client.call_tool("search", {"query": "neural"}))
    assert payload["hits"]
    hit = payload["hits"][0]
    assert hit["video_id"] == VID_A
    assert hit["kind"] == "transcript"
    assert hit["ts_s"] == 5.0
    assert "neural" in hit["snippet"]


async def test_search_rejects_unknown_kind(client):
    result = await client.call_tool("search", {"query": "x", "kind": "faces"})
    assert "unknown kind 'faces'" in error_text(result)


async def test_search_restricts_to_video_id_prefix(client):
    seed_video(VID_A, title="a")
    seed_video(VID_B, title="b")
    seed_segment(VID_A, 1.0, 2.0, "quantum computing intro")
    seed_segment(VID_B, 3.0, 4.0, "quantum computing outro")
    payload = result_payload(
        await client.call_tool("search", {"query": "quantum", "video_id": VID_A[:8]})
    )
    assert payload["hits"]
    assert all(h["video_id"] == VID_A for h in payload["hits"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k test_search`
Expected: 3 failed — `Unknown tool: search`

- [ ] **Step 3: Implement `search` in `mcp_server.py`**

```python
def search(
    query: str, video_id: str | None = None, kind: str | None = None, limit: int = 10
) -> dict:
    """Hybrid keyword + semantic search over transcripts and on-screen (OCR) text.

    Returns timestamped, scored hits. kind filters to 'transcript' or 'ocr';
    video_id (any unique prefix) restricts to one video. Follow up with
    get_transcript() around a hit's ts_s, or get_keyframe() to see the moment.
    """
    if kind is not None and kind not in ("transcript", "ocr"):
        _fail(f"unknown kind '{kind}'", "choose one of: transcript, ocr")
    from vidcp.search import search as run_search

    with _library() as conn:
        vid = _resolve(conn, video_id) if video_id else None
        hits = run_search(conn, query, video_id=vid, kind=kind, limit=limit)
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}
```

```python
_TOOLS = (search, list_videos, get_video)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: all pass (the empty `vec` table keeps the vector leg — and model loading — off)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP search tool"
```

---

### Task 4: `get_transcript` tool with windowing

**Files:**
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 1–3 helpers (`seed_video`, `seed_segment`, `seed_stage`)
- Produces: MCP tool `get_transcript(video_id: str, start_s: float | None = None, end_s: float | None = None) -> dict` returning `{"video_id", "segments": [{"id","start_s","end_s","text"}]}` (segment `words` deliberately omitted — word-level timestamps flood agent context)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
async def test_get_transcript_full(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "hello world")
    seed_segment(VID_A, 4.0, 8.0, "second segment")
    payload = result_payload(await client.call_tool("get_transcript", {"video_id": VID_A}))
    assert payload["video_id"] == VID_A
    assert [s["text"] for s in payload["segments"]] == ["hello world", "second segment"]
    assert set(payload["segments"][0]) == {"id", "start_s", "end_s", "text"}


async def test_get_transcript_window_overlap(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "before")
    seed_segment(VID_A, 3.0, 7.0, "overlaps start")
    seed_segment(VID_A, 8.0, 12.0, "inside")
    seed_segment(VID_A, 20.0, 24.0, "after")
    payload = result_payload(
        await client.call_tool(
            "get_transcript", {"video_id": VID_A, "start_s": 4.0, "end_s": 15.0}
        )
    )
    assert [s["text"] for s in payload["segments"]] == ["overlaps start", "inside"]


async def test_get_transcript_empty_window_is_not_an_error(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "hello")
    payload = result_payload(
        await client.call_tool(
            "get_transcript", {"video_id": VID_A, "start_s": 100.0, "end_s": 110.0}
        )
    )
    assert payload["segments"] == []


async def test_get_transcript_silent_video_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "skipped")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "no audio" in error_text(result)


async def test_get_transcript_not_finished_errors(client):
    seed_video(VID_A)  # no transcribe stage row at all
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "not available yet" in error_text(result)


async def test_get_transcript_no_speech_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "done")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "no speech" in error_text(result)


async def test_get_transcript_failed_stage_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "failed", error="model exploded")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "model exploded" in error_text(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k get_transcript`
Expected: 7 failed — `Unknown tool: get_transcript`

- [ ] **Step 3: Implement `get_transcript` in `mcp_server.py`**

```python
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


def get_transcript(
    video_id: str, start_s: float | None = None, end_s: float | None = None
) -> dict:
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
```

```python
_TOOLS = (search, list_videos, get_video, get_transcript)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP get_transcript tool with time windowing"
```

---

### Task 5: `list_scenes` tool

**Files:**
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 1–2 helpers
- Produces:
  - MCP tool `list_scenes(video_id: str) -> dict` returning `{"video_id", "scenes": [{"idx","start_s","end_s"}]}`
  - Test helper: `seed_scene(video_id, idx, start_s, end_s)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
def seed_scene(video_id, idx, start_s, end_s):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO scenes(video_id, idx, start_s, end_s) VALUES (?,?,?,?)",
            (video_id, idx, start_s, end_s),
        )
        conn.commit()
    finally:
        conn.close()


async def test_list_scenes_ordered_by_idx(client):
    seed_video(VID_A)
    seed_scene(VID_A, 1, 10.0, 20.0)
    seed_scene(VID_A, 0, 0.0, 10.0)
    payload = result_payload(await client.call_tool("list_scenes", {"video_id": VID_A[:8]}))
    assert payload["video_id"] == VID_A
    assert [s["idx"] for s in payload["scenes"]] == [0, 1]
    assert payload["scenes"][0] == {"idx": 0, "start_s": 0.0, "end_s": 10.0}


async def test_list_scenes_empty_is_not_an_error(client):
    seed_video(VID_A)
    payload = result_payload(await client.call_tool("list_scenes", {"video_id": VID_A}))
    assert payload["scenes"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k list_scenes`
Expected: 2 failed — `Unknown tool: list_scenes`

- [ ] **Step 3: Implement `list_scenes` in `mcp_server.py`**

```python
def list_scenes(video_id: str) -> dict:
    """List detected scene boundaries (idx, start_s, end_s) for a video."""
    with _library() as conn:
        vid = _resolve(conn, video_id)
        rows = conn.execute(
            "SELECT idx, start_s, end_s FROM scenes WHERE video_id=? ORDER BY idx", (vid,)
        ).fetchall()
    return {"video_id": vid, "scenes": [dict(row) for row in rows]}
```

```python
_TOOLS = (search, list_videos, get_video, get_transcript, list_scenes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP list_scenes tool"
```

---

### Task 6: `get_keyframe` tool (image content)

**Files:**
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `frames` table (`ts_s`, `path`, `kept`), Pillow (lazy import), `Image` from `mcp.server.fastmcp`
- Produces:
  - MCP tool `get_keyframe(video_id: str, ts_s: float)` — **no return annotation** — returning `[text describing the frame's actual timestamp, Image(JPEG, longest side <= 1280px)]`
  - Module constants `_MAX_EDGE = 1280`, `_JPEG_QUALITY = 85`
  - Test helper: `seed_frame(video_id, ts_s, tmp_path, size=(1920, 1080), kept=1) -> Path`

- [ ] **Step 1: Write the failing tests**

Add `import base64` and `import io` to the imports of `tests/test_mcp_server.py`, then add:

```python
def seed_frame(video_id, ts_s, tmp_path, size=(1920, 1080), kept=1):
    from PIL import Image as PILImage

    path = tmp_path / f"frame_{ts_s:.0f}_{kept}.jpg"
    PILImage.new("RGB", size, (200, 40, 40)).save(path, "JPEG")
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO frames(video_id, ts_s, path, kept) VALUES (?,?,?,?)",
            (video_id, ts_s, str(path), kept),
        )
        conn.commit()
    finally:
        conn.close()
    return path


async def test_get_keyframe_nearest_and_downscaled(client, tmp_path):
    from PIL import Image as PILImage

    seed_video(VID_A)
    seed_frame(VID_A, 10.0, tmp_path)
    seed_frame(VID_A, 50.0, tmp_path)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A[:8], "ts_s": 18.0})
    assert not result.isError, result.content
    texts = [c for c in result.content if c.type == "text"]
    images = [c for c in result.content if c.type == "image"]
    assert texts and "10.00s" in texts[0].text
    assert images and images[0].mimeType == "image/jpeg"
    decoded = PILImage.open(io.BytesIO(base64.b64decode(images[0].data)))
    assert max(decoded.size) <= 1280


async def test_get_keyframe_ignores_discarded_frames(client, tmp_path):
    seed_video(VID_A)
    seed_frame(VID_A, 10.0, tmp_path, kept=0)
    seed_frame(VID_A, 50.0, tmp_path)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 10.0})
    assert not result.isError, result.content
    texts = [c for c in result.content if c.type == "text"]
    assert "50.00s" in texts[0].text


async def test_get_keyframe_without_frames_errors(client):
    seed_video(VID_A)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 1.0})
    assert "no keyframes" in error_text(result)


async def test_get_keyframe_missing_file_errors(client, tmp_path):
    seed_video(VID_A)
    path = seed_frame(VID_A, 10.0, tmp_path)
    path.unlink()
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 10.0})
    assert "keyframe file missing" in error_text(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k get_keyframe`
Expected: 4 failed — `Unknown tool: get_keyframe`

- [ ] **Step 3: Implement `get_keyframe` in `mcp_server.py`**

Extend imports: add `io` to the stdlib imports, `from pathlib import Path`, and change the mcp import to `from mcp.server.fastmcp import FastMCP, Image`. Add:

```python
_MAX_EDGE = 1280
_JPEG_QUALITY = 85


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
```

```python
_TOOLS = (search, list_videos, get_video, get_transcript, list_scenes, get_keyframe)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP get_keyframe tool returning downscaled frame images"
```

---

### Task 7: `ingest` tool (async subprocess kickoff)

**Files:**
- Modify: `src/vidcp/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `store.sha256_file`, `pipeline.stages.probe.is_media_file` (lazy import; shells out to ffprobe), `subprocess`, `sys`; test uses the committed `speech_fixture` (conftest fixture) because `is_media_file` needs a real media file
- Produces:
  - MCP tool `ingest(path: str) -> dict` returning `{"video_id", "short_id", "status": "started" | "already_ingested"}`
  - `_spawn_ingest(path: Path, force: bool) -> None` — detached `python -m vidcp ingest [--force] <path>`
  - Test fixture: `spawn_recorder` (monkeypatches `Popen`, records `(cmd, kwargs)` calls)

**Design note (from the spec):** this tool performs NO DB/store writes — the child owns them. Pre-inserting the `videos` row would trip the CLI child's already-ingested skip. Completeness is checked via the `embed` stage: it is the DAG's terminal stage (transitively depends on every other stage), so `embed in (done, skipped)` ⇔ the whole pipeline finished — without importing the heavy stage modules that `default_stages()` pulls in.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
@pytest.fixture
def spawn_recorder(monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))

    monkeypatch.setattr("vidcp.mcp_server.subprocess.Popen", fake_popen)
    return calls


async def test_ingest_new_file_spawns_detached_ingest(client, spawn_recorder, speech_fixture):
    from vidcp.store import sha256_file

    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "started"
    assert payload["video_id"] == sha256_file(speech_fixture)
    ((cmd, kwargs),) = spawn_recorder
    assert cmd == [sys.executable, "-m", "vidcp", "ingest", str(speech_fixture)]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL


async def test_ingest_partial_prior_run_resumes_with_force(
    client, spawn_recorder, speech_fixture
):
    from vidcp.store import sha256_file

    video_id = sha256_file(speech_fixture)
    seed_video(video_id)
    seed_stage(video_id, "probe", "done")  # crashed mid-pipeline: embed never completed
    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "started"
    ((cmd, _),) = spawn_recorder
    assert cmd == [sys.executable, "-m", "vidcp", "ingest", "--force", str(speech_fixture)]


async def test_ingest_complete_video_short_circuits(client, spawn_recorder, speech_fixture):
    from vidcp.store import sha256_file

    video_id = sha256_file(speech_fixture)
    seed_video(video_id)
    seed_stage(video_id, "embed", "done")
    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "already_ingested"
    assert spawn_recorder == []


async def test_ingest_missing_file_errors(client, spawn_recorder):
    result = await client.call_tool("ingest", {"path": "/nope/missing.mp4"})
    assert "file not found" in error_text(result)
    assert spawn_recorder == []


async def test_ingest_non_media_file_errors(client, spawn_recorder, tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("definitely not a video")
    result = await client.call_tool("ingest", {"path": str(bogus)})
    assert "not a recognised media file" in error_text(result)
    assert spawn_recorder == []


async def test_all_seven_tools_registered(client):
    tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {
        "search",
        "list_videos",
        "get_video",
        "get_transcript",
        "list_scenes",
        "get_keyframe",
        "ingest",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k "test_ingest or test_all_seven"`
Expected: 6 failed — `Unknown tool: ingest` for the ingest tests; the tool-set test fails because `ingest` is missing

- [ ] **Step 3: Implement `ingest` in `mcp_server.py`**

Add `subprocess` and `sys` to the stdlib imports and `from vidcp.store import sha256_file` to the vidcp imports. Add:

```python
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
        start_new_session=True,
    )


def ingest(path: str) -> dict:
    """Ingest a video file into the library (asynchronous).

    Returns immediately with the video id; probing, transcription, OCR, and
    embedding continue in a background process. Poll get_video() until every
    stage is 'done' or 'skipped'.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        _fail(f"file not found: {file}")
    from vidcp.pipeline.stages.probe import is_media_file

    if not is_media_file(file):
        _fail(f"not a recognised media file: {file}")
    video_id = sha256_file(file)
    with _library() as conn:
        exists = (
            conn.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is not None
        )
        # embed is the DAG's terminal stage — it can only be done/skipped after
        # every other stage finished, so it doubles as a completeness check.
        embed = conn.execute(
            "SELECT status FROM stages WHERE video_id=? AND stage='embed'", (video_id,)
        ).fetchone()
    if exists and embed is not None and embed["status"] in ("done", "skipped"):
        return {"video_id": video_id, "short_id": video_id[:8], "status": "already_ingested"}
    # A pre-existing row with an unfinished pipeline needs --force: without it
    # the CLI child would hit its already-ingested check and skip the pipeline.
    _spawn_ingest(file, force=exists)
    return {"video_id": video_id, "short_id": video_id[:8], "status": "started"}
```

```python
_TOOLS = (search, list_videos, get_video, get_transcript, list_scenes, get_keyframe, ingest)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vidcp/mcp_server.py tests/test_mcp_server.py
git commit -m "Add MCP ingest tool with detached background pipeline"
```

---

### Task 8: `vidcp mcp` CLI command, README, final validation

**Files:**
- Modify: `src/vidcp/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `mcp_server.create_server()` (lazy import inside the command body — this is what keeps the `mcp` SDK off the normal CLI startup path)
- Produces: `vidcp mcp` CLI command

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_mcp_command_registered():
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_mcp_command_registered -v`
Expected: FAIL — exit code 2 (no such command "mcp")

- [ ] **Step 3: Add the command to `cli.py`**

Add after the `export` command (keep the lazy import inside the body):

```python
@app.command("mcp")
def mcp_command() -> None:
    """Run the MCP server over stdio, exposing the library to agents.

    Example: claude mcp add vidcp -- vidcp mcp
    """
    from vidcp.mcp_server import create_server

    create_server().run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 5: Add the README section**

Insert after the "How it works" section of `README.md`:

````markdown
## MCP server

`vidcp mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio so
agents can query the library directly:

```bash
claude mcp add vidcp -- vidcp mcp
```

Tools: `search`, `list_videos`, `get_video`, `get_transcript`, `list_scenes`,
`get_keyframe` (returns the nearest stored keyframe as an image), and `ingest`.
`ingest` returns immediately and processing continues in a background process —
poll `get_video` until every stage reports `done` or `skipped`.
````

- [ ] **Step 6: Full validation**

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not slow"
```

Expected: lint clean, all fast tests pass. Optional interactive smoke test:
`npx @modelcontextprotocol/inspector uv run vidcp mcp` (lists the 7 tools).

- [ ] **Step 7: Commit**

```bash
git add src/vidcp/cli.py README.md tests/test_cli.py
git commit -m "Add vidcp mcp command and README docs"
```
