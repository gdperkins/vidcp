# MCP Server for vidcp — Design

**Date:** 2026-07-12
**Status:** Approved

## Overview

Add an MCP (Model Context Protocol) server to vidcp so agents (Claude Code,
Claude Desktop, any MCP client) can query the video library directly: search
across transcripts and OCR text, pull transcript context, look at keyframes,
and kick off ingestion. The server is a thin wrapper over existing library
functions — no new pipeline or search behavior.

## Goals

- Expose the library's query surface as MCP tools over stdio.
- Let agents visually inspect moments via keyframe images.
- Let agents add videos (`ingest`) without blocking on the minutes-long
  pipeline.
- Preserve the project's conventions: lazy imports, `VidcpError` messaging,
  fast CLI startup.

## Non-goals (v1)

- `delete` / `reindex` tools (destructive or operational; stay in the CLI).
- HTTP/SSE transport, auth, MCP resources or prompts.
- Any change to search ranking, pipeline stages, or storage.

## Architecture

- New module: `src/vidcp/mcp_server.py`.
- New CLI command: `vidcp mcp` — starts the server on stdio. The `mcp` SDK is
  imported inside the command (lazy-import convention).
- New file: `src/vidcp/__main__.py` (two lines) so background ingest can be
  spawned as `python -m vidcp ...`.
- Dependency: official `mcp` Python SDK (FastMCP-style decorator API), added as
  a regular project dependency.
- DB access: each tool call opens/closes its own connection via `db.connect()`,
  matching how CLI commands behave. The server holds no long-lived connection.
- Registration example for docs: `claude mcp add vidcp -- vidcp mcp`.

## Tool surface

All `video_id` parameters accept any unique id prefix, resolved with the
existing `library.resolve_id`. Structured results reuse the existing Pydantic
models (`Hit`, `Video`, `StageState`, etc.) via `model_dump(mode="json")`.

### `search(query, video_id=None, kind=None, limit=10)`

Wraps `vidcp.search.search`. Returns the list of hits: `video_id`, `short_id`,
`kind` (`transcript` | `ocr`), `ts_s`, `text`, `snippet`, `score`.

### `list_videos()`

Returns all videos ordered by ingest time: id, short id, title, duration,
resolution, codecs, size, `ingested_at`.

### `get_video(video_id)`

Returns video metadata, artifact counts (scenes, frames, segments, OCR blocks),
and per-stage status rows (`stage`, `status`, `started_at`, `finished_at`,
`error`). This is also the polling endpoint for async ingest: an agent calls
`ingest`, then polls `get_video` until every stage is `done`/`skipped` or a
stage reports `failed`.

### `get_transcript(video_id, start_s=None, end_s=None)`

Returns transcript segments (start, end, text). With `start_s`/`end_s`, only
segments overlapping the window are returned — this is how an agent pulls
context around a search hit. Errors if the video has no transcript (silent
video or transcribe not finished), with a hint that distinguishes the two.

### `list_scenes(video_id)`

Returns scene rows: index, `start_s`, `end_s`.

### `get_keyframe(video_id, ts_s)`

Finds the nearest kept keyframe (`SELECT ... FROM frames WHERE video_id=? AND
kept=1 ORDER BY ABS(ts_s - ?) LIMIT 1`), loads it with Pillow, downscales to at
most 1280px on the longest side (JPEG, quality 85), and returns it as MCP image
content plus a text block stating the frame's actual timestamp. Errors if the
video has no keyframes yet.

### `ingest(path)`

Async kickoff. The tool computes the id but performs **no writes** — the
spawned CLI child owns `add_source`, the `videos` row insert, and the pipeline,
exactly as a normal ingest. (Pre-inserting the row here would make the child's
already-ingested check skip the pipeline entirely.)

1. Validate the path exists and `is_media_file` accepts it.
2. Hash the file (`sha256_file` — seconds, acceptable inline) to get the id.
3. Look up the id:
   - Row exists and every stage is `done`/`skipped` → return
     `{video_id, status: "already_ingested"}`; spawn nothing.
   - Row exists with incomplete/failed stages (a prior ingest crashed or
     failed) → spawn `vidcp ingest --force <path>`. `--force` only bypasses
     the CLI's already-ingested skip; the runner's `config_hash`/up-to-date
     checks still skip finished stages, so this is a resume, not a redo.
   - No row → spawn `vidcp ingest <path>`.
4. Spawn as a **detached subprocess** — `[sys.executable, "-m", "vidcp",
   "ingest", ...]` with stdio redirected to `DEVNULL` and
   `start_new_session=True` — and return `{video_id, status: "started"}`.

Why a subprocess instead of a thread: whisper/OCR memory lives and dies with
the child rather than accumulating in the long-lived server process, and the
runner's existing crash recovery (stale `running` rows reset to `pending` on
the next run) means a killed child leaves cleanly resumable state. The child
re-hashing the file is redundant but cheap, and keeps its behavior identical to
a normal CLI ingest. Concurrent duplicate calls are tolerated: the CLI's
already-ingested check and the stage state machine make the second run a cheap
no-op or resume.

## Error handling

- `VidcpError` raised by wrapped code is caught at the tool boundary and
  re-raised as an MCP tool error carrying `message` (and `hint` appended when
  present) — agents see the same wording CLI users do.
- Unknown or ambiguous id prefixes produce the existing `resolve_id` errors.
- Unexpected exceptions surface as generic tool errors; the server does not
  crash on a bad tool call.

## Testing

Use the SDK's in-memory client/server session so tests drive real MCP tool
calls without subprocesses or sockets:

- Tool listing: all 7 tools present with expected schemas.
- `search` / `get_transcript` / `list_scenes` / `get_video` against a seeded
  library (existing fixture-seeding helpers).
- `get_transcript` windowing returns only overlapping segments.
- `get_keyframe` returns image content; downscaling verified with an oversized
  fixture frame.
- `ingest` with `subprocess.Popen` mocked: fresh path spawns plain ingest
  (assert detached spawn args), partial prior ingest spawns with `--force`,
  fully-ingested returns `already_ingested` without spawning, invalid path
  errors.
- Anything that loads whisper/embedding models gets the `slow` marker (most of
  these tests need no models — search tests can seed FTS rows directly,
  consistent with existing tests).

## Documentation

- README: new "MCP server" section — what it exposes, the
  `claude mcp add vidcp -- vidcp mcp` one-liner, and a note that `ingest`
  returns immediately and is polled via `get_video`.
- `vidcp mcp --help` text describing the tool surface.
