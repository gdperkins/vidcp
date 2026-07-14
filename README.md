# vidcp

Local-first CLI that ingests video files and turns them into searchable
knowledge: metadata, scenes, keyframes, transcript, OCR text, and embeddings,
stored in SQLite. **CPU-only, no cloud APIs** — the only network access is a
one-time download of the whisper and embedding models.

## Install

```bash
uv tool install .
# or, from a checkout, run without installing:
uv run vidcp --help
```

Requires **Python 3.11+** and **ffmpeg/ffprobe** on your `PATH`. Run
`vidcp doctor` to check your environment.

## Quickstart

```bash
vidcp doctor                       # verify ffmpeg, DB, sqlite-vec, models
vidcp ingest talk.mp4              # probe → scenes/keyframes → transcribe → ocr → embed
vidcp list                         # what's in the library
vidcp search "neural networks"     # hybrid keyword + semantic search
vidcp search "whiteboard diagram" --kind visual   # CLIP visual search over keyframes
vidcp clip a1b2c3d4 --from 1:23 --to 1:45 -o moment.mp4
vidcp sync ~/Movies                # ingest anything new, resume anything unfinished
vidcp inspect a1b2c3d4 --stages    # details + per-stage status
vidcp transcript a1b2c3d4 --format srt
vidcp export a1b2c3d4 --format markdown -o notes.md
vidcp stats
```

Video ids are SHA-256 hashes shown truncated to 8 chars; any unique prefix works.

## How it works

`ingest` runs a small dependency-ordered pipeline. Independent chains run in
parallel; each stage is resumable and only re-runs when its inputs or config
change (`vidcp reindex <id> --stage <name>` forces a re-run of a stage and its
dependents).

```
probe ─┬─ audio ─ transcribe ─┐
       └─ scenes ─ keyframes ─┬─ ocr ─┴─ embed
                              └─ embed_frames
```

Everything lands in `~/.vidcp/library.db` (transcript + OCR text in FTS5,
vectors in `sqlite-vec` tables for text and CLIP keyframe embeddings); source
files and keyframes live under `~/.vidcp/store/`.

## MCP server

`vidcp mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio so
agents can query the library directly:

```bash
claude mcp add vidcp -- vidcp mcp
```

Each tool wraps the same functions the CLI uses, against the same
`~/.vidcp` library:

| Tool | What it does |
| --- | --- |
| `search` | Hybrid keyword + semantic search over transcript, OCR text, and visual keyframe content; optional `video_id` and `kind` (`transcript`\|`ocr`\|`visual`) filters |
| `list_videos` | Every video in the library, newest first |
| `get_video` | One video's metadata, artifact counts, and per-stage pipeline status |
| `get_transcript` | Transcript segments, optionally windowed to `[start_s, end_s]` |
| `list_scenes` | Detected scene boundaries with timestamps |
| `get_keyframe` | The stored keyframe nearest a timestamp, returned as a JPEG (longest side ≤ 1280 px) so the agent can look at the moment |
| `get_clip` | Extract `[start_s, end_s]` as an MP4 (stream copy, cached per range) and return its local path |
| `ingest` | Add a new video file; returns immediately while processing runs in a detached background process |

A typical agent flow: `search` for a phrase, `get_transcript` windowed around a
hit's `ts_s` for context, `get_keyframe` at that timestamp to see the frame, or
`get_clip` to extract the moment as a standalone video.

`ingest` is asynchronous — poll `get_video` until every stage reports `done` or
`skipped`. The video registers in the library shortly after `ingest` returns,
so poll `get_video` rather than treating an initial "no video matches" error as
failure; a stage reporting `failed` is also terminal.

## Configuration

Settings load from defaults → `~/.vidcp/config.toml` → `VIDCP_*` environment
variables (env wins).

| Setting | Env var | Default | Notes |
| --- | --- | --- | --- |
| home | `VIDCP_HOME` | `~/.vidcp` | Library + store location |
| whisper_model | `VIDCP_WHISPER_MODEL` | `small` | `tiny\|base\|small\|medium` |
| scene_threshold | `VIDCP_SCENE_THRESHOLD` | `27.0` | PySceneDetect ContentDetector |
| keyframe_min_interval_s | `VIDCP_KEYFRAME_MIN_INTERVAL_S` | `10.0` | Keyframe sampling floor |
| phash_max_distance | `VIDCP_PHASH_MAX_DISTANCE` | `6` | Perceptual-hash dedupe distance |
| ocr_enabled | `VIDCP_OCR_ENABLED` | `true` | Set false (or `--no-ocr`) to skip OCR |
| embed_model | `VIDCP_EMBED_MODEL` | `all-MiniLM-L6-v2` | 384-dim sentence-transformers |
| clip_model | `VIDCP_CLIP_MODEL` | `clip-ViT-B-32` | 512-dim CLIP for visual search |
| clip_enabled | `VIDCP_CLIP_ENABLED` | `true` | Set false to skip keyframe embedding |
| link_mode | `VIDCP_LINK_MODE` | `copy` | `copy\|hardlink` sources into the store |

## Constraints (v0.2)

- **CPU whisper is slow** — expect roughly real-time-ish transcription with the
  `small` model; use `--whisper-model tiny` (or `VIDCP_WHISPER_MODEL=tiny`) for
  speed at some accuracy cost.
- **OCR** reads sampled keyframes, not every frame, so fast-moving on-screen text
  can be missed; near-identical frames are perceptually de-duplicated.
- **First run downloads models** (whisper + embeddings) into the HuggingFace
  cache; everything after is offline.
- The CLIP model is a ~600 MB one-time download (set `VIDCP_CLIP_ENABLED=false`
  to opt out).
- Semantic search always returns the nearest matches, so an off-topic query
  still surfaces its closest results rather than "no matches".
- Videos ingested before v0.2 gain visual search on their next `vidcp sync` of
  a folder containing them, or via `vidcp reindex <id>`.
- Clip extraction stream-copies by default, so cut points snap to the nearest
  keyframes; use `--precise` for frame-accurate (re-encoded) cuts.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                 # includes slow (model) tests
uv run pytest -m "not slow"   # fast subset
```
