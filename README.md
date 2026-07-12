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
       └─ scenes ─ keyframes ─ ocr ─┴─ embed
```

Everything lands in `~/.vidcp/library.db` (transcript + OCR text in FTS5, vectors
in a `sqlite-vec` table); source files and keyframes live under `~/.vidcp/store/`.

## MCP server

`vidcp mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio so
agents can query the library directly:

```bash
claude mcp add vidcp -- vidcp mcp
```

Tools: `search`, `list_videos`, `get_video`, `get_transcript`, `list_scenes`,
`get_keyframe` (returns the nearest stored keyframe as an image), and `ingest`.
`ingest` returns immediately and processing continues in a background process —
poll `get_video` until every stage reports `done` or `skipped`. The video
registers in the library shortly after `ingest` returns, so poll `get_video`
rather than treating an initial "no video matches" error as failure; a stage
reporting `failed` is also terminal.

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
| link_mode | `VIDCP_LINK_MODE` | `copy` | `copy\|hardlink` sources into the store |

## Constraints (v0.1)

- **CPU whisper is slow** — expect roughly real-time-ish transcription with the
  `small` model; use `--whisper-model tiny` (or `VIDCP_WHISPER_MODEL=tiny`) for
  speed at some accuracy cost.
- **OCR** reads sampled keyframes, not every frame, so fast-moving on-screen text
  can be missed; near-identical frames are perceptually de-duplicated.
- **First run downloads models** (whisper + embeddings) into the HuggingFace
  cache; everything after is offline.
- Semantic search always returns the nearest matches, so an off-topic query
  still surfaces its closest results rather than "no matches".

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                 # includes slow (model) tests
uv run pytest -m "not slow"   # fast subset
```
