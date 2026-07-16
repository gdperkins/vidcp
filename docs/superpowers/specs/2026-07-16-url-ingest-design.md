# URL Ingest via yt-dlp — Design

**Date:** 2026-07-16
**Status:** Approved
**Scope:** v0.3 feature — `vidcp ingest <url>` downloads a single video with
yt-dlp, then runs the normal ingest pipeline.

## Goal

Make `vidcp ingest` accept video URLs alongside file paths. A URL is
downloaded with yt-dlp and then flows through the existing ingest pipeline
unchanged. Out of scope (deliberately, for later): playlist/channel URLs,
URL support in `vidcp sync`, URL support in the MCP `ingest` tool, and
quality/format selection flags.

## Interface

- `vidcp ingest` auto-detects URL arguments: anything starting with
  `http://` or `https://` is treated as a URL; everything else stays a file
  path. Mixed invocations work: `vidcp ingest talk.mp4 https://youtube.com/watch?v=abc`.
- No new flags. Existing `--force`, `--whisper-model`, and `--no-ocr` apply
  to downloaded videos exactly as they do to files.
- `vidcp sync` remains directories-only; the MCP `ingest` tool remains
  files-only.

## Dependency strategy

yt-dlp is an **external binary on PATH**, like ffmpeg/ffprobe — not a Python
dependency. Rationale: sites break old extractor versions constantly, and a
user-managed binary (brew/pipx) stays current independently of vidcp
releases; it also keeps the core install lean.

- `vidcp doctor` gains a `yt-dlp` row via the existing `_check_tool` helper.
  It is informational only — a missing yt-dlp does **not** fail doctor's
  exit code, because URL ingest is optional.
- Invoking `ingest` with a URL when yt-dlp is missing raises `VidcpError`
  with hint: install yt-dlp (`brew install yt-dlp` / `pipx install yt-dlp`).

## Components

### `src/vidcp/download.py` (new)

One public function:

```python
def download_url(url: str, dest_dir: Path) -> DownloadedVideo: ...

class DownloadedVideo(BaseModel):
    path: Path   # downloaded media file
    title: str   # video title from yt-dlp metadata
    url: str     # the original URL
```

- Shells out to `yt-dlp` with `--no-playlist` (single-video scope) and an
  output template into `dest_dir` (an empty per-run temp directory), then
  reads back the final file path and title.
- Subprocess only — no yt-dlp Python import anywhere.
- Errors become `VidcpError`: binary missing → install hint; nonzero exit →
  message containing the last line of yt-dlp's stderr.

### CLI integration (`src/vidcp/cli.py`)

- `ingest` partitions its arguments into URLs and paths before
  `_expand_paths` (which only sees the paths).
- Each URL is downloaded into a per-run temp dir under `VIDCP_HOME`, then
  fed through the existing `_ingest_one`.
- `_ingest_one` gains two optional parameters:
  - `origin: str | None` — stored in `videos.path` instead of the local
    file path (for URL ingests, the origin is the URL);
  - `title: str | None` — stored in `videos.title` (from yt-dlp metadata).
- The temp file is deleted after ingest (`try/finally`), including on
  failure. The content-addressed store keeps the canonical `source.*` copy;
  nothing else references the temp path.

## Data model

No schema migration. `videos.path` (informational; never read as a
filesystem path elsewhere in the codebase) holds the URL; `videos.title`
(already exists, currently always NULL) holds the yt-dlp title.

Dedup falls out of the existing content-hash check: re-downloading the same
bytes produces the same SHA-256 → "already ingested".

## Error handling

- A failed download is reported like today's skip lines
  (`skip <url>: <reason>`), increments the error counter, and yields exit
  code 1 if nothing else ingested — matching current `ingest` semantics.
- A playlist-only URL fails cleanly with yt-dlp's own error message.
- Temp files never outlive the run.

## Testing

No network access in tests:

- `download.py`: unit tests with a mocked `subprocess.run` (success parsing,
  nonzero exit → `VidcpError` with stderr tail, missing binary → install
  hint).
- CLI: monkeypatch `download_url` to copy a fixture into `dest_dir` and
  return a fake title; assert the video lands with `path` = URL and `title`
  set, temp dir is cleaned up, mixed file+URL invocations work, and download
  failure produces a skip line + correct exit code.

## Documentation

- README: quickstart line (`vidcp ingest <url>`), constraints note (yt-dlp
  must be on PATH and kept current), doctor row mention.
