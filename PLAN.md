# vidcp — Implementation Plan (v0.1, local-only)

## Context for the implementer

You are building `vidcp`, a local-first CLI that ingests video files and turns them into searchable knowledge: metadata, scenes, keyframes, transcript, OCR text, and embeddings, stored in SQLite. No cloud APIs, no network calls except one-time model downloads. Python 3.11+, CPU-only. Implement steps in order; each step ends with passing acceptance criteria. Do not start a step until the previous one's criteria pass. Prefer boring, readable code over cleverness. All timestamps in seconds as REAL. All user-facing IDs are SHA-256 hex, displayed truncated to 8 chars, accepted as any unique prefix.

## Global conventions

- Package manager: **uv**. Lint/format: **ruff**. Tests: **pytest**.
- Dependencies (top-level): `typer`, `rich`, `pydantic`, `pydantic-settings`, `scenedetect[opencv]`, `faster-whisper`, `rapidocr-onnxruntime`, `sentence-transformers`, `sqlite-vec`, `imagehash`, `pillow`.
- **Lazy imports**: `faster_whisper`, `rapidocr_onnxruntime`, `sentence_transformers` must be imported inside the functions that use them, never at module top level. `vidcp list` must start in under ~300ms.
- ffmpeg/ffprobe are invoked via `subprocess.run` with explicit arg lists (never `shell=True`), always with `-hide_banner -loglevel error`, and ffprobe always with `-print_format json`.
- Every read command supports `--json` for machine-readable output (plain `json.dumps` of pydantic models, `model_dump(mode="json")`).
- Errors: raise `VidcpError(message, hint=...)` subclasses; the CLI catches them and prints `message` in red and `hint` in dim via Rich, exit code 1. Never show a traceback to the user unless `--debug`.
- Home dir: `~/.vidcp/` overridable by env var `VIDCP_HOME` (critical for tests).

---

## Step 1 — Skeleton, config, DB, doctor

### Files

**`pyproject.toml`** — project name `vidcp`, `[project.scripts] vidcp = "vidcp.cli:app"`, deps as above, `requires-python = ">=3.11"`. Add `[tool.ruff]` with line-length 100.

**`src/vidcp/config.py`**

```python
class Settings(BaseSettings):
    home: Path = Path("~/.vidcp").expanduser()
    whisper_model: str = "small"          # tiny|base|small|medium
    scene_threshold: float = 27.0         # PySceneDetect ContentDetector
    keyframe_min_interval_s: float = 10.0 # floor for sparse-cut videos
    phash_max_distance: int = 6
    ocr_enabled: bool = True
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    link_mode: str = "copy"               # copy|hardlink
    model_config = SettingsConfigDict(env_prefix="VIDCP_", toml_file="~/.vidcp/config.toml")
```

Load order: defaults → config.toml → env vars. Provide `get_settings()` cached accessor. `home` derived paths: `db_path = home/"library.db"`, `store_path = home/"store"`.

**`src/vidcp/db.py`** — `connect()` returns a `sqlite3.Connection` with: `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, `row_factory=sqlite3.Row`, and `sqlite_vec` extension loaded (`conn.enable_load_extension(True); sqlite_vec.load(conn)`). Migrations: a `migrations/` list of SQL strings applied in order, tracked in `schema_version(version INT)` table. Migration 001 creates all tables from the schema below.

### Full schema (migration 001)

```sql
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
```

The `vec0` virtual table is created in migration 002 (Step 6) so earlier steps don't depend on sqlite-vec working:

```sql
CREATE VIRTUAL TABLE vec USING vec0(
  embedding float[384],
  video_id TEXT, kind TEXT, ref_id INT, ts_s REAL
);
```

**`src/vidcp/cli.py`** — Typer app with placeholder commands raising "not implemented yet" for everything in the surface list, plus a working:

**`vidcp doctor`** — checks and prints a Rich table: ffmpeg on PATH (+version), ffprobe on PATH, `~/.vidcp` writable, DB opens and migrations applied, sqlite-vec loads, and whether whisper/embedding models are already downloaded (check the HuggingFace cache dirs; report "will download on first use" rather than downloading). Exit 0 if ffmpeg+ffprobe+DB pass; nonzero otherwise.

### Acceptance criteria (Step 1)

`uv run vidcp doctor` prints the table and exits 0 on a machine with ffmpeg. `uv run vidcp list` errors gracefully ("not implemented"). `VIDCP_HOME=/tmp/x uv run vidcp doctor` creates `/tmp/x/library.db` with all migration-001 tables (verify in a pytest test using `sqlite3` directly). Ruff passes.

---

## Step 2 — Ingest core: hashing, store, probe, list/inspect/delete

**`src/vidcp/store.py`**

```python
def sha256_file(path: Path) -> str            # 1MB chunks
def artifact_dir(video_id: str) -> Path       # store/<id[:2]>/<id>/, mkdir -p
def add_source(path: Path, video_id: str) -> Path
    # copy or hardlink per settings.link_mode into artifact_dir/source<ext>
```

**`src/vidcp/models.py`** — pydantic models mirroring the tables: `Video`, `SceneRow`, `Segment`, `OcrBlock`, `StageState`. Include `Video.short_id` property (`id[:8]`).

**`src/vidcp/pipeline/base.py`**

```python
class VideoContext:
    video_id: str
    conn: sqlite3.Connection
    settings: Settings
    @property
    def source_path(self) -> Path      # resolve from artifact_dir
    @property
    def artifacts(self) -> Path

class Stage(ABC):
    name: ClassVar[str]
    depends_on: ClassVar[list[str]] = []
    def config_fingerprint(self, s: Settings) -> str:
        return ""   # override to include relevant settings; hashed with stage name
    @abstractmethod
    def run(self, ctx: VideoContext) -> None: ...
```

**`src/vidcp/pipeline/stages/probe.py`** — run ffprobe (`-show_format -show_streams`), parse JSON, populate the `videos` row: duration, width/height/fps (parse `avg_frame_rate` fraction) from the first video stream, `has_audio` from presence of an audio stream, `vcodec/acodec`, `size_bytes`, `created_at` from format tags if present, full JSON into `meta`. `title` defaults to filename stem.

**Minimal runner (full version in Step 7)** — `run_pipeline(ctx, stages)` executes stages sequentially in dependency order, writing `stages` rows: `running` → `done`/`failed` with timestamps and error text. If a stage row is `done` and `config_hash` matches, skip it. Ingest command flow:

```
vidcp ingest <paths...> [--force]
  for each path: validate exists + ffprobe recognises it (fail per-file, continue batch)
  id = sha256; if id in videos and not --force → print "already ingested <short_id>", skip
  add_source, INSERT videos (minimal row), run_pipeline([probe])
```

**Commands**: `list` (Rich table: short id, title, duration mm:ss, resolution, ingested date; `--json` gives full models), `inspect <id>` (all video fields; `--stages` adds the stages table), `delete <id> [--keep-artifacts]` (DELETE row — cascades — and rm artifact dir unless kept), plus `resolve_id(prefix)` helper used by every command: unique-prefix match against `videos.id`, `VidcpError` on ambiguous/missing.

### Acceptance criteria (Step 2)

Ingesting a fixture video creates the videos row with correct duration (±0.2s) and a `probe=done` stage row; re-ingesting prints "already ingested" and does no work (assert `ingested_at` unchanged); `list --json` round-trips through `json.loads`; `delete` removes row and artifacts; prefix resolution works with 4-char prefix and errors on ambiguity (test with two fixtures).

Also create **`tests/fixtures/generate.py`** now (invoked by a pytest session fixture, cached): three ~8s videos made with ffmpeg —
`color.mp4` (testsrc2, silent, no scene cuts), `cuts.mp4` (concat of 4 different solid-color+testsrc segments → 3 hard cuts, sine audio), `text.mp4` (dark background, `drawtext` showing `HELLO VIDCP 42` for the full duration, sine audio). Additionally commit one real ~5s speech WAV/MP4 saying a known phrase (generate once with any TTS or record; commit the file, do not synthesize in CI) — name it `speech.mp4`, expected transcript substring: define it in `tests/constants.py`.

---

## Step 3 — Scenes + keyframes

**`stages/scenes.py`** (`depends_on=["probe"]`) — use `scenedetect.detect(path, ContentDetector(threshold=settings.scene_threshold))`. If zero cuts detected, create a single scene spanning the whole video. Insert `scenes` rows with idx, start_s, end_s. `config_fingerprint` includes threshold.

**`stages/keyframes.py`** (`depends_on=["scenes"]`) — build the list of candidate timestamps: each scene midpoint, PLUS additional timestamps so no gap exceeds `keyframe_min_interval_s` (for long scenes, insert extra frames every N seconds within the scene; associate them with that scene). Extract all frames in **one ffmpeg call per video** using the select filter with `eq(n,...)` is fiddly — instead use one call per timestamp with `-ss <t> -i src -frames:v 1 -q:v 3 frames/f_<idx>.jpg`, but batched via `concurrent.futures.ThreadPoolExecutor(max_workers=4)` (ffmpeg seek-before-input is fast). Compute pHash per frame (`imagehash.phash`); walking in time order, drop any frame within Hamming distance ≤ `phash_max_distance` of the last **kept** frame. Update `scenes.keyframe_path`/`phash` for scene-midpoint frames; extra interval frames get their own scenes-table rows? — **No**: keep schema clean, store extra frames as additional rows in a new small table:

```sql
-- migration 003 (part of this step)
CREATE TABLE frames (
  id INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  scene_id INT REFERENCES scenes(id) ON DELETE CASCADE,
  ts_s REAL NOT NULL, path TEXT NOT NULL, phash TEXT, kept INT NOT NULL DEFAULT 1
);
```

All frames (scene midpoints included) live in `frames`; `scenes.keyframe_path` is a denormalised convenience pointing at its midpoint frame. Dropped duplicates: delete the JPEG, don't insert a row.

**`vidcp scenes <id> [--json]`** — table of idx, start→end, duration, keyframe path.

### Acceptance criteria (Step 3)

`cuts.mp4` yields 4 (±0) scenes with boundaries within 0.5s of the known cut points; `color.mp4` yields exactly 1 scene and, given the 8s duration and 10s floor, exactly 1 kept frame; JPEGs exist on disk; re-running ingest skips both stages (assert via `stages.finished_at` unchanged); changing `scene_threshold` in settings and re-ingesting re-runs scenes (config_hash invalidation test).

---

## Step 4 — Audio + transcription

**`stages/audio.py`** (`depends_on=["probe"]`) — if `has_audio=0`, mark own status `skipped` (the runner must support a stage setting skipped from inside `run` via raising `StageSkipped("no audio stream")`). Else: `ffmpeg -i src -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.

**`stages/transcribe.py`** (`depends_on=["audio"]`) — skipped if audio was skipped (runner rule: if any dependency is `skipped`, dependent is `skipped` too, recursively). Else:

```python
from faster_whisper import WhisperModel
model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
segments, info = model.transcribe(str(wav), vad_filter=True, word_timestamps=True)
```

Insert one `segments` row per whisper segment (text stripped; skip empty), `confidence = exp(avg_logprob)` clamped [0,1], `words` as JSON list of `{w, s, e}`. Also dump the raw result to `transcript.json` in the artifact dir. After inserting, populate FTS: one `fts` row per segment with `kind='transcript'`, `ref_id=segments.id`, `ts_s=start_s`. `config_fingerprint` includes the model name.

**`vidcp transcript <id> [--format txt|srt|vtt|json]`** — implement `export/srt.py` and `export/vtt.py` now as pure functions `segments → str` (timestamp formatting: SRT `HH:MM:SS,mmm`, VTT `HH:MM:SS.mmm`, header `WEBVTT`). `txt` = plain lines with `[mm:ss]` prefixes. Default prints Rich-formatted with timestamps.

Wire `audio, transcribe` into the ingest pipeline list. Add `--whisper-model` and `--no-ocr` (no-op until Step 5) flags to `ingest`, overriding settings for that run (note: override must flow into `config_fingerprint`).

### Acceptance criteria (Step 4)

`speech.mp4` transcript contains the known phrase (case-insensitive); `color.mp4` (silent? — it has no audio track in generate.py; ensure that) results in `audio=skipped, transcribe=skipped` and `transcript` command prints a friendly "no speech" message; SRT output validates (regex the timestamp lines, sequential indices); FTS query `SELECT * FROM fts WHERE fts MATCH 'phrase-word'` returns the segment. Mark the whisper test `@pytest.mark.slow` and make CI run slow tests (model `tiny` via env override to keep CI fast: `VIDCP_WHISPER_MODEL=tiny`).

---

## Step 5 — OCR

**`stages/ocr.py`** (`depends_on=["keyframes"]`, skipped when `settings.ocr_enabled` is false or `--no-ocr`) —

```python
from rapidocr_onnxruntime import RapidOCR
engine = RapidOCR()
result, _ = engine(frame_path)   # list of [bbox, text, score]
```

Per kept frame: join detected lines top-to-bottom into one text blob, keep per-line results too. Filter score < 0.5. Then **temporal dedupe**: walk frames in time order; if normalized similarity (`difflib.SequenceMatcher(None, a, b).ratio()`) with the previous block's text > 0.9, extend the previous block's `end_s` instead of inserting a new one. Insert `ocr_blocks` (start_s, end_s, joined text, mean confidence, bbox JSON = list of line boxes). FTS rows with `kind='ocr'`, `ts_s=start_s`.

Add `vidcp inspect <id> --ocr` or simply include OCR block count in inspect output; expose blocks via `vidcp search` (Step 6) rather than a dedicated command.

### Acceptance criteria (Step 5)

`text.mp4` produces ≥1 ocr_block whose text contains `HELLO VIDCP 42` (allow OCR noise: assert `"VIDCP" in text`); despite ~1 frame per 10s floor the 8s video yields exactly one block (temporal dedupe collapses if multiple frames); `--no-ocr` ingest marks stage `skipped`; FTS match on `VIDCP` returns kind `ocr`.

---

## Step 6 — Hybrid search

Migration 002 (`vec` table) applies here if not already.

**`stages/embed.py`** (`depends_on=["transcribe","ocr"]` — but must run even when both are skipped-with-no-rows? If there are zero FTS rows, mark skipped). Load sentence-transformers model once, `model.encode(texts, batch_size=64, normalize_embeddings=True)`, insert into `vec` via `INSERT INTO vec(embedding, video_id, kind, ref_id, ts_s) VALUES (?,...)` with `sqlite_vec.serialize_float32(vector)`. Embed both transcript segments and OCR blocks. Delete existing vec rows for the video first (idempotency).

Dependency subtlety for the runner: a stage whose dependencies are `skipped` normally cascades to skipped — embed must instead depend on "transcribe AND ocr have *finished* (done or skipped)". Introduce `depends_on` semantics = "finished", and cascade-skip only when **all** deps are skipped. Document this in runner docstring.

**`src/vidcp/search.py`**

```python
def search(conn, query, video_id=None, kind=None, limit=10) -> list[Hit]
```

FTS leg: `SELECT ref_id, kind, video_id, ts_s, bm25(fts) AS s FROM fts WHERE fts MATCH ? ORDER BY s LIMIT 50` (escape the query for FTS5: wrap each token in double quotes to avoid syntax errors on `:`/`-`). Vector leg: encode query, `SELECT ref_id, kind, video_id, ts_s, distance FROM vec WHERE embedding MATCH ? AND k = 50` (+ optional `video_id = ?` — sqlite-vec supports metadata filtering; if the installed version doesn't, over-fetch and filter in Python). RRF fusion with k=60 keyed on `(video_id, kind, ref_id)`. Hydrate hits: fetch text from segments/ocr_blocks, build snippet (±80 chars around first query-token match, else first 160 chars).

**`vidcp search "<q>" [--id] [--kind] [--limit] [--json]`** — Rich output: `short_id  [mm:ss]  (kind)  snippet`, semantically ordered. JSON includes exact `ts_s` and `ref_id`.

### Acceptance criteria (Step 6)

After ingesting all fixtures: exact keyword from speech transcript ranks that segment #1; a *paraphrase* of the speech phrase (choose one at fixture-creation time, put in constants) returns the segment in top 3 (vector leg working); `search --kind ocr "vidcp"` returns only OCR hits; `--id` filter restricts results; empty-result search prints "no matches" and exits 0.

---

## Step 7 — Runner hardening, resumability, parallelism, reindex

Finalize **`pipeline/runner.py`**:

1. Topological sort of the stage DAG (fail loudly on cycles at import time via a test).
2. Execute with `ThreadPoolExecutor(max_workers=2)`: submit any stage whose deps are all finished; the practical effect is the `audio→transcribe` chain runs parallel to `scenes→keyframes→ocr`. Whisper and OCR release the GIL in native code, so threads suffice; do not use multiprocessing.
3. State machine per stage row: `pending → running → done|failed|skipped`. On process crash, a row can be left `running`: on next ingest, treat stale `running` (no live pid — just always) as `pending`.
4. Skip logic: `done` + matching `config_hash` → skip silently (log line). `failed` → retry by default.
5. A failed stage fails its downstream (`failed` cascades as `pending`-but-blocked; report at end: "ingest completed with errors: transcribe failed: <err>. Run `vidcp reindex <id> --stage transcribe` to retry"). Exit code 2 for partial success.
6. Rich progress: one `Progress` with a task per stage, spinner + elapsed.

**`vidcp reindex <id> [--stage <name>] [--all]`** — `--stage X`: set X and all transitive dependents to `pending` (and delete their DB rows/artifacts: each Stage gains an optional `clean(ctx)` method — scenes deletes scenes+frames rows and JPEGs, transcribe deletes segments + its fts rows, etc. — implement `clean` for every stage now). No flag = rerun everything not `done`; `--all` = full wipe and rerun.

**`vidcp stats`** — counts: videos, total duration, scenes, segments, ocr blocks, vec rows, store size on disk, DB size.

### Acceptance criteria (Step 7)

Kill-resume test: monkeypatch transcribe stage to raise after audio completes; re-run ingest; audio is skipped (already done), transcribe retries. Reindex test: `reindex --stage scenes` clears scenes/frames/ocr/embed and reruns them but not transcribe. Parallelism smoke test: stage start/finish timestamps show audio ran overlapping scenes (assert intervals overlap on `cuts.mp4`). `stats` numbers match direct SQL counts.

---

## Step 8 — Exports, polish, docs

**`export/json.py`** — full knowledge object: video row + scenes (with frame paths) + segments (with words) + ocr_blocks, one nested document. This is the canonical "vidcp file format" — version it: `{"vidcp_export_version": 1, ...}`.

**`export/markdown.py`** — human-readable: title header, metadata table, "## Chapters" (scenes with timestamps), "## Transcript" (merged into paragraphs — join segments, break paragraph on gaps > 2s), "## On-screen text" (deduped OCR blocks with time ranges).

**`vidcp export <id> --format json|markdown|srt|vtt [-o path]`** — default stdout.

Polish checklist: `--version` flag; `vidcp` with no args prints help; every command's `--help` has a one-line description and an example; `ingest` accepts globs expanded by the shell and directories (recurse for `*.mp4 *.mov *.mkv *.webm *.avi`); consistent duration formatting helper (`mm:ss` under an hour, `h:mm:ss` over); README.md with install (`uv tool install`), quickstart (ingest → search → export), config reference table, and the "constraints" section (CPU whisper speed, OCR limits).

CI (GitHub Actions): ubuntu-latest, install ffmpeg via apt, `uv sync`, ruff check, pytest with `VIDCP_WHISPER_MODEL=tiny`, cache the HF model dir keyed on model names.

### Acceptance criteria (Step 8)

`export --format json | python -m json.tool` succeeds and contains all sections for `speech.mp4`; markdown export renders headers and transcript paragraphs; full test suite green in CI from a clean checkout; `uv tool install .` then `vidcp doctor && vidcp ingest fixture && vidcp search ...` works end-to-end outside the repo.

---

## Deferred (do not build now, but do not preclude)

URL ingestion (Fetcher plugin interface), vision-LLM scene descriptions (a new stage writing `kind='description'` into the same fts/vec tables), speaker diarization, the MCP server (a thin layer over `search.py` + exports — the reason every command has `--json`), and provider abstraction. The only forward-looking requirements binding on v0.1: the `Stage` ABC contract, the `kind` column discipline, and `--json` everywhere.
