# URL Ingest via yt-dlp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vidcp ingest` auto-detects `http(s)://` arguments, downloads them with the yt-dlp binary, and runs the normal ingest pipeline on the result.

**Architecture:** A new subprocess-only `src/vidcp/download.py` wraps the yt-dlp binary (on PATH, like ffmpeg — never a Python dependency) and returns the downloaded file plus its title. The CLI partitions `ingest` args into URLs and paths; each URL downloads into a per-run temp dir under `VIDCP_HOME`, flows through the existing `_ingest_one` (which gains optional `origin`/`title` parameters so `videos.path` stores the URL and `videos.title` the yt-dlp title), and the temp file is deleted — the content-addressed store keeps the canonical copy. No schema migration.

**Tech Stack:** Python 3.12 (uv), Typer/Rich CLI, yt-dlp external binary, pytest with mocked subprocess (no network in tests).

**Spec:** `docs/superpowers/specs/2026-07-16-url-ingest-design.md`

## Global Constraints

- Line length 100 (ruff); every module starts with `from __future__ import annotations`.
- No yt-dlp Python import anywhere — subprocess to the `yt-dlp` binary only.
- User-facing errors: raise `VidcpError(message, hint=...)`.
- Tests must not touch the network and must not require yt-dlp to be installed.
- Tests touching `VIDCP_*` env vars must call `get_settings.cache_clear()` (autouse `_isolated_home` handles setup/teardown).
- SQL migrations are append-only in `src/vidcp/db.py` — this feature adds NO migration.
- Lint before every commit: `uv run ruff check . && uv run ruff format .`
- Iterate with `uv run pytest -m "not slow"`.
- Commit messages: plain, descriptive, NO attribution footers of any kind.

---

## Task 1: `download.py` module

**Files:**
- Create: `src/vidcp/download.py`
- Test: `tests/test_download.py` (new)

**Interfaces:**
- Consumes: `VidcpError` from `vidcp.errors` (existing).
- Produces (used by Task 2):
  - `class DownloadedVideo(BaseModel)` with fields `path: Path`, `title: str`, `url: str`
  - `download_url(url: str, dest_dir: Path) -> DownloadedVideo`
  - `ensure_ytdlp() -> None` — raises `VidcpError` with an install hint when the binary is missing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_download.py`:

```python
"""download_url tests. subprocess and shutil.which are always mocked —
no network access and no yt-dlp binary required."""

import json
import subprocess
from pathlib import Path

import pytest

import vidcp.download as download
from vidcp.download import DownloadedVideo, download_url, ensure_ytdlp
from vidcp.errors import VidcpError

URL = "https://example.com/watch?v=abc123"


def _install_fake_run(
    monkeypatch,
    dest_dir: Path,
    *,
    returncode: int = 0,
    stderr: str = "",
    write_media: bool = True,
    write_info: bool = True,
    title: str = "A Talk",
):
    """Replace subprocess.run inside vidcp.download with a fake yt-dlp."""

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "yt-dlp"
        assert "--no-playlist" in cmd
        if returncode == 0:
            if write_media:
                (dest_dir / "A Talk [abc123].mp4").write_bytes(b"\x00" * 1024)
            if write_info:
                (dest_dir / "A Talk [abc123].info.json").write_text(
                    json.dumps({"title": title})
                )
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(download.subprocess, "run", fake_run)


@pytest.fixture
def _ytdlp_on_path(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: "/opt/bin/yt-dlp")


def test_download_url_success(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest)
    result = download_url(URL, dest)
    assert isinstance(result, DownloadedVideo)
    assert result.path.exists() and result.path.suffix == ".mp4"
    assert result.title == "A Talk"
    assert result.url == URL
    # the info.json sidecar is consumed and removed
    assert list(dest.glob("*.info.json")) == []


def test_download_url_title_falls_back_to_filename(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest, write_info=False)
    result = download_url(URL, dest)
    assert result.title == "A Talk [abc123]"


def test_download_url_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: None)

    def explode(*args, **kwargs):  # subprocess must never be reached
        raise AssertionError("subprocess.run called despite missing binary")

    monkeypatch.setattr(download.subprocess, "run", explode)
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, tmp_path / "dl")
    assert "yt-dlp" in str(excinfo.value)
    assert "install" in (excinfo.value.hint or "")


def test_ensure_ytdlp(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    with pytest.raises(VidcpError):
        ensure_ytdlp()
    monkeypatch.setattr(download.shutil, "which", lambda name: "/opt/bin/yt-dlp")
    ensure_ytdlp()  # no raise


def test_download_url_failure_surfaces_stderr_tail(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(
        monkeypatch, dest, returncode=1, stderr="warning: x\nERROR: Unsupported URL"
    )
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, dest)
    assert "Unsupported URL" in str(excinfo.value)


def test_download_url_no_output_file(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest, write_media=False, write_info=False)
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, dest)
    assert "no file" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidcp.download'`

- [ ] **Step 3: Write the implementation**

Create `src/vidcp/download.py`:

```python
"""Download a video URL with yt-dlp for ingestion.

yt-dlp is an external binary on PATH (like ffmpeg), never a Python
dependency: extractors churn fast, and a user-managed binary stays current
independently of vidcp releases. Everything here is subprocess-only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from vidcp.errors import VidcpError

_INSTALL_HINT = "install yt-dlp: brew install yt-dlp (or pipx install yt-dlp)"


class DownloadedVideo(BaseModel):
    path: Path
    title: str
    url: str


def ensure_ytdlp() -> None:
    """Raise a friendly error when the yt-dlp binary is not on PATH."""
    if shutil.which("yt-dlp") is None:
        raise VidcpError("yt-dlp not found on PATH", hint=_INSTALL_HINT)


def download_url(url: str, dest_dir: Path) -> DownloadedVideo:
    """Download a single video ``url`` into ``dest_dir`` with yt-dlp.

    ``dest_dir`` must be empty (a fresh temp dir): the downloaded media file
    is identified as the largest file left behind. The title comes from the
    ``--write-info-json`` sidecar, which is consumed and deleted. Raises
    ``VidcpError`` when yt-dlp is missing or the download fails.
    """
    ensure_ytdlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--write-info-json",
            "-o",
            str(dest_dir / "%(title).150B [%(id)s].%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "unknown error"
        raise VidcpError(f"download failed: {detail}")
    title = ""
    for info_file in dest_dir.glob("*.info.json"):
        try:
            title = json.loads(info_file.read_text()).get("title") or ""
        except (OSError, ValueError):
            title = ""
        info_file.unlink(missing_ok=True)
    media = [f for f in dest_dir.iterdir() if f.is_file()]
    if not media:
        raise VidcpError(f"yt-dlp produced no file for {url}")
    media_file = max(media, key=lambda f: f.stat().st_size)
    return DownloadedVideo(path=media_file, title=title or media_file.stem, url=url)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/vidcp/download.py tests/test_download.py
git commit -m "Add download module wrapping the yt-dlp binary"
```

---

## Task 2: URL support in `vidcp ingest`

**Files:**
- Modify: `src/vidcp/cli.py` (`_ingest_one`, `ingest`, new `_is_url` helper, add `tempfile` import)
- Test: `tests/test_ingest.py` (append)

**Interfaces:**
- Consumes: `download_url`, `ensure_ytdlp`, `DownloadedVideo` from Task 1 (lazy-imported inside `ingest`).
- Produces: `vidcp ingest` accepts `http(s)://` args mixed with files/dirs. `_ingest_one` gains keyword-only-style optional params `origin: str | None = None` (stored in `videos.path` instead of the local path) and `title: str | None = None` (stored in `videos.title`).

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py` already imports `json`, `pytest`, `CliRunner`, `app`, `get_settings`, `connect`, `VidcpError` and defines `runner = CliRunner()`. Add two imports **to the existing import block at the top of the file** (ruff flags mid-file module-level imports):

```python
import shutil
from pathlib import Path
```

Then append to the end of the file:

```python
# --- URL ingest (download_url is stubbed — no network, no yt-dlp) -----------


def _fake_download(fixture: Path, title: str = "Fake Talk"):
    from vidcp.download import DownloadedVideo

    def fake(url: str, dest_dir: Path) -> DownloadedVideo:
        dest = Path(dest_dir) / "Fake Talk [abc].mp4"
        shutil.copy2(fixture, dest)
        return DownloadedVideo(path=dest, title=title, url=url)

    return fake


URL = "https://example.com/watch?v=abc123"


def test_ingest_url_downloads_and_ingests(fixtures, monkeypatch):
    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)
    monkeypatch.setattr("vidcp.download.download_url", _fake_download(fixtures["color.mp4"]))
    result = runner.invoke(app, ["ingest", "--no-ocr", URL])
    assert result.exit_code == 0, result.output
    conn = connect()
    try:
        row = conn.execute("SELECT id, path, title FROM videos").fetchone()
        assert row is not None
        assert row["path"] == URL  # origin is the URL, not the temp file
        assert row["title"] == "Fake Talk"
        vid = row["id"]
    finally:
        conn.close()
    from vidcp.store import artifact_dir

    assert any(artifact_dir(vid).glob("source.*"))  # canonical copy in the store
    tmp_root = get_settings().home / "tmp"
    assert list(tmp_root.iterdir()) == []  # per-run temp dir cleaned up


def test_ingest_mixed_file_and_url(fixtures, tmp_path, monkeypatch):
    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)
    monkeypatch.setattr("vidcp.download.download_url", _fake_download(fixtures["color.mp4"]))
    local = tmp_path / "local.mp4"
    shutil.copy2(fixtures["cuts.mp4"], local)  # different content -> different hash
    result = runner.invoke(app, ["ingest", "--no-ocr", str(local), URL])
    assert result.exit_code == 0, result.output
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 2
    finally:
        conn.close()


def test_ingest_url_download_failure_is_skip(monkeypatch):
    from vidcp.errors import VidcpError

    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)

    def boom(url, dest_dir):
        raise VidcpError("download failed: ERROR: Unsupported URL")

    monkeypatch.setattr("vidcp.download.download_url", boom)
    result = runner.invoke(app, ["ingest", URL])
    assert result.exit_code == 1
    assert "skip" in result.output and "Unsupported URL" in result.output
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    finally:
        conn.close()
    tmp_root = get_settings().home / "tmp"
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_ingest_url_without_ytdlp_fails_fast(monkeypatch):
    import vidcp.download as download

    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    result = runner.invoke(app, ["ingest", URL])
    assert result.exit_code != 0
    assert "yt-dlp not found" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -m "not slow" -v`
Expected: the four new tests FAIL (URL arg treated as a nonexistent file path → "no video files found" / skip lines); all pre-existing tests still PASS.

- [ ] **Step 3: Write the implementation**

In `src/vidcp/cli.py`:

1. Add `tempfile` to the stdlib imports at the top (after `subprocess`):

```python
import tempfile
```

2. Add next to `_expand_paths`:

```python
def _is_url(item: str) -> bool:
    return item.startswith(("http://", "https://"))
```

3. Extend `_ingest_one`'s signature and INSERT (only these two spots change):

```python
def _ingest_one(
    conn,
    path: Path,
    settings: Settings,
    force: bool,
    console: Console = console,
    origin: str | None = None,
    title: str | None = None,
) -> str:
```

and replace the INSERT statement inside it with:

```python
        conn.execute(
            "INSERT INTO videos(id, path, title, ingested_at, has_audio) VALUES (?, ?, ?, ?, 1)",
            (video_id, origin or str(path.resolve()), title, now_iso()),
        )
```

and replace its docstring with:

```python
    """Ingest a single file. Returns:
    'ingested' | 'failed_stages' | 'already' | 'missing' | 'not_media'.

    ``console`` defaults to the module-level console; callers that need clean
    stdout (e.g. ``sync --json``) pass a stderr console instead so these Rich
    status lines don't land ahead of a JSON document on stdout.

    ``origin`` and ``title`` override the stored ``videos.path`` and
    ``videos.title`` — used by URL ingest, where the meaningful source is the
    URL, not the temp file the download landed in.
    """
```

4. Rework `ingest`: update the argument help and docstring, partition URLs, fail fast on a missing binary, and add the URL loop. The full new body:

```python
@app.command()
def ingest(
    paths: Optional[list[str]] = typer.Argument(
        None, help="Video files, directories, or http(s) URLs."
    ),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already present."),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model", help="Override the whisper model for this run."
    ),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Skip OCR for this run."),
) -> None:
    """Ingest video files, directories, or URLs into the library.

    URLs are downloaded with yt-dlp (must be on PATH), then ingested like any
    file. Example: vidcp ingest clip.mp4 https://youtube.com/watch?v=abc
    """
    if not paths:
        raise VidcpError("no paths given", hint="usage: vidcp ingest <file-dir-or-url> ...")
    settings = get_settings()
    overrides: dict[str, object] = {}
    if whisper_model:
        overrides["whisper_model"] = whisper_model
    if no_ocr:
        overrides["ocr_enabled"] = False
    if overrides:
        # A per-run copy so the overrides flow into stage config fingerprints.
        settings = settings.model_copy(update=overrides)
    urls = [item for item in paths if _is_url(item)]
    files = _expand_paths([item for item in paths if not _is_url(item)])
    if not files and not urls:
        raise VidcpError("no video files found in the given paths")
    if urls:
        from vidcp.download import ensure_ytdlp

        ensure_ytdlp()  # fail fast before any work if the binary is missing

    conn = connect()
    ingested = 0
    errors = 0
    had_failures = False
    try:
        for path in files:
            status = _ingest_one(conn, path, settings, force)
            if status == "missing":
                console.print(f"[red]skip[/red] {path}: file not found")
                errors += 1
            elif status == "not_media":
                console.print(f"[red]skip[/red] {path}: not a recognised media file")
                errors += 1
            elif status == "failed_stages":
                had_failures = True
                ingested += 1
            elif status == "ingested":
                ingested += 1

        if urls:
            from vidcp.download import download_url

            tmp_root = settings.home / "tmp"
            tmp_root.mkdir(parents=True, exist_ok=True)
            for url in urls:
                console.print(f"downloading {url}")
                # The store keeps the canonical source.* copy, so the download
                # itself is disposable — TemporaryDirectory guarantees cleanup.
                with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
                    try:
                        downloaded = download_url(url, Path(tmp))
                    except VidcpError as exc:
                        console.print(f"[red]skip[/red] {url}: {exc.message}")
                        errors += 1
                        continue
                    status = _ingest_one(
                        conn,
                        downloaded.path,
                        settings,
                        force,
                        origin=downloaded.url,
                        title=downloaded.title,
                    )
                    if status == "not_media":
                        console.print(
                            f"[red]skip[/red] {url}: downloaded file is not a recognised media file"
                        )
                        errors += 1
                    elif status == "failed_stages":
                        had_failures = True
                        ingested += 1
                    elif status == "ingested":
                        ingested += 1
    finally:
        conn.close()

    if had_failures:
        raise typer.Exit(code=2)  # partial success
    if errors and not ingested:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py tests/test_sync.py tests/test_cli.py -m "not slow" -v`
Expected: PASS (new URL tests plus all pre-existing ingest/sync/cli tests — `sync` and MCP are untouched).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/vidcp/cli.py tests/test_ingest.py
git commit -m "Accept URLs in vidcp ingest, downloading via yt-dlp"
```

---

## Task 3: doctor row and README docs

**Files:**
- Modify: `src/vidcp/cli.py` (`_check_tool` gains a `version_flag` param; `doctor` gains a yt-dlp row)
- Modify: `README.md` (quickstart line, constraints bullet)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: existing `_check_tool`, `_render_doctor_table`.
- Produces: `_check_tool(name: str, version_flag: str = "-version") -> tuple[bool, str]`; a `yt-dlp (optional)` doctor row that renders green `OK` when present and dim `info` when absent — never `FAIL`, and never affects doctor's exit code.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py` already imports `app` and defines `runner = CliRunner()` at the top — reuse both. Append to the end of the file (the `vidcp.cli` module import is function-local on purpose; module-level imports may only go at the top of the file):

```python
# --- doctor: optional yt-dlp row ---------------------------------------------


def test_doctor_reports_ytdlp_row():
    result = runner.invoke(app, ["doctor"])
    assert "yt-dlp" in result.output


def test_doctor_missing_ytdlp_is_not_fatal(monkeypatch):
    import vidcp.cli as cli_mod

    real_check = cli_mod._check_tool

    def fake_check(name, version_flag="-version"):
        if name == "yt-dlp":
            return False, "not found on PATH"
        return real_check(name, version_flag)

    monkeypatch.setattr(cli_mod, "_check_tool", fake_check)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output  # ffmpeg/ffprobe/db still OK
    assert "yt-dlp" in result.output
    assert "FAIL" not in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -m "not slow" -v -k ytdlp`
Expected: both FAIL (`"yt-dlp" not in result.output`; the second also fails on the unknown `version_flag` kwarg).

- [ ] **Step 3: Write the implementation**

In `src/vidcp/cli.py`:

1. Give `_check_tool` a version flag parameter (ffmpeg/ffprobe use `-version`, yt-dlp uses `--version`) — change only the signature and the command list:

```python
def _check_tool(name: str, version_flag: str = "-version") -> tuple[bool, str]:
    path = shutil.which(name)
    if path is None:
        return False, "not found on PATH"
    try:
        result = subprocess.run(
            [name, version_flag],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    first_line = result.stdout.splitlines()[0] if result.stdout else path
    return True, first_line
```

2. In `doctor`, after the ffprobe row:

```python
    # Optional: only needed for `vidcp ingest <url>`. Absence renders as dim
    # info (status None), not FAIL, and never affects the exit code.
    ytdlp_ok, detail = _check_tool("yt-dlp", "--version")
    if ytdlp_ok:
        rows.append(("yt-dlp (optional)", True, detail))
    else:
        rows.append(("yt-dlp (optional)", None, f"{detail} — needed only for URL ingest"))
```

The final exit-code condition (`if not (ffmpeg_ok and ffprobe_ok and db_ok)`) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -m "not slow" -v`
Expected: PASS (both new tests and all pre-existing ones).

- [ ] **Step 5: Update the README**

In `README.md`:

1. Quickstart: after the `vidcp ingest talk.mp4` line, add:

```bash
vidcp ingest https://youtube.com/watch?v=abc   # download with yt-dlp, then ingest
```

2. The install paragraph ("Requires **Python 3.11+** and **ffmpeg/ffprobe**...") — extend the last sentence:

```markdown
Requires **Python 3.11+** and **ffmpeg/ffprobe** on your `PATH`; URL ingest
additionally needs **yt-dlp** on your `PATH`. Run `vidcp doctor` to check
your environment.
```

3. Constraints section: add a bullet:

```markdown
- **URL ingest** shells out to `yt-dlp` (`brew install yt-dlp` or
  `pipx install yt-dlp`); keep it current — video sites regularly break older
  extractor versions. Downloads are single videos only (no playlists), and
  the library stores the source URL as the video's path.
```

- [ ] **Step 6: Full fast suite, lint, and commit**

```bash
uv run pytest -m "not slow"
uv run ruff check . && uv run ruff format .
git add src/vidcp/cli.py tests/test_cli.py README.md
git commit -m "Add optional yt-dlp doctor row and URL ingest docs"
```

---

## Self-review notes

- Spec coverage: auto-detect + mixed invocations (Task 2), PATH-binary strategy and missing-binary `VidcpError` (Tasks 1–2), informational doctor row (Task 3), `origin`/`title` data model with no migration (Task 2), temp cleanup incl. failure paths (Task 2, `TemporaryDirectory`), skip-line/exit-code semantics (Task 2), playlist scope via `--no-playlist` (Task 1), no-network tests throughout, README docs (Task 3).
- `sync` and the MCP `ingest` tool are intentionally untouched (spec: out of scope).
