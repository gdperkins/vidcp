"""vidcp command-line interface.

Functional so far: ``doctor`` (Step 1); ``ingest``/``list``/``inspect``/
``delete`` (Step 2); ``scenes`` (Step 3); ``transcript`` (Step 4);
``search`` (Step 6); ``reindex``/``stats`` (Step 7). ``export`` is the last
placeholder and raises :class:`~vidcp.errors.VidcpError` ("not implemented yet")
so it fails cleanly rather than with a traceback. Heavy libraries (whisper,
rapidocr, sentence-transformers) are never imported at module load time — this
keeps CLI startup fast.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from vidcp import __version__
from vidcp.config import Settings, get_settings
from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.export.srt import to_srt
from vidcp.export.vtt import to_vtt
from vidcp.library import artifact_counts, resolve_id
from vidcp.models import SceneRow, Segment, StageState, Video
from vidcp.pipeline import default_stages, transitive_dependents
from vidcp.pipeline.base import VideoContext
from vidcp.pipeline.runner import run_pipeline
from vidcp.pipeline.stages.probe import is_media_file
from vidcp.store import add_source, artifact_dir, sha256_file
from vidcp.util import format_duration, now_iso, parse_timestamp

app = typer.Typer(
    name="vidcp",
    help="Local-first CLI that turns video files into searchable knowledge.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

console = Console()

# Set from the global --debug flag; controls whether unexpected exceptions show
# a full traceback (see main()).
_DEBUG = False


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _expand_paths(raw: list[str]) -> list[Path]:
    """Expand directories to their contained video files; keep files as-is."""
    files: list[Path] = []
    for item in raw:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(f for f in path.rglob("*") if f.suffix.lower() in VIDEO_EXTS))
        else:
            files.append(path)
    return files


def _short_ts(value: str | None) -> str:
    return (value or "")[:19].replace("T", " ") if value else "-"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _db_size(db_path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{int(num)} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _run_with_progress(ctx, stages):
    """Run the pipeline with a live Rich progress bar (disabled off a TTY)."""
    tasks: dict[str, int] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    ) as progress:
        for stage in stages:
            tasks[stage.name] = progress.add_task(f"{stage.name}: pending", total=None)

        def callback(name: str, event: str) -> None:
            if name in tasks:
                progress.update(tasks[name], description=f"{name}: {event}")

        return run_pipeline(ctx, stages, progress=callback)


def _report_failures(outcomes, video_id: str) -> bool:
    """Print any failed stages + a retry hint. Return True if any failed."""
    failed = [o for o in outcomes if o.status == "failed"]
    if not failed:
        return False
    for outcome in failed:
        console.print(f"[red]{outcome.name} failed:[/red] {outcome.error}")
    console.print(
        f"[dim]Run `vidcp reindex {video_id[:8]} --stage {failed[0].name}` to retry.[/dim]"
    )
    return True


def _print_stages_table(stage_states: list[StageState]) -> None:
    table = Table(title="stages")
    for column in ("stage", "status", "started", "finished", "error"):
        table.add_column(column, overflow="fold")
    for state in stage_states:
        table.add_row(
            state.stage,
            state.status,
            _short_ts(state.started_at),
            _short_ts(state.finished_at),
            state.error or "-",
        )
    console.print(table)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vidcp {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the vidcp version and exit.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show full tracebacks instead of friendly errors.",
    ),
) -> None:
    """vidcp — local-first video knowledge extraction."""
    global _DEBUG
    _DEBUG = debug


# --------------------------------------------------------------------------- #
# doctor (functional)
# --------------------------------------------------------------------------- #


def _check_tool(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if path is None:
        return False, "not found on PATH"
    try:
        result = subprocess.run(
            [name, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    first_line = result.stdout.splitlines()[0] if result.stdout else path
    return True, first_line


def _check_home_writable(home: Path) -> tuple[bool, str]:
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".vidcp_write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return False, str(exc)
    return True, "ok"


def _check_db() -> tuple[bool, str]:
    try:
        conn = connect()
        try:
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version"
            ).fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return False, str(exc)
    return True, f"schema v{version}"


def _check_sqlite_vec() -> tuple[bool, str]:
    try:
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            version = conn.execute("SELECT vec_version()").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return False, str(exc)
    return True, str(version)


def _check_models(settings: Settings) -> list[tuple[str, str]]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    whisper_dir = hub / f"models--Systran--faster-whisper-{settings.whisper_model}"
    embed_slug = settings.embed_model.replace("/", "--")
    embed_dir = hub / f"models--{embed_slug}"
    downloaded = "downloaded"
    pending = "will download on first use"
    return [
        (
            f"whisper model ({settings.whisper_model})",
            downloaded if whisper_dir.exists() else pending,
        ),
        (
            f"embedding model ({settings.embed_model})",
            downloaded if embed_dir.exists() else pending,
        ),
    ]


def _render_doctor_table(rows: list[tuple[str, Optional[bool], str]]) -> None:
    table = Table(title="vidcp doctor")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for name, ok, detail in rows:
        if ok is True:
            status = "[green]OK[/green]"
        elif ok is False:
            status = "[red]FAIL[/red]"
        else:
            status = "[dim]info[/dim]"
        table.add_row(name, status, detail)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that the environment is ready to run vidcp.

    Example: vidcp doctor
    """
    settings = get_settings()
    rows: list[tuple[str, Optional[bool], str]] = []

    ffmpeg_ok, detail = _check_tool("ffmpeg")
    rows.append(("ffmpeg", ffmpeg_ok, detail))

    ffprobe_ok, detail = _check_tool("ffprobe")
    rows.append(("ffprobe", ffprobe_ok, detail))

    home_ok, detail = _check_home_writable(settings.home)
    rows.append((f"home writable ({settings.home})", home_ok, detail))

    db_ok, detail = _check_db()
    rows.append(("database + migrations", db_ok, detail))

    vec_ok, detail = _check_sqlite_vec()
    rows.append(("sqlite-vec extension", vec_ok, detail))

    for label, model_detail in _check_models(settings):
        rows.append((label, None, model_detail))

    _render_doctor_table(rows)

    if not (ffmpeg_ok and ffprobe_ok and db_ok):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# placeholder commands (implemented in later steps)
# --------------------------------------------------------------------------- #


def _ingest_one(conn, path: Path, settings: Settings, force: bool) -> str:
    """Ingest a single file. Returns:
    'ingested' | 'failed_stages' | 'already' | 'missing' | 'not_media'."""
    if not path.exists():
        return "missing"
    if not is_media_file(path):
        return "not_media"
    video_id = sha256_file(path)
    existing = conn.execute("SELECT id FROM videos WHERE id=?", (video_id,)).fetchone()
    if existing and not force:
        console.print(f"already ingested {video_id[:8]}  ({path.name})")
        return "already"
    add_source(path, video_id)
    if existing is None:
        conn.execute(
            "INSERT INTO videos(id, path, ingested_at, has_audio) VALUES (?, ?, ?, 1)",
            (video_id, str(path.resolve()), now_iso()),
        )
        conn.commit()
    outcomes = _run_with_progress(VideoContext(video_id, conn, settings), default_stages())
    if _report_failures(outcomes, video_id):
        console.print(f"[yellow]ingested with errors[/yellow] {video_id[:8]}  {path.name}")
        return "failed_stages"
    console.print(f"[green]ingested[/green] {video_id[:8]}  {path.name}")
    return "ingested"


@app.command()
def ingest(
    paths: Optional[list[str]] = typer.Argument(None, help="Video files or directories."),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already present."),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model", help="Override the whisper model for this run."
    ),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Skip OCR for this run."),
) -> None:
    """Ingest one or more video files into the library.

    Example: vidcp ingest clip.mp4 ~/Movies
    """
    if not paths:
        raise VidcpError("no paths given", hint="usage: vidcp ingest <file-or-dir> ...")
    settings = get_settings()
    overrides: dict[str, object] = {}
    if whisper_model:
        overrides["whisper_model"] = whisper_model
    if no_ocr:
        overrides["ocr_enabled"] = False
    if overrides:
        # A per-run copy so the overrides flow into stage config fingerprints.
        settings = settings.model_copy(update=overrides)
    files = _expand_paths(paths)
    if not files:
        raise VidcpError("no video files found in the given paths")

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
    finally:
        conn.close()

    if had_failures:
        raise typer.Exit(code=2)  # partial success
    if errors and not ingested:
        raise typer.Exit(code=1)


@app.command("list")
def list_videos(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List ingested videos.

    Example: vidcp list --json
    """
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC").fetchall()
    finally:
        conn.close()
    videos = [Video.from_row(row) for row in rows]

    if json_output:
        print(json.dumps([v.model_dump(mode="json") for v in videos]))
        return
    if not videos:
        console.print("No videos ingested yet. Run [bold]vidcp ingest <file>[/bold].")
        return

    table = Table(title="videos")
    table.add_column("id")
    table.add_column("title", overflow="fold")
    table.add_column("duration", justify="right")
    table.add_column("resolution")
    table.add_column("ingested")
    for video in videos:
        resolution = f"{video.width}x{video.height}" if video.width and video.height else "-"
        table.add_row(
            video.short_id,
            video.title or "-",
            format_duration(video.duration_s),
            resolution,
            _short_ts(video.ingested_at),
        )
    console.print(table)


@app.command()
def inspect(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    stages: bool = typer.Option(False, "--stages", help="Include the stages table."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show details for a single video.

    Example: vidcp inspect a1b2c3d4 --stages
    """
    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        video = Video.from_row(row)
        counts = artifact_counts(conn, vid)
        stage_states = [
            StageState.from_row(r)
            for r in conn.execute(
                "SELECT * FROM stages WHERE video_id=? ORDER BY stage", (vid,)
            ).fetchall()
        ]
    finally:
        conn.close()

    if json_output:
        payload = video.model_dump(mode="json")
        payload["counts"] = counts
        if stages:
            payload["stages"] = [s.model_dump(mode="json") for s in stage_states]
        print(json.dumps(payload))
        return

    table = Table(title=f"video {video.short_id}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")
    data = video.model_dump(mode="json")
    data.pop("meta", None)  # verbose ffprobe blob; omit from the human view
    for key, value in data.items():
        table.add_row(key, "-" if value is None else str(value))
    for key, value in counts.items():
        table.add_row(key, str(value))
    console.print(table)
    if stages:
        _print_stages_table(stage_states)


@app.command()
def delete(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    keep_artifacts: bool = typer.Option(False, "--keep-artifacts", help="Keep files on disk."),
) -> None:
    """Delete a video and its artifacts.

    Example: vidcp delete a1b2c3d4 --keep-artifacts
    """
    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        # fts and vec are virtual tables and can't carry FK constraints, so their
        # rows must be deleted explicitly (leaving them orphaned would inflate
        # stats and, via rowid reuse, mis-attribute future search hits).
        conn.execute("DELETE FROM fts WHERE video_id=?", (vid,))
        conn.execute("DELETE FROM vec WHERE video_id=?", (vid,))
        # FK cascade removes stages/scenes/segments/ocr_blocks/frames.
        conn.execute("DELETE FROM videos WHERE id=?", (vid,))
        conn.commit()
    finally:
        conn.close()
    if not keep_artifacts:
        shutil.rmtree(artifact_dir(vid, create=False), ignore_errors=True)
    console.print(f"deleted {vid[:8]}")


@app.command()
def scenes(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List detected scenes for a video.

    Example: vidcp scenes a1b2c3d4
    """
    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        rows = conn.execute("SELECT * FROM scenes WHERE video_id=? ORDER BY idx", (vid,)).fetchall()
        scene_models = [SceneRow.from_row(r) for r in rows]
    finally:
        conn.close()

    if json_output:
        print(json.dumps([s.model_dump(mode="json") for s in scene_models]))
        return
    if not scene_models:
        console.print("No scenes. Has this video been ingested?")
        return

    table = Table(title=f"scenes · {vid[:8]}")
    table.add_column("idx", justify="right")
    table.add_column("start", justify="right")
    table.add_column("end", justify="right")
    table.add_column("duration", justify="right")
    table.add_column("keyframe")
    for scene in scene_models:
        keyframe = Path(scene.keyframe_path).name if scene.keyframe_path else "-"
        table.add_row(
            str(scene.idx),
            f"{scene.start_s:.2f}",
            f"{scene.end_s:.2f}",
            format_duration(scene.end_s - scene.start_s),
            keyframe,
        )
    console.print(table)


@app.command()
def transcript(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    fmt: str = typer.Option("txt", "--format", help="Output format: txt|srt|vtt|json."),
) -> None:
    """Show or export a video transcript.

    Example: vidcp transcript a1b2c3d4 --format srt
    """
    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        rows = conn.execute(
            "SELECT * FROM segments WHERE video_id=? ORDER BY start_s", (vid,)
        ).fetchall()
        segments = [Segment.from_row(r) for r in rows]
    finally:
        conn.close()

    if not segments:
        console.print("No transcript: no speech detected (or the video has no audio).")
        return

    if fmt == "srt":
        print(to_srt(segments))
    elif fmt == "vtt":
        print(to_vtt(segments))
    elif fmt == "json":
        print(json.dumps([s.model_dump(mode="json") for s in segments]))
    elif fmt == "txt":
        for seg in segments:
            print(f"[{format_duration(seg.start_s)}] {seg.text}")
    else:
        raise VidcpError(
            f"unknown transcript format '{fmt}'",
            hint="choose one of: txt, srt, vtt, json",
        )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    video_id: Optional[str] = typer.Option(None, "--id", help="Restrict to one video."),
    kind: Optional[str] = typer.Option(None, "--kind", help="Filter by kind: transcript|ocr."),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Hybrid keyword + semantic search across the library.

    Example: vidcp search "machine learning" --kind transcript
    """
    from vidcp.search import search as run_search

    if kind is not None and kind not in ("transcript", "ocr"):
        raise VidcpError(f"unknown kind '{kind}'", hint="choose one of: transcript, ocr")

    conn = connect()
    try:
        vid = resolve_id(conn, video_id) if video_id else None
        hits = run_search(conn, query, video_id=vid, kind=kind, limit=limit)
    finally:
        conn.close()

    if json_output:
        print(json.dumps([h.model_dump(mode="json") for h in hits]))
        return
    if not hits:
        console.print("no matches")
        return

    table = Table(title=f"search · {query!r}")
    table.add_column("id")
    table.add_column("time", justify="right")
    table.add_column("kind")
    table.add_column("snippet", overflow="fold")
    for hit in hits:
        table.add_row(hit.short_id, format_duration(hit.ts_s), hit.kind, hit.snippet)
    console.print(table)


@app.command()
def reindex(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    stage: Optional[str] = typer.Option(None, "--stage", help="Stage (+ dependents) to rerun."),
    all_: bool = typer.Option(False, "--all", help="Full wipe and rerun."),
) -> None:
    """Rerun pipeline stages for a video.

    Example: vidcp reindex a1b2c3d4 --stage scenes
    """
    settings = get_settings()
    stages = default_stages()
    by_name = {s.name: s for s in stages}
    if stage is not None and stage not in by_name:
        raise VidcpError(f"unknown stage '{stage}'", hint="one of: " + ", ".join(by_name))

    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        ctx = VideoContext(vid, conn, settings)

        if all_:
            targets: Optional[set[str]] = {s.name for s in stages}
        elif stage is not None:
            targets = transitive_dependents(stages, stage)
        else:
            targets = None  # rerun everything not already done

        if targets is not None:
            for name in targets:
                by_name[name].clean(ctx)
                conn.execute("DELETE FROM stages WHERE video_id=? AND stage=?", (vid, name))
            conn.commit()

        outcomes = _run_with_progress(ctx, stages)
    finally:
        conn.close()

    if _report_failures(outcomes, vid):
        raise typer.Exit(code=2)
    console.print(f"reindexed {vid[:8]}")


@app.command()
def stats(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show library statistics.

    Example: vidcp stats
    """
    settings = get_settings()
    conn = connect()
    try:
        video_count, total_duration = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration_s), 0) FROM videos"
        ).fetchone()
        scenes = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        ocr_blocks = conn.execute("SELECT COUNT(*) FROM ocr_blocks").fetchone()[0]
        vec_rows = conn.execute("SELECT COUNT(*) FROM vec").fetchone()[0]
    finally:
        conn.close()

    store_bytes = _dir_size(settings.store_path)
    db_bytes = _db_size(settings.db_path)

    if json_output:
        print(
            json.dumps(
                {
                    "videos": video_count,
                    "total_duration_s": total_duration,
                    "scenes": scenes,
                    "segments": segments,
                    "ocr_blocks": ocr_blocks,
                    "vec_rows": vec_rows,
                    "store_bytes": store_bytes,
                    "db_bytes": db_bytes,
                }
            )
        )
        return

    table = Table(title="vidcp stats", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("videos", str(video_count))
    table.add_row("total duration", format_duration(total_duration))
    table.add_row("scenes", str(scenes))
    table.add_row("segments", str(segments))
    table.add_row("ocr blocks", str(ocr_blocks))
    table.add_row("vec rows", str(vec_rows))
    table.add_row("store size", _human_size(store_bytes))
    table.add_row("db size", _human_size(db_bytes))
    console.print(table)


@app.command()
def export(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|markdown|srt|vtt."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Write to a file."),
) -> None:
    """Export a video's knowledge object.

    Example: vidcp export a1b2c3d4 --format markdown -o notes.md
    """
    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
        if fmt == "json":
            from vidcp.export.json import to_export_dict

            content = json.dumps(to_export_dict(conn, vid), indent=2)
        elif fmt == "markdown":
            from vidcp.export.markdown import to_markdown

            content = to_markdown(conn, vid)
        elif fmt in ("srt", "vtt"):
            segments = [
                Segment.from_row(r)
                for r in conn.execute(
                    "SELECT * FROM segments WHERE video_id=? ORDER BY start_s", (vid,)
                )
            ]
            content = to_srt(segments) if fmt == "srt" else to_vtt(segments)
        else:
            raise VidcpError(
                f"unknown export format '{fmt}'",
                hint="choose one of: json, markdown, srt, vtt",
            )
    finally:
        conn.close()

    if output:
        Path(output).write_text(content)
        console.print(f"wrote {output}")
    else:
        print(content)


@app.command()
def clip(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    from_ts: str = typer.Option(..., "--from", help="Clip start (seconds, mm:ss, or h:mm:ss)."),
    to_ts: str = typer.Option(..., "--to", help="Clip end (seconds, mm:ss, or h:mm:ss)."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file path."),
    precise: bool = typer.Option(
        False, "--precise", help="Re-encode for frame-accurate cuts (slower)."
    ),
) -> None:
    """Extract a clip from a video into a standalone MP4.

    Example: vidcp clip a1b2c3d4 --from 1:23 --to 1:45 -o moment.mp4
    """
    from vidcp.clips import extract_clip

    try:
        start_s = parse_timestamp(from_ts)
        end_s = parse_timestamp(to_ts)
    except ValueError as exc:
        raise VidcpError(str(exc), hint="use seconds, mm:ss, or h:mm:ss") from None

    conn = connect()
    try:
        vid = resolve_id(conn, video_id)
    finally:
        conn.close()

    out = Path(output) if output else Path(f"{vid[:8]}_{start_s:g}-{end_s:g}.mp4")
    path = extract_clip(vid, start_s, end_s, out, precise=precise)
    console.print(f"wrote {path}")


@app.command("mcp")
def mcp_command() -> None:
    """Run the MCP server over stdio, exposing the library to agents.

    Example: claude mcp add vidcp -- vidcp mcp
    """
    from vidcp.mcp_server import create_server

    create_server().run()


# --------------------------------------------------------------------------- #
# console-script entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the CLI, rendering user-facing errors instead of tracebacks.

    This is the ``[project.scripts]`` target rather than the bare Typer ``app``
    so that *unexpected* exceptions become a short friendly message (exit 1)
    unless ``--debug`` was passed. ``VidcpError`` (a ``ClickException``) is
    already rendered by Click's standalone handling via its ``.show()``; the
    ``except ClickException`` branch below is a defensive fallback.
    """
    try:
        app()
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except Exception as exc:
        if _DEBUG:
            raise
        err = Console(stderr=True)
        err.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        err.print("[dim]Re-run with --debug for a full traceback.[/dim]")
        raise SystemExit(1) from None
