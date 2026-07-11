"""vidcp command-line interface.

Only ``doctor`` is functional in Step 1; every other command is a placeholder
that raises :class:`~vidcp.errors.VidcpError` ("not implemented yet") so it
fails cleanly rather than with a traceback. Heavy libraries (whisper, rapidocr,
sentence-transformers) are never imported at module load time — this keeps CLI
startup fast.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from rich.table import Table

from vidcp import __version__
from vidcp.config import Settings, get_settings
from vidcp.db import connect
from vidcp.errors import VidcpError

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


def _not_implemented(name: str) -> None:
    raise VidcpError(
        f"`vidcp {name}` is not implemented yet.",
        hint="This command is delivered in a later step of the build.",
    )


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
    """Check that the environment is ready to run vidcp."""
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


@app.command()
def ingest(
    paths: Optional[list[str]] = typer.Argument(None, help="Video files to ingest."),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already present."),
) -> None:
    """Ingest one or more video files into the library."""
    _not_implemented("ingest")


@app.command("list")
def list_videos(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List ingested videos."""
    _not_implemented("list")


@app.command()
def inspect(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    stages: bool = typer.Option(False, "--stages", help="Include the stages table."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show details for a single video."""
    _not_implemented("inspect")


@app.command()
def delete(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    keep_artifacts: bool = typer.Option(False, "--keep-artifacts", help="Keep files on disk."),
) -> None:
    """Delete a video and its artifacts."""
    _not_implemented("delete")


@app.command()
def scenes(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List detected scenes for a video."""
    _not_implemented("scenes")


@app.command()
def transcript(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    fmt: str = typer.Option("txt", "--format", help="Output format: txt|srt|vtt|json."),
) -> None:
    """Show or export a video transcript."""
    _not_implemented("transcript")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    video_id: Optional[str] = typer.Option(None, "--id", help="Restrict to one video."),
    kind: Optional[str] = typer.Option(None, "--kind", help="Filter by kind: transcript|ocr."),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Hybrid keyword + semantic search across the library."""
    _not_implemented("search")


@app.command()
def reindex(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    stage: Optional[str] = typer.Option(None, "--stage", help="Single stage to rerun."),
    all_: bool = typer.Option(False, "--all", help="Full wipe and rerun."),
) -> None:
    """Rerun pipeline stages for a video."""
    _not_implemented("reindex")


@app.command()
def stats() -> None:
    """Show library statistics."""
    _not_implemented("stats")


@app.command()
def export(
    video_id: str = typer.Argument(..., help="Video id (any unique prefix)."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|markdown|srt|vtt."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Write to a file."),
) -> None:
    """Export a video's knowledge object."""
    _not_implemented("export")


# --------------------------------------------------------------------------- #
# console-script entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the CLI, rendering user-facing errors instead of tracebacks.

    This is the ``[project.scripts]`` target rather than the bare Typer ``app``:
    with ``pretty_exceptions_enable=False`` Typer re-raises ``ClickException``
    (including :class:`~vidcp.errors.VidcpError`) instead of calling ``.show()``,
    so we render it here. Unexpected exceptions print a short message unless
    ``--debug`` was passed.
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
