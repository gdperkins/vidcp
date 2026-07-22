import sqlite3
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.config import get_settings
from vidcp.errors import VidcpError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()

MIGRATION_001_TABLES = {
    "videos",
    "stages",
    "scenes",
    "segments",
    "ocr_blocks",
    "fts",
    "schema_version",
}


def test_vidcp_error_exits_nonzero_gracefully():
    # A user-facing VidcpError (unknown id, empty library) exits non-zero.
    result = runner.invoke(app, ["inspect", "deadbeef"])
    assert result.exit_code != 0
    assert isinstance(result.exception, VidcpError)
    assert "no video matches" in result.exception.message.lower()


def test_vidcp_error_renders_friendly_not_traceback():
    # The real console-script entrypoint must render VidcpError as a friendly
    # message (exit 1), never a Python traceback.
    result = subprocess.run(
        ["uv", "run", "vidcp", "inspect", "deadbeef"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "no video matches" in combined.lower()
    assert "Traceback" not in combined


def test_doctor_runs_and_exits_zero():
    # Acceptance: `uv run vidcp doctor` exits 0 on a machine with ffmpeg.
    result = subprocess.run(
        ["uv", "run", "vidcp", "doctor"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_doctor_creates_migration_001_database():
    # Acceptance: doctor creates <VIDCP_HOME>/library.db with all migration-001
    # tables (verified here with raw sqlite3). VIDCP_HOME is inherited from the
    # autouse fixture's environment by the subprocess.
    subprocess.run(
        ["uv", "run", "vidcp", "doctor"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    db_path = get_settings().db_path
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert MIGRATION_001_TABLES <= names


def test_mcp_command_registered():
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output


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
