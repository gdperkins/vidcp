"""Clip extraction tests. Seeds the artifact store directly (no pipeline run),
using the committed speech.mp4 fixture — needs ffmpeg/ffprobe but no models."""

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.clips import extract_clip
from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.store import artifact_dir
from vidcp.util import now_iso

VID = "d" * 64


def _seed_source(speech_fixture: Path) -> Path:
    dest = artifact_dir(VID) / f"source{speech_fixture.suffix}"
    shutil.copy2(speech_fixture, dest)
    return dest


def _duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def test_extract_clip_stream_copy(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    out = tmp_path / "clip.mp4"
    result = extract_clip(VID, 0.0, 1.5, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert (
        0.0 < _duration(out) <= 2.2
    )  # stream copy cuts on keyframes; allow slack; bound stays below fixture's 2.9s duration to catch dropped -t


def test_extract_clip_precise_reencode(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    out = tmp_path / "precise.mp4"
    extract_clip(VID, 0.0, 1.0, out, precise=True)
    assert out.exists() and out.stat().st_size > 0
    assert 0.0 < _duration(out) <= 1.6  # re-encode is frame-accurate


def test_extract_clip_invalid_range(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    with pytest.raises(VidcpError):
        extract_clip(VID, 5.0, 5.0, tmp_path / "x.mp4")
    with pytest.raises(VidcpError):
        extract_clip(VID, -1.0, 2.0, tmp_path / "y.mp4")


def test_extract_clip_missing_source(tmp_path):
    with pytest.raises(VidcpError):
        extract_clip("e" * 64, 0.0, 1.0, tmp_path / "z.mp4")


runner = CliRunner()


def _seed_video_row():
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos(id, path, title, ingested_at) VALUES (?,?,?,?)",
            (VID, "/videos/talk.mp4", "talk", now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def test_clip_command_writes_file(tmp_path, speech_fixture, monkeypatch):
    _seed_source(speech_fixture)
    _seed_video_row()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "moment.mp4"
    result = runner.invoke(
        app, ["clip", VID[:8], "--from", "0:00", "--to", "0:01.5", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0
    assert "wrote" in result.output


def test_clip_command_default_output_name(tmp_path, speech_fixture, monkeypatch):
    _seed_source(speech_fixture)
    _seed_video_row()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["clip", VID[:8], "--from", "0", "--to", "1"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / f"{VID[:8]}_0-1.mp4").exists()


def test_clip_command_bad_timestamp(speech_fixture):
    _seed_source(speech_fixture)
    _seed_video_row()
    result = runner.invoke(app, ["clip", VID[:8], "--from", "bogus", "--to", "5"])
    assert result.exit_code != 0
    # Typer 0.26 dispatches through its own vendored click fork (typer._click),
    # which only recognizes its own ClickException subclass; VidcpError (a
    # subclass of the top-level click.ClickException) is therefore never
    # caught internally by CliRunner's command dispatch, so .show() never
    # runs and nothing reaches result.output/result.stderr here. That's
    # consistent with every other VidcpError assertion in this suite (see
    # test_cli.py::test_vidcp_error_exits_nonzero_gracefully) — assert on the
    # captured exception itself; the real console-script entrypoint (main()
    # in cli.py) is what actually renders VidcpError.show() to stderr for
    # users, and that path is covered by
    # test_cli.py::test_vidcp_error_renders_friendly_not_traceback.
    assert isinstance(result.exception, VidcpError)
    assert "invalid timestamp" in result.exception.message.lower()
