"""vidcp sync tests. Uses color.mp4 (no audio) with --no-ocr so the whole
pipeline completes without loading any ML model."""

import json
import shutil

from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.library import pipeline_complete

runner = CliRunner()


def _video_dir(tmp_path, fixtures):
    d = tmp_path / "videos"
    d.mkdir()
    shutil.copy2(fixtures["color.mp4"], d / "color.mp4")
    (d / "notes.txt").write_text("not a video")
    return d


def test_sync_dry_run_reports_without_ingesting(tmp_path, fixtures):
    d = _video_dir(tmp_path, fixtures)
    result = runner.invoke(app, ["sync", "--dry-run", "--no-ocr", str(d)])
    assert result.exit_code == 0, result.output
    assert "new" in result.output
    conn = connect()
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    conn.close()


def test_sync_ingests_then_reports_up_to_date(tmp_path, fixtures):
    d = _video_dir(tmp_path, fixtures)
    first = runner.invoke(app, ["sync", "--no-ocr", str(d)])
    assert first.exit_code == 0, first.output
    conn = connect()
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    conn.close()

    second = runner.invoke(app, ["sync", "--no-ocr", "--json", str(d)])
    assert second.exit_code == 0, second.output
    data = json.loads(second.output)
    assert data["new"] == 0
    assert data["up_to_date"] == 1


def test_sync_rejects_plain_files(tmp_path, fixtures):
    f = tmp_path / "clip.mp4"
    shutil.copy2(fixtures["color.mp4"], f)
    result = runner.invoke(app, ["sync", str(f)])
    assert result.exit_code != 0
    # A VidcpError is a real click.ClickException, but Typer's runner only
    # renders its own vendored ClickException subclass via .show(); matching
    # the established pattern in test_cli.py, assert on result.exception
    # instead of result.output (which stays empty for this exception type).
    assert isinstance(result.exception, VidcpError)
    assert "not a directory" in result.exception.message


def test_pipeline_complete_helper():
    conn = connect()
    try:
        vid = "f" * 64
        conn.execute(
            "INSERT INTO videos(id, path, ingested_at) VALUES (?,?,?)",
            (vid, "/x.mp4", "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO stages(video_id, stage, status) VALUES (?,?,?)", (vid, "probe", "done")
        )
        conn.commit()
        assert pipeline_complete(conn, vid, ["probe"])
        assert not pipeline_complete(conn, vid, ["probe", "scenes"])  # missing row
        conn.execute(
            "INSERT INTO stages(video_id, stage, status) VALUES (?,?,?)",
            (vid, "scenes", "failed"),
        )
        conn.commit()
        assert not pipeline_complete(conn, vid, ["probe", "scenes"])  # failed row
    finally:
        conn.close()
