import json

from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.config import get_settings
from vidcp.db import connect

runner = CliRunner()


def _ingest(path, *extra):
    result = runner.invoke(app, ["ingest", *extra, str(path)])
    assert result.exit_code == 0, result.output
    return result


def _only_vid(conn):
    return conn.execute("SELECT id FROM videos").fetchone()["id"]


def test_text_video_produces_single_ocr_block(fixtures):
    _ingest(fixtures["text.mp4"])
    conn = connect()
    try:
        vid = _only_vid(conn)
        blocks = conn.execute("SELECT text FROM ocr_blocks WHERE video_id=?", (vid,)).fetchall()
        assert len(blocks) == 1  # 8s video, one keyframe -> one block
        assert "VIDCP" in blocks[0]["text"]
    finally:
        conn.close()


def test_ocr_fts_match_returns_ocr_kind(fixtures):
    _ingest(fixtures["text.mp4"])
    conn = connect()
    try:
        hits = conn.execute("SELECT kind FROM fts WHERE fts MATCH 'VIDCP'").fetchall()
        assert any(h["kind"] == "ocr" for h in hits)
    finally:
        conn.close()


def test_no_ocr_flag_skips_stage(fixtures):
    _ingest(fixtures["text.mp4"], "--no-ocr")
    conn = connect()
    try:
        vid = _only_vid(conn)
        status = conn.execute(
            "SELECT status FROM stages WHERE video_id=? AND stage='ocr'", (vid,)
        ).fetchone()["status"]
        assert status == "skipped"
        assert (
            conn.execute("SELECT COUNT(*) FROM ocr_blocks WHERE video_id=?", (vid,)).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_ocr_disabled_setting_skips_stage(fixtures, monkeypatch):
    monkeypatch.setenv("VIDCP_OCR_ENABLED", "false")
    get_settings.cache_clear()
    _ingest(fixtures["text.mp4"])
    conn = connect()
    try:
        vid = _only_vid(conn)
        status = conn.execute(
            "SELECT status FROM stages WHERE video_id=? AND stage='ocr'", (vid,)
        ).fetchone()["status"]
        assert status == "skipped"
    finally:
        conn.close()


def test_inspect_reports_ocr_count(fixtures):
    _ingest(fixtures["text.mp4"])
    conn = connect()
    vid = _only_vid(conn)
    conn.close()
    data = json.loads(runner.invoke(app, ["inspect", vid[:8], "--json"]).output)
    assert data["counts"]["ocr_blocks"] == 1
