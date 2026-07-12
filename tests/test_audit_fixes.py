"""Regression tests for issues found in the full code audit."""

import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.pipeline.stages.keyframes import _scene_timestamps
from vidcp.pipeline.stages.probe import _safe_float, _safe_int

runner = CliRunner()


def _only_vid(conn):
    return conn.execute("SELECT id FROM videos").fetchone()["id"]


def _count(conn, table, vid):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE video_id=?", (vid,)).fetchone()[0]


# --- fast unit tests ---


def test_scene_timestamps_nonpositive_interval_does_not_hang():
    # interval <= 0 must not spin forever; only the midpoint is returned.
    assert _scene_timestamps(0.0, 30.0, 0.0) == [(15.0, True)]
    assert _scene_timestamps(0.0, 30.0, -5.0) == [(15.0, True)]


def test_scene_timestamps_basic_and_long_scene():
    assert _scene_timestamps(0.0, 8.0, 10.0) == [(4.0, True)]
    long_scene = _scene_timestamps(0.0, 30.0, 10.0)
    assert long_scene[0] == (15.0, True)
    assert len(long_scene) > 1


def test_probe_safe_casts_tolerate_non_numeric():
    assert _safe_float("N/A") is None
    assert _safe_float(None) is None
    assert _safe_float("8.0") == 8.0
    assert _safe_int("N/A") is None
    assert _safe_int("1234") == 1234


def test_snippet_stays_aligned_when_lowercase_changes_length():
    from vidcp.search import _snippet

    # "İ".lower() is two code points, which would shift a lower()-based window
    # off the match; locating against the original text keeps it aligned.
    text = "İ" * 100 + " " + "x" * 30 + " TARGETWORD " + "y" * 200
    assert "TARGETWORD" in _snippet(text, "targetword")


# --- slow integration tests (need the embed model) ---


@pytest.mark.slow
def test_delete_clears_fts_and_vec(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["text.mp4"])])  # OCR -> fts + vec
    conn = connect()
    vid = _only_vid(conn)
    assert _count(conn, "fts", vid) > 0
    assert _count(conn, "vec", vid) > 0
    conn.close()

    runner.invoke(app, ["delete", vid[:8]])

    conn = connect()
    try:
        assert _count(conn, "fts", vid) == 0
        assert _count(conn, "vec", vid) == 0
    finally:
        conn.close()


@pytest.mark.slow
def test_disabling_ocr_clears_stale_vectors(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["text.mp4"])])
    conn = connect()
    vid = _only_vid(conn)
    assert _count(conn, "vec", vid) == 1  # one OCR block embedded
    conn.close()

    # text.mp4 has no audio; disabling OCR leaves nothing to embed, so the old
    # OCR vector must be cleared (not left stale/searchable).
    runner.invoke(app, ["ingest", "--force", "--no-ocr", str(fixtures["text.mp4"])])
    conn = connect()
    try:
        status = {
            r["stage"]: r["status"]
            for r in conn.execute("SELECT stage, status FROM stages WHERE video_id=?", (vid,))
        }
        assert status["ocr"] == "skipped"
        assert _count(conn, "ocr_blocks", vid) == 0
        assert _count(conn, "vec", vid) == 0
    finally:
        conn.close()
