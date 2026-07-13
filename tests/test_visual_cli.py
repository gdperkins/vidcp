"""CLI integration for visual search: kind validation, delete, stats, doctor."""

import json

import sqlite_vec
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.errors import VidcpError

runner = CliRunner()

VID = "a" * 64


def _seed_frame_vector():
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos(id, path, ingested_at) VALUES (?,?,?)",
            (VID, "/x.mp4", "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO vec_frames(embedding, video_id, frame_id, ts_s) VALUES (?,?,?,?)",
            (sqlite_vec.serialize_float32([0.5] * 512), VID, 1, 3.0),
        )
        conn.commit()
    finally:
        conn.close()


def test_search_rejects_unknown_kind():
    result = runner.invoke(app, ["search", "x", "--kind", "bogus"])
    assert result.exit_code != 0
    assert isinstance(result.exception, VidcpError)
    assert "transcript, ocr, visual" in result.exception.hint


def test_stats_counts_frame_vectors():
    _seed_frame_vector()
    data = json.loads(runner.invoke(app, ["stats", "--json"]).output)
    assert data["vec_frames"] == 1


def test_delete_clears_frame_vectors():
    _seed_frame_vector()
    result = runner.invoke(app, ["delete", VID[:8]])
    assert result.exit_code == 0, result.output
    conn = connect()
    assert conn.execute("SELECT COUNT(*) FROM vec_frames").fetchone()[0] == 0
    conn.close()


def test_doctor_lists_clip_model():
    result = runner.invoke(app, ["doctor"])
    assert "clip model" in result.output.lower()
