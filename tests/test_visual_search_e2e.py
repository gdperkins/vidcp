"""End-to-end visual search with the real CLIP model. Slow: downloads
clip-ViT-B-32 (~600 MB) on first run. Uses color.mp4 (animated test pattern,
frames survive phash dedupe). Ranking quality is exercised manually — this
test proves the plumbing: ingest -> embed_frames -> visual hits with real
frame paths."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.config import get_settings
from vidcp.db import connect
from vidcp.search import search

runner = CliRunner()


@pytest.mark.slow
def test_visual_search_end_to_end(fixtures, monkeypatch):
    monkeypatch.setenv("VIDCP_CLIP_ENABLED", "true")
    get_settings.cache_clear()

    result = runner.invoke(app, ["ingest", "--no-ocr", str(fixtures["color.mp4"])])
    assert result.exit_code == 0, result.output

    conn = connect()
    try:
        stage = conn.execute("SELECT status FROM stages WHERE stage='embed_frames'").fetchone()
        assert stage["status"] == "done"
        assert conn.execute("SELECT COUNT(*) FROM vec_frames").fetchone()[0] > 0

        hits = search(conn, "a colorful test pattern", kind="visual", limit=5)
        assert hits
        assert all(h.kind == "visual" for h in hits)
        assert all(h.frame_path and Path(h.frame_path).exists() for h in hits)
    finally:
        conn.close()

    # CLI --json surface carries frame_path through.
    out = runner.invoke(app, ["search", "test pattern", "--kind", "visual", "--json"]).output
    data = json.loads(out)
    assert data and data[0]["frame_path"]
