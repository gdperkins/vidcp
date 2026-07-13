"""embed_frames stage tests. The CLIP model is stubbed — no downloads."""

import pytest
from PIL import Image

from vidcp.config import get_settings
from vidcp.db import connect
from vidcp.pipeline.base import StageSkipped, VideoContext
from vidcp.pipeline.stages.embed_frames import EmbedFramesStage

VID = "a" * 64


class _StubClip:
    def encode(self, items, batch_size=16, normalize_embeddings=True, **kwargs):
        return [[0.1] * 512 for _ in items]


def _seed(conn, tmp_path, n_frames=2):
    conn.execute(
        "INSERT INTO videos(id, path, ingested_at) VALUES (?,?,?)",
        (VID, "/x.mp4", "2026-01-01T00:00:00"),
    )
    for i in range(n_frames):
        frame = tmp_path / f"f_{i}.jpg"
        Image.new("RGB", (32, 32), (i * 40, 10, 10)).save(frame, "JPEG")
        conn.execute(
            "INSERT INTO frames(video_id, ts_s, path, kept) VALUES (?,?,?,1)",
            (VID, float(i), str(frame)),
        )
    conn.commit()


def test_embed_frames_inserts_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDCP_CLIP_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr("vidcp.pipeline.stages.embed_frames.load_model", lambda name: _StubClip())
    conn = connect()
    try:
        _seed(conn, tmp_path)
        EmbedFramesStage().run(VideoContext(VID, conn, get_settings()))
        rows = conn.execute(
            "SELECT frame_id, ts_s FROM vec_frames WHERE video_id=? ORDER BY ts_s", (VID,)
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["ts_s"] == 0.0
    finally:
        conn.close()


def test_embed_frames_skips_when_disabled(tmp_path):
    conn = connect()  # conftest leaves VIDCP_CLIP_ENABLED=false
    try:
        _seed(conn, tmp_path)
        with pytest.raises(StageSkipped):
            EmbedFramesStage().run(VideoContext(VID, conn, get_settings()))
        assert conn.execute("SELECT COUNT(*) FROM vec_frames").fetchone()[0] == 0
    finally:
        conn.close()


def test_embed_frames_skips_without_frames(monkeypatch):
    monkeypatch.setenv("VIDCP_CLIP_ENABLED", "true")
    get_settings.cache_clear()
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos(id, path, ingested_at) VALUES (?,?,?)",
            (VID, "/x.mp4", "2026-01-01T00:00:00"),
        )
        conn.commit()
        with pytest.raises(StageSkipped):
            EmbedFramesStage().run(VideoContext(VID, conn, get_settings()))
    finally:
        conn.close()


def test_embed_frames_registered_in_default_stages():
    from vidcp.pipeline import default_stages, transitive_dependents

    stages = default_stages()
    names = [s.name for s in stages]
    assert "embed_frames" in names
    assert "embed_frames" in transitive_dependents(stages, "keyframes")
