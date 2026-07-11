import constants as C

from vidcp.config import get_settings
from vidcp.db import connect
from vidcp.pipeline.base import Stage, VideoContext
from vidcp.pipeline.runner import run_pipeline
from vidcp.pipeline.stages.probe import ProbeStage
from vidcp.store import add_source, sha256_file


def _prepare_video(conn, src_path):
    vid = sha256_file(src_path)
    add_source(src_path, vid)
    conn.execute(
        "INSERT INTO videos(id, path, ingested_at, has_audio) VALUES (?, ?, ?, 1)",
        (vid, str(src_path), "2026-07-11T00:00:00"),
    )
    conn.commit()
    return vid


def test_probe_populates_video_row(fixtures):
    conn = connect()
    try:
        vid = _prepare_video(conn, fixtures["color.mp4"])
        run_pipeline(VideoContext(vid, conn, get_settings()), [ProbeStage()])
        row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        assert abs(row["duration_s"] - C.COLOR_DURATION_S) < 0.2
        assert row["width"] == C.COLOR_WIDTH
        assert row["height"] == C.COLOR_HEIGHT
        assert abs(row["fps"] - C.COLOR_FPS) < 0.5
        assert row["has_audio"] == 0  # color.mp4 is silent
        assert row["vcodec"] == "h264"
        assert row["title"] == "color"
        assert row["meta"] is not None
    finally:
        conn.close()


def test_probe_detects_audio(fixtures):
    conn = connect()
    try:
        vid = _prepare_video(conn, fixtures["cuts.mp4"])
        run_pipeline(VideoContext(vid, conn, get_settings()), [ProbeStage()])
        row = conn.execute("SELECT has_audio, acodec FROM videos WHERE id=?", (vid,)).fetchone()
        assert row["has_audio"] == 1
        assert row["acodec"] == "aac"
    finally:
        conn.close()


def test_runner_writes_stage_done(fixtures):
    conn = connect()
    try:
        vid = _prepare_video(conn, fixtures["color.mp4"])
        run_pipeline(VideoContext(vid, conn, get_settings()), [ProbeStage()])
        stage = conn.execute(
            "SELECT * FROM stages WHERE video_id=? AND stage='probe'", (vid,)
        ).fetchone()
        assert stage["status"] == "done"
        assert stage["started_at"] and stage["finished_at"]
        assert stage["config_hash"]
    finally:
        conn.close()


def test_runner_skips_completed_stage(fixtures):
    conn = connect()
    try:
        vid = _prepare_video(conn, fixtures["color.mp4"])
        ctx = VideoContext(vid, conn, get_settings())
        run_pipeline(ctx, [ProbeStage()])
        first = conn.execute(
            "SELECT finished_at FROM stages WHERE video_id=? AND stage='probe'", (vid,)
        ).fetchone()["finished_at"]
        run_pipeline(ctx, [ProbeStage()])  # done + config_hash match -> skip
        second = conn.execute(
            "SELECT finished_at FROM stages WHERE video_id=? AND stage='probe'", (vid,)
        ).fetchone()["finished_at"]
        assert first == second
    finally:
        conn.close()


def test_runner_marks_failed_stage(fixtures):
    class BoomStage(Stage):
        name = "boom"

        def run(self, ctx):
            raise RuntimeError("kaboom")

    conn = connect()
    try:
        vid = _prepare_video(conn, fixtures["color.mp4"])
        outcomes = run_pipeline(VideoContext(vid, conn, get_settings()), [BoomStage()])
        stage = conn.execute(
            "SELECT status, error FROM stages WHERE video_id=? AND stage='boom'", (vid,)
        ).fetchone()
        assert stage["status"] == "failed"
        assert "kaboom" in stage["error"]
        assert outcomes[0].status == "failed"
    finally:
        conn.close()
