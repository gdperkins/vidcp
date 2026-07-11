import json
from pathlib import Path

import constants as C
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


def _stage_finished(conn, vid):
    return {
        r["stage"]: r["finished_at"]
        for r in conn.execute("SELECT stage, finished_at FROM stages WHERE video_id=?", (vid,))
    }


def test_cuts_yields_four_scenes_at_known_boundaries(fixtures):
    _ingest(fixtures["cuts.mp4"])
    conn = connect()
    try:
        vid = _only_vid(conn)
        scenes = conn.execute(
            "SELECT start_s, end_s FROM scenes WHERE video_id=? ORDER BY idx", (vid,)
        ).fetchall()
        assert len(scenes) == C.CUTS_SCENE_COUNT  # 4
        starts = [s["start_s"] for s in scenes]
        for cut in C.CUTS_CUT_POINTS_S:  # (2, 4, 6)
            assert any(abs(start - cut) < 0.5 for start in starts), (cut, starts)
    finally:
        conn.close()


def test_color_yields_one_scene_and_one_kept_frame(fixtures):
    _ingest(fixtures["color.mp4"])
    conn = connect()
    try:
        vid = _only_vid(conn)
        scenes = conn.execute("SELECT * FROM scenes WHERE video_id=?", (vid,)).fetchall()
        assert len(scenes) == 1
        frames = conn.execute("SELECT * FROM frames WHERE video_id=?", (vid,)).fetchall()
        assert len(frames) == 1  # 8s video, 10s floor -> a single kept frame
        assert Path(frames[0]["path"]).exists()  # JPEG on disk
        assert scenes[0]["keyframe_path"] == frames[0]["path"]  # denormalised pointer
        assert scenes[0]["phash"] is not None
    finally:
        conn.close()


def test_reingest_force_skips_unchanged_stages(fixtures):
    _ingest(fixtures["color.mp4"])
    conn = connect()
    vid = _only_vid(conn)
    before = _stage_finished(conn, vid)
    conn.close()

    # --force re-runs the pipeline; unchanged stages must skip (finished_at same).
    _ingest(fixtures["color.mp4"], "--force")

    conn = connect()
    after = _stage_finished(conn, vid)
    conn.close()
    for stage in ("probe", "scenes", "keyframes"):
        assert before[stage] == after[stage], stage


def test_threshold_change_reruns_scenes_and_keyframes(fixtures, monkeypatch):
    _ingest(fixtures["cuts.mp4"])
    conn = connect()
    vid = _only_vid(conn)
    before = _stage_finished(conn, vid)
    conn.close()

    monkeypatch.setenv("VIDCP_SCENE_THRESHOLD", "5")
    get_settings.cache_clear()
    _ingest(fixtures["cuts.mp4"], "--force")

    conn = connect()
    after = _stage_finished(conn, vid)
    conn.close()
    assert after["probe"] == before["probe"]  # unaffected by threshold
    assert after["scenes"] != before["scenes"]  # config_hash invalidated
    assert after["keyframes"] != before["keyframes"]  # downstream invalidation


def test_scenes_command_json(fixtures):
    _ingest(fixtures["cuts.mp4"])
    conn = connect()
    vid = _only_vid(conn)
    conn.close()
    data = json.loads(runner.invoke(app, ["scenes", vid[:8], "--json"]).output)
    assert len(data) == C.CUTS_SCENE_COUNT
    assert data[0]["start_s"] == 0.0
    assert data[-1]["end_s"] > 0
