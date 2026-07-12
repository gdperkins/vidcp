import json
from datetime import datetime

import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.pipeline import default_stages
from vidcp.pipeline.base import Stage
from vidcp.pipeline.runner import topological_order
from vidcp.pipeline.stages.transcribe import TranscribeStage

runner = CliRunner()


def _only_vid(conn):
    return conn.execute("SELECT id FROM videos").fetchone()["id"]


def _stages(conn, vid):
    return {
        r["stage"]: (r["status"], r["started_at"], r["finished_at"])
        for r in conn.execute(
            "SELECT stage, status, started_at, finished_at FROM stages WHERE video_id=?",
            (vid,),
        )
    }


def _overlap(a, b) -> bool:
    a0, a1 = datetime.fromisoformat(a[0]), datetime.fromisoformat(a[1])
    b0, b1 = datetime.fromisoformat(b[0]), datetime.fromisoformat(b[1])
    return a0 < b1 and b0 < a1


# --- DAG (fast) ---


def test_default_pipeline_dag_is_acyclic():
    order = topological_order(default_stages())
    seen: set[str] = set()
    for stage in order:
        assert all(dep in seen for dep in stage.depends_on), stage.name
        seen.add(stage.name)


def test_topological_order_detects_cycle():
    class A(Stage):
        name = "a"
        depends_on = ["b"]

        def run(self, ctx): ...

    class B(Stage):
        name = "b"
        depends_on = ["a"]

        def run(self, ctx): ...

    with pytest.raises(ValueError, match="cycle"):
        topological_order([A(), B()])


# --- resumability / reindex / parallelism / stats (slow) ---


@pytest.mark.slow
def test_kill_resume_retries_failed_transcribe(speech_fixture, monkeypatch):
    original = TranscribeStage.run
    state = {"fail": True}

    def maybe_fail(self, ctx):
        if state["fail"]:
            raise RuntimeError("boom")
        return original(self, ctx)

    monkeypatch.setattr(TranscribeStage, "run", maybe_fail)

    r1 = runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(speech_fixture)])
    assert r1.exit_code == 2, r1.output  # partial success

    conn = connect()
    vid = _only_vid(conn)
    before = _stages(conn, vid)
    conn.close()
    assert before["audio"][0] == "done"
    assert before["transcribe"][0] == "failed"

    state["fail"] = False
    runner.invoke(app, ["ingest", "--force", "--whisper-model", "tiny", str(speech_fixture)])

    conn = connect()
    after = _stages(conn, vid)
    conn.close()
    assert after["audio"][0] == "done"
    assert after["audio"][2] == before["audio"][2]  # audio NOT re-run
    assert after["transcribe"][0] == "done"  # transcribe retried + succeeded


@pytest.mark.slow
def test_reindex_stage_scenes_reruns_dependents_not_transcribe(speech_fixture):
    runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(speech_fixture)])
    conn = connect()
    vid = _only_vid(conn)
    before = _stages(conn, vid)
    conn.close()

    result = runner.invoke(app, ["reindex", vid[:8], "--stage", "scenes"])
    assert result.exit_code == 0, result.output

    conn = connect()
    after = _stages(conn, vid)
    conn.close()
    for stage in ("scenes", "keyframes", "ocr", "embed"):
        assert after[stage][2] != before[stage][2], f"{stage} should have re-run"
    for stage in ("probe", "audio", "transcribe"):
        assert after[stage][2] == before[stage][2], f"{stage} should NOT have re-run"


@pytest.mark.slow
def test_audio_overlaps_scenes(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["cuts.mp4"])])
    conn = connect()
    vid = _only_vid(conn)
    stages = _stages(conn, vid)
    conn.close()
    audio = (stages["audio"][1], stages["audio"][2])
    scenes = (stages["scenes"][1], stages["scenes"][2])
    assert _overlap(audio, scenes), (audio, scenes)


@pytest.mark.slow
def test_stats_matches_sql(fixtures, speech_fixture):
    for path in (speech_fixture, fixtures["text.mp4"]):
        runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(path)])

    data = json.loads(runner.invoke(app, ["stats", "--json"]).output)
    conn = connect()
    try:
        for key, table in [
            ("videos", "videos"),
            ("scenes", "scenes"),
            ("segments", "segments"),
            ("ocr_blocks", "ocr_blocks"),
            ("vec_rows", "vec"),
        ]:
            assert data[key] == conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()
    assert data["videos"] == 2
    assert data["store_bytes"] > 0
    assert data["db_bytes"] > 0
