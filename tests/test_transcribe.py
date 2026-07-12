import re

import constants as C
import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.config import get_settings
from vidcp.db import connect
from vidcp.pipeline.stages.transcribe import TranscribeStage

runner = CliRunner()


def _ingest(path, *extra):
    result = runner.invoke(app, ["ingest", *extra, str(path)])
    assert result.exit_code == 0, result.output
    return result


def _only_vid(conn):
    return conn.execute("SELECT id FROM videos").fetchone()["id"]


def _stage_status(conn, vid):
    return {
        r["stage"]: r["status"]
        for r in conn.execute("SELECT stage, status FROM stages WHERE video_id=?", (vid,))
    }


# --- fast: silent video cascades to skipped (no whisper loaded) ---


def test_silent_video_skips_audio_and_transcribe(fixtures):
    _ingest(fixtures["color.mp4"])
    conn = connect()
    try:
        vid = _only_vid(conn)
        status = _stage_status(conn, vid)
        assert status["audio"] == "skipped"
        assert status["transcribe"] == "skipped"
        assert (
            conn.execute("SELECT COUNT(*) FROM segments WHERE video_id=?", (vid,)).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_transcript_command_reports_no_speech(fixtures):
    _ingest(fixtures["color.mp4"])
    conn = connect()
    vid = _only_vid(conn)
    conn.close()
    result = runner.invoke(app, ["transcript", vid[:8]])
    assert result.exit_code == 0
    assert "no speech" in result.output.lower()


def test_whisper_model_override_changes_config_fingerprint():
    settings = get_settings()
    other = settings.model_copy(update={"whisper_model": "base"})
    stage = TranscribeStage()
    assert stage.config_hash(settings) != stage.config_hash(other)


# --- slow: real transcription of committed speech.mp4 with the tiny model ---


@pytest.mark.slow
def test_speech_transcription_fts_and_srt(speech_fixture):
    _ingest(speech_fixture, "--whisper-model", "tiny")
    conn = connect()
    try:
        vid = _only_vid(conn)
        status = _stage_status(conn, vid)
        assert status["audio"] == "done"
        assert status["transcribe"] == "done"

        segs = conn.execute(
            "SELECT text FROM segments WHERE video_id=? ORDER BY start_s", (vid,)
        ).fetchall()
        joined = " ".join(s["text"] for s in segs).lower()
        assert C.SPEECH_EXPECTED_SUBSTRING in joined  # "natural language"

        hits = conn.execute(
            "SELECT kind FROM fts WHERE fts MATCH ?", (C.SPEECH_KEYWORD,)
        ).fetchall()  # "language"
        assert any(h["kind"] == "transcript" for h in hits)
    finally:
        conn.close()

    out = runner.invoke(app, ["transcript", vid[:8], "--format", "srt"]).output
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", out)
    assert out.strip().splitlines()[0] == "1"
