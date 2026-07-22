import json

import constants as C
import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.export.json import to_export_dict
from vidcp.export.markdown import to_markdown

runner = CliRunner()


def _only_vid(conn):
    return conn.execute("SELECT id FROM videos").fetchone()["id"]


@pytest.mark.slow
def test_json_export_has_all_sections(speech_fixture):
    runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(speech_fixture)])
    conn = connect()
    try:
        vid = _only_vid(conn)
        obj = to_export_dict(conn, vid)
    finally:
        conn.close()

    assert obj["vidcp_export_version"] == 1
    assert obj["video"]["id"] == vid
    for section in ("scenes", "segments", "ocr_blocks"):
        assert section in obj
    assert len(obj["scenes"]) >= 1
    assert "frames" in obj["scenes"][0]
    assert len(obj["segments"]) >= 1
    assert C.SPEECH_EXPECTED_SUBSTRING in obj["segments"][0]["text"].lower()
    # round-trips as JSON
    assert json.loads(json.dumps(obj))["vidcp_export_version"] == 1


@pytest.mark.slow
def test_markdown_export_renders_sections(speech_fixture):
    runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(speech_fixture)])
    conn = connect()
    try:
        vid = _only_vid(conn)
        md = to_markdown(conn, vid)
    finally:
        conn.close()

    assert md.startswith("# ")
    assert "## Chapters" in md
    assert "## Transcript" in md
    assert C.SPEECH_EXPECTED_SUBSTRING in md.lower()


@pytest.mark.slow
def test_export_command_json_and_file(tmp_path, speech_fixture):
    runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(speech_fixture)])
    conn = connect()
    vid = _only_vid(conn)
    conn.close()

    out = runner.invoke(app, ["export", vid[:8], "--format", "json"]).output
    assert json.loads(out)["vidcp_export_version"] == 1

    dest = tmp_path / "out.md"
    result = runner.invoke(app, ["export", vid[:8], "--format", "markdown", "-o", str(dest)])
    assert result.exit_code == 0
    assert dest.read_text(encoding="utf-8").startswith("# ")
