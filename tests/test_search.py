import json

import constants as C
import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.db import connect
from vidcp.search import search

runner = CliRunner()


def test_search_no_matches_exits_zero():
    # Empty library -> both legs empty -> "no matches", exit 0.
    result = runner.invoke(app, ["search", "zzznomatchqqq"])
    assert result.exit_code == 0
    assert "no matches" in result.output.lower()


@pytest.mark.slow
def test_hybrid_search_over_all_fixtures(fixtures, speech_fixture):
    for path in (
        speech_fixture,
        fixtures["text.mp4"],
        fixtures["cuts.mp4"],
        fixtures["color.mp4"],
    ):
        result = runner.invoke(app, ["ingest", "--whisper-model", "tiny", str(path)])
        assert result.exit_code == 0, result.output

    conn = connect()
    try:
        # Exact keyword ranks the speech transcript segment #1 (FTS leg).
        hits = search(conn, C.SPEECH_KEYWORD, limit=5)
        assert hits
        assert hits[0].kind == "transcript"
        assert C.SPEECH_KEYWORD in hits[0].text.lower()

        # A paraphrase (no shared keywords) surfaces the segment top-3 (vector leg).
        para = search(conn, C.SPEECH_PARAPHRASE, limit=3)
        assert any(C.SPEECH_EXPECTED_SUBSTRING in h.text.lower() for h in para)

        # --kind ocr returns only OCR hits.
        ocr_hits = search(conn, "vidcp", kind="ocr", limit=5)
        assert ocr_hits
        assert all(h.kind == "ocr" for h in ocr_hits)

        # --id restricts to a single video.
        speech_vid = conn.execute("SELECT id FROM videos WHERE title=?", ("speech",)).fetchone()[
            "id"
        ]
        id_hits = search(conn, C.SPEECH_KEYWORD, video_id=speech_vid, limit=5)
        assert id_hits
        assert all(h.video_id == speech_vid for h in id_hits)

        # A term-less query (punctuation only) matches nothing on both legs.
        assert search(conn, "...", limit=5) == []
    finally:
        conn.close()

    # Command-level: --kind ocr --json returns only ocr, with ts_s + ref_id.
    data = json.loads(runner.invoke(app, ["search", "vidcp", "--kind", "ocr", "--json"]).output)
    assert data
    assert all(d["kind"] == "ocr" for d in data)
    assert "ts_s" in data[0] and "ref_id" in data[0]
