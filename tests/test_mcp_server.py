"""MCP server tests: in-memory client sessions, no subprocesses or sockets.

Seeds rows directly through ``db.connect()`` (the autouse ``_isolated_home``
fixture gives every test a fresh ``VIDCP_HOME``). The vector search leg is
never exercised (the ``vec`` table stays empty), so no models are downloaded.
"""

import base64
import io
import json
import subprocess
import sys

import pytest
from mcp.shared.memory import create_connected_server_and_client_session as client_session

from vidcp.db import connect
from vidcp.mcp_server import create_server
from vidcp.util import now_iso

pytestmark = pytest.mark.anyio

VID_A = "a" * 64
VID_B = "b" * 64


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    server = create_server()
    async with client_session(server._mcp_server) as session:
        yield session


def result_payload(result) -> dict:
    """Unwrap a successful tool result into its dict payload."""
    assert not result.isError, result.content
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


def error_text(result) -> str:
    assert result.isError, result.content
    return result.content[0].text


def seed_video(video_id=VID_A, title="talk", duration_s=60.0) -> str:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos(id, path, title, duration_s, ingested_at) VALUES (?,?,?,?,?)",
            (video_id, f"/videos/{title}.mp4", title, duration_s, now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
    return video_id


def seed_stage(video_id, stage, status, error=None):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO stages(video_id, stage, status, error) VALUES (?,?,?,?) "
            "ON CONFLICT(video_id, stage) DO UPDATE SET status=excluded.status, "
            "error=excluded.error",
            (video_id, stage, status, error),
        )
        conn.commit()
    finally:
        conn.close()


def seed_all_stages_done(video_id):
    from vidcp.pipeline import default_stages

    for stage in default_stages():
        seed_stage(video_id, stage.name, "done")


def seed_segment(video_id, start_s, end_s, text) -> int:
    conn = connect()
    try:
        cur = conn.execute(
            "INSERT INTO segments(video_id, start_s, end_s, text) VALUES (?,?,?,?)",
            (video_id, start_s, end_s, text),
        )
        conn.execute(
            "INSERT INTO fts(text, video_id, kind, ref_id, ts_s) VALUES (?,?,?,?,?)",
            (text, video_id, "transcript", cur.lastrowid, start_s),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def seed_scene(video_id, idx, start_s, end_s):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO scenes(video_id, idx, start_s, end_s) VALUES (?,?,?,?)",
            (video_id, idx, start_s, end_s),
        )
        conn.commit()
    finally:
        conn.close()


def seed_frame(video_id, ts_s, tmp_path, size=(1920, 1080), kept=1):
    from PIL import Image as PILImage

    path = tmp_path / f"frame_{ts_s:.0f}_{kept}.jpg"
    PILImage.new("RGB", size, (200, 40, 40)).save(path, "JPEG")
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO frames(video_id, ts_s, path, kept) VALUES (?,?,?,?)",
            (video_id, ts_s, str(path), kept),
        )
        conn.commit()
    finally:
        conn.close()
    return path


async def test_lists_expected_tools(client):
    tools = {tool.name for tool in (await client.list_tools()).tools}
    assert "list_videos" in tools


async def test_list_videos_empty_library(client):
    payload = result_payload(await client.call_tool("list_videos", {}))
    assert payload == {"videos": []}


async def test_list_videos_returns_seeded_video(client):
    seed_video(VID_A, title="talk")
    payload = result_payload(await client.call_tool("list_videos", {}))
    assert len(payload["videos"]) == 1
    video = payload["videos"][0]
    assert video["id"] == VID_A
    assert video["short_id"] == VID_A[:8]
    assert video["title"] == "talk"
    assert "meta" not in video


async def test_get_video_metadata_counts_and_stages(client):
    seed_video(VID_A, title="talk")
    seed_stage(VID_A, "probe", "done")
    seed_stage(VID_A, "transcribe", "failed", error="boom")
    payload = result_payload(await client.call_tool("get_video", {"video_id": VID_A[:8]}))
    assert payload["id"] == VID_A
    assert payload["title"] == "talk"
    assert payload["counts"] == {"scenes": 0, "frames": 0, "segments": 0, "ocr_blocks": 0}
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert by_stage["probe"]["status"] == "done"
    assert by_stage["transcribe"]["status"] == "failed"
    assert by_stage["transcribe"]["error"] == "boom"


async def test_get_video_unknown_id_errors(client):
    result = await client.call_tool("get_video", {"video_id": "deadbeef"})
    assert "no video matches id 'deadbeef'" in error_text(result)


async def test_get_video_ambiguous_prefix_errors(client):
    seed_video("a" + "c" * 63)
    seed_video("a" + "d" * 63)
    result = await client.call_tool("get_video", {"video_id": "a"})
    assert "ambiguous" in error_text(result)


async def test_search_returns_fts_hit(client):
    seed_video(VID_A)
    seed_segment(VID_A, 5.0, 9.0, "neural networks are discussed here")
    payload = result_payload(await client.call_tool("search", {"query": "neural"}))
    assert payload["hits"]
    hit = payload["hits"][0]
    assert hit["video_id"] == VID_A
    assert hit["kind"] == "transcript"
    assert hit["ts_s"] == 5.0
    assert "neural" in hit["snippet"]


async def test_search_rejects_unknown_kind(client):
    result = await client.call_tool("search", {"query": "x", "kind": "faces"})
    assert "unknown kind 'faces'" in error_text(result)


async def test_search_restricts_to_video_id_prefix(client):
    seed_video(VID_A, title="a")
    seed_video(VID_B, title="b")
    seed_segment(VID_A, 1.0, 2.0, "quantum computing intro")
    seed_segment(VID_B, 3.0, 4.0, "quantum computing outro")
    payload = result_payload(
        await client.call_tool("search", {"query": "quantum", "video_id": VID_A[:8]})
    )
    assert payload["hits"]
    assert all(h["video_id"] == VID_A for h in payload["hits"])


async def test_get_transcript_full(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "hello world")
    seed_segment(VID_A, 4.0, 8.0, "second segment")
    payload = result_payload(await client.call_tool("get_transcript", {"video_id": VID_A}))
    assert payload["video_id"] == VID_A
    assert [s["text"] for s in payload["segments"]] == ["hello world", "second segment"]
    assert set(payload["segments"][0]) == {"id", "start_s", "end_s", "text"}


async def test_get_transcript_window_overlap(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "before")
    seed_segment(VID_A, 3.0, 7.0, "overlaps start")
    seed_segment(VID_A, 8.0, 12.0, "inside")
    seed_segment(VID_A, 20.0, 24.0, "after")
    payload = result_payload(
        await client.call_tool("get_transcript", {"video_id": VID_A, "start_s": 4.0, "end_s": 15.0})
    )
    assert [s["text"] for s in payload["segments"]] == ["overlaps start", "inside"]


async def test_get_transcript_empty_window_is_not_an_error(client):
    seed_video(VID_A)
    seed_segment(VID_A, 0.0, 4.0, "hello")
    payload = result_payload(
        await client.call_tool(
            "get_transcript", {"video_id": VID_A, "start_s": 100.0, "end_s": 110.0}
        )
    )
    assert payload["segments"] == []


async def test_get_transcript_silent_video_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "skipped")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "no audio" in error_text(result)


async def test_get_transcript_not_finished_errors(client):
    seed_video(VID_A)  # no transcribe stage row at all
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "not available yet" in error_text(result)


async def test_get_transcript_still_running_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "running")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "not available yet" in error_text(result)


async def test_get_transcript_no_speech_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "done")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "no speech" in error_text(result)


async def test_get_transcript_failed_stage_errors(client):
    seed_video(VID_A)
    seed_stage(VID_A, "transcribe", "failed", error="model exploded")
    result = await client.call_tool("get_transcript", {"video_id": VID_A})
    assert "model exploded" in error_text(result)


async def test_list_scenes_ordered_by_idx(client):
    seed_video(VID_A)
    seed_scene(VID_A, 1, 10.0, 20.0)
    seed_scene(VID_A, 0, 0.0, 10.0)
    payload = result_payload(await client.call_tool("list_scenes", {"video_id": VID_A[:8]}))
    assert payload["video_id"] == VID_A
    assert [s["idx"] for s in payload["scenes"]] == [0, 1]
    assert payload["scenes"][0] == {"idx": 0, "start_s": 0.0, "end_s": 10.0}


async def test_list_scenes_empty_is_not_an_error(client):
    seed_video(VID_A)
    payload = result_payload(await client.call_tool("list_scenes", {"video_id": VID_A}))
    assert payload["scenes"] == []


async def test_get_keyframe_nearest_and_downscaled(client, tmp_path):
    from PIL import Image as PILImage

    seed_video(VID_A)
    seed_frame(VID_A, 10.0, tmp_path)
    seed_frame(VID_A, 50.0, tmp_path)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A[:8], "ts_s": 18.0})
    assert not result.isError, result.content
    texts = [c for c in result.content if c.type == "text"]
    images = [c for c in result.content if c.type == "image"]
    assert texts and "10.00s" in texts[0].text
    assert images and images[0].mimeType == "image/jpeg"
    decoded = PILImage.open(io.BytesIO(base64.b64decode(images[0].data)))
    assert max(decoded.size) <= 1280


async def test_get_keyframe_ignores_discarded_frames(client, tmp_path):
    seed_video(VID_A)
    seed_frame(VID_A, 10.0, tmp_path, kept=0)
    seed_frame(VID_A, 50.0, tmp_path)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 10.0})
    assert not result.isError, result.content
    texts = [c for c in result.content if c.type == "text"]
    assert "50.00s" in texts[0].text


async def test_get_keyframe_without_frames_errors(client):
    seed_video(VID_A)
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 1.0})
    assert "no keyframes" in error_text(result)


async def test_get_keyframe_missing_file_errors(client, tmp_path):
    seed_video(VID_A)
    path = seed_frame(VID_A, 10.0, tmp_path)
    path.unlink()
    result = await client.call_tool("get_keyframe", {"video_id": VID_A, "ts_s": 10.0})
    assert "keyframe file missing" in error_text(result)


@pytest.fixture
def spawn_recorder(monkeypatch):
    from unittest.mock import MagicMock

    import subprocess as real_subprocess

    calls = []
    original_popen = real_subprocess.Popen
    vidcp_spawn_prefix = [sys.executable, "-m", "vidcp"]

    def fake_popen(cmd, **kwargs):
        # Record and fake any vidcp spawn regardless of kwargs, so a refactor that
        # drops start_new_session can't silently spawn a real detached process.
        # subprocess.run (ffprobe's internals via is_media_file) delegates to
        # Popen too and needs the real one, with full context-manager behavior.
        if cmd[:3] == vidcp_spawn_prefix:
            calls.append((cmd, kwargs))
            return MagicMock()
        return original_popen(cmd, **kwargs)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    return calls


async def test_ingest_new_file_spawns_detached_ingest(client, spawn_recorder, speech_fixture):
    from vidcp.store import sha256_file

    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "started"
    assert payload["video_id"] == sha256_file(speech_fixture)
    ((cmd, kwargs),) = spawn_recorder
    assert cmd == [sys.executable, "-m", "vidcp", "ingest", str(speech_fixture)]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL


async def test_ingest_partial_prior_run_resumes_with_force(client, spawn_recorder, speech_fixture):
    from vidcp.store import sha256_file

    video_id = sha256_file(speech_fixture)
    seed_video(video_id)
    seed_stage(video_id, "probe", "done")  # crashed mid-pipeline: embed never completed
    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "started"
    ((cmd, _),) = spawn_recorder
    assert cmd == [sys.executable, "-m", "vidcp", "ingest", "--force", str(speech_fixture)]


async def test_ingest_complete_video_short_circuits(client, spawn_recorder, speech_fixture):
    from vidcp.store import sha256_file

    video_id = sha256_file(speech_fixture)
    seed_video(video_id)
    seed_all_stages_done(video_id)
    payload = result_payload(await client.call_tool("ingest", {"path": str(speech_fixture)}))
    assert payload["status"] == "already_ingested"
    assert spawn_recorder == []


async def test_ingest_missing_file_errors(client, spawn_recorder):
    result = await client.call_tool("ingest", {"path": "/nope/missing.mp4"})
    assert "file not found" in error_text(result)
    assert spawn_recorder == []


async def test_ingest_non_media_file_errors(client, spawn_recorder, tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("definitely not a video")
    result = await client.call_tool("ingest", {"path": str(bogus)})
    assert "not a recognised media file" in error_text(result)
    assert spawn_recorder == []


async def test_get_clip_extracts_and_caches(client, tmp_path):
    import shutil
    from pathlib import Path

    from vidcp.store import artifact_dir

    speech = Path(__file__).parent / "fixtures" / "speech.mp4"
    if not speech.exists():
        pytest.skip("speech.mp4 fixture is missing")
    seed_video(VID_A)
    shutil.copy2(speech, artifact_dir(VID_A) / "source.mp4")

    result = await client.call_tool(
        "get_clip", {"video_id": VID_A[:8], "start_s": 0.0, "end_s": 1.0}
    )
    payload = result_payload(result)
    clip_path = Path(payload["path"])
    assert clip_path.exists() and payload["size_bytes"] > 0
    assert payload["video_id"] == VID_A

    again = result_payload(
        await client.call_tool("get_clip", {"video_id": VID_A[:8], "start_s": 0.0, "end_s": 1.0})
    )
    assert again["path"] == payload["path"]  # cached, same file


async def test_get_clip_invalid_range(client):
    seed_video(VID_A)
    result = await client.call_tool(
        "get_clip", {"video_id": VID_A[:8], "start_s": 5.0, "end_s": 2.0}
    )
    assert "invalid clip range" in error_text(result)


async def test_all_eight_tools_registered(client):
    tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {
        "search",
        "list_videos",
        "get_video",
        "get_transcript",
        "list_scenes",
        "get_keyframe",
        "get_clip",
        "ingest",
    }


def test_python_dash_m_vidcp_runs():
    result = subprocess.run(
        [sys.executable, "-m", "vidcp", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "vidcp" in result.stdout
