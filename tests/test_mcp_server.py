"""MCP server tests: in-memory client sessions, no subprocesses or sockets.

Seeds rows directly through ``db.connect()`` (the autouse ``_isolated_home``
fixture gives every test a fresh ``VIDCP_HOME``). The vector search leg is
never exercised (the ``vec`` table stays empty), so no models are downloaded.
"""

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


def test_python_dash_m_vidcp_runs():
    result = subprocess.run(
        [sys.executable, "-m", "vidcp", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "vidcp" in result.stdout
