import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidcp.cli import app
from vidcp.config import get_settings
from vidcp.db import connect
from vidcp.errors import VidcpError
from vidcp.library import resolve_id

runner = CliRunner()


def _insert_fake(conn, vid):
    conn.execute(
        "INSERT INTO videos(id, path, ingested_at, has_audio) VALUES (?, ?, ?, 1)",
        (vid, "/tmp/x.mp4", "2026-01-01T00:00:00"),
    )
    conn.commit()


# --- resolve_id -----------------------------------------------------------


def test_resolve_id_unique_4char_prefix():
    conn = connect()
    try:
        a = "aaaa1111" + "0" * 56
        b = "bbbb2222" + "0" * 56
        _insert_fake(conn, a)
        _insert_fake(conn, b)
        assert resolve_id(conn, "aaaa") == a  # 4-char prefix resolves uniquely
    finally:
        conn.close()


def test_resolve_id_ambiguous_raises():
    conn = connect()
    try:
        _insert_fake(conn, "aaaa1111" + "0" * 56)
        _insert_fake(conn, "aaaa2222" + "0" * 56)
        with pytest.raises(VidcpError):
            resolve_id(conn, "aaaa")
    finally:
        conn.close()


def test_resolve_id_missing_raises():
    conn = connect()
    try:
        with pytest.raises(VidcpError):
            resolve_id(conn, "zzzz")
    finally:
        conn.close()


# --- ingest / list / inspect / delete -------------------------------------


def test_ingest_creates_video_and_probe_done(fixtures):
    result = runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    assert result.exit_code == 0, result.output
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM videos").fetchone()
        assert row is not None
        assert abs(row["duration_s"] - 8.0) < 0.2
        assert row["title"] == "color"
        stage = conn.execute("SELECT status FROM stages WHERE stage='probe'").fetchone()
        assert stage["status"] == "done"
    finally:
        conn.close()


def test_reingest_is_noop(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    conn = connect()
    before = conn.execute("SELECT ingested_at FROM videos").fetchone()["ingested_at"]
    conn.close()

    result = runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    assert "already ingested" in result.output.lower()

    conn = connect()
    after = conn.execute("SELECT ingested_at FROM videos").fetchone()["ingested_at"]
    conn.close()
    assert before == after


def test_list_json_roundtrips(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    result = runner.invoke(app, ["list", "--json"])
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["duration_s"] is not None
    assert len(data[0]["id"]) == 64


def test_inspect_json_and_stages(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    conn = connect()
    vid = conn.execute("SELECT id FROM videos").fetchone()["id"]
    conn.close()

    data = json.loads(runner.invoke(app, ["inspect", vid[:8], "--json"]).output)
    assert data["id"] == vid
    assert data["width"] == 320

    with_stages = json.loads(runner.invoke(app, ["inspect", vid[:8], "--stages", "--json"]).output)
    assert any(s["stage"] == "probe" and s["status"] == "done" for s in with_stages["stages"])


def test_delete_removes_row_and_artifacts(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    conn = connect()
    vid = conn.execute("SELECT id FROM videos").fetchone()["id"]
    conn.close()
    artifact = get_settings().store_path / vid[:2] / vid
    assert artifact.exists()

    result = runner.invoke(app, ["delete", vid[:8]])
    assert result.exit_code == 0, result.output

    conn = connect()
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    conn.close()
    assert not artifact.exists()


def test_delete_keep_artifacts(fixtures):
    runner.invoke(app, ["ingest", str(fixtures["color.mp4"])])
    conn = connect()
    vid = conn.execute("SELECT id FROM videos").fetchone()["id"]
    conn.close()
    artifact = get_settings().store_path / vid[:2] / vid

    runner.invoke(app, ["delete", vid[:8], "--keep-artifacts"])
    conn = connect()
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    conn.close()
    assert artifact.exists()  # kept on disk


# --- URL ingest (download_url is stubbed — no network, no yt-dlp) -----------


def _fake_download(fixture: Path, title: str = "Fake Talk"):
    from vidcp.download import DownloadedVideo

    def fake(url: str, dest_dir: Path) -> DownloadedVideo:
        dest = Path(dest_dir) / "Fake Talk [abc].mp4"
        shutil.copy2(fixture, dest)
        return DownloadedVideo(path=dest, title=title, url=url)

    return fake


URL = "https://example.com/watch?v=abc123"


def test_ingest_url_downloads_and_ingests(fixtures, monkeypatch):
    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)
    monkeypatch.setattr("vidcp.download.download_url", _fake_download(fixtures["color.mp4"]))
    result = runner.invoke(app, ["ingest", "--no-ocr", URL])
    assert result.exit_code == 0, result.output
    conn = connect()
    try:
        row = conn.execute("SELECT id, path, title FROM videos").fetchone()
        assert row is not None
        assert row["path"] == URL  # origin is the URL, not the temp file
        assert row["title"] == "Fake Talk"
        vid = row["id"]
    finally:
        conn.close()
    from vidcp.store import artifact_dir

    assert any(artifact_dir(vid).glob("source.*"))  # canonical copy in the store
    tmp_root = get_settings().home / "tmp"
    assert list(tmp_root.iterdir()) == []  # per-run temp dir cleaned up


def test_ingest_mixed_file_and_url(fixtures, tmp_path, monkeypatch):
    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)
    monkeypatch.setattr("vidcp.download.download_url", _fake_download(fixtures["color.mp4"]))
    local = tmp_path / "local.mp4"
    shutil.copy2(fixtures["cuts.mp4"], local)  # different content -> different hash
    result = runner.invoke(app, ["ingest", "--no-ocr", str(local), URL])
    assert result.exit_code == 0, result.output
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 2
    finally:
        conn.close()


def test_ingest_url_download_failure_is_skip(monkeypatch):
    from vidcp.errors import VidcpError

    monkeypatch.setattr("vidcp.download.ensure_ytdlp", lambda: None)

    def boom(url, dest_dir):
        raise VidcpError("download failed: ERROR: Unsupported URL")

    monkeypatch.setattr("vidcp.download.download_url", boom)
    result = runner.invoke(app, ["ingest", URL])
    assert result.exit_code == 1
    assert "skip" in result.output and "Unsupported URL" in result.output
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    finally:
        conn.close()
    tmp_root = get_settings().home / "tmp"
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_ingest_url_without_ytdlp_fails_fast(monkeypatch):
    import vidcp.download as download

    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    result = runner.invoke(app, ["ingest", URL])
    assert result.exit_code != 0
    assert "yt-dlp not found" in result.output
