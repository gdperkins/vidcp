"""download_url tests. subprocess and shutil.which are always mocked —
no network access and no yt-dlp binary required."""

import json
import subprocess
from pathlib import Path

import pytest

import vidcp.download as download
from vidcp.download import DownloadedVideo, download_url, ensure_ytdlp
from vidcp.errors import VidcpError

URL = "https://example.com/watch?v=abc123"


def _install_fake_run(
    monkeypatch,
    dest_dir: Path,
    *,
    returncode: int = 0,
    stderr: str = "",
    write_media: bool = True,
    write_info: bool = True,
    title: str = "A Talk",
):
    """Replace subprocess.run inside vidcp.download with a fake yt-dlp."""

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "yt-dlp"
        assert "--no-playlist" in cmd
        if returncode == 0:
            if write_media:
                (dest_dir / "A Talk [abc123].mp4").write_bytes(b"\x00" * 1024)
            if write_info:
                (dest_dir / "A Talk [abc123].info.json").write_text(json.dumps({"title": title}))
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(download.subprocess, "run", fake_run)


@pytest.fixture
def _ytdlp_on_path(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: "/opt/bin/yt-dlp")


def test_download_url_success(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest)
    result = download_url(URL, dest)
    assert isinstance(result, DownloadedVideo)
    assert result.path.exists() and result.path.suffix == ".mp4"
    assert result.title == "A Talk"
    assert result.url == URL
    # the info.json sidecar is consumed and removed
    assert list(dest.glob("*.info.json")) == []


def test_download_url_title_falls_back_to_filename(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest, write_info=False)
    result = download_url(URL, dest)
    assert result.title == "A Talk [abc123]"


def test_download_url_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: None)

    def explode(*args, **kwargs):  # subprocess must never be reached
        raise AssertionError("subprocess.run called despite missing binary")

    monkeypatch.setattr(download.subprocess, "run", explode)
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, tmp_path / "dl")
    assert "yt-dlp" in str(excinfo.value)
    assert "install" in (excinfo.value.hint or "")


def test_ensure_ytdlp(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    with pytest.raises(VidcpError):
        ensure_ytdlp()
    monkeypatch.setattr(download.shutil, "which", lambda name: "/opt/bin/yt-dlp")
    ensure_ytdlp()  # no raise


def test_download_url_failure_surfaces_stderr_tail(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest, returncode=1, stderr="warning: x\nERROR: Unsupported URL")
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, dest)
    assert "Unsupported URL" in str(excinfo.value)


def test_download_url_no_output_file(tmp_path, monkeypatch, _ytdlp_on_path):
    dest = tmp_path / "dl"
    _install_fake_run(monkeypatch, dest, write_media=False, write_info=False)
    with pytest.raises(VidcpError) as excinfo:
        download_url(URL, dest)
    assert "no file" in str(excinfo.value)
