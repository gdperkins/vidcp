"""Download a video URL with yt-dlp for ingestion.

yt-dlp is an external binary on PATH (like ffmpeg), never a Python
dependency: extractors churn fast, and a user-managed binary stays current
independently of vidcp releases. Everything here is subprocess-only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from vidcp.errors import VidcpError

_INSTALL_HINT = "install yt-dlp: brew install yt-dlp (or pipx install yt-dlp)"


class DownloadedVideo(BaseModel):
    path: Path
    title: str
    url: str


def ensure_ytdlp() -> None:
    """Raise a friendly error when the yt-dlp binary is not on PATH."""
    if shutil.which("yt-dlp") is None:
        raise VidcpError("yt-dlp not found on PATH", hint=_INSTALL_HINT)


def download_url(url: str, dest_dir: Path) -> DownloadedVideo:
    """Download a single video ``url`` into ``dest_dir`` with yt-dlp.

    ``dest_dir`` must be empty (a fresh temp dir): the downloaded media file
    is identified as the largest file left behind. The title comes from the
    ``--write-info-json`` sidecar, which is consumed and deleted. Raises
    ``VidcpError`` when yt-dlp is missing or the download fails.
    """
    ensure_ytdlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--write-info-json",
            "-o",
            str(dest_dir / "%(title).150B [%(id)s].%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "unknown error"
        raise VidcpError(f"download failed: {detail}")
    title = ""
    for info_file in dest_dir.glob("*.info.json"):
        try:
            title = json.loads(info_file.read_text()).get("title") or ""
        except (OSError, ValueError):
            title = ""
        info_file.unlink(missing_ok=True)
    media = [f for f in dest_dir.iterdir() if f.is_file()]
    if not media:
        raise VidcpError(f"yt-dlp produced no file for {url}")
    media_file = max(media, key=lambda f: f.stat().st_size)
    return DownloadedVideo(path=media_file, title=title or media_file.stem, url=url)
