"""Probe stage: read media metadata with ffprobe into the videos row."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from vidcp.errors import VidcpError
from vidcp.pipeline.base import Stage, VideoContext


def _parse_fraction(value: str | None) -> float | None:
    """Parse ffprobe rate strings like ``"15/1"`` into a float."""
    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            denominator = float(den)
            return float(num) / denominator if denominator else None
        return float(value)
    except ValueError:
        return None


def ffprobe(path: Path) -> dict[str, Any]:
    """Return the parsed ffprobe JSON for a media file."""
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VidcpError(
            f"ffprobe failed for {path.name}",
            hint=result.stderr.strip() or "the file may not be a valid media file",
        )
    return json.loads(result.stdout)


def is_media_file(path: Path) -> bool:
    """Return True if ffprobe recognises the file as media with a video stream."""
    try:
        data = ffprobe(path)
    except VidcpError:
        return False
    return any(s.get("codec_type") == "video" for s in data.get("streams", []))


class ProbeStage(Stage):
    name = "probe"
    depends_on: list[str] = []

    def run(self, ctx: VideoContext) -> None:
        data = ffprobe(ctx.source_path)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = fmt.get("duration")
        duration_s = float(duration) if duration is not None else None
        size = fmt.get("size")
        size_bytes = int(size) if size is not None else None
        created_at = (fmt.get("tags") or {}).get("creation_time")

        width = video.get("width") if video else None
        height = video.get("height") if video else None
        fps = _parse_fraction(video.get("avg_frame_rate")) if video else None
        if not fps and video:
            fps = _parse_fraction(video.get("r_frame_rate"))
        vcodec = video.get("codec_name") if video else None
        acodec = audio.get("codec_name") if audio else None
        has_audio = 1 if audio else 0

        row = ctx.conn.execute("SELECT path FROM videos WHERE id=?", (ctx.video_id,)).fetchone()
        title = Path(row["path"]).stem if row and row["path"] else None

        ctx.conn.execute(
            """
            UPDATE videos SET
                title = COALESCE(title, ?),
                duration_s = ?, width = ?, height = ?, fps = ?,
                vcodec = ?, acodec = ?, size_bytes = ?, has_audio = ?,
                created_at = ?, meta = ?
            WHERE id = ?
            """,
            (
                title,
                duration_s,
                width,
                height,
                fps,
                vcodec,
                acodec,
                size_bytes,
                has_audio,
                created_at,
                json.dumps(data),
                ctx.video_id,
            ),
        )
        ctx.conn.commit()
