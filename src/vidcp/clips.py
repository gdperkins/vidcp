"""Clip extraction: cut [start_s, end_s] out of a stored source with ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vidcp.errors import VidcpError
from vidcp.store import source_path


def extract_clip(
    video_id: str,
    start_s: float,
    end_s: float,
    output: Path,
    precise: bool = False,
) -> Path:
    """Cut ``[start_s, end_s]`` from the stored source into ``output``.

    Stream-copies by default (fast; cut points land on the nearest keyframes,
    so the result may start slightly early). ``precise=True`` re-encodes for
    frame-accurate cuts at the cost of speed.
    """
    if start_s < 0 or end_s <= start_s:
        raise VidcpError(
            f"invalid clip range [{start_s:g}, {end_s:g}]",
            hint="end must be greater than start, and start must be >= 0",
        )
    src = source_path(video_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    codec_args = (
        ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac"] if precise else ["-c", "copy"]
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(src),
            "-t",
            f"{end_s - start_s:.3f}",
            *codec_args,
            "-avoid_negative_ts",
            "make_zero",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        detail = (
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        )
        raise VidcpError(f"ffmpeg failed to extract clip: {detail}")
    return output
