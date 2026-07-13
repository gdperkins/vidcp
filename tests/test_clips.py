"""Clip extraction tests. Seeds the artifact store directly (no pipeline run),
using the committed speech.mp4 fixture — needs ffmpeg/ffprobe but no models."""

import shutil
import subprocess
from pathlib import Path

import pytest

from vidcp.clips import extract_clip
from vidcp.errors import VidcpError
from vidcp.store import artifact_dir

VID = "d" * 64


def _seed_source(speech_fixture: Path) -> Path:
    dest = artifact_dir(VID) / f"source{speech_fixture.suffix}"
    shutil.copy2(speech_fixture, dest)
    return dest


def _duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def test_extract_clip_stream_copy(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    out = tmp_path / "clip.mp4"
    result = extract_clip(VID, 0.0, 1.5, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert (
        0.0 < _duration(out) <= 2.2
    )  # stream copy cuts on keyframes; allow slack; bound stays below fixture's 2.9s duration to catch dropped -t


def test_extract_clip_precise_reencode(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    out = tmp_path / "precise.mp4"
    extract_clip(VID, 0.0, 1.0, out, precise=True)
    assert out.exists() and out.stat().st_size > 0
    assert 0.0 < _duration(out) <= 1.6  # re-encode is frame-accurate


def test_extract_clip_invalid_range(tmp_path, speech_fixture):
    _seed_source(speech_fixture)
    with pytest.raises(VidcpError):
        extract_clip(VID, 5.0, 5.0, tmp_path / "x.mp4")
    with pytest.raises(VidcpError):
        extract_clip(VID, -1.0, 2.0, tmp_path / "y.mp4")


def test_extract_clip_missing_source(tmp_path):
    with pytest.raises(VidcpError):
        extract_clip("e" * 64, 0.0, 1.0, tmp_path / "z.mp4")
