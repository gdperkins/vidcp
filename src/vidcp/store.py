"""Content-addressed artifact store.

Sources and derived artifacts live under ``<home>/store/<id[:2]>/<id>/``. The
sharded first two hex chars keep any single directory from growing unbounded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from vidcp.config import get_settings

_CHUNK = 1024 * 1024  # 1 MB


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_dir(video_id: str, create: bool = True) -> Path:
    """Return the artifact directory for a video id (created unless create=False)."""
    directory = get_settings().store_path / video_id[:2] / video_id
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def add_source(path: Path, video_id: str) -> Path:
    """Copy or hardlink ``path`` into the artifact dir as ``source<ext>``.

    The link strategy follows ``settings.link_mode``; hardlinking falls back to
    a copy across filesystem boundaries.
    """
    settings = get_settings()
    dest = artifact_dir(video_id) / f"source{path.suffix}"
    if dest.exists():
        dest.unlink()
    if settings.link_mode == "hardlink":
        try:
            os.link(path, dest)
            return dest
        except OSError:
            pass  # cross-device or unsupported -> fall back to copy
    shutil.copy2(path, dest)
    return dest
