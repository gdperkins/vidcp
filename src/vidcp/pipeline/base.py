"""Pipeline context and the Stage contract.

``VideoContext`` bundles the identifiers and handles a stage needs. ``Stage`` is
the abstract base every processing step implements; ``config_hash`` combines the
stage name with its ``config_fingerprint`` so the runner can detect when a
stage's inputs changed and it must re-run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from vidcp.config import Settings
from vidcp.errors import VidcpError
from vidcp.store import artifact_dir


class VideoContext:
    def __init__(self, video_id: str, conn: sqlite3.Connection, settings: Settings) -> None:
        self.video_id = video_id
        self.conn = conn
        self.settings = settings

    @property
    def artifacts(self) -> Path:
        return artifact_dir(self.video_id)

    @property
    def source_path(self) -> Path:
        matches = sorted(self.artifacts.glob("source.*"))
        if not matches:
            raise VidcpError(
                f"no source file found for {self.video_id[:8]}",
                hint="the artifact store may be corrupted; re-ingest the video",
            )
        return matches[0]


class Stage(ABC):
    name: ClassVar[str]
    depends_on: ClassVar[list[str]] = []

    def config_fingerprint(self, settings: Settings) -> str:
        """Return a string capturing the settings this stage depends on.

        Overridden by stages whose output changes with configuration (e.g. the
        scene threshold). The default empty string means "never invalidated by
        config".
        """
        return ""

    def config_hash(self, settings: Settings) -> str:
        raw = f"{self.name}\x00{self.config_fingerprint(settings)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @abstractmethod
    def run(self, ctx: VideoContext) -> None: ...
