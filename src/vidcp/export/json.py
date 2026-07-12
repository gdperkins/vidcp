"""Canonical JSON export — the versioned vidcp knowledge object."""

from __future__ import annotations

import sqlite3
from typing import Any

from vidcp.models import OcrBlock, SceneRow, Segment, Video

EXPORT_VERSION = 1


def to_export_dict(conn: sqlite3.Connection, video_id: str) -> dict[str, Any]:
    """Build the nested knowledge object: video + scenes(+frames) + segments + ocr."""
    video = Video.from_row(
        conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    ).model_dump(mode="json")

    scenes: list[dict[str, Any]] = []
    for scene_row in conn.execute(
        "SELECT * FROM scenes WHERE video_id=? ORDER BY idx", (video_id,)
    ):
        scene = SceneRow.from_row(scene_row).model_dump(mode="json")
        scene["frames"] = [
            dict(frame)
            for frame in conn.execute(
                "SELECT id, ts_s, path, phash, kept FROM frames WHERE scene_id=? ORDER BY ts_s",
                (scene_row["id"],),
            )
        ]
        scenes.append(scene)

    segments = [
        Segment.from_row(r).model_dump(mode="json")
        for r in conn.execute(
            "SELECT * FROM segments WHERE video_id=? ORDER BY start_s", (video_id,)
        )
    ]
    ocr_blocks = [
        OcrBlock.from_row(r).model_dump(mode="json")
        for r in conn.execute(
            "SELECT * FROM ocr_blocks WHERE video_id=? ORDER BY start_s", (video_id,)
        )
    ]

    return {
        "vidcp_export_version": EXPORT_VERSION,
        "video": video,
        "scenes": scenes,
        "segments": segments,
        "ocr_blocks": ocr_blocks,
    }
