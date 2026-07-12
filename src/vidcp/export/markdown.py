"""Human-readable Markdown export."""

from __future__ import annotations

import sqlite3

from vidcp.models import OcrBlock, SceneRow, Segment, Video
from vidcp.util import format_duration

_PARAGRAPH_GAP_S = 2.0


def _merge_paragraphs(segments: list[Segment]) -> list[str]:
    """Join consecutive segments into paragraphs, breaking on gaps > 2s."""
    paragraphs: list[str] = []
    current: list[str] = []
    prev_end: float | None = None
    for seg in segments:
        if prev_end is not None and seg.start_s - prev_end > _PARAGRAPH_GAP_S:
            paragraphs.append(" ".join(current))
            current = []
        current.append(seg.text.strip())
        prev_end = seg.end_s
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def to_markdown(conn: sqlite3.Connection, video_id: str) -> str:
    video = Video.from_row(conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone())
    scenes = [
        SceneRow.from_row(r)
        for r in conn.execute("SELECT * FROM scenes WHERE video_id=? ORDER BY idx", (video_id,))
    ]
    segments = [
        Segment.from_row(r)
        for r in conn.execute(
            "SELECT * FROM segments WHERE video_id=? ORDER BY start_s", (video_id,)
        )
    ]
    ocr_blocks = [
        OcrBlock.from_row(r)
        for r in conn.execute(
            "SELECT * FROM ocr_blocks WHERE video_id=? ORDER BY start_s", (video_id,)
        )
    ]

    resolution = f"{video.width}x{video.height}" if video.width and video.height else "-"
    lines = [
        f"# {video.title or video.short_id}",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| id | {video.short_id} |",
        f"| duration | {format_duration(video.duration_s)} |",
        f"| resolution | {resolution} |",
        f"| fps | {video.fps if video.fps is not None else '-'} |",
        f"| video codec | {video.vcodec or '-'} |",
        f"| audio codec | {video.acodec or '-'} |",
        "",
    ]

    if scenes:
        lines += ["## Chapters", ""]
        for scene in scenes:
            span = f"{format_duration(scene.start_s)} – {format_duration(scene.end_s)}"
            lines.append(f"- **{span}** scene {scene.idx}")
        lines.append("")

    if segments:
        lines += ["## Transcript", ""]
        for paragraph in _merge_paragraphs(segments):
            lines += [paragraph, ""]

    if ocr_blocks:
        lines += ["## On-screen text", ""]
        for block in ocr_blocks:
            span = f"{format_duration(block.start_s)} – {format_duration(block.end_s)}"
            lines.append(f"- **{span}**: {block.text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
