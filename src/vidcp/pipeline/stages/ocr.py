"""OCR stage: read on-screen text from kept keyframes with RapidOCR."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from vidcp.config import Settings
from vidcp.pipeline.base import Stage, StageSkipped, VideoContext

_MIN_SCORE = 0.5
_SIMILARITY_THRESHOLD = 0.9


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_frame(path: Path) -> tuple[str, float | None, list[Any]]:
    """OCR a single frame -> (joined_text, mean_score, line_boxes)."""
    result, _ = _engine()(str(path))
    lines: list[tuple[float, str, float, list[list[float]]]] = []
    for box, text, score in result or []:
        if score is None or float(score) < _MIN_SCORE:
            continue
        stripped = (text or "").strip()
        if not stripped:
            continue
        norm_box = [[float(pt[0]), float(pt[1])] for pt in box]
        top = min(pt[1] for pt in norm_box)
        lines.append((top, stripped, float(score), norm_box))

    lines.sort(key=lambda ln: ln[0])  # top-to-bottom reading order
    text = " ".join(ln[1] for ln in lines)
    mean_score = sum(ln[2] for ln in lines) / len(lines) if lines else None
    boxes = [ln[3] for ln in lines]
    return text, mean_score, boxes


class OcrStage(Stage):
    name = "ocr"
    depends_on = ["keyframes"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"ocr_enabled={settings.ocr_enabled}"

    def run(self, ctx: VideoContext) -> None:
        if not ctx.settings.ocr_enabled:
            raise StageSkipped("ocr disabled")

        conn = ctx.conn
        # Idempotent: clear prior OCR rows (blocks + their FTS rows).
        conn.execute("DELETE FROM fts WHERE video_id=? AND kind='ocr'", (ctx.video_id,))
        conn.execute("DELETE FROM ocr_blocks WHERE video_id=?", (ctx.video_id,))

        frames = conn.execute(
            """
            SELECT f.scene_id, f.ts_s, f.path, s.end_s AS scene_end
            FROM frames f
            LEFT JOIN scenes s ON s.id = f.scene_id
            WHERE f.video_id=? AND f.kept=1
            ORDER BY f.ts_s
            """,
            (ctx.video_id,),
        ).fetchall()

        # Group frames into blocks, merging temporally-adjacent similar text.
        blocks: list[dict[str, Any]] = []
        for frame in frames:
            text, mean_score, boxes = _ocr_frame(Path(frame["path"]))
            if not text:
                continue
            end_s = frame["scene_end"] if frame["scene_end"] is not None else frame["ts_s"]
            prev = blocks[-1] if blocks else None
            if prev is not None and (
                SequenceMatcher(None, prev["text"], text).ratio() > _SIMILARITY_THRESHOLD
            ):
                prev["end_s"] = end_s
                if mean_score is not None:
                    prev["scores"].append(mean_score)
            else:
                blocks.append(
                    {
                        "scene_id": frame["scene_id"],
                        "start_s": frame["ts_s"],
                        "end_s": end_s,
                        "text": text,
                        "scores": [mean_score] if mean_score is not None else [],
                        "boxes": boxes,
                    }
                )

        for block in blocks:
            scores = block["scores"]
            confidence = sum(scores) / len(scores) if scores else None
            cur = conn.execute(
                "INSERT INTO ocr_blocks(video_id, scene_id, start_s, end_s, text, confidence, bbox) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ctx.video_id,
                    block["scene_id"],
                    block["start_s"],
                    block["end_s"],
                    block["text"],
                    confidence,
                    json.dumps(block["boxes"]),
                ),
            )
            conn.execute(
                "INSERT INTO fts(text, video_id, kind, ref_id, ts_s) VALUES (?, ?, 'ocr', ?, ?)",
                (block["text"], ctx.video_id, cur.lastrowid, block["start_s"]),
            )
        conn.commit()
