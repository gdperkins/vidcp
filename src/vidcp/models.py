"""Pydantic models mirroring the database tables.

Each model has a ``from_row`` classmethod that maps a ``sqlite3.Row`` (or plain
mapping) to the model, decoding JSON-typed columns. Models serialize cleanly for
``--json`` output via ``model_dump(mode="json")``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


class Video(BaseModel):
    id: str
    path: str
    title: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    vcodec: str | None = None
    acodec: str | None = None
    size_bytes: int | None = None
    has_audio: bool = True
    created_at: str | None = None
    ingested_at: str
    meta: dict[str, Any] | None = None

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @classmethod
    def from_row(cls, row: Any) -> "Video":
        data = dict(row)
        data["meta"] = _decode_json(data.get("meta"))
        return cls(**data)


class SceneRow(BaseModel):
    id: int
    video_id: str
    idx: int
    start_s: float
    end_s: float
    keyframe_path: str | None = None
    phash: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "SceneRow":
        return cls(**dict(row))


class Segment(BaseModel):
    id: int
    video_id: str
    start_s: float
    end_s: float
    text: str
    confidence: float | None = None
    words: list[dict[str, Any]] | None = None

    @classmethod
    def from_row(cls, row: Any) -> "Segment":
        data = dict(row)
        data["words"] = _decode_json(data.get("words"))
        return cls(**data)


class OcrBlock(BaseModel):
    id: int
    video_id: str
    scene_id: int | None = None
    start_s: float
    end_s: float
    text: str
    confidence: float | None = None
    bbox: list[Any] | None = None

    @classmethod
    def from_row(cls, row: Any) -> "OcrBlock":
        data = dict(row)
        data["bbox"] = _decode_json(data.get("bbox"))
        return cls(**data)


class StageState(BaseModel):
    video_id: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    config_hash: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "StageState":
        return cls(**dict(row))
