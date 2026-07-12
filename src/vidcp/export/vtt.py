"""WebVTT (.vtt) export."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vidcp.export import format_timestamp


def to_vtt(segments: Iterable[Any]) -> str:
    """Render transcript segments as WebVTT text.

    Each segment needs ``start_s``, ``end_s``, and ``text`` attributes.
    """
    blocks: list[str] = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp(seg.start_s, ".")
        end = format_timestamp(seg.end_s, ".")
        blocks.append(f"{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)
