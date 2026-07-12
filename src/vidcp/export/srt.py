"""SubRip (.srt) export."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vidcp.export import format_timestamp


def to_srt(segments: Iterable[Any]) -> str:
    """Render transcript segments as SubRip text.

    Each segment needs ``start_s``, ``end_s``, and ``text`` attributes.
    """
    blocks: list[str] = []
    for index, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start_s, ",")
        end = format_timestamp(seg.end_s, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)
