"""Export formats for vidcp knowledge objects (srt, vtt, and more in Step 8)."""

from __future__ import annotations


def format_timestamp(seconds: float, millis_sep: str = ",") -> str:
    """Format seconds as ``HH:MM:SS<sep>mmm`` (SRT uses ``,``; VTT uses ``.``)."""
    total_ms = round(max(0.0, seconds) * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"
