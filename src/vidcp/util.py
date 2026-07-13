"""Small shared formatting helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | None) -> str:
    """Format a duration as ``mm:ss`` (under an hour) or ``h:mm:ss`` (over)."""
    if seconds is None:
        return "-"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(value: str) -> float:
    """Parse a timestamp string into seconds.

    Accepts plain seconds ("83", "83.5"), "mm:ss", or "h:mm:ss". Raises
    ``ValueError`` on anything else (including negative or non-finite
    components such as "nan" or "inf").
    """
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(part == "" for part in parts):
        raise ValueError(f"invalid timestamp '{value}'")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"invalid timestamp '{value}'") from None
    if any(number < 0 or not math.isfinite(number) for number in numbers):
        raise ValueError(f"invalid timestamp '{value}'")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds
