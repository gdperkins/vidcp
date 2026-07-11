"""Small shared formatting helpers."""

from __future__ import annotations

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
