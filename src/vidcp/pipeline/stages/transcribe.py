"""Transcribe stage: faster-whisper -> segments + transcript.json + FTS rows."""

from __future__ import annotations

import json
import math
from functools import lru_cache

from vidcp.config import Settings
from vidcp.pipeline.base import Stage, StageSkipped, VideoContext


@lru_cache(maxsize=2)
def _load_model(name: str, device: str, compute_type: str):
    """Load (and cache) a WhisperModel so repeated calls don't reload it."""
    from faster_whisper import WhisperModel

    return WhisperModel(name, device=device, compute_type=compute_type)


def _confidence(avg_logprob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    return min(1.0, max(0.0, math.exp(avg_logprob)))


class TranscribeStage(Stage):
    name = "transcribe"
    depends_on = ["audio"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"model={settings.whisper_model}"

    def clean(self, ctx: VideoContext) -> None:
        conn = ctx.conn
        conn.execute("DELETE FROM fts WHERE video_id=? AND kind='transcript'", (ctx.video_id,))
        conn.execute("DELETE FROM segments WHERE video_id=?", (ctx.video_id,))
        conn.commit()
        (ctx.artifacts / "transcript.json").unlink(missing_ok=True)

    def run(self, ctx: VideoContext) -> None:
        wav = ctx.artifacts / "audio.wav"
        if not wav.exists():
            raise StageSkipped("no audio")

        self.clean(ctx)  # idempotent: clear prior transcript rows + json

        model = _load_model(ctx.settings.whisper_model, "cpu", "int8")
        segments, info = model.transcribe(str(wav), vad_filter=True, word_timestamps=True)

        conn = ctx.conn
        raw_segments = []
        for seg in segments:  # iterating drives the actual transcription
            text = (seg.text or "").strip()
            words = [{"w": w.word, "s": w.start, "e": w.end} for w in (seg.words or [])]
            raw_segments.append({"start": seg.start, "end": seg.end, "text": text, "words": words})
            if not text:
                continue
            cur = conn.execute(
                "INSERT INTO segments(video_id, start_s, end_s, text, confidence, words) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ctx.video_id,
                    seg.start,
                    seg.end,
                    text,
                    _confidence(seg.avg_logprob),
                    json.dumps(words),
                ),
            )
            conn.execute(
                "INSERT INTO fts(text, video_id, kind, ref_id, ts_s) "
                "VALUES (?, ?, 'transcript', ?, ?)",
                (text, ctx.video_id, cur.lastrowid, seg.start),
            )
        conn.commit()

        (ctx.artifacts / "transcript.json").write_text(
            json.dumps(
                {
                    "language": info.language,
                    "duration": info.duration,
                    "segments": raw_segments,
                },
                indent=2,
            )
        )
