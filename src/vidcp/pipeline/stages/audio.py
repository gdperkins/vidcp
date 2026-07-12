"""Audio stage: extract a 16 kHz mono WAV, or skip when there is no audio."""

from __future__ import annotations

import subprocess

from vidcp.pipeline.base import Stage, StageSkipped, VideoContext


class AudioStage(Stage):
    name = "audio"
    depends_on = ["probe"]

    def clean(self, ctx: VideoContext) -> None:
        (ctx.artifacts / "audio.wav").unlink(missing_ok=True)

    def run(self, ctx: VideoContext) -> None:
        row = ctx.conn.execute(
            "SELECT has_audio FROM videos WHERE id=?", (ctx.video_id,)
        ).fetchone()
        if not row or not row["has_audio"]:
            raise StageSkipped("no audio stream")

        out = ctx.artifacts / "audio.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(ctx.source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            check=True,
        )
