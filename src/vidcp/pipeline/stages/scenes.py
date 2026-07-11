"""Scenes stage: detect scene boundaries with PySceneDetect."""

from __future__ import annotations

from vidcp.config import Settings
from vidcp.pipeline.base import Stage, VideoContext


class ScenesStage(Stage):
    name = "scenes"
    depends_on = ["probe"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"threshold={settings.scene_threshold}"

    def run(self, ctx: VideoContext) -> None:
        from scenedetect import ContentDetector, detect

        conn = ctx.conn
        # Idempotent: clearing scenes cascades to frames, so a re-run starts clean.
        conn.execute("DELETE FROM scenes WHERE video_id=?", (ctx.video_id,))

        scene_list = detect(
            str(ctx.source_path),
            ContentDetector(threshold=ctx.settings.scene_threshold),
        )
        if scene_list:
            spans = [(start.seconds, end.seconds) for start, end in scene_list]
        else:
            # No cuts detected -> one scene spanning the whole video.
            row = conn.execute(
                "SELECT duration_s FROM videos WHERE id=?", (ctx.video_id,)
            ).fetchone()
            duration = row["duration_s"] if row and row["duration_s"] else 0.0
            spans = [(0.0, duration)]

        for idx, (start_s, end_s) in enumerate(spans):
            conn.execute(
                "INSERT INTO scenes(video_id, idx, start_s, end_s) VALUES (?, ?, ?, ?)",
                (ctx.video_id, idx, start_s, end_s),
            )
        conn.commit()
