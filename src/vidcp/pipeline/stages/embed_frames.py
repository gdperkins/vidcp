"""Embed-frames stage: encode kept keyframes with CLIP into vec_frames.

Enables visual semantic search: at query time the same CLIP model encodes the
text query, and nearest keyframes surface as kind='visual' hits. Skips itself
when ``clip_enabled`` is false or the video has no kept frames.
"""

from __future__ import annotations

from vidcp.config import Settings
from vidcp.embedding import load_model
from vidcp.pipeline.base import Stage, StageSkipped, VideoContext

# Keyframes are decoded to RGB at source resolution; encoding them all at once
# would hold multi-GB of image data in memory for long/4K videos. Chunk the
# decode+encode+insert cycle instead so only one batch's images are live.
_ENCODE_BATCH = 32


class EmbedFramesStage(Stage):
    name = "embed_frames"
    depends_on = ["keyframes"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"model={settings.clip_model};enabled={settings.clip_enabled}"

    def clean(self, ctx: VideoContext) -> None:
        ctx.conn.execute("DELETE FROM vec_frames WHERE video_id=?", (ctx.video_id,))
        ctx.conn.commit()

    def run(self, ctx: VideoContext) -> None:
        import sqlite_vec
        from PIL import Image

        conn = ctx.conn
        self.clean(ctx)  # replace this video's frame vectors
        if not ctx.settings.clip_enabled:
            raise StageSkipped("visual embeddings disabled")

        rows = conn.execute(
            "SELECT id, ts_s, path FROM frames WHERE video_id=? AND kept=1 ORDER BY ts_s",
            (ctx.video_id,),
        ).fetchall()
        if not rows:
            raise StageSkipped("no keyframes to embed")

        model = None
        readable = 0
        for chunk_start in range(0, len(rows), _ENCODE_BATCH):
            chunk = rows[chunk_start : chunk_start + _ENCODE_BATCH]
            images, meta = [], []
            for row in chunk:
                try:
                    images.append(Image.open(row["path"]).convert("RGB"))
                except (OSError, SyntaxError, ValueError):
                    continue  # missing/corrupt frame file -> skip it
                meta.append((row["id"], row["ts_s"]))
            if not images:
                continue
            readable += len(images)
            if model is None:  # defer the (possibly slow) model load until needed
                model = load_model(ctx.settings.clip_model)
            vectors = model.encode(images, batch_size=16, normalize_embeddings=True)
            for (frame_id, ts_s), vector in zip(meta, vectors):
                conn.execute(
                    "INSERT INTO vec_frames(embedding, video_id, frame_id, ts_s) VALUES (?, ?, ?, ?)",
                    (sqlite_vec.serialize_float32(vector), ctx.video_id, frame_id, ts_s),
                )
            conn.commit()
            # Chunk's decoded images go out of scope here, freeing their memory
            # before the next chunk is opened.

        if not readable:
            raise StageSkipped("no keyframes to embed")
