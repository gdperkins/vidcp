"""Embed stage: encode transcript segments + OCR blocks into the vec table.

Depends on ``transcribe`` and ``ocr`` having *finished* (done or skipped). It
runs whenever at least one of them produced rows, and self-skips when there is
nothing to embed.
"""

from __future__ import annotations

from vidcp.config import Settings
from vidcp.embedding import load_model
from vidcp.pipeline.base import Stage, StageSkipped, VideoContext


class EmbedStage(Stage):
    name = "embed"
    depends_on = ["transcribe", "ocr"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"model={settings.embed_model}"

    def run(self, ctx: VideoContext) -> None:
        import sqlite_vec

        conn = ctx.conn
        items: list[tuple[str, int, float, str]] = []  # (kind, ref_id, ts_s, text)
        for row in conn.execute(
            "SELECT id, start_s, text FROM segments WHERE video_id=? ORDER BY start_s",
            (ctx.video_id,),
        ):
            items.append(("transcript", row["id"], row["start_s"], row["text"]))
        for row in conn.execute(
            "SELECT id, start_s, text FROM ocr_blocks WHERE video_id=? ORDER BY start_s",
            (ctx.video_id,),
        ):
            items.append(("ocr", row["id"], row["start_s"], row["text"]))

        if not items:
            raise StageSkipped("nothing to embed")

        model = load_model(ctx.settings.embed_model)
        vectors = model.encode(
            [item[3] for item in items], batch_size=64, normalize_embeddings=True
        )

        # Idempotent: replace this video's vectors.
        conn.execute("DELETE FROM vec WHERE video_id=?", (ctx.video_id,))
        for (kind, ref_id, ts_s, _text), vector in zip(items, vectors):
            conn.execute(
                "INSERT INTO vec(embedding, video_id, kind, ref_id, ts_s) VALUES (?, ?, ?, ?, ?)",
                (sqlite_vec.serialize_float32(vector), ctx.video_id, kind, ref_id, ts_s),
            )
        conn.commit()
