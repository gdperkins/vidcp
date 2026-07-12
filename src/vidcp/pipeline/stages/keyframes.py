"""Keyframes stage: extract, perceptually dedupe, and index scene keyframes."""

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vidcp.config import Settings
from vidcp.pipeline.base import Stage, VideoContext


def _scene_timestamps(start: float, end: float, interval: float) -> list[tuple[float, bool]]:
    """Return (timestamp, is_midpoint) samples covering a scene.

    Always includes the midpoint; for scenes longer than ``interval`` it adds
    frames roughly every ``interval`` seconds so no gap exceeds it.
    """
    midpoint = (start + end) / 2
    samples = [(midpoint, True)]
    if interval > 0 and (end - start) > interval:
        t = start + interval
        while t < end:
            if abs(t - midpoint) > 0.5:
                samples.append((t, False))
            t += interval
    return samples


def _extract_frame(src: Path, ts: float, out: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out),
        ],
        check=False,
    )


class KeyframesStage(Stage):
    name = "keyframes"
    depends_on = ["scenes"]

    def config_fingerprint(self, settings: Settings) -> str:
        return f"interval={settings.keyframe_min_interval_s};phash={settings.phash_max_distance}"

    def clean(self, ctx: VideoContext) -> None:
        conn = ctx.conn
        conn.execute("DELETE FROM frames WHERE video_id=?", (ctx.video_id,))
        conn.execute(
            "UPDATE scenes SET keyframe_path=NULL, phash=NULL WHERE video_id=?",
            (ctx.video_id,),
        )
        conn.commit()
        frames_dir = ctx.artifacts / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)

    def run(self, ctx: VideoContext) -> None:
        import imagehash
        from PIL import Image

        conn = ctx.conn
        settings = ctx.settings

        self.clean(ctx)  # idempotent reset (rows + JPEGs)
        frames_dir = ctx.artifacts / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        scenes = conn.execute(
            "SELECT id, start_s, end_s FROM scenes WHERE video_id=? ORDER BY idx",
            (ctx.video_id,),
        ).fetchall()

        # Global, time-ordered candidate list: (scene_id, ts, is_midpoint).
        candidates: list[tuple[int, float, bool]] = []
        for scene in scenes:
            for ts, is_mid in _scene_timestamps(
                scene["start_s"], scene["end_s"], settings.keyframe_min_interval_s
            ):
                candidates.append((scene["id"], ts, is_mid))
        candidates.sort(key=lambda c: c[1])

        jobs = [
            (scene_id, ts, is_mid, frames_dir / f"f_{i:04d}.jpg")
            for i, (scene_id, ts, is_mid) in enumerate(candidates)
        ]

        # Extract all frames in parallel (seek-before-input is fast).
        src = ctx.source_path
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda job: _extract_frame(src, job[1], job[3]), jobs))

        # Pass 1 (no DB): hash each frame and dedupe against the last kept one.
        # Keeping the heavy phash work out of the write transaction avoids
        # holding the WAL write lock while parallel stages are writing.
        kept: list[tuple[int, float, bool, Path, str]] = []
        last_kept = None
        for scene_id, ts, is_mid, out in jobs:
            if not out.exists():
                continue
            try:
                phash = imagehash.phash(Image.open(out))
            except (OSError, SyntaxError, ValueError):
                out.unlink(missing_ok=True)  # unreadable/corrupt frame -> skip
                continue
            if last_kept is not None and (phash - last_kept) <= settings.phash_max_distance:
                out.unlink(missing_ok=True)  # duplicate -> delete JPEG, no row
                continue
            last_kept = phash
            kept.append((scene_id, ts, is_mid, out, str(phash)))

        # Pass 2 (short transaction): insert rows + scene keyframe pointers.
        for scene_id, ts, is_mid, out, phash_str in kept:
            conn.execute(
                "INSERT INTO frames(video_id, scene_id, ts_s, path, phash, kept) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (ctx.video_id, scene_id, ts, str(out), phash_str),
            )
            if is_mid:
                conn.execute(
                    "UPDATE scenes SET keyframe_path=?, phash=? WHERE id=?",
                    (str(out), phash_str, scene_id),
                )
        # Backfill: a scene whose midpoint frame was phash-deduped still gets a
        # keyframe — the kept frame nearest its midpoint. Referencing the outer
        # UPDATE table inside a subquery's ORDER BY ("scenes.start_s") raises
        # "no such column" on SQLite <= 3.47 (CI's bundled build), so rank the
        # candidate frames in a joined subquery instead.
        conn.execute(
            """
            UPDATE scenes SET keyframe_path = pick.path, phash = pick.phash
            FROM (
                SELECT f.scene_id AS scene_id, f.path AS path, f.phash AS phash,
                       ROW_NUMBER() OVER (
                           PARTITION BY f.scene_id
                           ORDER BY abs(f.ts_s - (s.start_s + s.end_s) / 2)
                       ) AS rn
                FROM frames f JOIN scenes s ON s.id = f.scene_id
                WHERE f.kept = 1 AND s.video_id = ?
            ) AS pick
            WHERE pick.scene_id = scenes.id AND pick.rn = 1
              AND scenes.keyframe_path IS NULL
            """,
            (ctx.video_id,),
        )
        conn.commit()
