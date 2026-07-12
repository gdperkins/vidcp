"""Resumable, parallel pipeline runner.

Stages run in dependency order on a small thread pool: independent chains (the
``audio -> transcribe`` chain and the ``scenes -> keyframes -> ocr`` chain) run
concurrently — the heavy work (whisper, OCR, ffmpeg) releases the GIL, so
threads suffice. Each stage runs on its own SQLite connection (WAL + busy
timeout make concurrent access safe).

State machine per stage row: ``pending -> running -> done|failed|skipped``.
Skip / invalidation rules:

* ``StageSkipped`` (e.g. audio on a silent video) -> ``skipped``. A dependent is
  cascade-skipped only when *all* its deps were skipped ("finished" semantics).
* A stage already ``done``/``skipped`` with a matching ``config_hash`` is not
  re-run, unless a dependency actually re-ran this invocation.
* A failed stage blocks its downstream (left ``pending`` with a blocked note);
  the batch continues and the failure is reported.

Resumability: a stale ``running`` row (from a crash) is reset to ``pending`` at
the start of each run, so the stage retries.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Optional

from vidcp.db import connect
from vidcp.pipeline.base import Stage, StageSkipped, VideoContext
from vidcp.util import now_iso as _now

MAX_WORKERS = 2


@dataclass
class StageOutcome:
    name: str
    status: str  # done | failed | skipped | blocked
    error: str | None = None


def topological_order(stages: list[Stage]) -> list[Stage]:
    """Return stages in dependency order, raising ValueError on a cycle."""
    by_name = {s.name: s for s in stages}
    ordered: list[Stage] = []
    state: dict[str, int] = {}  # 0/absent=unvisited, 1=visiting, 2=done

    def visit(stage: Stage) -> None:
        marker = state.get(stage.name, 0)
        if marker == 2:
            return
        if marker == 1:
            raise ValueError(f"cycle detected in stage DAG at {stage.name!r}")
        state[stage.name] = 1
        for dep in stage.depends_on:
            if dep in by_name:
                visit(by_name[dep])
        state[stage.name] = 2
        ordered.append(stage)

    for stage in stages:
        visit(stage)
    return ordered


def _upsert(conn, video_id, stage, status, config_hash, started, finished, error):
    conn.execute(
        """
        INSERT INTO stages(video_id, stage, status, started_at, finished_at, error, config_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, stage) DO UPDATE SET
            status=excluded.status, started_at=excluded.started_at,
            finished_at=excluded.finished_at, error=excluded.error,
            config_hash=excluded.config_hash
        """,
        (video_id, stage, status, started, finished, error, config_hash),
    )
    conn.commit()


def _run_stage(video_id, stage, settings, config_hash):
    """Worker: execute a stage on its own connection, recording status.

    Never raises — any error (including bookkeeping/DB errors) is converted to a
    ``("failed", ...)`` outcome so one worker's hiccup can't abort the whole
    pipeline (fut.result() would otherwise re-raise on the main thread).
    """
    try:
        conn = connect()
    except Exception as exc:  # pragma: no cover - connect rarely fails
        return ("failed", f"connect failed: {exc}")
    try:
        _upsert(conn, video_id, stage.name, "running", config_hash, _now(), None, None)
        ctx = VideoContext(video_id, conn, settings)
        try:
            stage.run(ctx)
        except StageSkipped as exc:
            status, error = "skipped", str(exc)
        except Exception as exc:
            status, error = "failed", str(exc)
        else:
            status, error = "done", None
        conn.execute(
            "UPDATE stages SET status=?, finished_at=?, error=? WHERE video_id=? AND stage=?",
            (status, _now(), error, video_id, stage.name),
        )
        conn.commit()
        return (status, error)
    except Exception as exc:  # bookkeeping/DB error around the stage
        return ("failed", f"stage bookkeeping error: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_pipeline(
    ctx: VideoContext,
    stages: list[Stage],
    progress: Optional[Callable[[str, str], None]] = None,
) -> list[StageOutcome]:
    order = topological_order(stages)
    by_name = {s.name: s for s in order}
    conn = ctx.conn
    settings = ctx.settings
    video_id = ctx.video_id

    # Resumability: treat any stale 'running' row as pending so it retries.
    conn.execute(
        "UPDATE stages SET status='pending' WHERE video_id=? AND status='running'",
        (video_id,),
    )
    conn.commit()

    resolved: dict[str, str] = {}  # name -> done|skipped|failed|blocked
    reran: set[str] = set()
    outcomes: dict[str, StageOutcome] = {}

    def notify(name, event):
        if progress:
            progress(name, event)

    def deps_of(stage):
        return [d for d in stage.depends_on if d in by_name]

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures: dict = {}
    submitted: set[str] = set()

    try:
        while len(resolved) < len(order) or futures:
            for stage in order:
                if stage.name in resolved or stage.name in submitted:
                    continue
                deps = deps_of(stage)
                if not all(d in resolved for d in deps):
                    continue

                dep_statuses = [resolved[d] for d in deps]
                config_hash = stage.config_hash(settings)

                # Failure cascade -> blocked (left pending for a future retry).
                blocking = [d for d in deps if resolved[d] in ("failed", "blocked")]
                if blocking:
                    _upsert(
                        conn,
                        video_id,
                        stage.name,
                        "pending",
                        config_hash,
                        None,
                        None,
                        f"blocked: {blocking[0]} did not complete",
                    )
                    resolved[stage.name] = "blocked"
                    outcomes[stage.name] = StageOutcome(stage.name, "blocked")
                    notify(stage.name, "blocked")
                    continue

                # Cascade skip only when ALL deps were skipped.
                if deps and all(st == "skipped" for st in dep_statuses):
                    existing = conn.execute(
                        "SELECT status, config_hash FROM stages WHERE video_id=? AND stage=?",
                        (video_id, stage.name),
                    ).fetchone()
                    if not (
                        existing
                        and existing["status"] == "skipped"
                        and existing["config_hash"] == config_hash
                    ):
                        # Newly skipped: clear any prior output so a stage that
                        # used to produce rows (e.g. embed of now-disabled OCR)
                        # doesn't leave stale data behind.
                        by_name[stage.name].clean(ctx)
                        _upsert(
                            conn,
                            video_id,
                            stage.name,
                            "skipped",
                            config_hash,
                            _now(),
                            _now(),
                            "dependency skipped",
                        )
                    resolved[stage.name] = "skipped"
                    outcomes[stage.name] = StageOutcome(stage.name, "skipped")
                    notify(stage.name, "skipped")
                    continue

                # Up-to-date: done/skipped + matching config, no dep re-ran.
                existing = conn.execute(
                    "SELECT status, config_hash FROM stages WHERE video_id=? AND stage=?",
                    (video_id, stage.name),
                ).fetchone()
                up_to_date = (
                    existing is not None
                    and existing["status"] in ("done", "skipped")
                    and existing["config_hash"] == config_hash
                )
                if up_to_date and not any(d in reran for d in deps):
                    resolved[stage.name] = existing["status"]
                    outcomes[stage.name] = StageOutcome(stage.name, "skipped")
                    notify(stage.name, "skipped")
                    continue

                # Otherwise: run it.
                fut = executor.submit(_run_stage, video_id, stage, settings, config_hash)
                futures[fut] = stage
                submitted.add(stage.name)
                notify(stage.name, "start")

            if futures:
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    stage = futures.pop(fut)
                    status, error = fut.result()
                    resolved[stage.name] = status
                    outcomes[stage.name] = StageOutcome(stage.name, status, error)
                    # Any stage that actually executed (done, or skipped/failed
                    # via a worker) may have changed its output, so it must
                    # invalidate up-to-date dependents — not just 'done'.
                    reran.add(stage.name)
                    notify(stage.name, status)
            elif len(resolved) < len(order):
                break  # nothing runnable and nothing running -> stop
    finally:
        executor.shutdown(wait=True)

    return [
        outcomes.get(stage.name, StageOutcome(stage.name, resolved.get(stage.name, "blocked")))
        for stage in order
    ]
