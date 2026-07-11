"""Minimal pipeline runner.

Runs stages in dependency order, recording each in the ``stages`` table
(``running`` -> ``done``/``failed``). A stage is skipped only when it is already
``done`` with a matching ``config_hash`` *and* none of its dependencies re-ran
in this invocation — so changing an upstream stage (e.g. the scene threshold)
transparently invalidates its downstream stages. Failures are recorded and
returned, not raised, so a batch can continue. The resumable, parallel runner
arrives in Step 7.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidcp.pipeline.base import Stage, VideoContext
from vidcp.util import now_iso as _now


@dataclass
class StageOutcome:
    name: str
    status: str  # done | failed | skipped
    error: str | None = None


def _order_stages(stages: list[Stage]) -> list[Stage]:
    """Depth-first topological order over ``depends_on`` (within this batch)."""
    by_name = {s.name: s for s in stages}
    ordered: list[Stage] = []
    visited: set[str] = set()

    def visit(stage: Stage) -> None:
        if stage.name in visited:
            return
        visited.add(stage.name)
        for dep in stage.depends_on:
            if dep in by_name:
                visit(by_name[dep])
        ordered.append(stage)

    for stage in stages:
        visit(stage)
    return ordered


def run_pipeline(ctx: VideoContext, stages: list[Stage]) -> list[StageOutcome]:
    conn = ctx.conn
    outcomes: list[StageOutcome] = []
    ran: set[str] = set()  # stages that actually (re)ran this invocation

    for stage in _order_stages(stages):
        config_hash = stage.config_hash(ctx.settings)
        existing = conn.execute(
            "SELECT status, config_hash FROM stages WHERE video_id=? AND stage=?",
            (ctx.video_id, stage.name),
        ).fetchone()
        up_to_date = (
            existing is not None
            and existing["status"] == "done"
            and existing["config_hash"] == config_hash
        )
        dependency_reran = any(dep in ran for dep in stage.depends_on)
        if up_to_date and not dependency_reran:
            outcomes.append(StageOutcome(stage.name, "skipped"))
            continue

        conn.execute(
            """
            INSERT INTO stages(video_id, stage, status, started_at, finished_at, error, config_hash)
            VALUES (?, ?, 'running', ?, NULL, NULL, ?)
            ON CONFLICT(video_id, stage) DO UPDATE SET
                status='running', started_at=excluded.started_at,
                finished_at=NULL, error=NULL, config_hash=excluded.config_hash
            """,
            (ctx.video_id, stage.name, _now(), config_hash),
        )
        conn.commit()

        try:
            stage.run(ctx)
        except Exception as exc:
            conn.execute(
                "UPDATE stages SET status='failed', finished_at=?, error=? "
                "WHERE video_id=? AND stage=?",
                (_now(), str(exc), ctx.video_id, stage.name),
            )
            conn.commit()
            outcomes.append(StageOutcome(stage.name, "failed", str(exc)))
        else:
            conn.execute(
                "UPDATE stages SET status='done', finished_at=? WHERE video_id=? AND stage=?",
                (_now(), ctx.video_id, stage.name),
            )
            conn.commit()
            ran.add(stage.name)
            outcomes.append(StageOutcome(stage.name, "done"))

    return outcomes
