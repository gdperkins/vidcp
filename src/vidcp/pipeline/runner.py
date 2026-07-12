"""Minimal pipeline runner.

Runs stages in dependency order, recording each in the ``stages`` table
(``running`` -> ``done``/``failed``/``skipped``). Skip rules:

* A stage that raises ``StageSkipped`` (e.g. audio on a silent video) is marked
  ``skipped``, and any stage depending on a skipped stage is skipped too.
* A stage already ``done``/``skipped`` with a matching ``config_hash`` is not
  re-run — unless one of its dependencies actually re-ran this invocation, so
  changing an upstream stage transparently invalidates its downstream stages.

Failures are recorded and returned, not raised, so a batch can continue. The
resumable, parallel runner arrives in Step 7.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidcp.pipeline.base import Stage, StageSkipped, VideoContext
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


def _set_status(conn, video_id, stage, status, config_hash, error=None):
    now = _now()
    conn.execute(
        """
        INSERT INTO stages(video_id, stage, status, started_at, finished_at, error, config_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, stage) DO UPDATE SET
            status=excluded.status, started_at=excluded.started_at,
            finished_at=excluded.finished_at, error=excluded.error,
            config_hash=excluded.config_hash
        """,
        (video_id, stage, status, now, now, error, config_hash),
    )
    conn.commit()


def run_pipeline(ctx: VideoContext, stages: list[Stage]) -> list[StageOutcome]:
    conn = ctx.conn
    outcomes: list[StageOutcome] = []
    ran: set[str] = set()  # stages that actually (re)ran this invocation
    status_by_stage: dict[str, str] = {}  # resulting DB status this invocation

    def dep_status(dep: str) -> str | None:
        if dep in status_by_stage:
            return status_by_stage[dep]
        row = conn.execute(
            "SELECT status FROM stages WHERE video_id=? AND stage=?",
            (ctx.video_id, dep),
        ).fetchone()
        return row["status"] if row else None

    for stage in _order_stages(stages):
        config_hash = stage.config_hash(ctx.settings)
        existing = conn.execute(
            "SELECT status, config_hash FROM stages WHERE video_id=? AND stage=?",
            (ctx.video_id, stage.name),
        ).fetchone()

        # Cascade: a stage whose dependency was skipped is skipped too.
        if any(dep_status(dep) == "skipped" for dep in stage.depends_on):
            already_skipped = (
                existing is not None
                and existing["status"] == "skipped"
                and existing["config_hash"] == config_hash
            )
            if not already_skipped:
                _set_status(
                    conn,
                    ctx.video_id,
                    stage.name,
                    "skipped",
                    config_hash,
                    error="dependency skipped",
                )
            status_by_stage[stage.name] = "skipped"
            outcomes.append(StageOutcome(stage.name, "skipped"))
            continue

        up_to_date = (
            existing is not None
            and existing["status"] in ("done", "skipped")
            and existing["config_hash"] == config_hash
        )
        dependency_reran = any(dep in ran for dep in stage.depends_on)
        if up_to_date and not dependency_reran:
            status_by_stage[stage.name] = existing["status"]
            outcomes.append(StageOutcome(stage.name, "skipped"))
            continue

        _set_status(conn, ctx.video_id, stage.name, "running", config_hash)
        try:
            stage.run(ctx)
        except StageSkipped as exc:
            conn.execute(
                "UPDATE stages SET status='skipped', finished_at=?, error=? "
                "WHERE video_id=? AND stage=?",
                (_now(), str(exc), ctx.video_id, stage.name),
            )
            conn.commit()
            status_by_stage[stage.name] = "skipped"
            outcomes.append(StageOutcome(stage.name, "skipped", str(exc)))
        except Exception as exc:
            conn.execute(
                "UPDATE stages SET status='failed', finished_at=?, error=? "
                "WHERE video_id=? AND stage=?",
                (_now(), str(exc), ctx.video_id, stage.name),
            )
            conn.commit()
            status_by_stage[stage.name] = "failed"
            outcomes.append(StageOutcome(stage.name, "failed", str(exc)))
        else:
            conn.execute(
                "UPDATE stages SET status='done', finished_at=? WHERE video_id=? AND stage=?",
                (_now(), ctx.video_id, stage.name),
            )
            conn.commit()
            status_by_stage[stage.name] = "done"
            ran.add(stage.name)
            outcomes.append(StageOutcome(stage.name, "done"))

    return outcomes
