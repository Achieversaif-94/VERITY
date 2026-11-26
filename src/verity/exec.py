"""Runs one task. Emits trace events. Enforces budget. Never trusts output text."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from .plan import Task
from .tools import REGISTRY, ToolContext
from .verify import Evidence, VerifyResult, validate

NodeState = Literal["pending", "running", "passed", "failed", "aborted"]


@dataclass
class ExecResult:
    state: NodeState
    evidence: Evidence | None
    verify: VerifyResult | None
    seconds: float
    tokens: int = 0


def execute_task(task: Task, ctx: ToolContext) -> ExecResult:
    tracer = ctx.tracer
    tracer.emit("node_start", node_id=task.id, tool=task.tool)
    t0 = time.monotonic()

    if task.tool not in REGISTRY:
        tracer.emit("node_end", node_id=task.id, state="failed", reason="unknown_tool")
        return ExecResult("failed", None, None, 0.0)

    try:
        ev = REGISTRY[task.tool]["run"](task.inputs, ctx, task.id)
    except Exception as e:
        elapsed = time.monotonic() - t0
        tracer.emit(
            "node_end", node_id=task.id, state="failed",
            reason=f"run_raised:{type(e).__name__}", seconds=elapsed,
        )
        return ExecResult("failed", None, None, elapsed)

    elapsed = time.monotonic() - t0
    if elapsed > task.budget.seconds:
        tracer.emit(
            "node_end", node_id=task.id, state="aborted",
            reason="budget_seconds", seconds=elapsed,
        )
        return ExecResult("aborted", ev, None, elapsed)

    res = validate(ev, REGISTRY[task.tool]["verify"])
    state: NodeState = "passed" if res.verdict == "PASS" else "failed"
    tracer.emit(
        "node_end", node_id=task.id, state=state, reason=res.reason,
        seconds=elapsed, artifact=ev.artifact_path, sha=ev.artifact_sha256,
    )
    return ExecResult(state, ev, res, elapsed)