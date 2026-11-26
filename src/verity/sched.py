"""Topological DAG scheduler with asyncio parallelism on independent branches.

Invariant: a node may only enter "running" if every dep is "passed".
If any dep is failed/aborted, the node is marked "aborted" immediately.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .exec import ExecResult, execute_task
from .plan import Plan, Task
from .tools import ToolContext


@dataclass
class RunResult:
    states: dict[str, str] = field(default_factory=dict)
    results: dict[str, ExecResult] = field(default_factory=dict)


async def _run_one(task: Task, ctx: ToolContext) -> tuple[str, ExecResult]:
    res = await asyncio.to_thread(execute_task, task, ctx)
    return task.id, res


async def run_plan(plan: Plan, ctx: ToolContext) -> RunResult:
    tracer = ctx.tracer
    tracer.emit("plan_start", nodes=[t.id for t in plan.tasks])

    states: dict[str, str] = {t.id: "pending" for t in plan.tasks}
    results: dict[str, ExecResult] = {}

    while True:
        ready = [
            t for t in plan.tasks
            if states[t.id] == "pending"
            and all(states[d] == "passed" for d in t.deps)
        ]
        blocked = [
            t for t in plan.tasks
            if states[t.id] == "pending"
            and any(states[d] in ("failed", "aborted") for d in t.deps)
        ]
        for t in blocked:
            states[t.id] = "aborted"
            tracer.emit("node_end", node_id=t.id, state="aborted", reason="dep_failed")

        if not ready and not blocked:
            break

        if ready:
            outs = await asyncio.gather(*(_run_one(t, ctx) for t in ready))
            for tid, res in outs:
                states[tid] = res.state
                results[tid] = res

    tracer.emit("plan_end", states=states)
    return RunResult(states=states, results=results)