"""Diff-only replanning.

On failure, ask the LLM for a minimal {add, remove, modify} diff -- never a full
rewrite. Validate: every remove/modify id exists, no id collisions, result is still
a well-formed Plan (acyclic, no dangling deps). Log the diff to the trace.

Caller caps replans per run; this module doesn't loop on its own.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from . import llm
from .plan import Plan, Task
from .trace import Tracer


class ReplanDiff(BaseModel):
    add: list[Task] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    modify: dict[str, dict] = Field(default_factory=dict)


class ReplanError(RuntimeError):
    pass


_SYSTEM = """You repair execution plans for a verifiable runtime.

Given a plan, the id of a node that FAILED verification, and the reason, propose a
minimal diff that could fix it. Output strict JSON with keys add, remove, modify.

Rules:
- Prefer modify over remove+add.
- New nodes may depend on existing nodes and on each other.
- Do not remove nodes that already passed (only failed or pending nodes).
- Every tool used must be one of: read_file, write_file, run_python, grep, csv_query, http_fetch.
- Node shape: {id, tool, inputs, deps, budget}. budget is optional.
- Return ONLY JSON, no prose, no fences.

Example output:
{"add":[{"id":"fetch_cfg","tool":"read_file","inputs":{"path":"config.json"},"deps":[]}],
 "remove":["n_fail"],
 "modify":{"n_downstream":{"inputs":{"path":"config.json"}}}}
"""


def apply_diff(plan: Plan, diff: ReplanDiff) -> Plan:
    by_id = {t.id: t for t in plan.tasks}

    for rid in diff.remove:
        if rid not in by_id:
            raise ReplanError(f"remove: unknown id {rid}")
        by_id.pop(rid)

    for mid in diff.modify:
        if mid not in by_id:
            raise ReplanError(f"modify: unknown id {mid}")

    for new in diff.add:
        if new.id in by_id:
            raise ReplanError(f"add: id {new.id} already exists")

    tasks: list[Task] = []
    for t in by_id.values():
        patch = diff.modify.get(t.id)
        if patch:
            t = t.model_copy(update=patch)
        tasks.append(t)
    tasks.extend(diff.add)

    try:
        return Plan(tasks=tasks)
    except (ValidationError, ValueError) as e:
        raise ReplanError(f"resulting plan invalid: {e}") from e


def replan(
    plan: Plan,
    failed_node_id: str,
    reason: str,
    *,
    tracer: Tracer,
    max_attempts: int = 2,
) -> tuple[Plan, ReplanDiff]:
    user = (
        f"PLAN:\n{plan.model_dump_json(indent=2)}\n\n"
        f"FAILED_NODE: {failed_node_id}\n"
        f"FAIL_REASON: {reason}\n\n"
        "Propose the diff."
    )

    diff: ReplanDiff | None = None
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            diff = llm.complete_json(
                _SYSTEM, user, ReplanDiff,
                tracer=tracer, node_id=failed_node_id,
            )
            new_plan = apply_diff(plan, diff)
            tracer.emit(
                "replan",
                node_id=failed_node_id,
                reason=reason,
                attempt=attempt,
                diff=diff.model_dump(),
                new_nodes=[t.id for t in new_plan.tasks],
            )
            return new_plan, diff
        except ReplanError as e:
            last_err = e
            tracer.emit(
                "replan_rejected",
                node_id=failed_node_id,
                attempt=attempt,
                reason=str(e),
            )
            user = (
                user
                + f"\n\nPrevious attempt rejected: {e}. Return a valid diff only."
            )
        except llm.LLMError as e:
            tracer.emit(
                "replan_llm_failed",
                node_id=failed_node_id,
                attempt=attempt,
                reason=str(e),
            )
            raise

    raise ReplanError(
        f"replan failed after {max_attempts} attempts: {last_err}"
    )