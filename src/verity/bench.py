"""Benchmark harness.

The harness must prove it can FAIL before we trust any tasks riding on it.
Every self-test includes at least one task whose grader returns False.

Design:
  - One tracer for the whole run. Node ids are prefixed per task (`<task_id>__<node_id>`),
    then stripped before the grader sees them, so graders use plain ids while the
    trace stays globally unique for replay.
  - Replan loop is here, not in sched.py. sched stays clean; the loop is
    an opt-in bench-level concern (Part H will share it with cli.py).
  - Tokens attributed per task from trace llm_call events, correlated by node-id prefix.
  - Graders are external Python files: bench/tasks/<category>/<task>/grader.py.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import replan as replan_mod
from .plan import Plan
from .sched import RunResult, run_plan
from .tools import ToolContext
from .trace import Tracer


@dataclass
class BenchTask:
    id: str
    category: str
    plan: Plan
    grader_path: Path
    dir: Path


@dataclass
class GraderContext:
    workdir: Path
    artifact_dir: Path
    trace_path: Path
    states: dict[str, str]
    task_t_start: float
    task_t_end: float


@dataclass
class TaskMetrics:
    task_id: str
    category: str
    passed: bool
    tokens: int
    latency_ms: float
    replans: int
    grader_reason: str


@dataclass
class RunSummary:
    run_id: str
    started: float
    tasks: list[TaskMetrics] = field(default_factory=list)
    trace_path: str = ""
    out_dir: str = ""


def discover_tasks(root: Path) -> list[BenchTask]:
    out: list[BenchTask] = []
    for task_yaml in sorted(root.rglob("task.yaml")):
        out.append(load_task(task_yaml.parent))
    return out


def load_task(task_dir: Path) -> BenchTask:
    cfg = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    plan = Plan.model_validate({"tasks": cfg["tasks"]})
    grader = task_dir / "grader.py"
    if not grader.exists():
        raise FileNotFoundError(f"{task_dir} missing grader.py")
    return BenchTask(
        id=str(cfg["id"]),
        category=str(cfg.get("category", task_dir.parent.name)),
        plan=plan,
        grader_path=grader,
        dir=task_dir,
    )


def _load_grader(grader_path: Path):
    name = f"_bench_grader_{abs(hash(str(grader_path)))}"
    spec = importlib.util.spec_from_file_location(name, grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {grader_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "grade"):
        raise RuntimeError(f"{grader_path} missing grade(ctx)")
    return mod.grade


def _failed_reason(result: RunResult, tid: str) -> str:
    r = result.results.get(tid)
    if r is None:
        return "no_result"
    if r.verify is not None:
        return r.verify.reason
    return "run_failed"


def _prefix(prefix: str, p: Plan) -> Plan:
    return Plan(tasks=[
        t.model_copy(update={
            "id": prefix + t.id,
            "deps": [prefix + d for d in t.deps],
        }) for t in p.tasks
    ])


def _strip_prefix(prefix: str, states: dict[str, str]) -> dict[str, str]:
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in states.items()
    }


def run_task_with_replan(
    task: BenchTask,
    ctx: ToolContext,
    max_replans: int = 3,
) -> tuple[RunResult, int]:
    prefix = task.id + "__"
    plan = _prefix(prefix, task.plan)
    replans = 0
    result = asyncio.run(run_plan(plan, ctx))

    while replans < max_replans:
        failed = [tid for tid, s in result.states.items() if s == "failed"]
        if not failed:
            break
        try:
            plan, _diff = replan_mod.replan(
                plan, failed[0], _failed_reason(result, failed[0]),
                tracer=ctx.tracer,
            )
        except Exception as e:
            ctx.tracer.emit(
                "bench_replan_abort", node_id=failed[0],
                reason=f"{type(e).__name__}: {e}",
            )
            break
        replans += 1
        result = asyncio.run(run_plan(plan, ctx))

    stripped = RunResult(
        states=_strip_prefix(prefix, result.states),
        results={k[len(prefix):] if k.startswith(prefix) else k: v
                 for k, v in result.results.items()},
    )
    return stripped, replans


def grade_task(task: BenchTask, ctx: ToolContext, result: RunResult,
               t_start: float, t_end: float) -> tuple[bool, str]:
    grade_fn = _load_grader(task.grader_path)
    gctx = GraderContext(
        workdir=ctx.workdir,
        artifact_dir=ctx.artifact_dir,
        trace_path=ctx.tracer.path,
        states=result.states,
        task_t_start=t_start,
        task_t_end=t_end,
    )
    try:
        return grade_fn(gctx)
    except Exception as e:
        return False, f"grader_raised:{type(e).__name__}: {e}"


def _attribute_tokens(trace_path: Path, task_ids: list[str]) -> dict[str, int]:
    by_task = {tid: 0 for tid in task_ids}
    if not trace_path.exists():
        return by_task
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("kind") != "llm_call":
            continue
        nid = ev.get("node_id") or ""
        for tid in task_ids:
            if nid.startswith(tid + "__"):
                p = ev["payload"]
                by_task[tid] += int(p.get("prompt_tokens", 0)) + int(p.get("completion_tokens", 0))
                break
    return by_task


def run_bench(
    task_roots: list[Path],
    out_root: Path,
    *,
    max_replans: int = 3,
    filter_pattern: str | None = None,
    on_progress=None,
) -> RunSummary:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    tracer = Tracer(trace_path, run_id=run_id)

    all_tasks: list[BenchTask] = []
    for root in task_roots:
        all_tasks.extend(discover_tasks(root))
    if filter_pattern:
        all_tasks = [t for t in all_tasks if filter_pattern in t.id]
    if not all_tasks:
        raise RuntimeError("no tasks discovered")

    tracer.emit("bench_start", tasks=[t.id for t in all_tasks], roots=[str(r) for r in task_roots])
    summary = RunSummary(
        run_id=run_id, started=time.time(),
        trace_path=str(trace_path), out_dir=str(out_dir),
    )

    for i, task in enumerate(all_tasks, 1):
        work = out_dir / "work" / task.id
        art = out_dir / "artifacts" / task.id
        work.mkdir(parents=True, exist_ok=True)
        art.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(workdir=work, artifact_dir=art, tracer=tracer)

        t_start = time.time()
        try:
            result, replans = run_task_with_replan(task, ctx, max_replans=max_replans)
            grader_ok, grader_reason = grade_task(task, ctx, result, t_start, time.time())
        except Exception as e:
            result = RunResult()
            replans = 0
            grader_ok, grader_reason = False, f"task_raised:{type(e).__name__}: {e}"
        t_end = time.time()

        tracer.emit(
            "bench_task_done", node_id=task.id,
            passed=grader_ok, replans=replans,
            reason=grader_reason, latency_ms=round((t_end - t_start) * 1000, 1),
        )
        summary.tasks.append(TaskMetrics(
            task_id=task.id,
            category=task.category,
            passed=grader_ok,
            tokens=0,
            latency_ms=(t_end - t_start) * 1000,
            replans=replans,
            grader_reason=grader_reason,
        ))
        if on_progress:
            on_progress(i, len(all_tasks), task.id, grader_ok)

    # Post-pass: attribute tokens per task by trace node-id prefix.
    tok_map = _attribute_tokens(trace_path, [t.task_id for t in summary.tasks])
    for m in summary.tasks:
        m.tokens = tok_map.get(m.task_id, 0)

    tracer.emit("bench_end", tasks=len(summary.tasks),
                passed=sum(1 for m in summary.tasks if m.passed))

    (out_dir / "summary.json").write_text(
        json.dumps(_summary_dict(summary), indent=2), encoding="utf-8",
    )
    (out_dir / "results.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _summary_dict(s: RunSummary) -> dict:
    return {
        "run_id": s.run_id,
        "started": s.started,
        "trace_path": s.trace_path,
        "out_dir": s.out_dir,
        "tasks": [
            {
                "task_id": m.task_id,
                "category": m.category,
                "passed": m.passed,
                "tokens": m.tokens,
                "latency_ms": round(m.latency_ms, 1),
                "replans": m.replans,
                "grader_reason": m.grader_reason,
            }
            for m in s.tasks
        ],
    }


def render_markdown(summary: RunSummary) -> str:
    by_cat: dict[str, list[TaskMetrics]] = {}
    for m in summary.tasks:
        by_cat.setdefault(m.category, []).append(m)

    lines = [
        "| Category | Tasks | Success | Mean tokens | Mean latency (ms) | Replans |",
        "|---|---|---|---|---|---|",
    ]
    all_metrics: list[TaskMetrics] = []
    for cat in sorted(by_cat):
        ms = by_cat[cat]
        all_metrics.extend(ms)
        lines.append(_row(cat, ms))
    lines.append(_row("**Overall**", all_metrics))
    return "\n".join(lines) + "\n"


def _row(label: str, ms: list[TaskMetrics]) -> str:
    n = len(ms)
    if n == 0:
        return f"| {label} | 0 | - | - | - | - |"
    passed = sum(1 for m in ms if m.passed)
    total_tokens = sum(m.tokens for m in ms)
    total_lat = sum(m.latency_ms for m in ms)
    total_replans = sum(m.replans for m in ms)
    pct = int(round(100.0 * passed / n))
    return (
        f"| {label} | {n} | {passed}/{n} ({pct}%) "
        f"| {total_tokens // n} | {total_lat / n:.0f} | {total_replans} |"
    )


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/selftest")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bench/runs")

    def _log(i, n, tid, ok):
        print(f"[{i}/{n}] {tid}: {'PASS' if ok else 'FAIL'}")

    s = run_bench([root], out, on_progress=_log)
    print()
    print(render_markdown(s))
    print(f"trace: {s.trace_path}")