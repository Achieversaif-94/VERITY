"""Part G: Offline benchmark harness for VERITY plans, tools, and trace integrity."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity.plan import Plan, Task
from verity.sched import run_plan
from verity.tools import ToolContext
from verity.trace import Tracer, verify_chain

OK_STATES = {"ok", "done", "succeeded", "passed", "verified"}


@dataclass
class BenchCase:
    name: str
    plan: Plan
    setup_files: dict[str, str] = field(default_factory=dict)
    expect_all_ok: bool = True


@dataclass
class CaseMetrics:
    name: str
    passed: bool
    elapsed_ms: float
    tasks_total: int
    tasks_ok: int
    trace_valid: bool
    trace_msg: str
    states: dict[str, str]


@dataclass
class BenchReport:
    total: int
    passed: int
    failed: int
    total_ms: float
    cases: list[CaseMetrics]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def default_suite() -> list[BenchCase]:
    """Deterministic offline benchmark cases exercising the DAG scheduler and tools."""
    return [
        BenchCase(
            name="write_then_read",
            plan=Plan(
                tasks=[
                    Task(
                        id="t1",
                        tool="write_file",
                        inputs={"path": "hello.txt", "content": "verity bench ok"},
                    ),
                    Task(
                        id="t2",
                        tool="read_file",
                        inputs={"path": "hello.txt"},
                        deps=["t1"],
                    ),
                ]
            ),
        ),
        BenchCase(
            name="seed_and_grep",
            setup_files={"notes.txt": "alpha\nverity_target_line\nomega\n"},
            plan=Plan(
                tasks=[
                    Task(
                        id="g1",
                        tool="grep",
                        inputs={"path": "notes.txt", "pattern": "verity_target"},
                    ),
                ]
            ),
        ),
        BenchCase(
            name="python_compute_dag",
            plan=Plan(
                tasks=[
                    Task(
                        id="p1",
                        tool="run_python",
                        inputs={"code": "print(sum(range(1, 101)))"},
                    ),
                    Task(
                        id="p2",
                        tool="write_file",
                        inputs={"path": "done.txt", "content": "5050"},
                        deps=["p1"],
                    ),
                ]
            ),
        ),
    ]


def run_case(case: BenchCase, root: Path) -> CaseMetrics:
    case_dir = root / case.name
    workdir = case_dir / "work"
    artifact_dir = case_dir / "artifacts"
    trace_path = case_dir / "trace.jsonl"
    workdir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in case.setup_files.items():
        target = workdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    tracer = Tracer(trace_path)
    ctx = ToolContext(workdir=workdir, artifact_dir=artifact_dir, tracer=tracer)

    t0 = time.perf_counter()
    res = asyncio.run(run_plan(case.plan, ctx))
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if hasattr(tracer, "close"):
        tracer.close()

    trace_valid, trace_msg = verify_chain(trace_path)
    tasks_total = len(case.plan.tasks)
    tasks_ok = sum(1 for s in res.states.values() if str(s).lower() in OK_STATES)
    all_ok = tasks_ok == tasks_total
    passed = trace_valid and (all_ok if case.expect_all_ok else not all_ok)

    return CaseMetrics(
        name=case.name,
        passed=passed,
        elapsed_ms=elapsed_ms,
        tasks_total=tasks_total,
        tasks_ok=tasks_ok,
        trace_valid=trace_valid,
        trace_msg=trace_msg,
        states={k: str(v) for k, v in res.states.items()},
    )


def run_suite(
    cases: list[BenchCase] | None = None,
    report_path: Path | None = None,
) -> BenchReport:
    suite = cases if cases is not None else default_suite()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="verity_bench_") as tmp:
        root = Path(tmp)
        metrics = [run_case(c, root) for c in suite]
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    passed = sum(1 for m in metrics if m.passed)
    report = BenchReport(
        total=len(metrics),
        passed=passed,
        failed=len(metrics) - passed,
        total_ms=total_ms,
        cases=metrics,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.to_json(), encoding="utf-8")
    return report


def main() -> int:
    report = run_suite()
    print(report.to_json())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())