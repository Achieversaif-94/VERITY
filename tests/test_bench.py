from pathlib import Path

import pytest

from verity import bench
from verity.exec import ExecResult
from verity.plan import Plan, Task
from verity.sched import RunResult
from verity.tools import ToolContext
from verity.trace import Tracer
from verity.verify import VerifyResult


def _ctx(tmp_path):
    work = tmp_path / "work"; work.mkdir()
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    return ToolContext(workdir=work, artifact_dir=art, tracer=t)


def _bench_task(tmp_path, task_id="fake", category="t"):
    g = tmp_path / "grader.py"
    g.write_text("def grade(ctx): return (True, 'ok')\n")
    return bench.BenchTask(
        id=task_id, category=category,
        plan=Plan(tasks=[Task(id="n1", tool="read_file", inputs={"path": "x"})]),
        grader_path=g, dir=tmp_path,
    )


def _failing_result():
    return RunResult(
        states={"n1": "failed"},
        results={"n1": ExecResult("failed", None, VerifyResult("FAIL", "no_file", None), 0.0)},
    )


def _passing_result():
    return RunResult(
        states={"n1": "passed"},
        results={"n1": ExecResult("passed", None, VerifyResult("PASS", "ok", None), 0.0)},
    )


def test_selftest_one_pass_one_fail(tmp_path):
    """The critical proof: harness correctly reports one pass and one fail."""
    selftest = Path("bench/selftest")
    if not selftest.exists():
        pytest.skip("bench/selftest not created")
    s = bench.run_bench([selftest], tmp_path)
    by_id = {m.task_id: m for m in s.tasks}
    assert by_id["pass_simple"].passed, by_id["pass_simple"].grader_reason
    assert not by_id["fail_wrong_content"].passed
    assert by_id["fail_wrong_content"].replans >= 0  # replan may abort due to network guard

    # outputs written
    assert (Path(s.out_dir) / "summary.json").exists()
    assert (Path(s.out_dir) / "results.md").exists()


def test_markdown_table_shape(tmp_path):
    selftest = Path("bench/selftest")
    if not selftest.exists():
        pytest.skip("bench/selftest not created")
    s = bench.run_bench([selftest], tmp_path)
    md = bench.render_markdown(s)
    assert "| Category |" in md
    assert "| selftest |" in md
    assert "**Overall**" in md
    # 2 tasks, 1 passed -> 50%
    assert "1/2 (50%)" in md


def test_replan_cap(tmp_path, monkeypatch):
    task = _bench_task(tmp_path)
    ctx = _ctx(tmp_path)

    async def fake_run_plan(plan, c):
        return _failing_result()

    def fake_replan(plan, node, reason, *, tracer, max_attempts=2):
        return plan, None

    monkeypatch.setattr("verity.bench.run_plan", fake_run_plan)
    monkeypatch.setattr("verity.bench.replan_mod.replan", fake_replan)

    result, replans = bench.run_task_with_replan(task, ctx, max_replans=3)
    assert replans == 3
    assert result.states == {"n1": "failed"}


def test_replan_recovers_on_second_attempt(tmp_path, monkeypatch):
    task = _bench_task(tmp_path)
    ctx = _ctx(tmp_path)
    calls = {"n": 0}

    async def fake_run_plan(plan, c):
        calls["n"] += 1
        return _failing_result() if calls["n"] == 1 else _passing_result()

    def fake_replan(plan, node, reason, *, tracer, max_attempts=2):
        return plan, None

    monkeypatch.setattr("verity.bench.run_plan", fake_run_plan)
    monkeypatch.setattr("verity.bench.replan_mod.replan", fake_replan)

    result, replans = bench.run_task_with_replan(task, ctx, max_replans=2)
    assert replans == 1
    assert result.states == {"n1": "passed"}


def test_grader_exception_reported_as_fail(tmp_path):
    g = tmp_path / "grader.py"
    g.write_text("def grade(ctx):\n    raise RuntimeError('boom')\n")
    task = bench.BenchTask(
        id="f", category="t",
        plan=Plan(tasks=[Task(id="n1", tool="read_file", inputs={"path": "x"})]),
        grader_path=g, dir=tmp_path,
    )
    ctx = _ctx(tmp_path)
    ok, reason = bench.grade_task(task, ctx, _passing_result(), 0.0, 1.0)
    assert not ok
    assert "grader_raised" in reason


def test_filter_pattern(tmp_path):
    selftest = Path("bench/selftest")
    if not selftest.exists():
        pytest.skip("bench/selftest not created")
    s = bench.run_bench([selftest], tmp_path, filter_pattern="pass_simple")
    assert len(s.tasks) == 1
    assert s.tasks[0].task_id == "pass_simple"