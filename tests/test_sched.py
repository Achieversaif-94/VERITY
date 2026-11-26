import asyncio

from verity.plan import Plan, Task
from verity.sched import run_plan
from verity.tools import ToolContext
from verity.trace import Tracer, verify_chain


def _ctx(tmp_path):
    work = tmp_path / "work"; work.mkdir()
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    return ToolContext(workdir=work, artifact_dir=art, tracer=t), work


def test_three_node_chain(tmp_path):
    """w1 -> r1 -> w2. Middle depends on first, last depends on middle."""
    ctx, work = _ctx(tmp_path)
    plan = Plan(tasks=[
        Task(id="w1", tool="write_file",
             inputs={"path": "a.txt", "content": "hello"}),
        Task(id="r1", tool="read_file",
             inputs={"path": "a.txt"}, deps=["w1"]),
        Task(id="w2", tool="write_file",
             inputs={"path": "b.txt", "content": "world"}, deps=["r1"]),
    ])
    result = asyncio.run(run_plan(plan, ctx))
    assert result.states == {"w1": "passed", "r1": "passed", "w2": "passed"}
    assert (work / "a.txt").read_text() == "hello"
    assert (work / "b.txt").read_text() == "world"
    ok, msg = verify_chain(ctx.tracer.path)
    assert ok, msg


def test_dep_failure_aborts_descendant(tmp_path):
    ctx, _ = _ctx(tmp_path)
    plan = Plan(tasks=[
        Task(id="missing", tool="read_file", inputs={"path": "nope.txt"}),
        Task(id="downstream", tool="read_file",
             inputs={"path": "also_nope.txt"}, deps=["missing"]),
    ])
    result = asyncio.run(run_plan(plan, ctx))
    assert result.states == {"missing": "failed", "downstream": "aborted"}


def test_cycle_rejected_at_construction():
    import pytest
    with pytest.raises(Exception):
        Plan(tasks=[
            Task(id="a", tool="read_file", inputs={"path": "x"}, deps=["b"]),
            Task(id="b", tool="read_file", inputs={"path": "y"}, deps=["a"]),
        ])