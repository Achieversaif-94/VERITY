from verity.exec import execute_task
from verity.plan import Task
from verity.tools import ToolContext
from verity.trace import Tracer


def _ctx(tmp_path):
    work = tmp_path / "work"; work.mkdir()
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    return ToolContext(workdir=work, artifact_dir=art, tracer=t), work


def test_read_file_task_passes(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "a.txt").write_text("hi")
    task = Task(id="n1", tool="read_file", inputs={"path": "a.txt"})
    res = execute_task(task, ctx)
    assert res.state == "passed"
    assert res.verify is not None and res.verify.verdict == "PASS"


def test_missing_file_task_fails(tmp_path):
    ctx, _ = _ctx(tmp_path)
    task = Task(id="n1", tool="read_file", inputs={"path": "nope.txt"})
    res = execute_task(task, ctx)
    assert res.state == "failed"


def test_unknown_tool_fails(tmp_path):
    ctx, _ = _ctx(tmp_path)
    task = Task(id="n1", tool="nonexistent", inputs={})
    res = execute_task(task, ctx)
    assert res.state == "failed"