import pytest

from verity.tools import REGISTRY, ToolContext
from verity.trace import Tracer, verify_chain
from verity.verify import validate


def test_read_file_passes_end_to_end(tmp_path):
    work = tmp_path / "work"; work.mkdir()
    (work / "hello.txt").write_text("hi")
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    ctx = ToolContext(workdir=work, artifact_dir=art, tracer=t)

    ev = REGISTRY["read_file"]["run"]({"path": "hello.txt"}, ctx, "n1")
    t.emit("node_run", node_id="n1", tool="read_file")
    res = validate(ev, REGISTRY["read_file"]["verify"])
    t.emit("node_verify", node_id="n1", verdict=res.verdict, reason=res.reason)

    assert res.verdict == "PASS"
    ok, msg = verify_chain(t.path)
    assert ok, msg


def test_missing_file_raises_in_run(tmp_path):
    """run() is allowed to raise; the scheduler will turn it into a FAIL later."""
    work = tmp_path / "work"; work.mkdir()
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    ctx = ToolContext(workdir=work, artifact_dir=art, tracer=t)
    with pytest.raises(FileNotFoundError):
        REGISTRY["read_file"]["run"]({"path": "nope.txt"}, ctx, "n1")
