import json

from verity.trace import Tracer, verify_chain


def test_chain_valid(tmp_path):
    p = tmp_path / "t.jsonl"
    t = Tracer(p, run_id="r1")
    t.emit("run_start")
    t.emit("node_run", node_id="a", tool="read_file")
    t.emit("node_verify", node_id="a", verdict="PASS")
    ok, msg = verify_chain(p)
    assert ok, msg
    assert len(p.read_text().splitlines()) == 3


def test_tamper_detected(tmp_path):
    p = tmp_path / "t.jsonl"
    t = Tracer(p, run_id="r1")
    t.emit("a")
    t.emit("b")
    lines = p.read_text().splitlines()
    ev = json.loads(lines[1])
    ev["payload"] = {"tampered": True}
    lines[1] = json.dumps(ev, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    ok, msg = verify_chain(p)
    assert not ok
    assert "hash mismatch" in msg


def test_resume_chain(tmp_path):
    p = tmp_path / "t.jsonl"
    t = Tracer(p, run_id="r1")
    t.emit("a")
    t.emit("b")
    t2 = Tracer(p, run_id="r1")
    assert t2._seq == 2
    t2.emit("c")
    ok, msg = verify_chain(p)
    assert ok, msg
