import pytest

from verity import replan as rp
from verity.plan import Plan, Task
from verity.trace import Tracer


def _plan():
    return Plan(tasks=[
        Task(id="a", tool="read_file", inputs={"path": "x.txt"}),
        Task(id="b", tool="write_file", inputs={"path": "y.txt", "content": "z"}, deps=["a"]),
    ])


def _fake_llm_returns(diff_dict):
    def fake(system, user, schema, *, tracer=None, node_id=None):
        return schema.model_validate(diff_dict)
    return fake


def test_apply_diff_add(tmp_path):
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({
        "add": [{"id": "c", "tool": "read_file", "inputs": {"path": "z.txt"}, "deps": ["a"]}],
    })
    new = rp.apply_diff(plan, diff)
    assert {t.id for t in new.tasks} == {"a", "b", "c"}


def test_apply_diff_remove(tmp_path):
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({"remove": ["b"]})
    new = rp.apply_diff(plan, diff)
    assert {t.id for t in new.tasks} == {"a"}


def test_apply_diff_modify(tmp_path):
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({
        "modify": {"a": {"inputs": {"path": "new.txt"}}}
    })
    new = rp.apply_diff(plan, diff)
    a = next(t for t in new.tasks if t.id == "a")
    assert a.inputs == {"path": "new.txt"}


def test_apply_diff_rejects_unknown_remove():
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({"remove": ["nope"]})
    with pytest.raises(rp.ReplanError, match="unknown id"):
        rp.apply_diff(plan, diff)


def test_apply_diff_rejects_duplicate_add():
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({
        "add": [{"id": "a", "tool": "read_file", "inputs": {"path": "x"}, "deps": []}],
    })
    with pytest.raises(rp.ReplanError, match="already exists"):
        rp.apply_diff(plan, diff)


def test_apply_diff_rejects_cycle():
    plan = _plan()
    diff = rp.ReplanDiff.model_validate({
        "modify": {"a": {"deps": ["b"]}},  # b already depends on a -> cycle
    })
    with pytest.raises(rp.ReplanError, match="invalid"):
        rp.apply_diff(plan, diff)


def test_replan_success_logs_event(tmp_path, monkeypatch):
    tracer = Tracer(tmp_path / "t.jsonl", run_id="r1")
    diff_dict = {
        "remove": ["b"],
        "add": [{"id": "c", "tool": "read_file", "inputs": {"path": "z.txt"}, "deps": ["a"]}],
    }
    monkeypatch.setattr("verity.replan.llm.complete_json", _fake_llm_returns(diff_dict))

    new_plan, diff = rp.replan(_plan(), "a", "file_missing", tracer=tracer)

    assert {t.id for t in new_plan.tasks} == {"a", "c"}
    events = (tmp_path / "t.jsonl").read_text().splitlines()
    assert any('"kind":"replan"' in e for e in events)


def test_replan_rejects_bad_diff_then_succeeds(tmp_path, monkeypatch):
    tracer = Tracer(tmp_path / "t.jsonl", run_id="r1")
    calls = {"n": 0}

    def fake(system, user, schema, *, tracer=None, node_id=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return schema.model_validate({"remove": ["nope"]})
        return schema.model_validate({"remove": ["b"]})

    monkeypatch.setattr("verity.replan.llm.complete_json", fake)
    new_plan, _ = rp.replan(_plan(), "a", "x", tracer=tracer)
    assert {t.id for t in new_plan.tasks} == {"a"}
    assert calls["n"] == 2