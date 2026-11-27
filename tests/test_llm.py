import pytest
from pydantic import BaseModel

from verity import llm
from verity.trace import Tracer, verify_chain


class Out(BaseModel):
    answer: int


def test_groq_success(tmp_path, monkeypatch):
    tracer = Tracer(tmp_path / "t.jsonl", run_id="r1")
    monkeypatch.setattr(
        "verity.llm._groq_call",
        lambda s, u, timeout=30.0: ('{"answer": 42}', {"prompt_tokens": 10, "completion_tokens": 5}),
    )
    monkeypatch.setattr(
        "verity.llm._gemini_call",
        lambda s, u, timeout=30.0: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )
    out = llm.complete_json("sys", "user", Out, tracer=tracer, node_id="n1")
    assert out.answer == 42
    ok, msg = verify_chain(tracer.path)
    assert ok, msg
    events = (tmp_path / "t.jsonl").read_text().splitlines()
    assert any('"kind":"llm_call"' in e and '"status":"ok"' in e for e in events)


def test_groq_invalid_json_falls_back_to_gemini(tmp_path, monkeypatch):
    tracer = Tracer(tmp_path / "t.jsonl", run_id="r1")
    monkeypatch.setattr(
        "verity.llm._groq_call",
        lambda s, u, timeout=30.0: ("not json at all", {"prompt_tokens": 5, "completion_tokens": 2}),
    )
    monkeypatch.setattr(
        "verity.llm._gemini_call",
        lambda s, u, timeout=30.0: ('{"answer": 7}', {"prompt_tokens": 8, "completion_tokens": 3}),
    )
    out = llm.complete_json("sys", "user", Out, tracer=tracer, node_id="n1")
    assert out.answer == 7


def test_all_providers_fail_raises_llm_error(monkeypatch):
    monkeypatch.setattr(
        "verity.llm._groq_call",
        lambda s, u, timeout=30.0: ("garbage", {}),
    )
    monkeypatch.setattr(
        "verity.llm._gemini_call",
        lambda s, u, timeout=30.0: ("also garbage", {}),
    )
    with pytest.raises(llm.LLMError):
        llm.complete_json("sys", "user", Out)


def test_schema_validation_rejects_wrong_shape(monkeypatch):
    monkeypatch.setattr(
        "verity.llm._groq_call",
        lambda s, u, timeout=30.0: ('{"wrong": "field"}', {}),
    )
    monkeypatch.setattr(
        "verity.llm._gemini_call",
        lambda s, u, timeout=30.0: ('{"also": "wrong"}', {}),
    )
    with pytest.raises(llm.LLMError):
        llm.complete_json("sys", "user", Out)


def test_code_fence_tolerated(monkeypatch):
    fenced = '```json\n{"answer": 99}\n```'
    monkeypatch.setattr(
        "verity.llm._groq_call",
        lambda s, u, timeout=30.0: (fenced, {}),
    )
    out = llm.complete_json("sys", "user", Out)
    assert out.answer == 99