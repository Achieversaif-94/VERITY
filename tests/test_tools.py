from unittest.mock import Mock

from verity.tools import REGISTRY, ToolContext
from verity.trace import Tracer
from verity.verify import validate


def _ctx(tmp_path):
    work = tmp_path / "work"; work.mkdir()
    art = tmp_path / "art"; art.mkdir()
    t = Tracer(tmp_path / "t.jsonl", run_id="r1")
    return ToolContext(workdir=work, artifact_dir=art, tracer=t), work


def _run_verify(tool, inputs, ctx, node_id="n1"):
    ev = REGISTRY[tool]["run"](inputs, ctx, node_id)
    return validate(ev, REGISTRY[tool]["verify"])


def test_run_python_passes(tmp_path):
    ctx, _ = _ctx(tmp_path)
    res = _run_verify("run_python", {
        "code": "print('hello')",
        "expected_stdout_contains": "hello",
    }, ctx)
    assert res.verdict == "PASS", res.reason


def test_run_python_wrong_stdout_fails(tmp_path):
    ctx, _ = _ctx(tmp_path)
    res = _run_verify("run_python", {
        "code": "print('world')",
        "expected_stdout_contains": "hello",
    }, ctx)
    assert res.verdict == "FAIL"
    assert res.reason == "stdout_missing_substring"


def test_run_python_nonzero_exit_fails(tmp_path):
    ctx, _ = _ctx(tmp_path)
    res = _run_verify("run_python", {
        "code": "raise SystemExit(1)",
        "expected_exit_code": 0,
    }, ctx)
    assert res.verdict == "FAIL"
    assert "exit_code" in res.reason


def test_grep_counts_matches(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "data.txt").write_text("apple\nbanana\napricot\n")
    res = _run_verify("grep", {
        "path": "data.txt", "pattern": "^ap", "expected_count": 2,
    }, ctx)
    assert res.verdict == "PASS", res.reason


def test_grep_wrong_count_fails(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "data.txt").write_text("apple\nbanana\n")
    res = _run_verify("grep", {
        "path": "data.txt", "pattern": "^ap", "expected_count": 5,
    }, ctx)
    assert res.verdict == "FAIL"
    assert "count" in res.reason


def test_csv_sum(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "nums.csv").write_text("a,b\n1,10\n2,20\n3,30\n")
    res = _run_verify("csv_query", {
        "path": "nums.csv", "op": "sum", "column": "a", "expected_value": 6,
    }, ctx)
    assert res.verdict == "PASS", res.reason


def test_csv_wrong_value_fails(tmp_path):
    ctx, work = _ctx(tmp_path)
    (work / "nums.csv").write_text("a,b\n1,10\n2,20\n")
    res = _run_verify("csv_query", {
        "path": "nums.csv", "op": "sum", "column": "a", "expected_value": 999,
    }, ctx)
    assert res.verdict == "FAIL"
    assert "value" in res.reason


def test_http_fetch_verify_with_mock(tmp_path, monkeypatch):
    ctx, _ = _ctx(tmp_path)

    def fake_get(url, **kwargs):
        m = Mock()
        m.status_code = 200
        m.text = "hello world"
        m.url = url
        return m

    monkeypatch.setattr("verity.tools.httpx.get", fake_get)

    res = _run_verify("http_fetch", {
        "url": "https://example.test/",
        "expect_status": 200,
        "expect_body_contains": "hello",
    }, ctx)
    assert res.verdict == "PASS", res.reason


def test_http_fetch_wrong_body_fails(tmp_path, monkeypatch):
    ctx, _ = _ctx(tmp_path)

    def fake_get(url, **kwargs):
        m = Mock()
        m.status_code = 200
        m.text = "nope"
        m.url = url
        return m

    monkeypatch.setattr("verity.tools.httpx.get", fake_get)

    res = _run_verify("http_fetch", {
        "url": "https://example.test/",
        "expect_body_contains": "hello",
    }, ctx)
    assert res.verdict == "FAIL"
    assert res.reason == "body_missing_substring"