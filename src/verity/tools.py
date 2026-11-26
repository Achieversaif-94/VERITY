"""Tool registry. Each tool = {run(inputs, ctx, node_id) -> Evidence, verify(ev) -> (bool, str)}.

verify() is pure: same artifact bytes + same metadata => same result.
No network, no clock, no randomness inside verify().
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from .trace import Tracer
from .verify import Evidence


@dataclass
class ToolContext:
    workdir: Path
    artifact_dir: Path
    tracer: Tracer


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _resolve(ctx: ToolContext, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ctx.workdir / path)


def _write_evidence(
    ctx: ToolContext,
    node_id: str,
    tool: str,
    artifact_name: str,
    payload: bytes,
    metadata: dict | None = None,
) -> Evidence:
    d = ctx.artifact_dir / node_id
    d.mkdir(parents=True, exist_ok=True)
    art = d / artifact_name
    art.write_bytes(payload)
    return Evidence(
        tool=tool,
        artifact_path=str(art),
        artifact_sha256=_sha(payload),
        schema_ok=True,
        verify_passed=False,
        produced_at_node=node_id,
        metadata=metadata or {},
    )


def _check_hash_and_read(ev: Evidence) -> tuple[bytes | None, str]:
    try:
        data = Path(ev.artifact_path).read_bytes()
    except FileNotFoundError:
        return None, "artifact_missing"
    if _sha(data) != ev.artifact_sha256:
        return None, "artifact_hash_mismatch"
    return data, "ok"


# ---------- read_file ----------

def read_file_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    src = _resolve(ctx, inputs["path"])
    data = src.read_bytes()
    return _write_evidence(ctx, node_id, "read_file", "content.bin", data)


def read_file_verify(ev: Evidence) -> tuple[bool, str]:
    data, msg = _check_hash_and_read(ev)
    return (data is not None, msg)


# ---------- write_file ----------

def write_file_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    dst = _resolve(ctx, inputs["path"])
    content = inputs["content"]
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return _write_evidence(ctx, node_id, "write_file", "written.bin", data)


def write_file_verify(ev: Evidence) -> tuple[bool, str]:
    return read_file_verify(ev)


# ---------- run_python ----------

def run_python_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    code = inputs["code"]
    timeout = float(inputs.get("timeout_seconds", 30.0))
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=str(ctx.workdir),
        capture_output=True,
        timeout=timeout,
    )
    payload = {
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
        "returncode": proc.returncode,
    }
    metadata = {
        "expected_stdout_contains": inputs.get("expected_stdout_contains"),
        "expected_exit_code": inputs.get("expected_exit_code", 0),
    }
    return _write_evidence(
        ctx, node_id, "run_python", "result.json",
        json.dumps(payload).encode("utf-8"), metadata,
    )


def run_python_verify(ev: Evidence) -> tuple[bool, str]:
    data, msg = _check_hash_and_read(ev)
    if data is None:
        return False, msg
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False, "artifact_not_json"
    expected_exit = ev.metadata.get("expected_exit_code", 0)
    if payload.get("returncode") != expected_exit:
        return False, f"exit_code {payload.get('returncode')} != {expected_exit}"
    sub = ev.metadata.get("expected_stdout_contains")
    if sub is not None and sub not in payload.get("stdout", ""):
        return False, "stdout_missing_substring"
    return True, "ok"


# ---------- grep ----------

def grep_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    src = _resolve(ctx, inputs["path"])
    pattern = re.compile(inputs["pattern"])
    lines = src.read_text(encoding="utf-8").splitlines()
    matches = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(lines)
        if pattern.search(line)
    ]
    metadata = {"expected_count": inputs.get("expected_count")}
    return _write_evidence(
        ctx, node_id, "grep", "matches.json",
        json.dumps(matches).encode("utf-8"), metadata,
    )


def grep_verify(ev: Evidence) -> tuple[bool, str]:
    data, msg = _check_hash_and_read(ev)
    if data is None:
        return False, msg
    try:
        matches = json.loads(data)
    except json.JSONDecodeError:
        return False, "artifact_not_json"
    expected = ev.metadata.get("expected_count")
    if expected is not None and len(matches) != expected:
        return False, f"count {len(matches)} != {expected}"
    return True, "ok"


# ---------- csv_query ----------

def csv_query_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    src = _resolve(ctx, inputs["path"])
    op = inputs["op"]
    column = inputs.get("column")
    with src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if op == "count":
        result: float = float(len(rows))
    elif op == "unique":
        if not column:
            raise ValueError("column required for unique")
        result = float(len({r[column] for r in rows}))
    elif op in ("sum", "mean", "min", "max"):
        if not column:
            raise ValueError(f"column required for {op}")
        vals = [float(r[column]) for r in rows]
        if op == "sum":
            result = sum(vals)
        elif op == "mean":
            result = (sum(vals) / len(vals)) if vals else 0.0
        elif op == "min":
            result = min(vals) if vals else 0.0
        else:
            result = max(vals) if vals else 0.0
    else:
        raise ValueError(f"unknown op {op}")
    payload = {"op": op, "column": column, "value": result}
    metadata = {"expected_value": inputs.get("expected_value")}
    return _write_evidence(
        ctx, node_id, "csv_query", "result.json",
        json.dumps(payload).encode("utf-8"), metadata,
    )


def csv_query_verify(ev: Evidence) -> tuple[bool, str]:
    data, msg = _check_hash_and_read(ev)
    if data is None:
        return False, msg
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False, "artifact_not_json"
    expected = ev.metadata.get("expected_value")
    if expected is not None:
        if abs(float(payload["value"]) - float(expected)) > 1e-6:
            return False, f"value {payload['value']} != {expected}"
    return True, "ok"


# ---------- http_fetch ----------

def http_fetch_run(inputs: dict, ctx: ToolContext, node_id: str) -> Evidence:
    url = inputs["url"]
    timeout = float(inputs.get("timeout_seconds", 10.0))
    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    payload = {"status": resp.status_code, "url": str(resp.url), "body": resp.text}
    metadata = {
        "expect_status": inputs.get("expect_status"),
        "expect_body_contains": inputs.get("expect_body_contains"),
    }
    return _write_evidence(
        ctx, node_id, "http_fetch", "response.json",
        json.dumps(payload).encode("utf-8"), metadata,
    )


def http_fetch_verify(ev: Evidence) -> tuple[bool, str]:
    data, msg = _check_hash_and_read(ev)
    if data is None:
        return False, msg
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False, "artifact_not_json"
    expected_status = ev.metadata.get("expect_status")
    if expected_status is not None and payload.get("status") != expected_status:
        return False, f"status {payload.get('status')} != {expected_status}"
    expected_body = ev.metadata.get("expect_body_contains")
    if expected_body is not None and expected_body not in payload.get("body", ""):
        return False, "body_missing_substring"
    return True, "ok"


REGISTRY: dict[str, dict] = {
    "read_file":  {"run": read_file_run,  "verify": read_file_verify},
    "write_file": {"run": write_file_run, "verify": write_file_verify},
    "run_python": {"run": run_python_run, "verify": run_python_verify},
    "grep":       {"run": grep_run,       "verify": grep_verify},
    "csv_query":  {"run": csv_query_run,  "verify": csv_query_verify},
    "http_fetch": {"run": http_fetch_run, "verify": http_fetch_verify},
}