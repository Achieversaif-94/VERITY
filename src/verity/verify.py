"""The load-bearing file.

A task can pass ONLY if it produced an Evidence object whose fresh re-run of the
tool's deterministic verify() returns (True, _). No evidence => FAIL.
No exceptions. Raw LLM text is not evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Verdict = Literal["PASS", "FAIL"]
Verifier = Callable[["Evidence"], tuple[bool, str]]


@dataclass(frozen=True)
class Evidence:
    tool: str
    artifact_path: str
    artifact_sha256: str
    schema_ok: bool
    verify_passed: bool
    produced_at_node: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyResult:
    verdict: Verdict
    reason: str
    evidence: Evidence | None = None


def validate(evidence: Evidence | None, verifier: Verifier | None) -> VerifyResult:
    if evidence is None:
        return VerifyResult("FAIL", "no_evidence")
    if not evidence.artifact_path:
        return VerifyResult("FAIL", "no_artifact_path", evidence)
    if not evidence.artifact_sha256:
        return VerifyResult("FAIL", "no_artifact_hash", evidence)
    if not evidence.schema_ok:
        return VerifyResult("FAIL", "schema_mismatch", evidence)
    if verifier is None:
        return VerifyResult("FAIL", "unknown_tool", evidence)
    try:
        ok, detail = verifier(evidence)
    except Exception as e:
        return VerifyResult("FAIL", f"verifier_raised:{type(e).__name__}", evidence)
    if not ok:
        return VerifyResult("FAIL", detail or "verify_failed", evidence)
    return VerifyResult("PASS", "ok", evidence)