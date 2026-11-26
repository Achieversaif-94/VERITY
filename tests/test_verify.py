from verity.verify import Evidence, validate


def _verifier(ev: Evidence) -> tuple[bool, str]:
    return (True, "ok")


def test_no_evidence_fails():
    res = validate(None, _verifier)
    assert res.verdict == "FAIL"
    assert res.reason == "no_evidence"


def test_missing_artifact_path_fails():
    ev = Evidence(
        tool="read_file", artifact_path="", artifact_sha256="abc",
        schema_ok=True, verify_passed=True, produced_at_node="n1",
    )
    res = validate(ev, _verifier)
    assert res.verdict == "FAIL"
    assert res.reason == "no_artifact_path"


def test_verify_passed_field_is_not_trusted():
    """An LLM claiming verify_passed=True must not be believed; only the tool decides."""
    ev = Evidence(
        tool="read_file", artifact_path="/tmp/x", artifact_sha256="abc",
        schema_ok=True, verify_passed=True, produced_at_node="n1",
    )
    res = validate(ev, lambda e: (False, "verify_failed"))
    assert res.verdict == "FAIL"


def test_verifier_exception_is_fail():
    ev = Evidence(
        tool="read_file", artifact_path="/tmp/x", artifact_sha256="abc",
        schema_ok=True, verify_passed=True, produced_at_node="n1",
    )

    def boom(_ev):
        raise RuntimeError("nope")

    res = validate(ev, boom)
    assert res.verdict == "FAIL"
    assert "verifier_raised" in res.reason
