"""Selftest A: harness should report PASS."""
from pathlib import Path


def grade(ctx) -> tuple[bool, str]:
    p = Path(ctx.workdir) / "hello.txt"
    if not p.exists():
        return False, "hello.txt missing"
    if p.read_text(encoding="utf-8") != "hello verity":
        return False, f"content mismatch: {p.read_text(encoding='utf-8')!r}"
    if ctx.states.get("w1") != "passed" or ctx.states.get("r1") != "passed":
        return False, f"unexpected states: {ctx.states}"
    return True, "ok"