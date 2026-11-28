"""Selftest B: harness should report FAIL. This grader is adversarial on purpose --
if the harness ever reports this task as passed, the harness is broken."""


def grade(ctx) -> tuple[bool, str]:
    return False, "deliberate fail: grader rejects any content"