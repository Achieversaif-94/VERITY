from __future__ import annotations

import json
from pathlib import Path

from verity.bench import BenchCase, run_suite
from verity.plan import Plan, Task


def test_default_benchmark_suite_passes(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    report = run_suite(report_path=out)

    assert report.total == 3
    assert report.passed == 3
    assert report.failed == 0
    assert all(c.trace_valid for c in report.cases)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["passed"] == 3
    assert len(saved["cases"]) == 3


def test_benchmark_detects_failing_case() -> None:
    bad_case = BenchCase(
        name="missing_file_fail",
        plan=Plan(
            tasks=[
                Task(id="r1", tool="read_file", inputs={"path": "does_not_exist.txt"}),
            ]
        ),
        expect_all_ok=True,
    )
    report = run_suite(cases=[bad_case])
    assert report.total == 1
    assert report.failed == 1
    assert report.cases[0].passed is False
    assert report.cases[0].trace_valid is True