from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tool_agent import run_tool_calling_workflow


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CASES = ROOT / "data" / "eval" / "resume_eval_cases.jsonl"


@dataclass(frozen=True)
class EvalAssertion:
    name: str
    passed: bool
    expected: Any
    actual: Any


def load_eval_cases(path: str | Path = DEFAULT_EVAL_CASES) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval case file not found: {eval_path}")

    for line_number, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        case = json.loads(stripped)
        if "id" not in case or "resume_text" not in case or "expected" not in case:
            raise ValueError(f"Invalid eval case at line {line_number}: id, resume_text and expected are required")
        cases.append(case)
    return cases


def run_eval_suite(
    cases_path: str | Path = DEFAULT_EVAL_CASES,
    out_path: str | Path | None = None,
    redact: bool = False,
) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    case_results = [_run_case(case, redact=redact) for case in cases]
    passed_cases = sum(1 for item in case_results if item["passed"])
    failed_cases = len(case_results) - passed_cases
    total_assertions = sum(len(item["assertions"]) for item in case_results)
    passed_assertions = sum(
        1
        for item in case_results
        for assertion in item["assertions"]
        if assertion["passed"]
    )

    report = {
        "suite": "resume_screen_agent_regression_eval",
        "cases_path": str(cases_path),
        "summary": {
            "total_cases": len(case_results),
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "total_assertions": total_assertions,
            "passed_assertions": passed_assertions,
            "failed_assertions": total_assertions - passed_assertions,
            "pass_rate": round(passed_cases / len(case_results), 4) if case_results else 0,
        },
        "cases": case_results,
    }

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def _run_case(case: dict[str, Any], redact: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        resume_path = Path(tmp) / f"{case['id']}.txt"
        resume_path.write_text(case["resume_text"], encoding="utf-8")
        workflow = run_tool_calling_workflow(resume_path=resume_path, redact=redact)

    final_report = workflow["final_report"]
    tool_names = [item["name"] for item in workflow["tool_trace"]]
    assertions = _evaluate_expectations(
        expected=case["expected"],
        final_report=final_report,
        tool_names=tool_names,
    )
    passed = all(item.passed for item in assertions)

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "passed": passed,
        "score": final_report.get("score"),
        "level": final_report.get("level"),
        "must_have_result": final_report.get("must_have_result"),
        "tool_trace": tool_names,
        "assertions": [
            {
                "name": item.name,
                "passed": item.passed,
                "expected": item.expected,
                "actual": item.actual,
            }
            for item in assertions
        ],
        "risks": final_report.get("risks", []),
        "missing_information": final_report.get("missing_information", []),
    }


def _evaluate_expectations(
    expected: dict[str, Any],
    final_report: dict[str, Any],
    tool_names: list[str],
) -> list[EvalAssertion]:
    assertions: list[EvalAssertion] = []

    if "must_have_result" in expected:
        assertions.append(
            _equals("must_have_result", expected["must_have_result"], final_report.get("must_have_result"))
        )
    if "level" in expected:
        assertions.append(_equals("level", expected["level"], final_report.get("level")))
    if "min_score" in expected:
        assertions.append(
            _compare("min_score", f">= {expected['min_score']}", final_report.get("score"), lambda v: v >= expected["min_score"])
        )
    if "max_score" in expected:
        assertions.append(
            _compare("max_score", f"<= {expected['max_score']}", final_report.get("score"), lambda v: v <= expected["max_score"])
        )
    if "required_tools" in expected:
        actual = [tool for tool in expected["required_tools"] if tool in tool_names]
        assertions.append(
            EvalAssertion(
                name="required_tools",
                passed=set(expected["required_tools"]).issubset(set(tool_names)),
                expected=expected["required_tools"],
                actual=actual,
            )
        )
    if "required_missing_contains" in expected:
        assertions.append(
            _contains_all(
                name="required_missing_contains",
                expected=expected["required_missing_contains"],
                actual=final_report.get("missing_information", []),
            )
        )
    if "required_risks_contains" in expected:
        assertions.append(
            _contains_all(
                name="required_risks_contains",
                expected=expected["required_risks_contains"],
                actual=final_report.get("risks", []),
            )
        )
    if "max_failed_evidence" in expected:
        evidence_check = final_report.get("evidence_verification", {})
        failed_count = evidence_check.get("failed_count", 0)
        assertions.append(
            _compare(
                "max_failed_evidence",
                f"<= {expected['max_failed_evidence']}",
                failed_count,
                lambda v: v <= expected["max_failed_evidence"],
            )
        )

    return assertions


def _equals(name: str, expected: Any, actual: Any) -> EvalAssertion:
    return EvalAssertion(name=name, passed=actual == expected, expected=expected, actual=actual)


def _compare(name: str, expected: Any, actual: Any, predicate: Any) -> EvalAssertion:
    try:
        passed = predicate(actual)
    except TypeError:
        passed = False
    return EvalAssertion(name=name, passed=passed, expected=expected, actual=actual)


def _contains_all(name: str, expected: list[str], actual: list[Any]) -> EvalAssertion:
    actual_text = "\n".join(str(item) for item in actual)
    hits = [item for item in expected if item in actual_text]
    return EvalAssertion(
        name=name,
        passed=len(hits) == len(expected),
        expected=expected,
        actual=hits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic regression evals for the resume screen agent.")
    parser.add_argument("--cases", default=str(DEFAULT_EVAL_CASES), help="JSONL eval case file.")
    parser.add_argument("--out", default=str(ROOT / "results" / "agent_eval_report.json"), help="Output JSON report path.")
    parser.add_argument("--redact", action="store_true", help="Redact basic personal info before workflow execution.")
    args = parser.parse_args()

    report = run_eval_suite(cases_path=args.cases, out_path=args.out, redact=args.redact)
    summary = report["summary"]
    print(
        f"Eval complete: {summary['passed_cases']}/{summary['total_cases']} cases passed, "
        f"{summary['passed_assertions']}/{summary['total_assertions']} assertions passed."
    )
    print(f"Report written to {args.out}")
    if summary["failed_cases"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
