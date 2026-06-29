from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent.eval import load_eval_cases, run_eval_suite


class AgentEvalTests(unittest.TestCase):
    def test_default_eval_cases_pass(self) -> None:
        report = run_eval_suite()

        self.assertEqual(report["summary"]["failed_cases"], 0)
        self.assertGreaterEqual(report["summary"]["total_cases"], 3)

    def test_eval_detects_regression(self) -> None:
        case = """{"id":"regression_case","description":"impossible expectation","resume_text":"Python 后端项目。","expected":{"must_have_result":"pass","min_score":100}}"""
        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            cases_path.write_text(case, encoding="utf-8")
            report = run_eval_suite(cases_path)

        self.assertEqual(report["summary"]["failed_cases"], 1)

    def test_load_eval_cases_ignores_comments_and_blank_lines(self) -> None:
        content = """
# comment
{"id":"case_1","resume_text":"Python Agent 项目。","expected":{"must_have_result":"pass"}}
"""
        with tempfile.TemporaryDirectory() as tmp:
            cases_path = Path(tmp) / "cases.jsonl"
            cases_path.write_text(content, encoding="utf-8")
            cases = load_eval_cases(cases_path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["id"], "case_1")


if __name__ == "__main__":
    unittest.main()
