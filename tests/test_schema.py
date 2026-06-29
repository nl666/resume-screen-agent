from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from resume_screen_agent.schema import extract_json_object, validate_result


class SchemaTests(unittest.TestCase):
    def test_extract_json_from_fence(self) -> None:
        data = extract_json_object('```json\n{"candidate_name":"Candidate A"}\n```')
        self.assertEqual(data["candidate_name"], "Candidate A")

    def test_validate_result(self) -> None:
        data = {
            "candidate_name": "Candidate A",
            "must_have_result": "pass",
            "score": 82,
            "level": "review",
            "strengths": ["Agent project"],
            "risks": [],
            "missing_information": [],
            "evidence": [
                {
                    "criterion": "Agent project delivery",
                    "score": 3,
                    "resume_text": "Built an internal knowledge-base Agent.",
                }
            ],
            "recommended_next_step": "Technical review",
            "human_review_required": True,
        }
        validated = validate_result(data)
        self.assertEqual(validated["score"], 82)

    def test_level_is_derived_from_score(self) -> None:
        data = {
            "candidate_name": "Candidate B",
            "must_have_result": "pass",
            "score": 55,
            "level": "backup",
            "strengths": [],
            "risks": [],
            "missing_information": [],
            "evidence": [
                {
                    "criterion": "Backend basics",
                    "score": 5,
                    "resume_text": "Python scripts and Go experience.",
                }
            ],
            "recommended_next_step": "Hold",
            "human_review_required": False,
        }
        validated = validate_result(data)
        self.assertEqual(validated["level"], "weak_match")

    def test_fail_gate_overrides_score(self) -> None:
        data = {
            "candidate_name": "Candidate C",
            "must_have_result": "fail",
            "score": 80,
            "level": "review",
            "strengths": [],
            "risks": [],
            "missing_information": [],
            "evidence": [
                {
                    "criterion": "Backend basics",
                    "score": 8,
                    "resume_text": "Python scripts.",
                }
            ],
            "recommended_next_step": "Hold",
            "human_review_required": False,
        }
        validated = validate_result(data)
        self.assertEqual(validated["level"], "not_recommended")


if __name__ == "__main__":
    unittest.main()
