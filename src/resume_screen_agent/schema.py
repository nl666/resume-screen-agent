from __future__ import annotations

import json
import re
from typing import Any


MUST_HAVE_VALUES = {"pass", "fail", "unclear"}
LEVEL_VALUES = {"strong_match", "review", "backup", "weak_match", "not_recommended"}


class ResultValidationError(ValueError):
    pass


def derive_level(score: int, must_have_result: str) -> str:
    """Derive the final level from deterministic screening rules."""
    if must_have_result == "fail":
        return "not_recommended"
    if score >= 85:
        return "strong_match"
    if score >= 75:
        return "review"
    if score >= 65:
        return "backup"
    if score >= 50:
        return "weak_match"
    return "not_recommended"


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output, allowing accidental fenced output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])

    if not isinstance(data, dict):
        raise ResultValidationError("Model output must be a JSON object.")
    return data


def validate_result(data: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "candidate_name",
        "must_have_result",
        "score",
        "level",
        "strengths",
        "risks",
        "missing_information",
        "evidence",
        "recommended_next_step",
        "human_review_required",
    }

    missing = required_keys - data.keys()
    if missing:
        raise ResultValidationError(f"Missing required keys: {', '.join(sorted(missing))}")

    if data["must_have_result"] not in MUST_HAVE_VALUES:
        raise ResultValidationError("must_have_result must be one of: pass, fail, unclear")

    if data["level"] not in LEVEL_VALUES:
        raise ResultValidationError("level must be one of: strong_match, review, backup, weak_match, not_recommended")

    score = data["score"]
    if not isinstance(score, (int, float)) or not 0 <= score <= 100:
        raise ResultValidationError("score must be a number between 0 and 100")
    data["score"] = int(round(score))
    data["level"] = derive_level(data["score"], data["must_have_result"])

    for key in ("strengths", "risks", "missing_information"):
        if not isinstance(data[key], list):
            raise ResultValidationError(f"{key} must be a list")

    if not isinstance(data["evidence"], list):
        raise ResultValidationError("evidence must be a list")

    for item in data["evidence"]:
        if not isinstance(item, dict):
            raise ResultValidationError("each evidence item must be an object")
        for key in ("criterion", "score", "resume_text"):
            if key not in item:
                raise ResultValidationError(f"evidence item missing key: {key}")

    if not isinstance(data["human_review_required"], bool):
        raise ResultValidationError("human_review_required must be a boolean")

    if data["must_have_result"] == "unclear" or 65 <= data["score"] <= 79:
        data["human_review_required"] = True

    return data
