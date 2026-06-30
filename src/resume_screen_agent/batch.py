from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .extract import SUPPORTED_EXTENSIONS
from .llm import ChatModel
from .screen import DEFAULT_JD, DEFAULT_STANDARD, DEFAULT_SYSTEM_PROMPT, screen_resume_file


def iter_resume_files(resume_dir: str | Path) -> list[Path]:
    root = Path(resume_dir)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def write_jsonl(results: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file",
        "candidate_name",
        "must_have_result",
        "score",
        "level",
        "extraction_status",
        "extracted_text_chars",
        "human_review_required",
        "recommended_next_step",
        "strengths",
        "risks",
        "missing_information",
    ]

    with out_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "source_file": item.get("source_file", ""),
                    "candidate_name": item.get("candidate_name", ""),
                    "must_have_result": item.get("must_have_result", ""),
                    "score": item.get("score", ""),
                    "level": item.get("level", ""),
                    "extraction_status": item.get("extraction_status", ""),
                    "extracted_text_chars": item.get("extracted_text_chars", ""),
                    "human_review_required": item.get("human_review_required", ""),
                    "recommended_next_step": item.get("recommended_next_step", ""),
                    "strengths": "；".join(map(str, item.get("strengths", []))),
                    "risks": "；".join(map(str, item.get("risks", []))),
                    "missing_information": "；".join(map(str, item.get("missing_information", []))),
                }
            )


def batch_screen(
    resume_dir: str | Path,
    out_dir: str | Path,
    jd_path: str | Path = DEFAULT_JD,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT,
    standard_path: str | Path = DEFAULT_STANDARD,
    redact: bool = False,
    model: ChatModel | None = None,
) -> list[dict[str, Any]]:
    chat_model = model or ChatModel()
    results = []

    for resume_path in iter_resume_files(resume_dir):
        try:
            result = screen_resume_file(
                resume_path=resume_path,
                jd_path=jd_path,
                system_prompt_path=system_prompt_path,
                standard_path=standard_path,
                redact=redact,
                model=chat_model,
            )
            result["source_file"] = str(resume_path)
        except Exception as exc:  # noqa: BLE001 - batch jobs should continue and record failures.
            result = {
                "source_file": str(resume_path),
                "candidate_name": "",
                "must_have_result": "unclear",
                "score": 0,
                "level": "not_recommended",
                "extraction_status": "failed",
                "extracted_text_chars": 0,
                "strengths": [],
                "risks": [f"screening_failed: {exc}"],
                "missing_information": [],
                "evidence": [],
                "recommended_next_step": "人工检查该简历解析或模型调用失败原因",
                "human_review_required": True,
            }
        results.append(result)

    out_root = Path(out_dir)
    write_jsonl(results, out_root / "screening_results.jsonl")
    write_csv(results, out_root / "screening_results.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch screen resumes.")
    parser.add_argument("--resume-dir", required=True, help="Directory containing resumes.")
    parser.add_argument("--out-dir", default="results", help="Output directory.")
    parser.add_argument("--jd", default=str(DEFAULT_JD), help="Path to JD text file.")
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT), help="Path to system prompt file.")
    parser.add_argument("--standard", default=str(DEFAULT_STANDARD), help="Path to screening standard file.")
    parser.add_argument("--redact", action="store_true", help="Redact sensitive resume info before sending to model.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--base-url", help="Override OpenAI-compatible base URL.")
    args = parser.parse_args()

    chat_model = ChatModel(model=args.model, base_url=args.base_url)
    results = batch_screen(
        resume_dir=args.resume_dir,
        out_dir=args.out_dir,
        jd_path=args.jd,
        system_prompt_path=args.system_prompt,
        standard_path=args.standard,
        redact=args.redact,
        model=chat_model,
    )
    print(f"Screened {len(results)} resume(s). Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
