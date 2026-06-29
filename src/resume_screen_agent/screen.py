from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .extract import read_text, read_text_with_diagnostics
from .llm import ChatModel
from .redact import redact_basic_personal_info
from .schema import extract_json_object, validate_result


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = ROOT / "prompts" / "screening_system_prompt.md"
DEFAULT_STANDARD = ROOT / "standards" / "resume_screening_standard.md"
DEFAULT_JD = ROOT / "data" / "jd.txt"


def build_system_prompt(system_prompt: str, standard: str) -> str:
    return f"{system_prompt.strip()}\n\n以下是必须执行的评分标准：\n\n{standard.strip()}"


def build_user_prompt(jd: str, resume_text: str) -> str:
    return f"""请根据岗位 JD 和候选人简历进行初筛评分。

岗位 JD：
{jd.strip()}

候选人简历：
{resume_text.strip()}
"""


def screen_resume_text(
    jd: str,
    resume_text: str,
    system_prompt: str,
    standard: str,
    model: ChatModel | None = None,
) -> dict[str, Any]:
    chat_model = model or ChatModel()
    full_system_prompt = build_system_prompt(system_prompt, standard)
    user_prompt = build_user_prompt(jd, resume_text)
    raw = chat_model.complete_json(full_system_prompt, user_prompt)
    data = extract_json_object(raw)
    return validate_result(data)


def screen_resume_file(
    resume_path: str | Path,
    jd_path: str | Path = DEFAULT_JD,
    system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT,
    standard_path: str | Path = DEFAULT_STANDARD,
    redact: bool = False,
    model: ChatModel | None = None,
) -> dict[str, Any]:
    jd = read_text(jd_path)
    extracted = read_text_with_diagnostics(resume_path)
    resume_text = extracted.text
    if redact:
        resume_text = redact_basic_personal_info(resume_text)
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
    standard = Path(standard_path).read_text(encoding="utf-8")
    result = screen_resume_text(jd, resume_text, system_prompt, standard, model=model)
    result["extracted_text_chars"] = extracted.char_count
    result["extraction_status"] = extracted.status
    if extracted.status != "ok":
        result.setdefault("risks", []).append(f"resume_text_extraction_status: {extracted.status}")
        result["human_review_required"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen one resume and output JSON.")
    parser.add_argument("--resume", required=True, help="Path to resume file.")
    parser.add_argument("--jd", default=str(DEFAULT_JD), help="Path to JD text file.")
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT), help="Path to system prompt file.")
    parser.add_argument("--standard", default=str(DEFAULT_STANDARD), help="Path to screening standard file.")
    parser.add_argument("--out", help="Output JSON path. If omitted, print to stdout.")
    parser.add_argument("--redact", action="store_true", help="Redact basic personal info before sending to model.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--base-url", help="Override OpenAI-compatible base URL.")
    args = parser.parse_args()

    chat_model = ChatModel(model=args.model, base_url=args.base_url)
    result = screen_resume_file(
        resume_path=args.resume,
        jd_path=args.jd,
        system_prompt_path=args.system_prompt,
        standard_path=args.standard,
        redact=args.redact,
        model=chat_model,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
