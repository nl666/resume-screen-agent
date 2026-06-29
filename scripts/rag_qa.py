from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from resume_screen_agent.rag import query_knowledge_base  # noqa: E402
from resume_screen_agent.llm import ChatModel  # noqa: E402

DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Q&A over the resume screening knowledge base."
    )
    parser.add_argument("--question", required=True, help="Question to ask the knowledge base.")
    parser.add_argument(
        "--knowledge-dir",
        default=str(DEFAULT_KNOWLEDGE_DIR),
        help="Directory containing knowledge documents.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--out", help="Output JSON path. If omitted, print to stdout.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--base-url", help="Override OpenAI-compatible base URL.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vector", "keyword"],
        default="hybrid",
        help="Retrieval mode: vector search, keyword search, or hybrid search.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the local vector index before querying.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only return retrieved chunks; do not call the LLM.",
    )
    args = parser.parse_args()

    chat_model = None
    if not args.retrieval_only:
        chat_model = ChatModel(model=args.model, base_url=args.base_url)

    result = query_knowledge_base(
        question=args.question,
        knowledge_dir=args.knowledge_dir,
        top_k=args.top_k,
        use_llm=not args.retrieval_only,
        model=chat_model,
        retrieval_mode=args.retrieval_mode,
        rebuild_index=args.rebuild_index,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Answer written to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
