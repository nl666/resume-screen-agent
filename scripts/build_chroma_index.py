from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from resume_screen_agent.chroma_rag import (  # noqa: E402
    DEFAULT_BGE_MODEL,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DIR,
    build_chroma_index,
)
from resume_screen_agent.rag import DEFAULT_KNOWLEDGE_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local BGE + Chroma RAG index.")
    parser.add_argument(
        "--knowledge-dir",
        default=str(DEFAULT_KNOWLEDGE_DIR),
        help="Directory containing knowledge documents.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_CHROMA_DIR),
        help="Directory for Chroma persistent storage.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BGE_MODEL,
        help="SentenceTransformers embedding model name.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete the existing collection before building.",
    )
    args = parser.parse_args()

    result = build_chroma_index(
        knowledge_dir=args.knowledge_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        reset=not args.no_reset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
