from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from resume_screen_agent.rag import (  # noqa: E402
    DEFAULT_KNOWLEDGE_DIR,
    DEFAULT_VECTOR_INDEX_PATH,
    build_vector_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local vector index for RAG.")
    parser.add_argument(
        "--knowledge-dir",
        default=str(DEFAULT_KNOWLEDGE_DIR),
        help="Directory containing knowledge documents.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_VECTOR_INDEX_PATH),
        help="Output vector index JSON path.",
    )
    args = parser.parse_args()

    payload = build_vector_index(args.knowledge_dir, args.out)
    summary = {
        "index_path": str(Path(args.out)),
        "chunks": len(payload.get("chunks", [])),
        "embedding": payload.get("embedding", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
