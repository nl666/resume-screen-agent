from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Resume Screen Agent web/API service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Web app requires uvicorn. Run: pip install -r requirements.txt") from exc

    uvicorn.run(
        "resume_screen_agent.web_app:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
