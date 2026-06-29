from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from .eval import DEFAULT_EVAL_CASES, run_eval_suite
from .rag import DEFAULT_KNOWLEDGE_DIR, query_knowledge_base
from .tool_agent import run_dynamic_tool_calling_agent, run_tool_calling_workflow


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "resume-screen-agent"
SERVER_VERSION = "0.1.0"


def get_mcp_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "screen_resume",
            "title": "Screen Resume",
            "description": "Run the resume screening Agent over one resume. Defaults to the dynamic Planner/Tool/Observation loop.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resume_path": {"type": "string", "description": "Resume path, relative to project root or absolute inside project."},
                    "screening_json_path": {"type": "string", "description": "Optional existing screening JSON to normalize and verify."},
                    "out_path": {"type": "string", "description": "Optional output JSON path, relative to project root."},
                    "mode": {"type": "string", "enum": ["dynamic", "fixed"], "default": "dynamic"},
                    "redact": {"type": "boolean", "default": True},
                    "max_steps": {"type": "integer", "default": 12, "minimum": 1, "maximum": 30},
                },
                "required": ["resume_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "rag_query",
            "title": "RAG Query",
            "description": "Query the local knowledge base and return answer text plus cited chunks. Defaults to retrieval-only mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "knowledge_dir": {"type": "string", "description": "Optional knowledge directory, relative to project root."},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                    "use_llm": {"type": "boolean", "default": False},
                    "answer_mode": {
                        "type": "string",
                        "enum": ["retrieval", "strict", "mixed", "free"],
                        "default": "retrieval",
                        "description": "retrieval returns chunks only; strict answers only from retrieved context; mixed separates knowledge-base evidence and model supplement; free skips retrieval.",
                    },
                    "retrieval_mode": {
                        "type": "string",
                        "enum": ["hybrid", "vector", "keyword"],
                        "default": "hybrid",
                        "description": "hybrid uses vector similarity plus keyword matching; vector uses vector similarity only.",
                    },
                    "vector_store": {
                        "type": "string",
                        "enum": ["local", "chroma"],
                        "default": "local",
                        "description": "local uses the dependency-free JSON vector index; chroma uses BGE embeddings with a local Chroma database.",
                    },
                    "rebuild_index": {"type": "boolean", "default": False},
                    "out_path": {"type": "string", "description": "Optional output JSON path, relative to project root."},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_eval",
            "title": "Run Agent Eval",
            "description": "Run deterministic regression eval cases for the resume screening Agent.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cases_path": {"type": "string", "description": "Optional JSONL eval case path, relative to project root."},
                    "out_path": {"type": "string", "description": "Optional output report path, relative to project root."},
                    "redact": {"type": "boolean", "default": False},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "export_report",
            "title": "Export Report",
            "description": "Write a JSON report payload to a project-local output path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "report": {"type": "object", "description": "JSON object to write."},
                    "out_path": {"type": "string", "description": "Output path, relative to project root."},
                },
                "required": ["report", "out_path"],
                "additionalProperties": False,
            },
        },
    ]


def call_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "screen_resume": _tool_screen_resume,
        "rag_query": _tool_rag_query,
        "run_eval": _tool_run_eval,
        "export_report": _tool_export_report,
    }
    if name not in handlers:
        raise ValueError(f"Unknown tool: {name}")
    return handlers[name](args)


def handle_jsonrpc(message: dict[str, Any]) -> dict[str, Any] | None:
    message_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    try:
        if method == "initialize":
            return _response(
                message_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "title": "Resume Screen Agent MCP Server",
                        "version": SERVER_VERSION,
                    },
                    "instructions": "Expose resume screening, RAG query, eval, and report export tools.",
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return _response(message_id, {})
        if method == "tools/list":
            return _response(message_id, {"tools": get_mcp_tools()})
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return _response(message_id, _tool_error("arguments must be an object"))
            result = call_mcp_tool(str(tool_name), arguments)
            return _response(message_id, _tool_success(result))
        return _error(message_id, -32601, f"Method not found: {method}")
    except ValueError as exc:
        return _error(message_id, -32602, str(exc))
    except Exception as exc:
        return _response(message_id, _tool_error(str(exc)))


def run_stdio_server() -> None:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
            response = handle_jsonrpc(message)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"Parse error: {exc}")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def _tool_screen_resume(args: dict[str, Any]) -> dict[str, Any]:
    resume_path = _resolve_read_path(_require_str(args, "resume_path"))
    screening_json_path = _optional_read_path(args.get("screening_json_path"))
    out_path = _optional_write_path(args.get("out_path"))
    mode = args.get("mode", "dynamic")
    redact = bool(args.get("redact", True))
    max_steps = int(args.get("max_steps", 12))

    if mode == "fixed":
        result = run_tool_calling_workflow(
            resume_path=resume_path,
            screening_json_path=screening_json_path,
            out_path=out_path,
            redact=redact,
        )
    elif mode == "dynamic":
        result = run_dynamic_tool_calling_agent(
            resume_path=resume_path,
            screening_json_path=screening_json_path,
            out_path=out_path,
            redact=redact,
            max_steps=max_steps,
        )
    else:
        raise ValueError("mode must be dynamic or fixed")

    return _json_safe(result)


def _tool_rag_query(args: dict[str, Any]) -> dict[str, Any]:
    question = _require_str(args, "question")
    top_k = int(args.get("top_k", 5))
    use_llm = bool(args.get("use_llm", False))
    answer_mode = str(args.get("answer_mode", "retrieval"))
    retrieval_mode = str(args.get("retrieval_mode", "hybrid"))
    vector_store = str(args.get("vector_store", "local"))
    rebuild_index = bool(args.get("rebuild_index", False))
    knowledge_dir = _optional_read_path(args.get("knowledge_dir")) or DEFAULT_KNOWLEDGE_DIR
    out_path = _optional_write_path(args.get("out_path"))

    result = query_knowledge_base(
        question=question,
        knowledge_dir=knowledge_dir,
        top_k=top_k,
        use_llm=use_llm,
        answer_mode=answer_mode,
        retrieval_mode=retrieval_mode,
        vector_store=vector_store,
        rebuild_index=rebuild_index,
    )

    if out_path:
        _write_json(out_path, result)
    return _json_safe(result)


def _tool_run_eval(args: dict[str, Any]) -> dict[str, Any]:
    cases_path = _optional_read_path(args.get("cases_path")) or DEFAULT_EVAL_CASES
    out_path = _optional_write_path(args.get("out_path")) or (ROOT / "results" / "agent_eval_report.json")
    result = run_eval_suite(cases_path=cases_path, out_path=out_path, redact=bool(args.get("redact", False)))
    return _json_safe(result)


def _tool_export_report(args: dict[str, Any]) -> dict[str, Any]:
    report = args.get("report")
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    out_path = _resolve_write_path(_require_str(args, "out_path"))
    _write_json(out_path, report)
    return {"written": True, "out_path": str(out_path), "bytes": out_path.stat().st_size}


def _tool_success(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=2)}],
        "structuredContent": data,
        "isError": False,
    }


def _tool_error(message: str) -> dict[str, Any]:
    data = {"error": message}
    return {
        "content": [{"type": "text", "text": message}],
        "structuredContent": data,
        "isError": True,
    }


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required and must be a non-empty string")
    return value


def _optional_read_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("path value must be a string")
    return _resolve_read_path(value)


def _optional_write_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("path value must be a string")
    return _resolve_write_path(value)


def _resolve_read_path(value: str) -> Path:
    path = _resolve_project_path(value)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path


def _resolve_write_path(value: str) -> Path:
    path = _resolve_project_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_project_path(value: str) -> Path:
    raw = Path(value)
    path = raw if raw.is_absolute() else ROOT / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must stay inside project root: {ROOT}") from exc
    return resolved


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    run_stdio_server()


if __name__ == "__main__":
    main()
