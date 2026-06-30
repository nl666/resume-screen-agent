from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

from .eval import DEFAULT_EVAL_CASES, run_eval_suite
from .extract import SUPPORTED_EXTENSIONS
from .rag import DEFAULT_KNOWLEDGE_DIR, query_knowledge_base
from .tool_agent import run_dynamic_tool_calling_agent, run_tool_calling_workflow


ROOT = Path(__file__).resolve().parents[2]
WEB_RESULTS_DIR = ROOT / "results" / "web"
WEB_UPLOADS_DIR = ROOT / "data" / "uploads"
WEB_STATIC_DIR = ROOT / "web" / "static"
TASKS: dict[str, dict[str, Any]] = {}
TASK_LOCK = Lock()
TASK_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def create_app() -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Web API requires fastapi, uvicorn and python-multipart. Run: pip install -r requirements.txt") from exc
    globals()["UploadFile"] = UploadFile

    app = FastAPI(
        title="Resume Screen Agent API",
        description="Web/API service for resume screening, RAG query, eval and result browsing.",
        version="0.1.0",
    )

    if WEB_STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        html_path = WEB_STATIC_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="web/static/index.html not found")
        return FileResponse(html_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "resume-screen-agent",
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        }

    @app.post("/api/screen-resume")
    def screen_resume(
        file: UploadFile = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        saved_path = _save_upload(file)
        result_id = _make_result_id("screen", saved_path.stem)
        out_path = WEB_RESULTS_DIR / f"{result_id}.json"

        if mode == "fixed":
            result = run_tool_calling_workflow(saved_path, out_path=out_path, redact=redact)
        elif mode == "dynamic":
            result = run_dynamic_tool_calling_agent(saved_path, out_path=out_path, redact=redact, max_steps=max_steps)
        else:
            raise HTTPException(status_code=400, detail="mode must be dynamic or fixed")

        return {
            "result_id": result_id,
            "result_file": str(out_path),
            "summary": _summarize_screening_result(result),
            "result": result,
        }

    @app.post("/api/batch-screen")
    def batch_screen(
        files: list[UploadFile] = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        _validate_screen_mode(mode)
        batch_id = _make_result_id("batch", "resumes")
        batch_dir = WEB_RESULTS_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        items = []

        for file in files:
            saved_path = _save_upload(file, batch_id=batch_id)
            out_path = batch_dir / f"{saved_path.stem}.json"
            try:
                result = _screen_saved_resume(saved_path, out_path, mode, redact, max_steps)
                items.append(
                    {
                        "source_file": saved_path.name,
                        "result_file": str(out_path),
                        "ok": True,
                        "summary": _summarize_screening_result(result),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - batch endpoint should return per-file errors.
                error_result = {"source_file": saved_path.name, "ok": False, "error": str(exc)}
                _write_json(out_path, error_result)
                items.append(error_result)

        summary = {
            "total": len(items),
            "ok": sum(1 for item in items if item.get("ok")),
            "failed": sum(1 for item in items if not item.get("ok")),
        }
        batch_report = {"batch_id": batch_id, "summary": summary, "items": items}
        _write_json(WEB_RESULTS_DIR / f"{batch_id}.json", batch_report)
        return batch_report

    @app.post("/api/batch-screen-async")
    def batch_screen_async(
        files: list[UploadFile] = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        try:
            _validate_screen_mode(mode)
            if not files:
                raise ValueError("files are required")
            batch_id = _make_result_id("batch", "resumes")
            saved_paths = [_save_upload(file, batch_id=batch_id) for file in files]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        task_id = _make_result_id("task", "batch")
        task = _create_batch_task(task_id, batch_id, saved_paths)
        TASK_EXECUTOR.submit(_run_batch_screen_task, task_id, batch_id, saved_paths, mode, redact, max_steps)
        return task

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = _get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task

    @app.post("/api/rag-query")
    async def rag_query(payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        top_k = int(payload.get("top_k", 5))
        use_llm = bool(payload.get("use_llm", False))
        retrieval_mode = str(payload.get("retrieval_mode", "hybrid"))
        vector_store = str(payload.get("vector_store", "local"))
        answer_mode = str(payload.get("answer_mode", "retrieval"))
        rebuild_index = bool(payload.get("rebuild_index", False))
        result = query_knowledge_base(
            question=question,
            knowledge_dir=DEFAULT_KNOWLEDGE_DIR,
            top_k=top_k,
            use_llm=use_llm,
            retrieval_mode=retrieval_mode,
            vector_store=vector_store,
            answer_mode=answer_mode,
            rebuild_index=rebuild_index,
        )

        out_path = WEB_RESULTS_DIR / f"{_make_result_id('rag', 'query')}.json"
        _write_json(out_path, result)
        return {"result_file": str(out_path), "result": result}

    @app.post("/api/run-eval")
    def run_eval(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        out_path = WEB_RESULTS_DIR / f"{_make_result_id('eval', 'report')}.json"
        result = run_eval_suite(
            cases_path=DEFAULT_EVAL_CASES,
            out_path=out_path,
            redact=bool(payload.get("redact", False)),
        )
        return {"result_file": str(out_path), "summary": result["summary"], "result": result}

    @app.get("/api/results")
    def list_results() -> dict[str, Any]:
        return {"items": list_result_files()}

    @app.get("/api/results/{result_name}")
    def get_result(result_name: str) -> dict[str, Any]:
        try:
            return _read_result_file(result_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/result")
    def get_result_by_name(name: str) -> dict[str, Any]:
        try:
            return _read_result_file(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _validate_screen_mode(mode: str) -> None:
    if mode not in {"dynamic", "fixed"}:
        raise ValueError("mode must be dynamic or fixed")


def _screen_saved_resume(saved_path: Path, out_path: Path, mode: str, redact: bool, max_steps: int) -> dict[str, Any]:
    if mode == "fixed":
        return run_tool_calling_workflow(saved_path, out_path=out_path, redact=redact)
    if mode == "dynamic":
        return run_dynamic_tool_calling_agent(saved_path, out_path=out_path, redact=redact, max_steps=max_steps)
    raise ValueError("mode must be dynamic or fixed")


def _create_batch_task(task_id: str, batch_id: str, saved_paths: list[Path]) -> dict[str, Any]:
    now = int(time.time())
    task = {
        "task_id": task_id,
        "batch_id": batch_id,
        "kind": "batch_screen",
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "result_file": str(WEB_RESULTS_DIR / f"{batch_id}.json"),
        "progress": {
            "total": len(saved_paths),
            "processed": 0,
            "ok": 0,
            "failed": 0,
            "percent": 0,
            "current_file": "",
        },
        "items": [],
    }
    with TASK_LOCK:
        TASKS[task_id] = task
    return deepcopy(task)


def _get_task(task_id: str) -> dict[str, Any] | None:
    with TASK_LOCK:
        task = TASKS.get(task_id)
        return deepcopy(task) if task else None


def _update_task(task_id: str, **updates: Any) -> dict[str, Any]:
    with TASK_LOCK:
        task = TASKS[task_id]
        task.update(updates)
        task["updated_at"] = int(time.time())
        return deepcopy(task)


def _set_task_progress(
    task_id: str,
    *,
    total: int,
    processed: int,
    ok: int,
    failed: int,
    current_file: str = "",
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    percent = int((processed / total) * 100) if total else 100
    progress = {
        "total": total,
        "processed": processed,
        "ok": ok,
        "failed": failed,
        "percent": min(100, max(0, percent)),
        "current_file": current_file,
    }
    updates: dict[str, Any] = {"progress": progress}
    if items is not None:
        updates["items"] = deepcopy(items)
    return _update_task(task_id, **updates)


def _run_batch_screen_task(
    task_id: str,
    batch_id: str,
    saved_paths: list[Path],
    mode: str,
    redact: bool,
    max_steps: int,
) -> None:
    total = len(saved_paths)
    batch_dir = WEB_RESULTS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    ok_count = 0
    failed_count = 0

    try:
        _update_task(task_id, status="running")
        for index, saved_path in enumerate(saved_paths, start=1):
            _set_task_progress(
                task_id,
                total=total,
                processed=index - 1,
                ok=ok_count,
                failed=failed_count,
                current_file=saved_path.name,
                items=items,
            )
            out_path = batch_dir / f"{saved_path.stem}.json"
            try:
                result = _screen_saved_resume(saved_path, out_path, mode, redact, max_steps)
                item = {
                    "source_file": saved_path.name,
                    "result_file": str(out_path),
                    "ok": True,
                    "summary": _summarize_screening_result(result),
                }
                ok_count += 1
            except Exception as exc:  # noqa: BLE001 - batch jobs should continue and record per-file errors.
                item = {"source_file": saved_path.name, "ok": False, "error": str(exc)}
                _write_json(out_path, item)
                failed_count += 1
            items.append(item)
            _set_task_progress(
                task_id,
                total=total,
                processed=index,
                ok=ok_count,
                failed=failed_count,
                current_file=saved_path.name,
                items=items,
            )

        summary = {"total": total, "ok": ok_count, "failed": failed_count}
        batch_report = {"batch_id": batch_id, "summary": summary, "items": items}
        result_file = WEB_RESULTS_DIR / f"{batch_id}.json"
        _write_json(result_file, batch_report)
        _update_task(
            task_id,
            status="completed",
            result_file=str(result_file),
            summary=summary,
            result=batch_report,
        )
    except Exception as exc:  # noqa: BLE001 - preserve task state for UI instead of dropping the failure.
        _update_task(task_id, status="failed", error=str(exc))


def list_result_files() -> list[dict[str, Any]]:
    if not WEB_RESULTS_DIR.exists():
        return []

    items = []
    for path in sorted(WEB_RESULTS_DIR.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        relative = path.relative_to(WEB_RESULTS_DIR).as_posix()
        items.append(
            {
                "name": relative,
                "size": path.stat().st_size,
                "modified_at": int(path.stat().st_mtime),
                "summary": _summarize_result_file(data),
            }
        )
    return items


def _save_upload(file: Any, batch_id: str | None = None) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {supported}")

    upload_dir = WEB_UPLOADS_DIR / (batch_id or time.strftime("%Y%m%d"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{safe_filename(file.filename or 'resume.txt')}"
    path = upload_dir / filename
    with path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return path


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "resume.txt"
    name = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return name[:120]


def _make_result_id(prefix: str, name: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{safe_filename(name).replace('.', '_')}"


def _resolve_result_file(result_name: str) -> Path:
    raw = Path(result_name)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Invalid result file name")
    path = (WEB_RESULTS_DIR / raw).resolve()
    try:
        path.relative_to(WEB_RESULTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Invalid result file name") from exc
    if not path.exists() or path.suffix.lower() != ".json":
        raise FileNotFoundError(f"Result not found: {result_name}")
    return path


def _read_result_file(result_name: str) -> dict[str, Any]:
    path = _resolve_result_file(result_name)
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_screening_result(result: dict[str, Any]) -> dict[str, Any]:
    final_report = result.get("final_report", result)
    return {
        "candidate_name": final_report.get("candidate_name", ""),
        "must_have_result": final_report.get("must_have_result", ""),
        "score": final_report.get("score", ""),
        "level": final_report.get("level", ""),
        "human_review_required": final_report.get("human_review_required", ""),
        "extraction_status": final_report.get("extraction_status", ""),
    }


def _summarize_result_file(data: dict[str, Any]) -> dict[str, Any]:
    if "summary" in data and isinstance(data["summary"], dict):
        return data["summary"]
    if "final_report" in data:
        return _summarize_screening_result(data)
    if "result" in data and isinstance(data["result"], dict):
        return _summarize_result_file(data["result"])
    if "answer" in data:
        return {"answer": str(data.get("answer", ""))[:120], "confidence": data.get("confidence", "")}
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


try:
    app = create_app()
except RuntimeError:
    app = None
