from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
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
WEB_LOG_DIR = ROOT / "logs"
WEB_APP_LOG_PATH = WEB_LOG_DIR / "web_app.log"
WEB_OPERATIONS_PATH = WEB_LOG_DIR / "operations.jsonl"
TASKS: dict[str, dict[str, Any]] = {}
TASK_LOCK = Lock()
OPERATION_LOCK = Lock()
TASK_EXECUTOR = ThreadPoolExecutor(max_workers=2)
LOGGER = logging.getLogger("resume_screen_agent.web_app")


def create_app() -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Web API requires fastapi, uvicorn and python-multipart. Run: pip install -r requirements.txt") from exc
    globals()["UploadFile"] = UploadFile
    _configure_app_logging()

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
            "operation_log": str(WEB_OPERATIONS_PATH),
            "app_log": str(WEB_APP_LOG_PATH),
        }

    @app.post("/api/screen-resume")
    def screen_resume(
        file: UploadFile = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        operation_id = _make_operation_id("screen")
        started = time.perf_counter()
        _record_operation(
            operation_id=operation_id,
            kind="screen_resume",
            status="started",
            message="单份简历筛选开始",
            details={"filename": safe_filename(file.filename or ""), "mode": mode, "redact": redact, "max_steps": max_steps},
        )
        saved_path: Path | None = None
        try:
            _validate_screen_mode(mode)
            saved_path = _save_upload(file)
            result_id = _make_result_id("screen", saved_path.stem)
            out_path = WEB_RESULTS_DIR / f"{result_id}.json"
            result = _screen_saved_resume(saved_path, out_path, mode, redact, max_steps)
            summary = _summarize_screening_result(result)
            response = {
                "result_id": result_id,
                "result_file": str(out_path),
                "operation_id": operation_id,
                "summary": summary,
                "result": result,
            }
            _record_operation(
                operation_id=operation_id,
                kind="screen_resume",
                status="succeeded",
                message="单份简历筛选完成",
                duration_ms=_duration_ms(started),
                result_path=out_path,
                summary=summary,
                details={"uploaded_file": saved_path.name},
            )
            return response
        except ValueError as exc:
            _record_operation(
                operation_id=operation_id,
                kind="screen_resume",
                status="failed",
                message="单份简历筛选参数错误",
                duration_ms=_duration_ms(started),
                error=str(exc),
                details={"uploaded_file": saved_path.name if saved_path else ""},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("screen_resume failed")
            _record_operation(
                operation_id=operation_id,
                kind="screen_resume",
                status="failed",
                message="单份简历筛选失败",
                duration_ms=_duration_ms(started),
                error=str(exc),
                details={"uploaded_file": saved_path.name if saved_path else ""},
            )
            raise

    @app.post("/api/batch-screen")
    def batch_screen(
        files: list[UploadFile] = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        operation_id = _make_operation_id("batch")
        started = time.perf_counter()
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen",
            status="started",
            message="同步批量筛选开始",
            details={"file_count": len(files), "mode": mode, "redact": redact, "max_steps": max_steps},
        )
        try:
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
                    item = {
                        "source_file": saved_path.name,
                        "result_file": str(out_path),
                        "ok": True,
                        "summary": _summarize_screening_result(result),
                    }
                    _record_operation(
                        operation_id=operation_id,
                        kind="batch_screen_file",
                        status="succeeded",
                        message="批量筛选单文件完成",
                        result_path=out_path,
                        details={"source_file": saved_path.name},
                        summary=item["summary"],
                    )
                    items.append(item)
                except Exception as exc:  # noqa: BLE001 - batch endpoint should return per-file errors.
                    error_result = {"source_file": saved_path.name, "ok": False, "error": str(exc)}
                    _write_json(out_path, error_result)
                    _record_operation(
                        operation_id=operation_id,
                        kind="batch_screen_file",
                        status="failed",
                        message="批量筛选单文件失败",
                        result_path=out_path,
                        error=str(exc),
                        details={"source_file": saved_path.name},
                    )
                    items.append(error_result)

            summary = {
                "total": len(items),
                "ok": sum(1 for item in items if item.get("ok")),
                "failed": sum(1 for item in items if not item.get("ok")),
            }
            batch_report = {"batch_id": batch_id, "operation_id": operation_id, "summary": summary, "items": items}
            result_path = WEB_RESULTS_DIR / f"{batch_id}.json"
            _write_json(result_path, batch_report)
            _record_operation(
                operation_id=operation_id,
                kind="batch_screen",
                status="succeeded",
                message="同步批量筛选完成",
                duration_ms=_duration_ms(started),
                result_path=result_path,
                summary=summary,
            )
            return batch_report
        except ValueError as exc:
            _record_operation(
                operation_id=operation_id,
                kind="batch_screen",
                status="failed",
                message="同步批量筛选参数错误",
                duration_ms=_duration_ms(started),
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("batch_screen failed")
            _record_operation(
                operation_id=operation_id,
                kind="batch_screen",
                status="failed",
                message="同步批量筛选失败",
                duration_ms=_duration_ms(started),
                error=str(exc),
            )
            raise

    @app.post("/api/batch-screen-async")
    def batch_screen_async(
        files: list[UploadFile] = File(...),
        mode: str = Form("dynamic"),
        redact: bool = Form(True),
        max_steps: int = Form(12),
    ) -> dict[str, Any]:
        operation_id = _make_operation_id("batch")
        started = time.perf_counter()
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen_async",
            status="started",
            message="异步批量筛选任务创建中",
            details={"file_count": len(files), "mode": mode, "redact": redact, "max_steps": max_steps},
        )
        try:
            _validate_screen_mode(mode)
            if not files:
                raise ValueError("files are required")
            batch_id = _make_result_id("batch", "resumes")
            saved_paths = [_save_upload(file, batch_id=batch_id) for file in files]
        except ValueError as exc:
            _record_operation(
                operation_id=operation_id,
                kind="batch_screen_async",
                status="failed",
                message="异步批量筛选任务创建失败",
                duration_ms=_duration_ms(started),
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        task_id = _make_result_id("task", "batch")
        task = _create_batch_task(task_id, batch_id, saved_paths, operation_id=operation_id)
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen_async",
            status="queued",
            message="异步批量筛选任务已创建",
            duration_ms=_duration_ms(started),
            details={"task_id": task_id, "batch_id": batch_id, "saved_files": [path.name for path in saved_paths]},
        )
        TASK_EXECUTOR.submit(_run_batch_screen_task, task_id, batch_id, saved_paths, mode, redact, max_steps, operation_id)
        return task

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = _get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task

    @app.post("/api/rag-query")
    async def rag_query(payload: dict[str, Any]) -> dict[str, Any]:
        operation_id = _make_operation_id("rag")
        started = time.perf_counter()
        question = str(payload.get("question", "")).strip()
        if not question:
            _record_operation(
                operation_id=operation_id,
                kind="rag_query",
                status="failed",
                message="知识库问答缺少问题",
                error="question is required",
            )
            raise HTTPException(status_code=400, detail="question is required")

        top_k = int(payload.get("top_k", 5))
        use_llm = bool(payload.get("use_llm", False))
        retrieval_mode = str(payload.get("retrieval_mode", "hybrid"))
        vector_store = str(payload.get("vector_store", "local"))
        answer_mode = str(payload.get("answer_mode", "retrieval"))
        rebuild_index = bool(payload.get("rebuild_index", False))
        _record_operation(
            operation_id=operation_id,
            kind="rag_query",
            status="started",
            message="知识库问答开始",
            details={
                "question_preview": _preview_text(question),
                "top_k": top_k,
                "use_llm": use_llm,
                "retrieval_mode": retrieval_mode,
                "vector_store": vector_store,
                "answer_mode": answer_mode,
                "rebuild_index": rebuild_index,
            },
        )
        try:
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
            summary = {
                "confidence": result.get("confidence", ""),
                "citations": result.get("citation_summary", {}).get("total", len(result.get("sources", []))),
                "answer_mode": result.get("retrieval", {}).get("answer_mode", answer_mode),
            }
            _record_operation(
                operation_id=operation_id,
                kind="rag_query",
                status="succeeded",
                message="知识库问答完成",
                duration_ms=_duration_ms(started),
                result_path=out_path,
                summary=summary,
            )
            return {"result_file": str(out_path), "operation_id": operation_id, "result": result}
        except Exception as exc:
            LOGGER.exception("rag_query failed")
            _record_operation(
                operation_id=operation_id,
                kind="rag_query",
                status="failed",
                message="知识库问答失败",
                duration_ms=_duration_ms(started),
                error=str(exc),
            )
            raise

    @app.post("/api/run-eval")
    def run_eval(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        operation_id = _make_operation_id("eval")
        started = time.perf_counter()
        redact = bool(payload.get("redact", False))
        _record_operation(
            operation_id=operation_id,
            kind="run_eval",
            status="started",
            message="系统效果检查开始",
            details={"redact": redact},
        )
        try:
            out_path = WEB_RESULTS_DIR / f"{_make_result_id('eval', 'report')}.json"
            result = run_eval_suite(
                cases_path=DEFAULT_EVAL_CASES,
                out_path=out_path,
                redact=redact,
            )
            _record_operation(
                operation_id=operation_id,
                kind="run_eval",
                status="succeeded",
                message="系统效果检查完成",
                duration_ms=_duration_ms(started),
                result_path=out_path,
                summary=result["summary"],
            )
            return {"result_file": str(out_path), "operation_id": operation_id, "summary": result["summary"], "result": result}
        except Exception as exc:
            LOGGER.exception("run_eval failed")
            _record_operation(
                operation_id=operation_id,
                kind="run_eval",
                status="failed",
                message="系统效果检查失败",
                duration_ms=_duration_ms(started),
                error=str(exc),
            )
            raise

    @app.get("/api/results")
    def list_results() -> dict[str, Any]:
        return {"items": list_result_files()}

    @app.get("/api/operations")
    def list_operations(limit: int = 80) -> dict[str, Any]:
        return {"items": list_operation_events(limit=limit)}

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


def _create_batch_task(task_id: str, batch_id: str, saved_paths: list[Path], operation_id: str) -> dict[str, Any]:
    now = int(time.time())
    task = {
        "task_id": task_id,
        "operation_id": operation_id,
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
    operation_id: str,
) -> None:
    started = time.perf_counter()
    total = len(saved_paths)
    batch_dir = WEB_RESULTS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    ok_count = 0
    failed_count = 0

    try:
        _update_task(task_id, status="running")
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen_async",
            status="running",
            message="异步批量筛选任务开始执行",
            details={"task_id": task_id, "batch_id": batch_id, "total": total},
        )
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
                _record_operation(
                    operation_id=operation_id,
                    kind="batch_screen_file",
                    status="succeeded",
                    message=f"异步批量筛选文件完成（{index}/{total}）",
                    result_path=out_path,
                    details={"task_id": task_id, "source_file": saved_path.name, "index": index, "total": total},
                    summary=item["summary"],
                )
            except Exception as exc:  # noqa: BLE001 - batch jobs should continue and record per-file errors.
                item = {"source_file": saved_path.name, "ok": False, "error": str(exc)}
                _write_json(out_path, item)
                failed_count += 1
                _record_operation(
                    operation_id=operation_id,
                    kind="batch_screen_file",
                    status="failed",
                    message=f"异步批量筛选文件失败（{index}/{total}）",
                    result_path=out_path,
                    error=str(exc),
                    details={"task_id": task_id, "source_file": saved_path.name, "index": index, "total": total},
                )
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
        batch_report = {"batch_id": batch_id, "operation_id": operation_id, "summary": summary, "items": items}
        result_file = WEB_RESULTS_DIR / f"{batch_id}.json"
        _write_json(result_file, batch_report)
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen_async",
            status="succeeded",
            message="异步批量筛选任务完成",
            duration_ms=_duration_ms(started),
            result_path=result_file,
            summary=summary,
            details={"task_id": task_id, "batch_id": batch_id},
        )
        _update_task(
            task_id,
            status="completed",
            result_file=str(result_file),
            summary=summary,
            result=batch_report,
        )
    except Exception as exc:  # noqa: BLE001 - preserve task state for UI instead of dropping the failure.
        LOGGER.exception("batch_screen_async task failed")
        _record_operation(
            operation_id=operation_id,
            kind="batch_screen_async",
            status="failed",
            message="异步批量筛选任务失败",
            duration_ms=_duration_ms(started),
            error=str(exc),
            details={"task_id": task_id, "batch_id": batch_id},
        )
        _update_task(task_id, status="failed", error=str(exc))


def _configure_app_logging() -> None:
    WEB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    target = str(WEB_APP_LOG_PATH.resolve())
    for handler in LOGGER.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename).resolve() == Path(target):
            return
    _close_app_logging()
    handler = RotatingFileHandler(target, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)


def _close_app_logging() -> None:
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()


def _make_operation_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _record_operation(
    *,
    operation_id: str,
    kind: str,
    status: str,
    message: str,
    duration_ms: int | None = None,
    result_path: str | Path | None = None,
    details: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    event = {
        "event_id": _make_result_id("op", kind),
        "operation_id": operation_id,
        "kind": kind,
        "status": status,
        "message": message,
        "created_at": int(time.time()),
        "created_at_iso": _now_iso(),
    }
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    if result_path is not None:
        event["result_file"] = str(result_path)
        result_name = _result_name_for_path(Path(result_path))
        if result_name:
            event["result_name"] = result_name
    if details:
        event["details"] = _sanitize_for_log(details)
    if summary:
        event["summary"] = _sanitize_for_log(summary)
    if error:
        event["error"] = _preview_text(error, limit=500)

    WEB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with OPERATION_LOCK:
        with WEB_OPERATIONS_PATH.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
    log_level = logging.ERROR if status == "failed" else logging.INFO
    LOGGER.log(log_level, "%s %s %s %s", operation_id, kind, status, message)
    return event


def list_operation_events(limit: int = 80) -> list[dict[str, Any]]:
    if not WEB_OPERATIONS_PATH.exists():
        return []
    safe_limit = max(1, min(int(limit or 80), 500))
    items: list[dict[str, Any]] = []
    try:
        lines = WEB_OPERATIONS_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(items) >= safe_limit:
            break
    return items


def _result_name_for_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(WEB_RESULTS_DIR.resolve()).as_posix()
    except (ValueError, FileNotFoundError):
        return ""


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_log(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value[:50]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _preview_text(value, limit=240)
    return value


def _preview_text(value: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(value)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
