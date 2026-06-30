from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent import web_app


class WebAppHelperTests(unittest.TestCase):
    def test_safe_filename_keeps_extension_and_removes_path(self) -> None:
        self.assertEqual(web_app.safe_filename("../简历 test.pdf"), "简历_test.pdf")

    def test_list_result_files_summarizes_screening_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = web_app.WEB_RESULTS_DIR
            web_app.WEB_RESULTS_DIR = Path(tmp)
            try:
                report_path = web_app.WEB_RESULTS_DIR / "screen.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "final_report": {
                                "candidate_name": "Candidate",
                                "score": 72,
                                "level": "backup",
                                "must_have_result": "pass",
                            }
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                items = web_app.list_result_files()
            finally:
                web_app.WEB_RESULTS_DIR = old_dir

        self.assertEqual(items[0]["name"], "screen.json")
        self.assertEqual(items[0]["summary"]["score"], 72)

    def test_resolve_result_file_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = web_app.WEB_RESULTS_DIR
            web_app.WEB_RESULTS_DIR = Path(tmp)
            try:
                with self.assertRaises(ValueError):
                    web_app._resolve_result_file("../secret.json")
            finally:
                web_app.WEB_RESULTS_DIR = old_dir

    def test_operation_log_records_and_lists_recent_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_log_dir = web_app.WEB_LOG_DIR
            old_app_log = web_app.WEB_APP_LOG_PATH
            old_operations = web_app.WEB_OPERATIONS_PATH
            old_results = web_app.WEB_RESULTS_DIR
            web_app.WEB_LOG_DIR = Path(tmp) / "logs"
            web_app.WEB_APP_LOG_PATH = web_app.WEB_LOG_DIR / "web_app.log"
            web_app.WEB_OPERATIONS_PATH = web_app.WEB_LOG_DIR / "operations.jsonl"
            web_app.WEB_RESULTS_DIR = Path(tmp) / "results"
            try:
                result_path = web_app.WEB_RESULTS_DIR / "sample.json"
                web_app._write_json(result_path, {"ok": True})
                event = web_app._record_operation(
                    operation_id="op_test",
                    kind="rag_query",
                    status="succeeded",
                    message="知识库问答完成",
                    result_path=result_path,
                    summary={"confidence": "high"},
                )
                items = web_app.list_operation_events()
            finally:
                web_app._close_app_logging()
                web_app.WEB_LOG_DIR = old_log_dir
                web_app.WEB_APP_LOG_PATH = old_app_log
                web_app.WEB_OPERATIONS_PATH = old_operations
                web_app.WEB_RESULTS_DIR = old_results

        self.assertEqual(event["result_name"], "sample.json")
        self.assertEqual(items[0]["operation_id"], "op_test")
        self.assertEqual(items[0]["summary"]["confidence"], "high")


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "fastapi is not installed")
class WebAppFastApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        app = web_app.create_app()
        client = TestClient(app)
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertIn("operation_log", response.json())

    def test_operations_endpoint_returns_recent_events(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            old_log_dir = web_app.WEB_LOG_DIR
            old_app_log = web_app.WEB_APP_LOG_PATH
            old_operations = web_app.WEB_OPERATIONS_PATH
            web_app.WEB_LOG_DIR = Path(tmp) / "logs"
            web_app.WEB_APP_LOG_PATH = web_app.WEB_LOG_DIR / "web_app.log"
            web_app.WEB_OPERATIONS_PATH = web_app.WEB_LOG_DIR / "operations.jsonl"
            try:
                web_app._record_operation(
                    operation_id="op_api",
                    kind="run_eval",
                    status="started",
                    message="系统效果检查开始",
                )
                app = web_app.create_app()
                client = TestClient(app)
                response = client.get("/api/operations")
            finally:
                web_app._close_app_logging()
                web_app.WEB_LOG_DIR = old_log_dir
                web_app.WEB_APP_LOG_PATH = old_app_log
                web_app.WEB_OPERATIONS_PATH = old_operations

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["operation_id"], "op_api")

    def test_screen_resume_upload_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            old_results = web_app.WEB_RESULTS_DIR
            old_uploads = web_app.WEB_UPLOADS_DIR
            old_log_dir = web_app.WEB_LOG_DIR
            old_app_log = web_app.WEB_APP_LOG_PATH
            old_operations = web_app.WEB_OPERATIONS_PATH
            web_app.WEB_RESULTS_DIR = Path(tmp) / "results"
            web_app.WEB_UPLOADS_DIR = Path(tmp) / "uploads"
            web_app.WEB_LOG_DIR = Path(tmp) / "logs"
            web_app.WEB_APP_LOG_PATH = web_app.WEB_LOG_DIR / "web_app.log"
            web_app.WEB_OPERATIONS_PATH = web_app.WEB_LOG_DIR / "operations.jsonl"
            try:
                app = web_app.create_app()
                client = TestClient(app)
                resume_bytes = (PROJECT_ROOT / "examples" / "sample_resume.txt").read_bytes()
                response = client.post(
                    "/api/screen-resume",
                    files={"file": ("sample_resume.txt", resume_bytes, "text/plain")},
                    data={"mode": "dynamic", "redact": "true", "max_steps": "12"},
                )
            finally:
                web_app._close_app_logging()
                web_app.WEB_RESULTS_DIR = old_results
                web_app.WEB_UPLOADS_DIR = old_uploads
                web_app.WEB_LOG_DIR = old_log_dir
                web_app.WEB_APP_LOG_PATH = old_app_log
                web_app.WEB_OPERATIONS_PATH = old_operations

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("operation_id", body)
        self.assertEqual(body["result"]["agent_type"], "dynamic_tool_calling_resume_screen_agent")
        self.assertIn("final_report", body["result"])

    def test_async_batch_screen_endpoint_reports_progress(self) -> None:
        from fastapi.testclient import TestClient

        def fake_dynamic_agent(resume_path: Path, out_path: Path, redact: bool, max_steps: int) -> dict:
            result = {
                "agent_type": "dynamic_tool_calling_resume_screen_agent",
                "final_report": {
                    "candidate_name": Path(resume_path).stem,
                    "score": 70,
                    "level": "backup",
                    "must_have_result": "pass",
                    "human_review_required": False,
                },
            }
            web_app._write_json(Path(out_path), result)
            return result

        with tempfile.TemporaryDirectory() as tmp:
            old_results = web_app.WEB_RESULTS_DIR
            old_uploads = web_app.WEB_UPLOADS_DIR
            old_log_dir = web_app.WEB_LOG_DIR
            old_app_log = web_app.WEB_APP_LOG_PATH
            old_operations = web_app.WEB_OPERATIONS_PATH
            web_app.WEB_RESULTS_DIR = Path(tmp) / "results"
            web_app.WEB_UPLOADS_DIR = Path(tmp) / "uploads"
            web_app.WEB_LOG_DIR = Path(tmp) / "logs"
            web_app.WEB_APP_LOG_PATH = web_app.WEB_LOG_DIR / "web_app.log"
            web_app.WEB_OPERATIONS_PATH = web_app.WEB_LOG_DIR / "operations.jsonl"
            with web_app.TASK_LOCK:
                web_app.TASKS.clear()
            try:
                app = web_app.create_app()
                client = TestClient(app)
                with patch.object(web_app, "run_dynamic_tool_calling_agent", side_effect=fake_dynamic_agent):
                    response = client.post(
                        "/api/batch-screen-async",
                        files=[
                            ("files", ("one.txt", b"Python FastAPI RAG", "text/plain")),
                            ("files", ("two.txt", b"Agent Tool Calling", "text/plain")),
                        ],
                        data={"mode": "dynamic", "redact": "true", "max_steps": "12"},
                    )
                    self.assertEqual(response.status_code, 200)
                    task_id = response.json()["task_id"]

                    task = response.json()
                    for _ in range(30):
                        task_response = client.get(f"/api/tasks/{task_id}")
                        self.assertEqual(task_response.status_code, 200)
                        task = task_response.json()
                        if task["status"] == "completed":
                            break
                        time.sleep(0.05)
                    operations_response = client.get("/api/operations")
            finally:
                web_app._close_app_logging()
                web_app.WEB_RESULTS_DIR = old_results
                web_app.WEB_UPLOADS_DIR = old_uploads
                web_app.WEB_LOG_DIR = old_log_dir
                web_app.WEB_APP_LOG_PATH = old_app_log
                web_app.WEB_OPERATIONS_PATH = old_operations
                with web_app.TASK_LOCK:
                    web_app.TASKS.clear()

        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["progress"]["processed"], 2)
        self.assertEqual(task["progress"]["percent"], 100)
        self.assertEqual(task["summary"], {"total": 2, "ok": 2, "failed": 0})
        self.assertEqual(len(task["result"]["items"]), 2)
        operation_kinds = [item["kind"] for item in operations_response.json()["items"]]
        self.assertIn("batch_screen_async", operation_kinds)
        self.assertIn("batch_screen_file", operation_kinds)


if __name__ == "__main__":
    unittest.main()
