from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent.mcp_server import call_mcp_tool, handle_jsonrpc


class McpServerTests(unittest.TestCase):
    def test_tools_list_exposes_expected_tools(self) -> None:
        response = handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        self.assertIsNotNone(response)
        tools = response["result"]["tools"]
        names = {tool["name"] for tool in tools}
        self.assertEqual({"screen_resume", "rag_query", "run_eval", "export_report"}, names)

    def test_tools_call_rag_query_returns_structured_content(self) -> None:
        response = handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "rag_query",
                    "arguments": {"question": "Tool Calling Agent 有哪些工具？", "top_k": 2},
                },
            }
        )

        self.assertFalse(response["result"]["isError"])
        structured = response["result"]["structuredContent"]
        self.assertIn("answer", structured)
        self.assertTrue(structured["sources"])

    def test_screen_resume_tool_runs_dynamic_agent(self) -> None:
        result = call_mcp_tool(
            "screen_resume",
            {"resume_path": "examples/sample_resume.txt", "mode": "dynamic", "max_steps": 12},
        )

        self.assertEqual(result["agent_type"], "dynamic_tool_calling_resume_screen_agent")
        self.assertEqual(result["loop_status"], "finished")
        self.assertIn("final_report", result)

    def test_run_eval_tool_returns_summary(self) -> None:
        result = call_mcp_tool("run_eval", {})

        self.assertEqual(result["summary"]["failed_cases"], 0)

    def test_export_report_writes_project_local_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            rel_path = Path(tmp).relative_to(PROJECT_ROOT) / "report.json"
            result = call_mcp_tool("export_report", {"report": {"ok": True}, "out_path": str(rel_path)})
            written = Path(result["out_path"])

            self.assertTrue(written.exists())
            self.assertEqual(json.loads(written.read_text(encoding="utf-8")), {"ok": True})

    def test_export_report_rejects_outside_project_path(self) -> None:
        with self.assertRaises(ValueError):
            call_mcp_tool("export_report", {"report": {"ok": True}, "out_path": "C:/Windows/report.json"})


if __name__ == "__main__":
    unittest.main()
