from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent.tool_agent import (
    classify_evidence_strength,
    run_dynamic_tool_calling_agent,
    run_tool_calling_workflow,
)


class ToolAgentTests(unittest.TestCase):
    def test_tool_workflow_records_trace_and_report(self) -> None:
        resume = """候选人：测试
项目：AI Agent 简历初筛系统
使用 Python、DeepSeek API、Prompt、结构化 JSON 输出实现简历筛选。
补充 RAG 知识库模块，支持文档解析、Chunk、Embedding、向量检索和引用溯源。
设计 Tool Calling 工具调用流程，将读取简历、证据校验和报告导出封装为工具。
使用单元测试和回归测试检查输出稳定性。
"""
        with tempfile.TemporaryDirectory() as tmp:
            resume_path = Path(tmp) / "resume.txt"
            resume_path.write_text(resume, encoding="utf-8")
            result = run_tool_calling_workflow(resume_path)

        tool_names = [item["name"] for item in result["tool_trace"]]
        self.assertIn("read_resume", tool_names)
        self.assertIn("check_must_have", tool_names)
        self.assertIn("verify_evidence", tool_names)
        self.assertEqual(result["final_report"]["must_have_result"], "pass")
        self.assertGreater(result["final_report"]["score"], 0)

    def test_tool_workflow_exports_report(self) -> None:
        resume = "Python 后端项目，使用 Agent、RAG、Tool Calling 实现简历初筛系统。"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resume_path = root / "resume.txt"
            out_path = root / "tool_report.json"
            resume_path.write_text(resume, encoding="utf-8")
            result = run_tool_calling_workflow(resume_path, out_path=out_path)

            self.assertTrue(out_path.exists())
            self.assertEqual(result["tool_trace"][-1]["name"], "export_report")

    def test_dynamic_agent_records_reasoning_trace(self) -> None:
        resume = """候选人：动态 Agent 测试
项目：AI Agent 简历初筛系统
使用 Python、FastAPI、DeepSeek、Prompt、RAG、Tool Calling 和回归测试实现简历筛选。
"""
        with tempfile.TemporaryDirectory() as tmp:
            resume_path = Path(tmp) / "resume.txt"
            resume_path.write_text(resume, encoding="utf-8")
            result = run_dynamic_tool_calling_agent(resume_path)

        tool_names = [item["name"] for item in result["tool_trace"]]
        self.assertEqual(result["agent_type"], "dynamic_tool_calling_resume_screen_agent")
        self.assertEqual(result["loop_status"], "finished")
        self.assertIn("lookup_screening_rules", tool_names)
        self.assertIn("finalize_report", tool_names)
        self.assertTrue(result["reasoning_trace"])

    def test_dynamic_agent_uses_existing_screening_json_branch(self) -> None:
        resume = "Python 后端项目，使用 Agent、RAG、Tool Calling 实现简历初筛系统。"
        screening_result = {
            "candidate_name": "Candidate",
            "must_have_result": "pass",
            "score": 78,
            "level": "review",
            "strengths": ["Agent project"],
            "risks": [],
            "missing_information": [],
            "evidence": [
                {
                    "criterion": "Agent project",
                    "score": 5,
                    "resume_text": "Python 后端项目，使用 Agent、RAG、Tool Calling 实现简历初筛系统。",
                }
            ],
            "recommended_next_step": "Review",
            "human_review_required": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resume_path = root / "resume.txt"
            json_path = root / "screening.json"
            resume_path.write_text(resume, encoding="utf-8")
            json_path.write_text(json.dumps(screening_result, ensure_ascii=False), encoding="utf-8")
            result = run_dynamic_tool_calling_agent(resume_path, screening_json_path=json_path)

        tool_names = [item["name"] for item in result["tool_trace"]]
        self.assertIn("load_screening_result", tool_names)
        self.assertIn("build_report_from_screening_result", tool_names)
        self.assertNotIn("keyword_score_resume", tool_names)

    def test_planned_capabilities_are_not_scored_as_implemented(self) -> None:
        resume = """候选人：规划型测试
项目：AI 应用学习项目
使用 Python 和 FastAPI 完成后端接口开发。
计划后续接入 RAG、Embedding、向量数据库、Hybrid Search 和 Rerank。
准备学习 Tool Calling、MCP Server、Agent Eval、Benchmark 和回归测试。
"""
        with tempfile.TemporaryDirectory() as tmp:
            resume_path = Path(tmp) / "resume.txt"
            resume_path.write_text(resume, encoding="utf-8")
            result = run_tool_calling_workflow(resume_path)

        report = result["final_report"]
        weak_items = [
            item
            for item in report["evidence"]
            if item.get("evidence_strength") in {"planned", "learning"}
        ]
        self.assertTrue(weak_items)
        self.assertLessEqual(report["score"], 55)
        self.assertIn("缺少 Tool Calling 项目证据", report["risks"])
        self.assertIn("MCP Server / Client 或工具协议封装经验", report["missing_information"])

    def test_evidence_strength_classifier_distinguishes_implemented_and_planned(self) -> None:
        implemented, _ = classify_evidence_strength("实现 MCP Server，并暴露 tools/list 和 tools/call。")
        planned, _ = classify_evidence_strength("计划后续接入 MCP Server 和 Tool Calling。")
        not_implemented, _ = classify_evidence_strength("尚未实现 MCP Server 和 Tool Calling。")

        self.assertEqual(implemented, "implemented")
        self.assertEqual(planned, "planned")
        self.assertEqual(not_implemented, "not_implemented")


if __name__ == "__main__":
    unittest.main()
