from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent.rag import (
    EMBEDDING_MODEL_NAME,
    answer_with_context,
    build_vector_index,
    chunk_text,
    load_documents,
    query_knowledge_base,
    retrieve,
    retrieve_vector,
)


class FakeChatModel:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        if "不查询知识库" in system_prompt or "自由回答" in user_prompt:
            return '{"answer":"这是模型自由回答。","sources":[],"confidence":"medium"}'
        return '{"answer":"这是基于资料的回答。","sources":[{"file":"rag.md","chunk_id":"rag.md#0","quote":"RAG 链路"}],"confidence":"high"}'


class RagTests(unittest.TestCase):
    def test_load_documents_supports_markdown_csv_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rules.md").write_text("RAG 能力包括 Chunk、Embedding、向量检索。", encoding="utf-8")
            (root / "results.csv").write_text("candidate,score\nA,75\n", encoding="utf-8")
            (root / "results.jsonl").write_text('{"candidate":"A","level":"review"}\n', encoding="utf-8")

            docs = load_documents(root)
            sources = {doc["source"] for doc in docs}

        self.assertEqual(sources, {"rules.md", "results.csv", "results.jsonl"})

    def test_retrieve_returns_relevant_chunk(self) -> None:
        docs = [
            {"source": "rag.md", "text": "RAG 经验包括文档切分、Chunk、Embedding、向量检索和引用溯源。"},
            {"source": "backend.md", "text": "后端工程能力包括 API、数据库、缓存和消息队列。"},
        ]
        chunks = chunk_text(docs, min_chars=10, max_chars=80)
        top = retrieve("什么能证明 RAG 经验？", chunks, top_k=1)

        self.assertEqual(top[0].source_file, "rag.md")

    def test_retrieve_short_query_prefers_query_coverage(self) -> None:
        docs = [
            {"source": "tool-calling-agent.md", "text": "Tool Calling Agent 已实现工具：read_resume、check_must_have、verify_evidence、derive_level、export_report。"},
            {"source": "tool-calling.md", "text": "Tool Calling 判断标准包括 Function Calling、MCP、系统集成、数据库和第三方 API。"},
        ]
        chunks = chunk_text(docs, min_chars=10, max_chars=120)
        top = retrieve("Tool Calling Agent 有哪些工具？", chunks, top_k=1)

        self.assertEqual(top[0].source_file, "tool-calling-agent.md")

    def test_vector_retrieve_returns_relevant_chunk(self) -> None:
        docs = [
            {"source": "mcp-server.md", "text": "MCP Server 暴露 screen_resume、rag_query、run_eval、export_report 四个工具。"},
            {"source": "scoring-rules.md", "text": "评分规则包括硬性门槛、项目经验、工程能力和证据强度。"},
        ]
        chunks = chunk_text(docs, min_chars=10, max_chars=120)
        top = retrieve_vector("MCP Server 有哪些工具？", chunks, top_k=1)

        self.assertEqual(top[0].source_file, "mcp-server.md")

    def test_query_knowledge_base_builds_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "vector_index.json"
            (root / "rag.md").write_text(
                "RAG 链路包含文档解析、Chunk、Embedding、向量索引、余弦相似度检索和引用溯源。",
                encoding="utf-8",
            )
            (root / "backend.md").write_text(
                "后端系统包含 FastAPI、PostgreSQL、Redis、Docker 和日志监控。",
                encoding="utf-8",
            )

            payload = build_vector_index(root, index_path=index_path, min_chars=10, max_chars=120)
            result = query_knowledge_base(
                "RAG 如何做向量检索？",
                knowledge_dir=root,
                top_k=1,
                use_llm=False,
                retrieval_mode="vector",
                index_path=index_path,
            )

            self.assertTrue(index_path.exists())
            self.assertEqual(payload["embedding"]["model"], EMBEDDING_MODEL_NAME)
            self.assertEqual(result["sources"][0]["file"], "rag.md")
            self.assertEqual(result["retrieval"]["mode"], "vector")
            self.assertEqual(result["citations"][0]["citation_id"], "S1")
            self.assertIn("score", result["citations"][0])
            self.assertIn("match_reason", result["citations"][0])
            self.assertEqual(result["citation_summary"]["total"], 1)
            self.assertIn("引用来源", result["answer"])

    def test_answer_with_context_supports_mixed_mode_prompt(self) -> None:
        model = FakeChatModel()
        chunks = [next(iter(chunk_text([{"source": "rag.md", "text": "RAG 链路包含 Chunk 和 Embedding。"}], min_chars=10, max_chars=120)))]

        result = answer_with_context("RAG 是什么？", chunks, model=model, answer_mode="mixed")

        self.assertEqual(result["confidence"], "high")
        self.assertIn("混合增强模式", model.prompts[0][0])

    def test_query_knowledge_base_free_mode_skips_retrieval(self) -> None:
        result = query_knowledge_base(
            "如何优化 Agent 项目？",
            use_llm=True,
            answer_mode="free",
            model=FakeChatModel(),
        )

        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["citation_summary"]["total"], 0)
        self.assertEqual(result["retrieval"]["answer_mode"], "free")
        self.assertEqual(result["retrieval"]["mode"], "none")

    def test_answer_with_context_is_enriched_with_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rag.md").write_text(
                "RAG 项目需要保留引用证据，包含 citation_id、chunk_id、相关度和匹配原因。",
                encoding="utf-8",
            )

            result = query_knowledge_base(
                "RAG 引用证据应该包含什么？",
                knowledge_dir=root,
                top_k=1,
                use_llm=True,
                model=FakeChatModel(),
                answer_mode="strict",
            )

        citation = result["citations"][0]
        self.assertEqual(citation["citation_id"], "S1")
        self.assertEqual(citation["file"], "rag.md")
        self.assertIn(citation["relevance"], {"high", "medium", "low"})
        self.assertIn("rag.md", result["citation_summary"]["files"])
        self.assertEqual(result["retrieval"]["citation_count"], 1)


if __name__ == "__main__":
    unittest.main()
