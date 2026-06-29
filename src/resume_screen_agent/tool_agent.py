from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .extract import read_text_with_diagnostics
from .llm import ChatModel
from .redact import redact_basic_personal_info
from .schema import derive_level, extract_json_object, validate_result


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class ToolContext:
    resume_path: Path
    redact: bool = False
    screening_json_path: Path | None = None
    out_path: Path | None = None
    resume_text: str = ""
    extraction_status: str = ""
    extracted_text_chars: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    tool_trace: list[ToolCallRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AgentAction:
    action_type: str
    thought: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


class AgentPlanner(Protocol):
    planner_name: str

    def next_action(
        self,
        context: ToolContext,
        available_tools: dict[str, ToolSpec],
        reasoning_trace: list[dict[str, Any]],
    ) -> AgentAction:
        ...


def _safe_args(arguments: dict[str, Any]) -> dict[str, Any]:
    safe = {key: _json_safe(value) for key, value in arguments.items()}
    if "resume_text" in safe:
        safe["resume_text"] = f"<{len(str(safe['resume_text']))} chars>"
    if "report" in safe:
        safe["report"] = "<report>"
    return safe


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


def _call_tool(
    context: ToolContext,
    name: str,
    func: Callable[..., dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    result = func(context=context, **kwargs)
    context.tool_trace.append(
        ToolCallRecord(name=name, arguments=_safe_args(kwargs), result=_summarize_tool_result(result))
    )
    return result


def _summarize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = dict(result)
    if "text" in summary:
        summary["text"] = f"<{len(str(summary['text']))} chars>"
    if "report" in summary:
        report = summary["report"]
        summary["report"] = {
            "candidate_name": report.get("candidate_name", ""),
            "must_have_result": report.get("must_have_result", ""),
            "score": report.get("score", ""),
            "level": report.get("level", ""),
        }
    return _json_safe(summary)


def read_resume_tool(context: ToolContext) -> dict[str, Any]:
    extracted = read_text_with_diagnostics(context.resume_path)
    text = extracted.text
    if context.redact:
        text = redact_basic_personal_info(text)

    context.resume_text = text
    context.extraction_status = extracted.status
    context.extracted_text_chars = extracted.char_count
    context.memory["resume_loaded"] = True
    context.memory["extraction_status"] = extracted.status
    context.memory["extracted_text_chars"] = extracted.char_count
    return {
        "text": text,
        "extraction_status": extracted.status,
        "extracted_text_chars": extracted.char_count,
    }


def load_screening_result_tool(context: ToolContext, screening_json_path: str | Path | None = None) -> dict[str, Any]:
    path_value = screening_json_path or context.screening_json_path
    if not path_value:
        context.memory["screening_result"] = None
        return {"screening_result": None}

    path = Path(path_value)
    data = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_result(data)
    context.memory["screening_result"] = validated
    return {"screening_result": validated}


def lookup_screening_rules_tool(context: ToolContext, query: str = "AI Agent 简历初筛评分标准", top_k: int = 3) -> dict[str, Any]:
    from .rag import chunk_text, load_documents, retrieve

    docs = load_documents(ROOT / "data" / "knowledge")
    chunks = chunk_text(docs)
    selected = retrieve(query, chunks, top_k=top_k)
    sources = [
        {
            "file": chunk.source_file,
            "chunk_id": chunk.chunk_id,
            "quote": chunk.text[:240],
        }
        for chunk in selected
    ]
    context.memory["screening_rules"] = sources
    return {"sources": sources}


def check_must_have_tool(context: ToolContext) -> dict[str, Any]:
    text = context.resume_text
    checks = [
        {
            "name": "主力开发语言",
            "matched": _has_any(text, ["python", "java", "go", "golang", "typescript", "javascript"]),
            "evidence": _find_evidence(text, ["Python", "Java", "Go", "Golang", "TypeScript", "JavaScript"]),
        },
        {
            "name": "后端开发或系统集成经验",
            "matched": _has_any(text, ["后端", "api", "rest", "数据库", "sql", "redis", "kafka", "系统集成", "fastapi"]),
            "evidence": _find_evidence(text, ["后端", "API", "REST", "数据库", "SQL", "Redis", "Kafka", "系统集成", "FastAPI"]),
        },
        {
            "name": "LLM / Agent / RAG / Tool Calling / Prompt 相关经验",
            "matched": _has_any(text, ["llm", "大模型", "agent", "rag", "tool calling", "function calling", "prompt", "deepseek", "openai"]),
            "evidence": _find_evidence(text, ["LLM", "大模型", "Agent", "RAG", "Tool Calling", "Function Calling", "Prompt", "DeepSeek", "OpenAI"]),
        },
        {
            "name": "真实项目证据",
            "matched": _has_any(text, ["项目", "系统", "平台", "上线", "github", "side project", "开发", "构建", "实现"]),
            "evidence": _find_evidence(text, ["项目", "系统", "平台", "上线", "GitHub", "Side Project", "开发", "构建", "实现"]),
        },
    ]
    matched_count = sum(1 for item in checks if item["matched"])
    if matched_count >= 3:
        result = "pass"
    elif matched_count == 2:
        result = "unclear"
    else:
        result = "fail"

    output = {
        "must_have_result": result,
        "matched_count": matched_count,
        "checks": checks,
    }
    context.memory["must_have"] = output
    return output


def keyword_score_tool(context: ToolContext, must_have_result: str | None = None) -> dict[str, Any]:
    text = context.resume_text
    if must_have_result is None:
        must_have_result = context.memory.get("must_have", {}).get("must_have_result", "unclear")
    evidence: list[dict[str, Any]] = []
    score = 0

    score += _score_item(text, evidence, "编程语言与代码能力", 8, ["Python", "Java", "Go", "TypeScript", "JavaScript"])
    score += _score_item(text, evidence, "API、数据库、缓存、消息队列、搜索等基础组件", 7, ["API", "数据库", "SQL", "Redis", "Kafka", "Elasticsearch"])
    score += _score_item(text, evidence, "系统设计、异步任务、稳定性、性能优化", 6, ["系统设计", "异步", "稳定", "性能", "优化"])
    score += _score_item(text, evidence, "Docker、K8s、CI/CD、部署经验", 4, ["Docker", "K8s", "Kubernetes", "CI/CD", "部署"])

    score += _score_item(text, evidence, "调用并集成主流 LLM API", 5, ["LLM API", "DeepSeek", "OpenAI", "Claude", "Qwen", "大模型"])
    score += _score_item(text, evidence, "Prompt Engineering、结构化输出、上下文管理", 5, ["Prompt", "结构化", "JSON", "上下文"])
    score += _score_item(text, evidence, "Agent 任务拆解、Planning、Memory、Tool Use", 7, ["Agent", "任务", "Planning", "Memory", "Tool Use", "工具调用"])
    score += _score_item(text, evidence, "Agent / LLM 框架实践", 5, ["LangGraph", "LangChain", "Dify", "AutoGen", "CrewAI", "LlamaIndex"])
    score += _score_item(text, evidence, "真实 Agent 项目落地经验", 3, ["Agent 简历初筛", "Agent 项目", "AI Agent"])

    score += _score_item(text, evidence, "文档解析、Chunk、Embedding、向量检索", 5, ["文档解析", "Chunk", "Embedding", "向量检索"])
    score += _score_item(text, evidence, "向量数据库、Hybrid Search、Rerank", 5, ["向量数据库", "Milvus", "Qdrant", "pgvector", "Hybrid Search", "Rerank"])
    score += _score_item(text, evidence, "引用溯源、幻觉控制、知识库权限或多租户隔离", 5, ["引用溯源", "幻觉", "知识库", "权限", "多租户"])

    score += _score_item(text, evidence, "Function Calling / Tool Calling 实践", 5, ["Function Calling", "Tool Calling", "工具调用"])
    score += _score_item(text, evidence, "MCP Server / Client 或工具协议封装经验", 5, ["MCP", "Server", "Client", "工具协议"])
    score += _score_item(text, evidence, "对接数据库、内部系统、第三方 API、搜索引擎等经验", 5, ["对接", "数据库", "第三方 API", "搜索引擎", "系统集成"])

    score += _score_item(text, evidence, "日志、监控、链路追踪、可观测性", 3, ["日志", "监控", "链路追踪", "可观测"])
    score += _score_item(text, evidence, "Agent Eval、Benchmark、回归测试", 4, ["Eval", "Benchmark", "回归测试", "单元测试"])
    score += _score_item(text, evidence, "Prompt Injection、防越权、敏感操作审批、沙箱隔离", 4, ["Prompt Injection", "防越权", "权限", "沙箱", "脱敏"])
    score += _score_item(text, evidence, "Token 成本、模型路由、缓存、TTFT 优化", 4, ["Token", "模型路由", "缓存", "TTFT", "成本"])

    score += _score_item(text, evidence, "沟通与产品落地", 5, ["业务需求", "技术方案", "设计文档", "复盘", "协作", "交付"])

    score = min(score, 100)
    report = {
        "candidate_name": context.resume_path.stem,
        "must_have_result": must_have_result,
        "score": score,
        "level": derive_level(score, must_have_result),
        "strengths": _derive_strengths(evidence),
        "risks": _derive_risks(score, must_have_result, evidence),
        "missing_information": _derive_missing_information(evidence),
        "evidence": evidence[:12],
        "recommended_next_step": _recommended_next_step(score, must_have_result),
        "human_review_required": must_have_result == "unclear" or 65 <= score <= 79,
    }
    validated = validate_result(report)
    context.memory["report"] = validated
    return {"report": validated}


def build_report_from_screening_result_tool(context: ToolContext) -> dict[str, Any]:
    screening_result = context.memory.get("screening_result")
    if not screening_result:
        raise ValueError("screening_result is required before build_report_from_screening_result")

    must_have_result = context.memory.get("must_have", {}).get(
        "must_have_result",
        screening_result.get("must_have_result", "unclear"),
    )
    report = dict(screening_result)
    report["must_have_result"] = must_have_result
    report["level"] = derive_level(int(report["score"]), must_have_result)
    validated = validate_result(report)
    context.memory["report"] = validated
    return {"report": validated}


def verify_evidence_tool(context: ToolContext, report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report or context.memory.get("report")
    if not report:
        raise ValueError("report is required before verify_evidence")
    text = context.resume_text
    verified = 0
    failed: list[dict[str, Any]] = []
    for item in report.get("evidence", []):
        quote = str(item.get("resume_text", "")).strip()
        if quote and quote in text:
            verified += 1
        else:
            failed.append(
                {
                    "criterion": item.get("criterion", ""),
                    "resume_text": quote[:120],
                }
            )
    output = {
        "verified_count": verified,
        "failed_count": len(failed),
        "failed_items": failed,
    }
    context.memory["evidence_check"] = output
    return output


def derive_level_tool(context: ToolContext, score: int | None = None, must_have_result: str | None = None) -> dict[str, Any]:
    report = context.memory.get("report", {})
    score_value = int(score if score is not None else report.get("score", 0))
    must_have_value = must_have_result or report.get("must_have_result") or context.memory.get("must_have", {}).get("must_have_result", "unclear")
    level = derive_level(score_value, must_have_value)
    if report:
        report["level"] = level
        context.memory["report"] = report
    context.memory["level"] = level
    return {"level": level}


def finalize_report_tool(context: ToolContext) -> dict[str, Any]:
    report = dict(context.memory.get("report") or {})
    if not report:
        raise ValueError("report is required before finalize_report")

    evidence_check = context.memory.get("evidence_check")
    if evidence_check is None:
        evidence_check = verify_evidence_tool(context, report)

    report["extraction_status"] = context.extraction_status
    report["extracted_text_chars"] = context.extracted_text_chars
    report["evidence_verification"] = evidence_check
    report["level"] = derive_level(int(report["score"]), report["must_have_result"])

    if context.extraction_status != "ok":
        report.setdefault("risks", []).append(f"resume_text_extraction_status: {context.extraction_status}")
        report["human_review_required"] = True
    if evidence_check["failed_count"]:
        report.setdefault("risks", []).append("部分证据片段未能在简历原文中逐字校验")
        report["human_review_required"] = True

    context.memory["final_report"] = report
    return {"report": report}


def export_report_tool(context: ToolContext, report: dict[str, Any], out_path: str | Path) -> dict[str, Any]:
    path = Path(out_path or context.out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"written": True, "out_path": str(path)}


def get_tool_registry() -> dict[str, ToolSpec]:
    return {
        "read_resume": ToolSpec(
            name="read_resume",
            description="读取简历文本，并记录文本提取状态。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=read_resume_tool,
        ),
        "lookup_screening_rules": ToolSpec(
            name="lookup_screening_rules",
            description="从本地知识库检索简历初筛评分规则和能力证据说明。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": [],
            },
            handler=lookup_screening_rules_tool,
        ),
        "load_screening_result": ToolSpec(
            name="load_screening_result",
            description="读取已有模型初筛 JSON，并做 schema 校验和等级修正。",
            parameters={
                "type": "object",
                "properties": {"screening_json_path": {"type": "string"}},
                "required": [],
            },
            handler=load_screening_result_tool,
        ),
        "check_must_have": ToolSpec(
            name="check_must_have",
            description="检查候选人是否满足硬性门槛。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=check_must_have_tool,
        ),
        "keyword_score_resume": ToolSpec(
            name="keyword_score_resume",
            description="没有已有模型初筛结果时，按关键词和证据做保守评分。",
            parameters={
                "type": "object",
                "properties": {"must_have_result": {"type": "string", "enum": ["pass", "unclear", "fail"]}},
                "required": [],
            },
            handler=keyword_score_tool,
        ),
        "build_report_from_screening_result": ToolSpec(
            name="build_report_from_screening_result",
            description="基于已有模型初筛结果生成报告，并用当前硬性门槛结果修正等级。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=build_report_from_screening_result_tool,
        ),
        "verify_evidence": ToolSpec(
            name="verify_evidence",
            description="校验证据片段是否能在简历原文中逐字找到。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=verify_evidence_tool,
        ),
        "derive_level": ToolSpec(
            name="derive_level",
            description="根据 score 和 must_have_result 重新计算候选人分层。",
            parameters={
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "must_have_result": {"type": "string", "enum": ["pass", "unclear", "fail"]},
                },
                "required": [],
            },
            handler=derive_level_tool,
        ),
        "finalize_report": ToolSpec(
            name="finalize_report",
            description="汇总提取状态、证据校验、风险和最终分层，生成 final_report。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=finalize_report_tool,
        ),
    }


class RuleBasedPlanner:
    planner_name = "rule_based_planner"

    def next_action(
        self,
        context: ToolContext,
        available_tools: dict[str, ToolSpec],
        reasoning_trace: list[dict[str, Any]],
    ) -> AgentAction:
        called = {record.name for record in context.tool_trace}

        if "read_resume" not in called:
            return AgentAction("tool_call", "需要先读取简历，确认文本是否可解析。", "read_resume")
        if "lookup_screening_rules" not in called:
            return AgentAction(
                "tool_call",
                "读取评分规则，避免后续筛选脱离岗位标准。",
                "lookup_screening_rules",
                {"query": "AI Agent 开发工程师 简历初筛 RAG Tool Calling Eval 评分标准", "top_k": 3},
            )
        if context.screening_json_path and "load_screening_result" not in called:
            return AgentAction(
                "tool_call",
                "发现已有模型初筛结果，先读取并校验它。",
                "load_screening_result",
                {"screening_json_path": str(context.screening_json_path)},
            )
        if "check_must_have" not in called:
            return AgentAction("tool_call", "检查硬性门槛，决定候选人是否具备基本匹配度。", "check_must_have")
        if context.memory.get("screening_result") and "build_report_from_screening_result" not in called:
            return AgentAction(
                "tool_call",
                "已有模型初筛 JSON 可复用，但需要用当前门槛结果修正报告。",
                "build_report_from_screening_result",
            )
        if not context.memory.get("report") and "keyword_score_resume" not in called:
            must_have_result = context.memory.get("must_have", {}).get("must_have_result", "unclear")
            return AgentAction(
                "tool_call",
                "没有可复用初筛结果，使用本地关键词证据做保守评分。",
                "keyword_score_resume",
                {"must_have_result": must_have_result},
            )
        if "verify_evidence" not in called:
            return AgentAction("tool_call", "校验证据片段，防止报告引用不存在的简历内容。", "verify_evidence")
        if "derive_level" not in called:
            report = context.memory.get("report", {})
            return AgentAction(
                "tool_call",
                "用确定性规则重新计算最终分层。",
                "derive_level",
                {
                    "score": report.get("score", 0),
                    "must_have_result": report.get("must_have_result", "unclear"),
                },
            )
        if "finalize_report" not in called:
            return AgentAction("tool_call", "汇总工具观察结果，生成最终报告。", "finalize_report")
        return AgentAction("finish", "已获得完整 final_report，可以停止工具调用。")


class JsonModelPlanner:
    planner_name = "json_model_planner"

    def __init__(self, model: ChatModel | None = None) -> None:
        self.model = model or ChatModel()

    def next_action(
        self,
        context: ToolContext,
        available_tools: dict[str, ToolSpec],
        reasoning_trace: list[dict[str, Any]],
    ) -> AgentAction:
        tool_specs = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in available_tools.values()
        ]
        memory_summary = {
            "resume_loaded": bool(context.memory.get("resume_loaded")),
            "has_screening_json": bool(context.screening_json_path),
            "has_loaded_screening_result": bool(context.memory.get("screening_result")),
            "has_report": bool(context.memory.get("report")),
            "has_evidence_check": bool(context.memory.get("evidence_check")),
            "has_final_report": bool(context.memory.get("final_report")),
            "called_tools": [record.name for record in context.tool_trace],
        }
        system_prompt = """你是一个真实的 Tool Calling Agent Planner。
你不能直接完成简历筛选，只能选择下一步工具调用，或在 final_report 已生成后 finish。
必须返回 JSON，不要输出额外文本。

返回格式只能是两种之一：
{"action_type":"tool_call","thought":"为什么要调用这个工具","tool_name":"工具名","arguments":{}}
{"action_type":"finish","thought":"为什么可以停止","arguments":{}}
"""
        user_prompt = json.dumps(
            {
                "task": "筛选一份 AI Agent 开发工程师候选人简历",
                "memory_summary": memory_summary,
                "available_tools": tool_specs,
                "recent_reasoning_trace": reasoning_trace[-6:],
            },
            ensure_ascii=False,
            indent=2,
        )
        data = extract_json_object(self.model.complete_json(system_prompt, user_prompt))
        return AgentAction(
            action_type=str(data.get("action_type", "")),
            thought=str(data.get("thought", "")),
            tool_name=str(data.get("tool_name", "")),
            arguments=data.get("arguments") if isinstance(data.get("arguments"), dict) else {},
        )


def run_dynamic_tool_calling_agent(
    resume_path: str | Path,
    screening_json_path: str | Path | None = None,
    out_path: str | Path | None = None,
    redact: bool = False,
    planner: AgentPlanner | None = None,
    max_steps: int = 12,
) -> dict[str, Any]:
    context = ToolContext(
        resume_path=Path(resume_path),
        redact=redact,
        screening_json_path=Path(screening_json_path) if screening_json_path else None,
        out_path=Path(out_path) if out_path else None,
    )
    tools = get_tool_registry()
    active_planner = planner or RuleBasedPlanner()
    reasoning_trace: list[dict[str, Any]] = []
    loop_status = "max_steps_reached"

    for step in range(1, max_steps + 1):
        action = active_planner.next_action(context, tools, reasoning_trace)
        if action.action_type == "finish":
            loop_status = "finished"
            reasoning_trace.append(
                {
                    "step": step,
                    "thought": action.thought,
                    "action": {"type": "finish"},
                    "observation": {"status": "finished"},
                }
            )
            break

        if action.action_type != "tool_call":
            raise ValueError(f"Invalid agent action_type: {action.action_type}")
        if action.tool_name not in tools:
            raise ValueError(f"Unknown tool requested by planner: {action.tool_name}")

        result = _execute_agent_tool(context, tools[action.tool_name], action.arguments)
        reasoning_trace.append(
            {
                "step": step,
                "thought": action.thought,
                "action": {
                    "type": "tool_call",
                    "tool_name": action.tool_name,
                    "arguments": _safe_args(action.arguments),
                },
                "observation": _summarize_tool_result(result),
            }
        )

    final_report = context.memory.get("final_report") or context.memory.get("report") or {}
    final_output = {
        "agent_type": "dynamic_tool_calling_resume_screen_agent",
        "planner": active_planner.planner_name,
        "loop_status": loop_status,
        "resume_file": str(context.resume_path),
        "reasoning_trace": reasoning_trace,
        "tool_trace": [
            {"name": item.name, "arguments": item.arguments, "result": item.result}
            for item in context.tool_trace
        ],
        "final_report": final_report,
    }

    if out_path:
        _call_tool(context, "export_report", export_report_tool, report=final_output, out_path=out_path)
        final_output["tool_trace"] = [
            {"name": item.name, "arguments": item.arguments, "result": item.result}
            for item in context.tool_trace
        ]
        Path(out_path).write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    return final_output


def _execute_agent_tool(context: ToolContext, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    args = dict(arguments)
    if tool.name == "load_screening_result" and not args.get("screening_json_path") and context.screening_json_path:
        args["screening_json_path"] = str(context.screening_json_path)
    if tool.name == "keyword_score_resume" and not args.get("must_have_result"):
        args["must_have_result"] = context.memory.get("must_have", {}).get("must_have_result", "unclear")
    return _call_tool(context, tool.name, tool.handler, **args)


def run_tool_calling_workflow(
    resume_path: str | Path,
    screening_json_path: str | Path | None = None,
    out_path: str | Path | None = None,
    redact: bool = False,
) -> dict[str, Any]:
    context = ToolContext(resume_path=Path(resume_path), redact=redact)

    _call_tool(context, "read_resume", read_resume_tool)
    loaded = _call_tool(context, "load_screening_result", load_screening_result_tool, screening_json_path=screening_json_path)
    must_have = _call_tool(context, "check_must_have", check_must_have_tool)

    screening_result = loaded.get("screening_result")
    if screening_result:
        report = dict(screening_result)
        report["must_have_result"] = must_have["must_have_result"]
        report["level"] = derive_level(int(report["score"]), report["must_have_result"])
        report = validate_result(report)
    else:
        scored = _call_tool(
            context,
            "keyword_score_resume",
            keyword_score_tool,
            must_have_result=must_have["must_have_result"],
        )
        report = scored["report"]

    evidence_check = _call_tool(context, "verify_evidence", verify_evidence_tool, report=report)
    level_result = _call_tool(
        context,
        "derive_level",
        derive_level_tool,
        score=report["score"],
        must_have_result=report["must_have_result"],
    )
    report["level"] = level_result["level"]
    report["extraction_status"] = context.extraction_status
    report["extracted_text_chars"] = context.extracted_text_chars
    report["evidence_verification"] = evidence_check

    if context.extraction_status != "ok":
        report.setdefault("risks", []).append(f"resume_text_extraction_status: {context.extraction_status}")
        report["human_review_required"] = True
    if evidence_check["failed_count"]:
        report.setdefault("risks", []).append("部分证据片段未能在简历原文中逐字校验")
        report["human_review_required"] = True

    final_output = {
        "agent_type": "tool_calling_resume_screen_agent",
        "resume_file": str(context.resume_path),
        "tool_trace": [
            {"name": item.name, "arguments": item.arguments, "result": item.result}
            for item in context.tool_trace
        ],
        "final_report": report,
    }

    if out_path:
        _call_tool(context, "export_report", export_report_tool, report=final_output, out_path=out_path)
        final_output["tool_trace"] = [
            {"name": item.name, "arguments": item.arguments, "result": item.result}
            for item in context.tool_trace
        ]
        Path(out_path).write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    return final_output


def _has_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _find_evidence(text: str, keywords: list[str]) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if part.strip()]
    lower_keywords = [keyword.lower() for keyword in keywords]
    candidates: list[tuple[int, int, int, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        keyword_hits = sum(1 for keyword in lower_keywords if keyword in lowered)
        if keyword_hits:
            strength, _ = classify_evidence_strength(sentence)
            strength_rank = {
                "implemented": 5,
                "used": 4,
                "mentioned": 3,
                "learning": 2,
                "planned": 1,
                "not_implemented": 0,
            }.get(strength, 0)
            candidates.append((strength_rank, keyword_hits, -len(sentence), sentence[:240]))
    if not candidates:
        return ""
    return max(candidates)[3]


def classify_evidence_strength(text: str) -> tuple[str, str]:
    lowered = text.lower()
    planned_cues = [
        "计划",
        "规划",
        "后续",
        "预留",
        "可扩展",
        "准备接入",
        "准备学习",
        "打算",
        "未来",
        "下一步",
        "待接入",
        "待实现",
        "将会",
        "希望",
    ]
    negative_cues = [
        "未实现",
        "尚未实现",
        "未接入",
        "尚未接入",
        "未完成",
        "尚未完成",
        "没有",
        "无 ",
        "无相关",
        "缺少",
        "不足",
    ]
    learning_cues = [
        "学习",
        "了解",
        "熟悉",
        "掌握基础",
        "入门",
        "自学",
        "有兴趣",
        "持续实践兴趣",
        "愿意",
        "尝试",
    ]
    implemented_cues = [
        "实现",
        "完成",
        "构建",
        "搭建",
        "封装",
        "开发",
        "上线",
        "落地",
        "运行",
        "生成",
        "记录",
        "校验",
        "暴露",
        "提供",
        "输出",
        "建立",
        "加入",
    ]
    used_cues = ["使用", "基于", "接入", "集成", "采用", "负责", "参与", "设计"]

    negative_hit = _first_cue(lowered, negative_cues)
    if negative_hit:
        return "not_implemented", f"出现否定类表述“{negative_hit}”，不作为有效能力证据"

    planned_hit = _first_cue(lowered, planned_cues)
    if planned_hit:
        return "planned", f"出现规划类表述“{planned_hit}”，不按已落地能力计满分"

    implemented_hit = _first_cue(lowered, implemented_cues)
    if implemented_hit:
        return "implemented", f"出现落地类动作“{implemented_hit}”"

    used_hit = _first_cue(lowered, used_cues)
    if used_hit:
        return "used", f"出现使用/参与类动作“{used_hit}”"

    learning_hit = _first_cue(lowered, learning_cues)
    if learning_hit:
        return "learning", f"出现学习/了解类表述“{learning_hit}”"

    return "mentioned", "仅出现相关关键词，缺少明确动作证据"


def _first_cue(text: str, cues: list[str]) -> str:
    for cue in cues:
        if cue.lower() in text:
            return cue
    return ""


def _score_item(text: str, evidence: list[dict[str, Any]], criterion: str, points: int, keywords: list[str]) -> int:
    quote = _find_evidence(text, keywords)
    if not quote:
        return 0

    keyword_hits = sum(1 for keyword in keywords if keyword.lower() in quote.lower())
    if keyword_hits >= 3:
        keyword_factor = 1.0
    elif keyword_hits == 2:
        keyword_factor = 0.7
    else:
        keyword_factor = 0.4

    strength, reason = classify_evidence_strength(quote)
    strength_factor = {
        "implemented": 1.0,
        "used": 0.85,
        "mentioned": 0.55,
        "learning": 0.4,
        "planned": 0.15,
        "not_implemented": 0.0,
    }.get(strength, 0.4)
    score = round(points * keyword_factor * strength_factor)

    evidence.append(
        {
            "criterion": criterion,
            "score": score,
            "evidence_strength": strength,
            "strength_reason": reason,
            "resume_text": quote,
        }
    )
    return score


def _derive_strengths(evidence: list[dict[str, Any]]) -> list[str]:
    strengths = []
    for item in evidence:
        if item["score"] >= 4 and _is_credible_evidence(item):
            strengths.append(item["criterion"])
    return strengths[:6]


def _derive_risks(score: int, must_have_result: str, evidence: list[dict[str, Any]]) -> list[str]:
    risks = []
    if must_have_result != "pass":
        risks.append("硬性门槛未完全满足")
    if score < 65:
        risks.append("总分未达到 backup 阈值")
    covered = {item["criterion"] for item in evidence if _is_credible_evidence(item)}
    if not any("RAG" in item or "文档解析" in item for item in covered):
        risks.append("缺少 RAG 项目证据")
    if not any("Tool Calling" in item or "Function Calling" in item for item in covered):
        risks.append("缺少 Tool Calling 项目证据")
    weak_evidence = [
        item["criterion"]
        for item in evidence
        if item.get("evidence_strength") in {"planned", "learning", "not_implemented"}
    ]
    if weak_evidence:
        risks.append("部分能力仅为计划或学习表述，未按已落地证据计满分：" + "、".join(weak_evidence[:3]))
    return risks


def _derive_missing_information(evidence: list[dict[str, Any]]) -> list[str]:
    covered = {item["criterion"] for item in evidence if _is_credible_evidence(item)}
    missing = []
    if not any("MCP" in item for item in covered):
        missing.append("MCP Server / Client 或工具协议封装经验")
    if not any("Benchmark" in item or "回归测试" in item for item in covered):
        missing.append("Agent Eval / Benchmark / 回归测试")
    if not any("向量数据库" in item for item in covered):
        missing.append("向量数据库 / Hybrid Search / Rerank")
    return missing


def _is_credible_evidence(item: dict[str, Any]) -> bool:
    return item.get("evidence_strength", "implemented") in {"implemented", "used"}


def _recommended_next_step(score: int, must_have_result: str) -> str:
    if must_have_result == "fail":
        return "暂不推荐进入技术面，可补充 AI / Agent 项目后复评"
    if score >= 75:
        return "建议人工复核后进入技术面"
    if score >= 65:
        return "建议进入备选池并补充项目细节"
    return "建议继续补充 RAG、Tool Calling、Eval 等项目证据"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tool-calling workflow over one resume.")
    parser.add_argument("--resume", required=True, help="Path to resume file.")
    parser.add_argument("--screening-json", help="Optional existing screening JSON to normalize and verify.")
    parser.add_argument("--out", help="Output JSON path. If omitted, print to stdout.")
    parser.add_argument("--redact", action="store_true", help="Redact basic personal info before tool calls.")
    parser.add_argument("--mode", choices=["fixed", "dynamic"], default="fixed", help="fixed keeps the old deterministic workflow; dynamic runs a planner/tool loop.")
    parser.add_argument("--llm-planner", action="store_true", help="Use the configured model to choose the next tool in dynamic mode.")
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum planner/tool iterations in dynamic mode.")
    parser.add_argument("--model", help="Override planner model name when --llm-planner is used.")
    parser.add_argument("--base-url", help="Override planner OpenAI-compatible base URL when --llm-planner is used.")
    args = parser.parse_args()

    if args.mode == "dynamic":
        planner: AgentPlanner | None = None
        if args.llm_planner:
            planner = JsonModelPlanner(model=ChatModel(model=args.model, base_url=args.base_url))
        result = run_dynamic_tool_calling_agent(
            resume_path=args.resume,
            screening_json_path=args.screening_json,
            out_path=args.out,
            redact=args.redact,
            planner=planner,
            max_steps=args.max_steps,
        )
    else:
        result = run_tool_calling_workflow(
            resume_path=args.resume,
            screening_json_path=args.screening_json,
            out_path=args.out,
            redact=args.redact,
        )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        print(f"Tool agent report written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
