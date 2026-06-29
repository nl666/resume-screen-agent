# Tool Calling Agent 模块

## 模块目标

Tool Calling Agent 用于把简历初筛流程拆成多个可调用工具，让筛选过程不再是一次性模型输出，而是具备可追踪的执行链路。

## 已实现工具

1. `read_resume`：读取 PDF / DOCX / TXT / Markdown 简历，并返回文本提取状态。
2. `lookup_screening_rules`：从本地知识库检索评分规则和能力证据说明。
3. `load_screening_result`：读取已有模型初筛 JSON，并进行结构校验。
4. `check_must_have`：检查硬性门槛，包括主力开发语言、后端或系统集成经验、LLM/Agent/RAG/Tool Calling 经验、真实项目证据。
5. `keyword_score_resume`：在没有模型初筛结果时，基于关键词和证据片段做保守评分。
6. `build_report_from_screening_result`：复用已有模型初筛 JSON，并用当前硬性门槛结果修正最终分层。
7. `verify_evidence`：校验证据片段是否真的出现在简历原文中。
8. `derive_level`：根据 score 和 must_have_result 强制计算最终分层，避免模型分层漂移。
9. `finalize_report`：汇总提取状态、证据校验、风险和最终报告。
10. `export_report`：导出完整工具调用报告。

## 动态 Agent Loop

新版 Tool Calling Agent 支持 Planner / Tool Call / Observation / Next Action 循环：

1. Planner 根据当前 memory 和已调用工具选择下一步 action。
2. 执行器只允许调用工具注册表中的工具，避免任意代码执行。
3. 工具返回 observation，并把关键状态写入 agent memory。
4. Planner 根据 observation 决定继续调用工具或 finish。
5. 最终输出 `reasoning_trace`、`tool_trace` 和 `final_report`。

当提供已有初筛 JSON 时，Agent 会走 `load_screening_result` + `build_report_from_screening_result` 分支；没有已有初筛 JSON 时，Agent 会走 `keyword_score_resume` 分支。

## 输出结构

Tool Calling Agent 输出包括：

- `agent_type`：固定流程为 `tool_calling_resume_screen_agent`，动态循环为 `dynamic_tool_calling_resume_screen_agent`。
- `resume_file`：被分析的简历文件。
- `reasoning_trace`：Planner 每轮决策、工具调用和 observation。
- `tool_trace`：工具调用轨迹，每一步包含工具名、参数摘要和结果摘要。
- `final_report`：最终初筛报告，包括分数、分层、硬性门槛、证据校验和人工复核建议。

## 价值

该模块覆盖 Agent 开发 JD 中的 Tool Calling、工具封装、流程编排、证据校验、结构化输出和可解释性要求。
