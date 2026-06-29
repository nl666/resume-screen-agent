# MCP Server 模块

## 模块目标

MCP Server 用于把简历初筛项目包装成可被外部 Agent 调用的工具服务。外部 Agent 不需要直接调用 Python 脚本，只需要通过 MCP 的 `tools/list` 和 `tools/call` 使用项目能力。

## 暴露工具

1. `screen_resume`
   - 运行单份简历筛选。
   - 默认使用 `dynamic` 模式，也就是 Planner / Tool Call / Observation / Next Action 循环。
   - 支持传入已有 `screening_json_path` 做校验和分层修正。

2. `rag_query`
   - 查询本地知识库。
   - 默认是 retrieval-only 模式，不依赖大模型。
   - `use_llm=true` 时调用模型生成自然语言答案，并保留 sources 引用。

3. `run_eval`
   - 运行 Agent Eval / 回归测试。
   - 输出总样例数、通过样例数、失败断言和详细 case 结果。

4. `export_report`
   - 将 JSON 报告导出到项目目录内。
   - 只允许写入项目根目录内部，避免 MCP 客户端误写系统路径。

## 协议能力

MCP Server 支持：

- `initialize`：返回 server info 和 tools capability。
- `tools/list`：返回工具名称、说明和 inputSchema。
- `tools/call`：根据工具名和 arguments 执行项目能力。
- `ping`：连通性检查。

## 工程价值

MCP Server 让项目从“本地脚本工具”升级为“可被 Agent 编排的工具服务”。这覆盖了 AI Agent 开发岗位中常见的 MCP、工具协议封装、外部系统集成、结构化工具输入输出和可观测性要求。
