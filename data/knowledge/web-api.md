# Web / API 服务模块

## 模块目标

Web / API 服务把简历初筛 Agent 从命令行工具升级为可视化工作台。用户可以通过浏览器上传简历、查看筛选结果、执行 RAG 问答、运行 Eval 回归测试，并查看历史 JSON 报告。

## 后端接口

FastAPI 后端提供以下接口：

1. `GET /api/health`
   - 检查服务是否在线。
   - 返回支持的简历文件扩展名。

2. `POST /api/screen-resume`
   - 上传单份简历。
   - 默认使用动态 Tool Calling Agent。
   - 返回 `summary`、`final_report`、`reasoning_trace` 和 `tool_trace`。

3. `POST /api/batch-screen`
   - 上传多份简历。
   - 对每份简历独立执行筛选。
   - 返回批次 summary 和每份简历的结果文件路径。

4. `POST /api/rag-query`
   - 对本地知识库执行检索问答。
   - 默认 retrieval-only，不依赖大模型。
   - 可通过 `use_llm=true` 调用模型生成自然语言答案。

5. `POST /api/run-eval`
   - 运行 Agent Eval / 回归测试。
   - 返回通过样例数、失败样例数和断言统计。

6. `GET /api/results`
   - 查看历史结果列表。

7. `GET /api/result?name=...`
   - 查看某个历史 JSON 结果。

## 页面功能

前端页面包含：

- 单份简历上传筛选。
- 批量简历上传筛选。
- RAG 知识库问答。
- Eval 回归测试。
- 历史结果列表和详情查看。

## 工程价值

Web / API 模块让项目具备产品化展示能力，覆盖 FastAPI 后端、文件上传、批量处理、历史结果管理、前后端交互和 Agent 工作流服务化能力。
