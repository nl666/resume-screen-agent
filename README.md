# Resume Screen Agent

AI Agent 开发工程师简历初筛 MVP。

这个项目用于读取岗位 JD、评分标准和候选人简历，调用 DeepSeek/OpenAI 兼容模型，输出结构化 JSON 初筛结果，并支持批量导出 CSV。

## 适合第一版做什么

- 单份简历初筛
- 批量简历初筛
- 固定评分标准
- 固定 JSON 输出
- 人工复核标记
- 简单个人信息脱敏

第一版不引入 LangChain / LangGraph。当前任务本质是文档读取、规则评分和 JSON 输出，普通 API 调用更稳。

## 目录结构

```text
resume-screen-agent/
  data/
    eval/
      resume_eval_cases.jsonl
    jd.txt
    knowledge/
    resumes/
  prompts/
    rag_system_prompt.md
    screening_system_prompt.md
    tool_agent_system_prompt.md
  standards/
    resume_screening_standard.md
  results/
  scripts/
    batch_screen.py
    build_chroma_index.py
    build_vector_index.py
    eval_agent.py
    mcp_server.py
    rag_qa.py
    screen_one.py
    tool_agent.py
    web_app.py
  src/
    resume_screen_agent/
      batch.py
      extract.py
      eval.py
      llm.py
      mcp_server.py
      rag.py
      redact.py
      schema.py
      screen.py
      tool_agent.py
      web_app.py
  tests/
    test_eval.py
    test_mcp_server.py
    test_rag.py
    test_schema.py
    test_tool_agent.py
    test_web_app.py
  web/
    static/
      index.html
  .env.example
  requirements.txt
  requirements-bge-chroma.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```text
DEEPSEEK_API_KEY=sk-...
MODEL_NAME=deepseek-chat
BASE_URL=https://api.deepseek.com
```

不要把 `.env` 提交或发给别人。

## 单份简历初筛

把简历文本放到 `data/resumes/`，支持 `.txt`、`.md`、`.docx`、`.pdf`。

```bash
python scripts/screen_one.py --resume data/resumes/candidate.txt --out results/candidate.json
```

如果想先做基本脱敏：

```bash
python scripts/screen_one.py --resume data/resumes/candidate.txt --out results/candidate.json --redact
```

## 批量初筛

```bash
python scripts/batch_screen.py --resume-dir data/resumes --out-dir results
```

输出：

```text
results/screening_results.jsonl
results/screening_results.csv
```

CSV 中的 `extraction_status` 用于检查 PDF 是否被正确读取：

```text
ok：文本提取正常
too_short_check_ocr：提取文字过少，可能是扫描版 PDF，需要 OCR
empty：未提取到文字
failed：解析或模型调用失败
```

## RAG 知识库问答

知识库目录：

```text
data/knowledge/
```

构建本地向量索引：

```bash
python scripts/build_vector_index.py
```

可选：安装并构建 BGE + Chroma 本地向量库：

```bash
pip install -r requirements-bge-chroma.txt
python scripts/build_chroma_index.py
```

运行问答：

```bash
python scripts/rag_qa.py --question "什么样的证据能证明 RAG 经验？" --retrieval-only --out results/rag_answer.json
```

默认使用“向量相似度 + 关键词匹配”的混合检索。如果只想验证向量召回：

```bash
python scripts/rag_qa.py --question "MCP Server 暴露了哪些工具？" --retrieval-mode vector --retrieval-only
```

使用 BGE + Chroma 查询：

```bash
python scripts/rag_qa.py --question "MCP Server 暴露了哪些工具？" --vector-store chroma --retrieval-only
```

三种回答模式：

```bash
# 严格根据知识库回答：先检索资料，再要求 LLM 只根据资料回答
python scripts/rag_qa.py --question "MCP Server 暴露了哪些工具？" --answer-mode strict

# 知识库 + 模型补充：先给知识库依据，再允许模型补充通用建议
python scripts/rag_qa.py --question "这个 Agent 项目还能怎么优化？" --answer-mode mixed

# 模型自由回答：不查询知识库，直接让 LLM 根据通用能力回答
python scripts/rag_qa.py --question "AI Agent 学习路线怎么安排？" --answer-mode free
```

输出包含：

```text
answer：基于知识库的回答
sources / citations：带 [S1] 编号的引用证据，包含来源文件、chunk_id、原文片段、相关度、匹配原因
citation_summary：引用概览，包含命中片段数、覆盖文件数、高相关证据数、最高相关度
confidence：high / medium / low
retrieval：回答模式、检索方式、向量模型、索引路径、知识片段数量、引用数量、覆盖文件
```

说明：

```text
默认 local 模式：无需额外依赖，使用本地 hashing embedding + JSON 向量索引。
增强 chroma 模式：使用 BGE embedding + Chroma 本地向量数据库，语义检索效果更接近真实生产 RAG。
```

## Tool Calling Agent

Tool Calling Agent 会把简历初筛拆成可追踪工具调用：

```text
read_resume
lookup_screening_rules
load_screening_result
check_must_have
keyword_score_resume
build_report_from_screening_result
verify_evidence
derive_level
finalize_report
export_report
```

只基于简历本身运行工具调用流程：

```bash
python scripts/tool_agent.py --resume examples/sample_resume.txt --out results/tool_agent_sample.json --redact
```

运行更真实的 Planner / Tool / Observation 循环：

```bash
python scripts/tool_agent.py --mode dynamic --resume examples/sample_resume.txt --out results/dynamic_tool_agent_sample.json --redact
```

用模型决定下一步工具调用：

```bash
python scripts/tool_agent.py --mode dynamic --llm-planner --resume examples/sample_resume.txt --out results/llm_planner_tool_agent_sample.json --redact
```

结合已有模型初筛结果做校验与分层修正：

```bash
python scripts/tool_agent.py --resume data/resumes/南亮_AI应用开发简历.pdf --screening-json results/nanliang.json --out results/nanliang_tool_report.json --redact
```

输出包含：

```text
tool_trace：每一步工具调用轨迹
reasoning_trace：Planner 每轮决策、工具调用和 observation
final_report：最终初筛报告
evidence_verification：证据片段是否能在简历原文中校验
```

## Agent Eval / 回归测试

Eval 用来防止改提示词、评分规则或工具链之后，结果悄悄漂移。

默认黄金样例在：

```text
data/eval/resume_eval_cases.jsonl
```

一键运行：

```bash
python scripts/eval_agent.py
```

指定输出路径：

```bash
python scripts/eval_agent.py --out results/agent_eval_report.json
```

每个 eval case 包含：

```text
resume_text：脱敏简历样例
expected：期望断言，例如 must_have_result、level、min_score、max_score、required_tools
```

评测报告包含：

```text
summary：通过样例数、失败样例数、断言通过率
cases：每个样例的分数、等级、工具调用链、失败断言
```

## MCP Server

MCP Server 用于把项目能力暴露给外部 Agent 或 MCP 客户端调用。

启动 stdio server：

```bash
python scripts/mcp_server.py
```

暴露工具：

```text
screen_resume：运行简历筛选 Agent，默认使用 dynamic Planner/Tool/Observation 循环
rag_query：查询本地知识库，默认检索模式，use_llm=true 时调用模型生成答案
run_eval：运行 Agent Eval 回归测试
export_report：导出 JSON 报告到项目目录内
```

MCP 客户端配置示例：

```json
{
  "mcpServers": {
    "resume-screen-agent": {
      "command": "python",
      "args": ["scripts/mcp_server.py"],
      "cwd": "C:\\Users\\nnn\\Documents\\Codex\\2026-06-27\\wo\\outputs\\resume-screen-agent"
    }
  }
}
```

## Web / API 服务

安装依赖后启动：

```bash
python scripts/web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

主要接口：

```text
GET  /api/health
POST /api/screen-resume
POST /api/batch-screen
POST /api/batch-screen-async
POST /api/rag-query
POST /api/run-eval
GET  /api/results
GET  /api/result?name=...
GET  /api/operations
```

页面支持：

```text
单份简历上传筛选
批量简历上传筛选
RAG 知识库问答
Eval 回归测试
历史结果查看
操作记录查看
批量任务进度追踪
RAG 引用证据展示
```

日志与操作记录：

```text
logs/web_app.log：后端运行日志，主要用于排查异常和服务状态。
logs/operations.jsonl：结构化操作记录，每行一个 JSON 事件。
```

操作记录会记录：

```text
operation_id：一次操作的唯一编号
kind：操作类型，例如 screen_resume、batch_screen_async、rag_query、run_eval
status：started / queued / running / succeeded / failed
message：中文操作说明
duration_ms：耗时
result_name：可回放的结果文件名
summary：分数、引用数、通过率等关键摘要
error：失败原因摘要
```

## 本地校验

不需要 API Key，先跑 schema 测试：

```bash
python -m unittest discover -s tests
```

## 下一步

1. 放入真实 JD 到 `data/jd.txt`。
2. 放入 3-5 份脱敏简历样例。
3. 跑单份初筛，检查 JSON 是否稳定。
4. 再跑批量初筛，检查 CSV 字段是否够用。
5. 根据人工复核结果微调评分标准和系统提示词。
