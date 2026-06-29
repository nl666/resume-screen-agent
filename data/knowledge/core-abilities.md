# 岗位核心能力要求

## AI Agent 开发工程师最看重的能力（按优先级排序）

### 第一梯队：LLM / Agent 实战能力（25 分）

这是岗位的核心竞争力，权重最高：

1. **Agent 任务拆解、Planning、Memory、Tool Use（7 分）**
   - 能否将复杂任务拆成可执行的步骤
   - 是否理解多轮对话中的上下文管理
   - 有没有 Tool Use 实践经验

2. **LLM API 集成（5 分）**
   - 调用主流 LLM API（OpenAI、DeepSeek 等）
   - 处理流式输出、异常重试、超时控制

3. **Prompt Engineering、结构化输出（5 分）**
   - 设计系统提示词与评分标准
   - 实现结构化 JSON 输出并校验

4. **Agent 框架实践（5 分）**
   - LangGraph、LangChain、Dify、AutoGen、CrewAI 至少一种
   - 理解工作流编排、条件分支、状态管理

5. **真实 Agent 项目落地（3 分）**
   - 线上运行的项目，而非仅 demo
   - 有实际上线数据和效果反馈

### 第二梯队：后端工程能力（25 分）

1. **编程语言与代码能力（8 分）**：Python 主力 + 第二语言
2. **基础组件（7 分）**：API、数据库、缓存、消息队列、搜索
3. **系统设计（6 分）**：异步任务、稳定性、性能优化
4. **部署运维（4 分）**：Docker、K8s、CI/CD

### 第三梯队：RAG / 知识库（15 分）+ Tool Calling / MCP（15 分）

- 文档解析、Chunk、Embedding、向量检索
- Function Calling / Tool Calling / MCP
- 对接数据库、内部系统、第三方 API

### 第四梯队：工程化 + 安全 + 成本（15 分）+ 沟通（5 分）

- 日志、监控、链路追踪
- Agent Eval、Benchmark
- Prompt Injection 防护
- Token 成本优化
- 业务需求拆解、文档撰写、跨团队协作

## 封顶规则

- 无真实 AI / Agent / RAG 项目证据 → 总分 ≤ 65
- 只有后端经验但无 LLM/Agent 证据 → 总分 ≤ 60
- must_have_result = fail → 不推荐进入技术面（无论分数多少）
