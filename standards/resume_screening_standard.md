# 简历初筛评分标准 v1

一、硬性门槛：先判断 `pass / fail / unclear`。

二、能力评分：满分 100。

三、候选人分层：`strong_match / review / backup / weak_match / not_recommended`。

四、每项评分必须引用简历证据。

## 封顶规则

- 如果没有任何真实 AI / Agent / RAG 项目证据，总分最高不超过 65 分。
- 如果只有后端经验但没有 LLM / Agent 相关证据，总分最高不超过 60 分。
- 如果硬性门槛为 `fail`，即使分数较高，也不得推荐进入技术面。

## 硬性门槛

1. 有 Python / Java / Go / TypeScript 中至少一门主力开发语言。
2. 有后端开发、AI 应用开发或系统集成经验。
3. 有 LLM / Agent / RAG / Tool Calling / Prompt 工程相关项目经验。
4. 能看到真实项目证据：上线项目、企业项目、Side Project、GitHub、技术文章均可。

硬性门槛判断：

- 满足 3 项及以上：`must_have_result = "pass"`。
- 满足 2 项：`must_have_result = "unclear"`。
- 满足 0-1 项：`must_have_result = "fail"`。

## 给分规则

- 0 分：未提及。
- 40% 分值：只提到关键词或学习经历。
- 70% 分值：有项目经历，但缺少技术细节或结果。
- 100% 分值：有项目 + 技术栈 + 个人职责 + 结果证据。

## 分值明细

### 后端工程能力：25 分

- 编程语言与代码能力：8
- API、数据库、缓存、消息队列、搜索等基础组件：7
- 系统设计、异步任务、稳定性、性能优化：6
- Docker、K8s、CI/CD、部署经验：4

### LLM / Agent 实战能力：25 分

- 调用并集成主流 LLM API：5
- Prompt Engineering、结构化输出、上下文管理：5
- Agent 任务拆解、Planning、Memory、Tool Use：7
- LangGraph / LangChain / Dify / AutoGen / CrewAI 等框架实践：5
- 有真实 Agent 项目落地经验：3

### RAG / 知识库能力：15 分

- 文档解析、Chunk、Embedding、向量检索：5
- 向量数据库、Elasticsearch、Hybrid Search、Rerank：5
- 引用溯源、幻觉控制、知识库权限或多租户隔离：5

### Tool Calling / MCP / 系统集成：15 分

- Function Calling / Tool Calling 实践：5
- MCP Server / Client 或工具协议封装经验：5
- 对接数据库、内部系统、第三方 API、搜索引擎等经验：5

### 工程化、评测、安全、成本：15 分

- 日志、监控、链路追踪、可观测性：3
- Agent Eval、Benchmark、回归测试：4
- Prompt Injection、防越权、敏感操作审批、沙箱隔离：4
- Token 成本、模型路由、缓存、TTFT 优化：4

### 沟通与产品落地：5 分

- 能把业务需求拆成技术方案：2
- 能写清楚设计文档、复盘、技术说明：2
- 有跨团队协作经验：1

## 分层标准

- 85-100：`strong_match`，建议直接进入技术面。
- 75-84：`review`，建议 HR/技术负责人复核后面试。
- 65-74：`backup`，可培养或方向接近，建议放入备选池。
- 50-64：`weak_match`，除非岗位放宽否则不优先。
- 50 以下：`not_recommended`。

`level` 必须根据最终分数、硬性门槛和封顶规则共同决定。

## 人工复核触发条件

- 简历描述很少，但方向可能相关。
- 有 Agent / RAG 项目，但技术细节不足。
- 分数在 65-79 分之间。
- 候选人有强工程背景但 AI 经验证据不充分。
- 简历中存在明显夸大或无法验证的表述。
- `must_have_result = "unclear"`。

如果触发任何人工复核条件，`human_review_required` 必须为 `true`。

## 合规与公平

评分必须基于简历证据，不可任意编造，不能有任何歧视。不得基于姓名、性别、年龄、照片、婚育、籍贯、民族、宗教、残障、住址、学校偏见、空窗期偏见等非岗位能力因素评分。
