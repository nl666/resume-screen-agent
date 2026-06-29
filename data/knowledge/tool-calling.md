# Tool Calling 评分判断标准

## Tool Calling / MCP / 系统集成（15 分）

### 1. Function Calling / Tool Calling 实践（5 分）

**判断标准：**

**0 分**：未提及
**2 分（40%）**：提到 Function Calling 概念或看过文档
**3.5 分（70%）**：使用过 Function Calling，但仅限简单的单工具调用
**5 分（100%）**：实现过多工具编排、参数校验、错误处理、超时重试、并行调用

**强证据：**
- 使用 OpenAI Function Calling / DeepSeek Tool Calling
- 实现了多个 Tool 的注册、描述、参数 Schema 定义
- 处理了 Tool 调用异常、超时、重试
- 工具返回结果回写对话上下文
- 支持并行调用多个工具

**弱证据（不充分）：**
- 仅"了解 Function Calling"
- 只调用了一个 API 就说做了 Tool Calling

### 2. MCP Server / Client 或工具协议封装经验（5 分）

**判断标准：**

**0 分**：未提及
**2 分（40%）**：了解 MCP 协议
**3.5 分（70%）**：使用过 MCP Client 或开发过简单 MCP Server
**5 分（100%）**：开发过 MCP Server + Client，实现了多工具注册与发现

**强证据：**
- 实现过 MCP Server，暴露多个 Tool
- 对接了 MCP 生态（如使用 mcp-go、mcp-python 等 SDK）
- 实现过工具协议封装（非 MCP 的自定义协议也算）
- 有工具注册、发现、权限控制机制

### 3. 对接数据库、内部系统、第三方 API、搜索引擎等经验（5 分）

**判断标准：**

**0 分**：未提及
**2 分（40%）**：仅理论了解或简单 HTTP 调用
**3.5 分（70%）**：对接过数据库或第三方 API，但无 Agent 工具封装
**5 分（100%）**：将数据库/API 封装为 Agent 可调用的标准化工具

**强证据：**
- 将数据库查询封装为 Tool（含参数 Schema）
- 对接搜索引擎（Elasticsearch、SerpAPI 等）
- 对接内部系统（工单系统、CRM、OA 等）
- 实现了工具的认证、鉴权、限流

## 常见误判

1. **"用过 API" ≠ Function Calling**：普通的 REST API 调用不算 Function Calling，必须是为 LLM 设计的工具调用机制
2. **"用过 Postman" ≠ 系统集成**：必须是开发工作中对接系统，不只是测试接口
3. **"了解 MCP" ≠ 有 MCP 经验**：需要有实际开发或使用经验
