# 候选人分层标准

## 五个等级

| 等级 | 分数区间 | 含义 | 建议动作 |
|------|---------|------|---------|
| strong_match | 85-100 | 高度匹配 | 直接进入技术面 |
| review | 75-84 | 需要复核 | HR/技术负责人复核后面试 |
| backup | 65-74 | 备选池 | 可培养或方向接近，放入备选池 |
| weak_match | 50-64 | 弱匹配 | 除非岗位放宽否则不优先 |
| not_recommended | <50 | 不推荐 | 不推荐进入技术面 |

## 进入 backup 的条件

候选人需要满足以下全部条件才能进入 backup（65-74 分）：

### 必要条件
1. **must_have_result = "pass" 或 "unclear"**：硬性门槛不能是 fail
2. **分数在 65-74 分之间**
3. **至少有一项 AI/Agent/RAG 相关经验**（即使不深入）

### 典型 backup 画像
- 有后端工程经验但 AI 经验较浅（如 1-2 年后端 + 个人 AI 项目）
- 应届生有较强的 AI 项目经验但无企业工作经验
- 方向接近（如 NLP、搜索、推荐）转 AI Agent 方向
- 有扎实编程基础但 Agent 框架经验不足

### 与 review（75-84）的区别
- review 候选人：AI/Agent 经验充分，可能有 1-2 个小短板
- backup 候选人：有基础但缺少关键经验（如无 Agent 框架、无 RAG、无 Tool Calling），需要培养

### 与 weak_match（50-64）的区别
- weak_match：仅 API 调用层面，缺少系统级 Agent 理解
- backup：有系统级理解或工程基础，但 Agent 实战偏少

## 等级决定规则

`level` 由分数、硬性门槛和封顶规则共同决定：

- must_have_result = "fail" → 强制 not_recommended
- 无 AI/Agent/RAG 项目 → 封顶 65 分 → 最高 backup
- 仅后端无 LLM → 封顶 60 分 → 最高 weak_match
- 最终分数决定最终等级区间
