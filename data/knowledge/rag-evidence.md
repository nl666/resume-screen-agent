# RAG 经验证据类型

## 什么样的证据能证明 RAG 经验？

RAG（检索增强生成）能力评分共 15 分，分为三个子项：

### 1. 文档解析、Chunk、Embedding、向量检索（5 分）

**需要看到的证据：**
- 使用过文档解析工具/库：PyPDF2、pypdf、python-docx、Unstructured、LangChain Document Loaders
- 实现过 Chunk 切分策略：固定长度切分、语义切分、滑动窗口、父子 Chunk
- 使用过 Embedding 模型：OpenAI text-embedding-3、BGE、M3E、Jina Embeddings
- 实现过向量检索流程：文档 → 解析 → Chunk → Embedding → 存入向量库 → 相似度检索

**40% 分（2 分）**：提到 RAG 概念或看过相关文章
**70% 分（3.5 分）**：有 RAG 项目但缺少技术细节
**100% 分（5 分）**：详细描述文档解析→Chunk→Embedding→检索完整链路

### 2. 向量数据库、Elasticsearch、Hybrid Search、Rerank（5 分）

**需要看到的证据：**
- 使用过向量数据库：Milvus、Qdrant、Pinecone、Weaviate、Chroma、FAISS
- 实现过混合检索：向量检索 + BM25/关键词检索（Hybrid Search）
- 实现过 Rerank：Cross-encoder、Cohere Rerank、BGE Reranker
- 使用过 Elasticsearch 做全文检索

**40% 分（2 分）**：仅提到向量数据库名称
**70% 分（3.5 分）**：使用过向量数据库但未涉及混合检索或 Rerank
**100% 分（5 分）**：有向量检索 + 混合检索 + Rerank 完整方案

### 3. 引用溯源、幻觉控制、知识库权限或多租户隔离（5 分）

**需要看到的证据：**
- 实现过答案引用溯源：引用原文片段、标注来源文档和页码
- 做过幻觉控制：限定回答范围、拒绝回答知识库外问题
- 实现过知识库权限：按用户/部门/角色隔离文档访问
- 支持多租户：多知识库管理、租户间数据隔离

**40% 分（2 分）**：提到相关概念
**70% 分（3.5 分）**：实现过引用溯源
**100% 分（5 分）**：引用溯源 + 幻觉控制 + 权限/多租户

## 简历中如何有效展示 RAG 经验

推荐写法示例：
> "基于 LangChain + Milvus 构建 RAG 问答系统，使用 BGE Embedding 和 BM25 混合检索，配合 BGE Reranker 重排序。实现文档解析（PDF/DOCX/TXT）、语义 Chunk 切分（512 token 滑动窗口）、向量检索 Top-20 + Rerank Top-5。系统包含引用溯源功能，答案中标注来源文档和原文片段。知识库支持按部门隔离权限。"

不推荐的写法：
> "了解 RAG 技术"
> "做过知识库问答"
