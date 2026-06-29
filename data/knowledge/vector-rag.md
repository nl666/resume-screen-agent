# 本项目的向量 RAG 实现

## 模块目标

知识库问答不再只做关键词搜索，而是按真实 RAG 链路运行：

1. 读取 `data/knowledge/` 下的 Markdown、TXT、CSV、JSONL、DOCX、PDF 文档。
2. 将文档切分成可引用的 Chunk，每个 Chunk 保留来源文件和编号。
3. 使用本地 hashing embedding 将 Chunk 转成稀疏向量。
4. 将向量索引写入 `data/vector_index/knowledge_index.json`。
5. 用户提问时，将问题同样转成向量。
6. 使用余弦相似度召回相关 Chunk。
7. 默认使用向量相似度 + 关键词匹配的混合检索，提升中文短问题的命中稳定性。
8. 返回答案、引用来源、检索方式、索引片段数量。

## 当前 Embedding 方案

当前使用 `local-hashing-embedding-v1`，这是一个本地可运行的 deterministic hashing embedding：

- 优点：不依赖外部 API，不需要向量数据库服务，单元测试稳定。
- 优点：能完整展示文档解析、Chunk、Embedding、向量索引、余弦检索、引用溯源链路。
- 限制：它不是深度语义模型，对同义词、复杂语义改写的理解弱于 BGE、Jina、OpenAI Embedding。

## 后续可升级方向

如果要做生产级 RAG，可以把 `embed_text()` 替换为外部 embedding 模型：

- OpenAI `text-embedding-3-small` 或 `text-embedding-3-large`
- BGE / BGE-M3
- Jina Embeddings
- 其他 OpenAI-compatible embedding API

向量存储也可以从本地 JSON 索引升级为：

- Chroma：适合本地 Demo 和轻量项目
- FAISS：适合本地高性能向量检索
- Qdrant / Milvus：适合服务化、权限、多租户、较大规模知识库

## BGE + Chroma 增强模式

本项目已支持可选增强后端：

```text
知识库文档 -> Chunk -> BGE Embedding -> Chroma 本地向量库 -> 向量相似度检索 -> 引用溯源
```

安装依赖：

```bash
pip install -r requirements-bge-chroma.txt
```

构建 Chroma 索引：

```bash
python scripts/build_chroma_index.py
```

查询：

```bash
python scripts/rag_qa.py --question "MCP Server 暴露了哪些工具？" --vector-store chroma --retrieval-only
```

默认模型是 `BAAI/bge-small-zh-v1.5`。第一次运行会下载模型，下载完成后可复用本地缓存。

## 三种回答模式

知识库问答支持三种用户可切换的回答模式：

1. 严格根据知识库
   - 先从向量库检索相关 Chunk。
   - LLM 只能根据检索到的 context 回答。
   - 如果 context 不足，应回答知识库中没有足够信息。
   - 适合岗位标准、评分规则、项目模块说明等需要依据明确的问题。

2. 知识库 + 模型补充
   - 先从向量库检索相关 Chunk。
   - 回答中优先列出知识库依据。
   - 如果资料不足或用户需要建议，允许 LLM 追加通用经验，但必须标注为模型补充。
   - 适合项目优化、简历表达、技术方案扩展等问题。

3. 模型自由回答
   - 不查询向量库。
   - 直接调用 LLM 根据通用知识回答。
   - sources 为空，retrieval 中会标记未使用向量库。
   - 适合学习路线、开放性 brainstorming、非项目知识库问题。

## 对简历筛选系统的价值

向量 RAG 主要用于解释和校准筛选依据：

- 查询岗位能力要求
- 查询评分标准
- 查询什么证据能证明 RAG、Tool Calling、MCP、Eval 经验
- 为 Agent 的工具调用提供可引用的规则片段
- 输出来源文件和 Chunk，减少无依据回答
