# 代码模块说明

## 项目结构

```
resume-screen-agent/
├── src/resume_screen_agent/
│   ├── __init__.py          # 包初始化
│   ├── extract.py           # 文档提取（PDF/DOCX/TXT/MD）
│   ├── llm.py               # LLM 调用封装（ChatModel）
│   ├── schema.py            # JSON 解析、校验、等级推导
│   ├── screen.py            # 单份简历筛选入口
│   ├── batch.py             # 批量筛选逻辑
│   ├── redact.py            # 个人信息脱敏
│   └── rag.py               # RAG 知识库问答
├── scripts/
│   ├── screen_one.py        # 单份简历筛选 CLI
│   ├── batch_screen.py      # 批量筛选 CLI
│   └── rag_qa.py            # 知识库问答 CLI
├── data/
│   ├── jd.txt               # 岗位 JD
│   ├── resumes/              # 待筛选简历
│   └── knowledge/            # 知识库文档
├── prompts/
│   ├── screening_system_prompt.md  # 筛选系统提示词
│   └── rag_system_prompt.md        # RAG 问答系统提示词
├── standards/
│   └── resume_screening_standard.md  # 评分标准
├── results/                 # 输出结果
├── requirements.txt
└── .env                     # API 密钥配置
```

## 模块职责

### extract.py
- `read_text(path)` — 读取文档文本（支持 .txt .md .docx .pdf）
- `read_text_with_diagnostics(path)` — 读取并返回字符数、状态

### llm.py
- `load_dotenv(path)` — 加载 .env 环境变量
- `ChatModel` — LLM 调用封装，提供 `complete_json()` 方法

### schema.py
- `extract_json_object(text)` — 从模型输出解析 JSON（容错）
- `validate_result(data)` — 校验筛选结果并修正 level
- `derive_level(score, must_have_result)` — 根据分数和硬性门槛推导等级

### screen.py
- `screen_resume_text()` — 纯文本筛选
- `screen_resume_file()` — 文件级别筛选（含提取→脱敏→评分）
- `main()` — CLI 入口

### batch.py
- `screen_resume_dir()` — 批量筛选目录下所有简历
- `main()` — CLI 入口

### rag.py
- `load_documents()` — 加载知识库目录中的所有文档
- `chunk_text()` — 将文档切分为 500-800 字的 Chunk
- `retrieve()` — 基于关键词匹配检索 Top-K 相关 Chunk
- `answer_with_context()` — 将 Chunk + 问题发给模型生成回答

### redact.py
- `redact_basic_personal_info()` — 脱敏姓名、电话、邮箱等

## 扩展建议

### 添加新的文档格式支持
编辑 `extract.py` 的 `SUPPORTED_EXTENSIONS` 和 `read_text_with_diagnostics()`

### 添加新的评分维度
编辑 `standards/resume_screening_standard.md` 和 `prompts/screening_system_prompt.md`

### 添加知识库文档
在 `data/knowledge/` 下添加 `.md` 或 `.txt` 文件即可被 RAG 自动加载

### 替换检索策略
当前 `rag.py` 使用关键词匹配（Jaccard），如需语义检索可替换为 Embedding + 向量数据库
