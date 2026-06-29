from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extract import read_text, SUPPORTED_EXTENSIONS
from .llm import ChatModel
from .schema import extract_json_object

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
DEFAULT_RAG_SYSTEM_PROMPT = ROOT / "prompts" / "rag_system_prompt.md"
DEFAULT_VECTOR_INDEX_PATH = ROOT / "data" / "vector_index" / "knowledge_index.json"

VECTOR_INDEX_VERSION = 1
EMBEDDING_MODEL_NAME = "local-hashing-embedding-v1"
EMBEDDING_DIMENSIONS = 512

_VALID_CONFIDENCE = {"high", "medium", "low"}
_VALID_RETRIEVAL_MODES = {"keyword", "vector", "hybrid"}

SparseVector = dict[int, float]


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_id: str
    source_file: str


@dataclass(frozen=True)
class VectorRecord:
    chunk: Chunk
    vector: SparseVector


def load_documents(knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR) -> list[dict[str, str]]:
    """Read all supported documents from the knowledge directory.

    Supports .txt, .md, .csv, .jsonl, .docx, .pdf. Returns a list of
    ``{"text": ..., "source": "filename"}`` dicts.
    """
    docs: list[dict[str, str]] = []
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.is_dir():
        return docs

    for file_path in sorted(knowledge_path.iterdir()):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS and suffix not in {".csv", ".jsonl"}:
            continue
        try:
            text = read_text(file_path)
            if text.strip():
                docs.append({"text": text, "source": file_path.name})
        except Exception:
            continue
    return docs


def chunk_text(
    docs: list[dict[str, str]],
    min_chars: int = 500,
    max_chars: int = 800,
) -> list[Chunk]:
    """Split documents into chunks of roughly *min_chars* to *max_chars*."""
    chunks: list[Chunk] = []

    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        paragraphs = re.split(r"\n\s*\n", text)

        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) <= max_chars:
                buffer += ("\n\n" if buffer else "") + para
            else:
                if buffer and len(buffer) >= min_chars:
                    chunks.append(
                        Chunk(text=buffer, chunk_id=f"{source}#{chunk_idx}", source_file=source)
                    )
                    chunk_idx += 1
                    buffer = para
                elif len(para) > max_chars:
                    if buffer:
                        chunks.append(
                            Chunk(text=buffer, chunk_id=f"{source}#{chunk_idx}", source_file=source)
                        )
                        chunk_idx += 1
                        buffer = ""
                    for sub in _split_long_paragraph(para, max_chars):
                        chunks.append(
                            Chunk(text=sub, chunk_id=f"{source}#{chunk_idx}", source_file=source)
                        )
                        chunk_idx += 1
                else:
                    buffer += ("\n\n" if buffer else "") + para

        if buffer:
            chunks.append(
                Chunk(text=buffer, chunk_id=f"{source}#{chunk_idx}", source_file=source)
            )

    return chunks


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Split a single long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    result: list[str] = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) <= max_chars:
            buffer += sent
        else:
            if buffer:
                result.append(buffer)
            buffer = sent
    if buffer:
        result.append(buffer)
    return result if result else [text]


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese with character bigrams and English with word tokens."""
    tokens: list[str] = []

    for m in re.finditer(r"[a-zA-Z0-9_]+", text.lower()):
        tokens.append(m.group())

    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in chinese_runs:
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])

    return [t for t in tokens if len(t) > 1]


def embed_text(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> SparseVector:
    """Create a deterministic local embedding vector.

    This is a dependency-free hashing embedding. It is not a learned semantic
    model, but it produces real numeric vectors that can be indexed and searched
    with cosine similarity. The function can later be replaced by OpenAI,
    DeepSeek-compatible embedding APIs, BGE, or another embedding backend.
    """
    counts = Counter(_tokenize(text))
    if not counts:
        return {}

    vector: dict[int, float] = {}
    for token, count in counts.items():
        hashed = _stable_hash(token)
        index = hashed % dimensions
        sign = 1.0 if ((hashed >> 63) & 1) == 0 else -1.0
        value = sign * (1.0 + math.log(count))
        vector[index] = vector.get(index, 0.0) + value

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items() if value}


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Return cosine similarity for two normalized sparse vectors."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def build_vector_records(chunks: list[Chunk]) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for chunk in chunks:
        embedded_text = f"{chunk.source_file}\n{chunk.text}"
        records.append(VectorRecord(chunk=chunk, vector=embed_text(embedded_text)))
    return records


def build_vector_index(
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    index_path: str | Path | None = DEFAULT_VECTOR_INDEX_PATH,
    min_chars: int = 500,
    max_chars: int = 800,
) -> dict[str, Any]:
    """Build and optionally persist a vector index for the knowledge base."""
    knowledge_path = Path(knowledge_dir)
    docs = load_documents(knowledge_path)
    chunks = chunk_text(docs, min_chars=min_chars, max_chars=max_chars)
    records = build_vector_records(chunks)
    payload = _vector_index_payload(knowledge_path, records)

    if index_path is not None:
        output_path = Path(index_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def load_vector_index(index_path: str | Path = DEFAULT_VECTOR_INDEX_PATH) -> dict[str, Any]:
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if data.get("version") != VECTOR_INDEX_VERSION:
        raise ValueError("vector index version mismatch")
    embedding = data.get("embedding", {})
    if embedding.get("model") != EMBEDDING_MODEL_NAME:
        raise ValueError("vector index embedding model mismatch")
    if int(embedding.get("dimensions", 0)) != EMBEDDING_DIMENSIONS:
        raise ValueError("vector index dimensions mismatch")
    return data


def load_or_build_vector_index(
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    index_path: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Load a fresh vector index or rebuild it when documents changed."""
    knowledge_path = Path(knowledge_dir)
    resolved_index_path = _resolve_index_path(knowledge_path, index_path)
    expected_signature = _knowledge_signature(knowledge_path)

    if not rebuild and resolved_index_path.exists():
        try:
            payload = load_vector_index(resolved_index_path)
            if payload.get("source_signature") == expected_signature:
                return payload
        except Exception:
            pass

    return build_vector_index(knowledge_path, resolved_index_path)


def records_from_index(payload: dict[str, Any]) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for item in payload.get("chunks", []):
        chunk = Chunk(
            text=str(item.get("text", "")),
            chunk_id=str(item.get("chunk_id", "")),
            source_file=str(item.get("source_file", "")),
        )
        vector = {int(index): float(value) for index, value in item.get("vector", {}).items()}
        records.append(VectorRecord(chunk=chunk, vector=vector))
    return records


def _vector_index_payload(knowledge_dir: Path, records: list[VectorRecord]) -> dict[str, Any]:
    return {
        "version": VECTOR_INDEX_VERSION,
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "type": "local_hashing_sparse_vector",
        },
        "knowledge_dir": str(knowledge_dir.resolve()),
        "source_signature": _knowledge_signature(knowledge_dir),
        "chunks": [
            {
                "source_file": record.chunk.source_file,
                "chunk_id": record.chunk.chunk_id,
                "text": record.chunk.text,
                "vector": {str(index): round(value, 10) for index, value in record.vector.items()},
            }
            for record in records
        ],
    }


def _knowledge_signature(knowledge_dir: Path) -> list[dict[str, Any]]:
    if not knowledge_dir.is_dir():
        return []

    signature: list[dict[str, Any]] = []
    for file_path in sorted(knowledge_dir.iterdir()):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS and suffix not in {".csv", ".jsonl"}:
            continue
        stat = file_path.stat()
        signature.append(
            {
                "source": file_path.name,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return signature


def _resolve_index_path(knowledge_dir: Path, index_path: str | Path | None) -> Path:
    if index_path is not None:
        return Path(index_path)

    try:
        if knowledge_dir.resolve() == DEFAULT_KNOWLEDGE_DIR.resolve():
            return DEFAULT_VECTOR_INDEX_PATH
    except FileNotFoundError:
        pass

    return knowledge_dir / ".vector_index.json"


def retrieve_keyword(question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    """Retrieve chunks by lexical query-term coverage."""
    return [chunk for chunk, _ in _score_keyword(question, chunks)[:top_k]]


def retrieve_vector(question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    """Retrieve chunks by vector cosine similarity."""
    records = build_vector_records(chunks)
    return [chunk for chunk, _ in _score_vector_records(question, records)[:top_k]]


def retrieve_from_index(
    question: str,
    payload: dict[str, Any],
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[Chunk]:
    """Retrieve from a persisted vector index."""
    records = records_from_index(payload)
    return _retrieve_from_records(question, records, top_k=top_k, mode=mode)


def retrieve(
    question: str,
    chunks: list[Chunk],
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[Chunk]:
    """Retrieve *top_k* relevant chunks.

    Modes:
    - ``vector``: cosine similarity over local embedding vectors
    - ``keyword``: lexical coverage scoring
    - ``hybrid``: vector similarity plus keyword scoring
    """
    mode = _normalize_retrieval_mode(mode)
    records = build_vector_records(chunks)
    return _retrieve_from_records(question, records, top_k=top_k, mode=mode)


def _retrieve_from_records(
    question: str,
    records: list[VectorRecord],
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[Chunk]:
    if not records:
        return []

    mode = _normalize_retrieval_mode(mode)
    if mode == "keyword":
        chunks = [record.chunk for record in records]
        return retrieve_keyword(question, chunks, top_k=top_k)

    vector_scores = _score_vector_records(question, records)
    if mode == "vector":
        return [chunk for chunk, _ in vector_scores[:top_k]]

    keyword_scores = dict(_score_keyword(question, [record.chunk for record in records]))
    max_vector = max((score for _, score in vector_scores), default=0.0) or 1.0
    max_keyword = max(keyword_scores.values(), default=0.0) or 1.0

    combined: list[tuple[Chunk, float]] = []
    for chunk, vector_score in vector_scores:
        lexical_score = keyword_scores.get(chunk, 0.0)
        score = (0.65 * (vector_score / max_vector)) + (0.35 * (lexical_score / max_keyword))
        combined.append((chunk, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in combined[:top_k]]


def _score_vector_records(question: str, records: list[VectorRecord]) -> list[tuple[Chunk, float]]:
    query_vector = embed_text(question)
    if not query_vector:
        return [(record.chunk, 0.0) for record in records]

    scored = [
        (record.chunk, cosine_similarity(query_vector, record.vector))
        for record in records
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _score_keyword(question: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []

    q_terms = set(_tokenize(question))
    if not q_terms:
        return [(chunk, 0.0) for chunk in chunks]

    scored: list[tuple[Chunk, float]] = []
    for chunk in chunks:
        c_terms = set(_tokenize(chunk.text))
        if not c_terms:
            scored.append((chunk, 0.0))
            continue
        intersection = len(q_terms & c_terms)
        query_coverage = intersection / len(q_terms)
        chunk_density = intersection / len(c_terms)
        source_terms = set(_tokenize(chunk.source_file))
        source_boost = 0.15 * len(q_terms & source_terms)
        exact_boost = 0.2 if question.lower() in chunk.text.lower() else 0.0
        score = query_coverage + (0.2 * chunk_density) + source_boost + exact_boost
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _normalize_retrieval_mode(mode: str) -> str:
    normalized = str(mode or "hybrid").strip().lower()
    if normalized not in _VALID_RETRIEVAL_MODES:
        return "hybrid"
    return normalized


def query_knowledge_base(
    question: str,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    top_k: int = 5,
    use_llm: bool = False,
    model: ChatModel | None = None,
    retrieval_mode: str = "hybrid",
    index_path: str | Path | None = None,
    rebuild_index: bool = False,
) -> dict[str, Any]:
    """Query the knowledge base with vector/hybrid retrieval and citations."""
    mode = _normalize_retrieval_mode(retrieval_mode)

    if mode == "keyword":
        docs = load_documents(knowledge_dir)
        chunks = chunk_text(docs)
        selected = retrieve_keyword(question, chunks, top_k=top_k)
        resolved_index_path: Path | None = None
        chunk_count = len(chunks)
    else:
        index_payload = load_or_build_vector_index(
            knowledge_dir=knowledge_dir,
            index_path=index_path,
            rebuild=rebuild_index,
        )
        selected = retrieve_from_index(question, index_payload, top_k=top_k, mode=mode)
        resolved_index_path = _resolve_index_path(Path(knowledge_dir), index_path)
        chunk_count = len(index_payload.get("chunks", []))

    metadata = {
        "mode": mode,
        "embedding_model": EMBEDDING_MODEL_NAME if mode != "keyword" else None,
        "embedding_dimensions": EMBEDDING_DIMENSIONS if mode != "keyword" else None,
        "index_path": str(resolved_index_path) if resolved_index_path else None,
        "chunk_count": chunk_count,
    }

    if use_llm:
        result = answer_with_context(question, selected, model=model)
    else:
        result = {
            "answer": f"已通过{_retrieval_mode_label(mode)}检索到 {len(selected)} 个相关知识片段。打开“生成完整回答”后，可基于这些片段生成自然语言答案。",
            "sources": [
                {
                    "file": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                    "quote": chunk.text[:300],
                }
                for chunk in selected
            ],
            "confidence": "high" if selected else "low",
        }

    result["retrieval"] = metadata
    return result


def _retrieval_mode_label(mode: str) -> str:
    labels = {
        "keyword": "关键词",
        "vector": "向量",
        "hybrid": "向量+关键词混合",
    }
    return labels.get(mode, "向量+关键词混合")


def answer_with_context(
    question: str,
    chunks: list[Chunk],
    model: ChatModel | None = None,
    system_prompt_path: str | Path = DEFAULT_RAG_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build context from *chunks*, send to the LLM, and return a structured answer.

    Returns a dict with keys ``answer``, ``sources``, and ``confidence``.
    """
    chat_model = model or ChatModel()

    context_parts: list[str] = []
    for chunk in chunks:
        context_parts.append(f"[{chunk.chunk_id}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    user_prompt = f"""context:
{context}

question:
{question}

请返回以下 JSON 格式（不要包含其他内容）：
{{
  "answer": "基于知识库的回答",
  "sources": [
    {{"file": "来源文件名", "chunk_id": "chunk编号", "quote": "引用的原文片段"}}
  ],
  "confidence": "high / medium / low"
}}"""

    max_retries = 3
    data = None
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = chat_model.complete_json(system_prompt, user_prompt)
            data = extract_json_object(raw)
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                continue
    if data is None:
        raise RuntimeError(f"RAG model call failed after {max_retries} attempts: {last_error}") from last_error

    for key in ("answer", "sources", "confidence"):
        if key not in data:
            raise ValueError(f"RAG response missing required key: {key}")

    if data["confidence"] not in _VALID_CONFIDENCE:
        data["confidence"] = "medium"

    if not isinstance(data["sources"], list):
        data["sources"] = []

    return data
