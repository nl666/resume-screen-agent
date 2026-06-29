from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import ChatModel
from .rag import Chunk, DEFAULT_KNOWLEDGE_DIR, answer_with_context, chunk_text, load_documents


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
DEFAULT_CHROMA_COLLECTION = "resume_screen_knowledge"
DEFAULT_BGE_MODEL = "BAAI/bge-small-zh-v1.5"


@dataclass(frozen=True)
class ChromaQueryResult:
    chunks: list[Chunk]
    sources: list[dict[str, Any]]
    chunk_count: int
    persist_dir: Path
    model_name: str
    collection_name: str


def build_chroma_index(
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    persist_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    model_name: str = DEFAULT_BGE_MODEL,
    reset: bool = True,
) -> dict[str, Any]:
    """Build a local Chroma index with BGE embeddings."""
    chromadb, SentenceTransformer = _load_optional_dependencies()
    knowledge_path = Path(knowledge_dir)
    output_dir = Path(persist_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(knowledge_path)
    chunks = chunk_text(docs)
    embedder = _load_embedder(SentenceTransformer, model_name, local_files_only=False)

    client = chromadb.PersistentClient(path=str(output_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        documents = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {"source_file": chunk.source_file, "chunk_id": chunk.chunk_id}
            for chunk in chunks
        ]
        embeddings = _encode_texts(
            embedder,
            [f"{chunk.source_file}\n{chunk.text}" for chunk in chunks],
        )
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return {
        "vector_store": "chroma",
        "persist_dir": str(output_dir),
        "collection": collection_name,
        "embedding_model": model_name,
        "chunk_count": len(chunks),
        "knowledge_dir": str(knowledge_path),
    }


def query_chroma_knowledge_base(
    question: str,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    top_k: int = 5,
    use_llm: bool = False,
    model: ChatModel | None = None,
    persist_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    model_name: str = DEFAULT_BGE_MODEL,
    rebuild_index: bool = False,
) -> dict[str, Any]:
    """Query local Chroma using BGE embeddings and return cited chunks."""
    result = retrieve_chroma(
        question=question,
        knowledge_dir=knowledge_dir,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        rebuild_index=rebuild_index,
    )

    if use_llm:
        payload = answer_with_context(question, result.chunks, model=model)
    else:
        payload = {
            "answer": f"已通过 BGE + Chroma 本地向量库检索到 {len(result.sources)} 个相关知识片段。打开“生成完整回答”后，可基于这些片段生成自然语言答案。",
            "sources": result.sources,
            "confidence": "high" if result.sources else "low",
        }

    payload["retrieval"] = {
        "mode": "vector",
        "vector_store": "chroma",
        "embedding_model": result.model_name,
        "persist_dir": str(result.persist_dir),
        "collection": result.collection_name,
        "chunk_count": result.chunk_count,
    }
    return payload


def retrieve_chroma(
    question: str,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    top_k: int = 5,
    persist_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    model_name: str = DEFAULT_BGE_MODEL,
    rebuild_index: bool = False,
) -> ChromaQueryResult:
    chromadb, SentenceTransformer = _load_optional_dependencies()
    output_dir = Path(persist_dir)
    if rebuild_index or not output_dir.exists():
        build_chroma_index(
            knowledge_dir=knowledge_dir,
            persist_dir=output_dir,
            collection_name=collection_name,
            model_name=model_name,
            reset=True,
        )

    client = chromadb.PersistentClient(path=str(output_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        build_chroma_index(
            knowledge_dir=knowledge_dir,
            persist_dir=output_dir,
            collection_name=collection_name,
            model_name=model_name,
            reset=True,
        )
        collection = client.get_collection(collection_name)

    if collection.count() == 0:
        build_chroma_index(
            knowledge_dir=knowledge_dir,
            persist_dir=output_dir,
            collection_name=collection_name,
            model_name=model_name,
            reset=True,
        )
        collection = client.get_collection(collection_name)

    embedder = _load_embedder(SentenceTransformer, model_name, local_files_only=True)
    query_embedding = _encode_texts(embedder, [question])[0]
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, int(top_k)),
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    chunks: list[Chunk] = []
    sources: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        chunk = Chunk(
            text=str(document or ""),
            chunk_id=str(metadata.get("chunk_id", "")),
            source_file=str(metadata.get("source_file", "")),
        )
        chunks.append(chunk)
        sources.append(
            {
                "file": chunk.source_file,
                "chunk_id": chunk.chunk_id,
                "quote": chunk.text[:300],
                "distance": round(float(distance), 6),
            }
        )

    return ChromaQueryResult(
        chunks=chunks,
        sources=sources,
        chunk_count=collection.count(),
        persist_dir=output_dir,
        model_name=model_name,
        collection_name=collection_name,
    )


def _encode_texts(embedder: Any, texts: list[str]) -> list[list[float]]:
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vector) for vector in embeddings]


def _load_embedder(SentenceTransformer: Any, model_name: str, local_files_only: bool) -> Any:
    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except TypeError:
        return SentenceTransformer(model_name)


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE + Chroma requires optional dependencies. "
            "Run: pip install -r requirements-bge-chroma.txt"
        ) from exc
    return chromadb, SentenceTransformer
