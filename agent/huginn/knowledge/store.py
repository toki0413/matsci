"""Local RAG knowledge base with ChromaDB and sentence-transformers."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from huginn.utils.cache import TimedLRUCache

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"
SEED_DIR = Path(__file__).parent / "seed"


class _EmbeddingModel:
    """Wrapper that prefers ChromaDB's cached ONNX embedder, falling back to sentence-transformers.

    The underlying embedding models are loaded once and reused across
    ``KnowledgeBase`` instances to avoid repeated initialization overhead.
    Computed embeddings are also cached by content hash so identical documents
    do not need to be re-encoded.
    """

    _ef: Any | None = None
    _st: Any | None = None
    _use_chroma: bool = False
    _initialized: bool = False
    _embedding_cache: TimedLRUCache[np.ndarray] = TimedLRUCache(
        max_size=1024, ttl=3600.0
    )

    def __init__(self) -> None:
        if not _EmbeddingModel._initialized:
            try:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                ef = DefaultEmbeddingFunction()
                _ = ef(["test"])
                _EmbeddingModel._ef = ef
                _EmbeddingModel._use_chroma = True
            except Exception:
                _EmbeddingModel._use_chroma = False
            _EmbeddingModel._initialized = True

    def encode(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        if cache_key:
            cached = _EmbeddingModel._embedding_cache.get(cache_key)
            if cached is not None:
                return cached

        if _EmbeddingModel._use_chroma and _EmbeddingModel._ef is not None:
            vectors = _EmbeddingModel._ef(texts)
            result = np.asarray(vectors, dtype=np.float32)
        else:
            if _EmbeddingModel._st is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise RuntimeError(
                        "Embedding requires sentence-transformers or chromadb's default embedder. "
                        "Install: pip install sentence-transformers"
                    ) from e
                _EmbeddingModel._st = SentenceTransformer(EMBED_MODEL)
            result = _EmbeddingModel._st.encode(texts)

        if cache_key:
            _EmbeddingModel._embedding_cache.set(cache_key, result)
        return result


def _chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Simple sliding-window chunking by character."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from supported file types.

    Images and scanned PDFs fall back to OCR when normal extraction is empty.
    """
    from huginn.knowledge.ocr_loader import extract_text_with_ocr, is_image_file

    lower = filename.lower()
    if is_image_file(filename):
        return extract_text_with_ocr(filename, content)

    if lower.endswith(".pdf"):
        try:
            import fitz  # pymupdf
        except ImportError as e:
            raise RuntimeError(
                "PDF support requires pymupdf. Install: pip install pymupdf"
            ) from e
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text())
        text = "\n".join(parts)
        # Fall back to OCR for scanned/image-based PDFs.
        if not text.strip():
            ocr_text = extract_text_with_ocr(filename, content)
            if ocr_text.strip():
                return ocr_text
        return text

    if lower.endswith((".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml")):
        return content.decode("utf-8", errors="ignore")

    # Best-effort for anything else
    return content.decode("utf-8", errors="ignore")


class KnowledgeBase:
    """A local vector knowledge base backed by ChromaDB."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir(exist_ok=True)

        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "Knowledge base requires chromadb. Install: pip install chromadb"
            ) from e

        self.client = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.collection = self.client.get_or_create_collection("huginn_kb")
        self._model: Any | None = None
        self._query_cache: TimedLRUCache[list[dict[str, Any]]] = TimedLRUCache(
            max_size=256, ttl=60.0
        )
        # 语义缓存: gptcache 可选, 没装就退回上面的 TimedLRUCache
        self._semantic_cache: Any | None = None
        try:
            from gptcache import Cache
            from gptcache.embedding import Onnx
            from gptcache.manager.factory import manager_factory
            from gptcache.processor.pre import get_prompt
            from gptcache.similarity_evaluation import SearchDistanceEvaluation

            onnx = Onnx(model_name="all-MiniLM-L6-v2")
            data_manager = manager_factory(
                "sqlite,faiss",
                data_dir=str(self.root / "semantic_cache"),
                vector_params={"dimension": onnx.dimension},
            )
            self._semantic_cache = Cache()
            self._semantic_cache.init(
                pre_embedding_func=get_prompt,
                data_manager=data_manager,
                # max_distance=0.4 对应 cosine similarity 约 0.92
                # (L2 距离 d, sim = 1 - d^2/2, d=0.4 -> sim=0.92)
                similarity_evaluation=SearchDistanceEvaluation(max_distance=0.4),
                embedding_func=onnx.to_embeddings,
            )
        except Exception:
            # gptcache 没装或初始化失败 (模型下载/faiss 缺失等), 优雅降级
            self._semantic_cache = None

    @property
    def model(self) -> _EmbeddingModel:
        if self._model is None:
            self._model = _EmbeddingModel()
        return self._model

    def _flush_semantic_cache(self) -> None:
        """知识库变更时清掉语义缓存, 避免返回过期结果."""
        if self._semantic_cache is not None:
            try:
                self._semantic_cache.flush()
            except Exception:
                pass

    def add_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Ingest a document, chunk it, and store embeddings."""
        doc_id = uuid.uuid4().hex[:12]
        text = _extract_text(filename, content)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")

        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("Document is empty after chunking")

        chunk_hash = hashlib.sha256("".join(chunks).encode()).hexdigest()
        embeddings = self.model.encode(chunks, cache_key=chunk_hash).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "chunk": i}
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._query_cache.clear()
        self._flush_semantic_cache()

        safe_name = f"{doc_id}_{Path(filename).name}"
        doc_path = self.docs_dir / safe_name
        doc_path.write_bytes(content)

        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique documents stored in the collection."""
        data = self.collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            doc_id = meta.get("doc_id")
            if not doc_id or doc_id in docs:
                continue
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
            }
        return sorted(docs.values(), key=lambda d: d["filename"])

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the knowledge base."""
        data = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            self._query_cache.clear()
            self._flush_semantic_cache()
        for path in self.docs_dir.glob(f"{doc_id}_*"):
            path.unlink(missing_ok=True)
        return len(ids) > 0

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        if not text.strip():
            return []

        # 语义缓存优先: 相近的 query 直接复用历史结果
        if self._semantic_cache is not None:
            try:
                hit = self._semantic_cache.get(prompt=text.strip())
                if hit is not None:
                    return hit
            except Exception:
                pass  # 缓存查询出错不影响正常流程

        cache_key = (text.strip(), top_k)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self.model.encode([text]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        self._query_cache.set(cache_key, chunks)

        # 回写语义缓存, 下次相似 query 能命中
        if self._semantic_cache is not None:
            try:
                self._semantic_cache.put(prompt=text.strip(), data=chunks)
            except Exception:
                pass

        return chunks

    def count(self) -> int:
        return self.collection.count()


def seed_knowledge_base(kb: KnowledgeBase, force: bool = False) -> dict[str, Any]:
    """Ingest built-in seed reference documents into a knowledge base.

    Seeds are loaded from ``huginn/knowledge/seed/*.md``. Each seed is
    identified by ``seed:<sha256>`` so that unchanged files are skipped on
    subsequent runs. Passing ``force=True`` removes all existing seed entries
    and re-ingests them.
    """
    if not SEED_DIR.is_dir():
        return {"added": 0, "skipped": 0, "failed": 0}

    seed_files = sorted(SEED_DIR.glob("*.md"))
    existing_seed_ids = {
        doc["doc_id"]
        for doc in kb.list_documents()
        if doc["doc_id"].startswith("seed:")
    }

    if force:
        for doc_id in list(existing_seed_ids):
            kb.delete_document(doc_id)
        existing_seed_ids = set()

    added = skipped = failed = 0
    for path in seed_files:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        doc_id = f"seed:{digest}"
        if doc_id in existing_seed_ids:
            skipped += 1
            continue

        try:
            text = _extract_text(path.name, content)
            chunks = _chunk_text(text)
            if not chunks:
                skipped += 1
                continue
            embeddings = kb.model.encode(chunks, cache_key=digest).tolist()
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "doc_id": doc_id,
                    "filename": path.name,
                    "chunk": i,
                    "seed": True,
                }
                for i in range(len(chunks))
            ]
            kb.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            added += 1
        except Exception:
            failed += 1

    return {"added": added, "skipped": skipped, "failed": failed}


_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base(workspace: str = ".") -> KnowledgeBase:
    """Get or create the singleton knowledge base for a workspace.

    Built-in seed documents are automatically ingested the first time the
    knowledge base is created.
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(Path(workspace) / ".huginn_kb")
        seed_knowledge_base(_knowledge_base, force=False)
    return _knowledge_base
