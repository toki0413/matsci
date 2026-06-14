"""Local RAG knowledge base with ChromaDB and sentence-transformers."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
    """Extract plain text from supported file types."""
    lower = filename.lower()
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
        return "\n".join(parts)

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

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "Embedding requires sentence-transformers. "
                    "Install: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def add_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Ingest a document, chunk it, and store embeddings."""
        doc_id = uuid.uuid4().hex[:12]
        text = _extract_text(filename, content)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")

        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("Document is empty after chunking")

        embeddings = self.model.encode(chunks).tolist()
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
        for path in self.docs_dir.glob(f"{doc_id}_*"):
            path.unlink(missing_ok=True)
        return len(ids) > 0

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        if not text.strip():
            return []
        embedding = self.model.encode([text]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            chunks.append({
                "chunk_id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def count(self) -> int:
        return self.collection.count()


_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base(workspace: str = ".") -> KnowledgeBase:
    """Get or create the singleton knowledge base for a workspace."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(Path(workspace) / ".huginn_kb")
    return _knowledge_base
