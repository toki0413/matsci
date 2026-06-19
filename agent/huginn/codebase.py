"""Codebase semantic search index.

Indexes text files in a workspace and exposes semantic / hybrid search.
Uses the same embedding model as the knowledge base but a separate
ChromaDB collection so code and documents don't contaminate each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    ".huginn_kb",
    ".chroma",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".kimi",
    ".kimi-code",
    ".claude",
    ".cursor",
}

CODE_EXTENSIONS = {
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".f90",
    ".f",
    ".jl",
    ".go",
    ".java",
    ".kt",
}


def _chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _should_index(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    if path.suffix.lower() not in CODE_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return False
    except Exception:
        return False
    try:
        data = path.read_bytes()
        if b"\x00" in data:
            return False
    except Exception:
        return False
    return True


class CodebaseIndex:
    """Semantic index of a workspace codebase."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self._model: Any | None = None

        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError("Codebase search requires chromadb") from e

        self.client = chromadb.PersistentClient(
            path=str(self.root / ".huginn_kb" / "chroma")
        )
        self.collection = self.client.get_or_create_collection("huginn_codebase")

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "Codebase search requires sentence-transformers"
                ) from e
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def index_workspace(self) -> dict[str, Any]:
        """Re-index the entire workspace."""
        # Clear existing code index
        all_ids = self.collection.get(include=[]).get("ids") or []
        if all_ids:
            self.collection.delete(ids=all_ids)

        indexed = 0
        chunks_total = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if not _should_index(path):
                continue
            try:
                rel = path.relative_to(self.root)
                text = path.read_text(encoding="utf-8", errors="ignore")
                if not text.strip():
                    continue
                chunks = _chunk_text(text)
                if not chunks:
                    continue
                embeddings = self.model.encode(chunks).tolist()
                ids = [f"{rel.as_posix()}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "path": rel.as_posix(),
                        "chunk": i,
                        "language": path.suffix.lstrip(".").lower() or "txt",
                    }
                    for i in range(len(chunks))
                ]
                self.collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                indexed += 1
                chunks_total += len(chunks)
            except Exception:
                continue

        return {
            "indexed_files": indexed,
            "chunks": chunks_total,
            "root": str(self.root),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search over indexed code."""
        if not query.strip():
            return []
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            meta = results["metadatas"][0][i]
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "path": meta.get("path"),
                    "language": meta.get("language"),
                    "chunk": meta.get("chunk", 0),
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return chunks

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "indexed_files": len(
                {
                    m.get("path")
                    for m in (
                        self.collection.get(include=["metadatas"]).get("metadatas")
                        or []
                    )
                }
            ),
            "chunks": self.collection.count(),
            "root": str(self.root),
        }


_codebase_index: CodebaseIndex | None = None


def get_codebase_index(workspace: str = ".") -> CodebaseIndex:
    """Get or create the singleton codebase index for a workspace."""
    global _codebase_index
    if _codebase_index is None:
        _codebase_index = CodebaseIndex(Path(workspace).resolve())
    return _codebase_index
