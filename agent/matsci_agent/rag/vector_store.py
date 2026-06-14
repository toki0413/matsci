"""Local vector store using ChromaDB for material science RAG.

Supports document ingestion, embedding, semantic search, and optional
encryption at rest. Uses keyword fallback if embedding model is not cached.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _embedding_model_cached() -> bool:
    """Check if ChromaDB's default ONNX model is already downloaded."""
    cache_path = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
    return cache_path.exists() and any(cache_path.iterdir())


class VectorStore:
    """Local vector store for material science documents."""

    DEFAULT_COLLECTION = "matsci_knowledge"

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.persist_dir = persist_dir or self._default_persist_dir()
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._embedding_available = False
        self._checked = False

    def _default_persist_dir(self) -> str:
        base = Path.home() / ".matsci" / "rag"
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _check_embedding(self):
        """Lazy check for embedding availability. Only runs once."""
        if self._checked:
            return self._embedding_available
        self._checked = True

        if not _embedding_model_cached():
            self._embedding_available = False
            return False

        try:
            import importlib
            ef_module = importlib.import_module("chromadb.utils.embedding_functions")
            DefaultEF = getattr(ef_module, "DefaultEmbeddingFunction")
            self._embedding_fn = DefaultEF()
            _ = self._embedding_fn(["test"])
            self._embedding_available = True
        except Exception:
            self._embedding_available = False

        return self._embedding_available

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            ef = self._get_embedding_fn()
            if ef:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collection

    def _get_embedding_fn(self):
        if self._embedding_fn is not None:
            return self._embedding_fn
        if self._check_embedding():
            return self._embedding_fn
        return None

    def _compute_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        if not self._check_embedding():
            return None
        ef = self._get_embedding_fn()
        if ef is None:
            return None
        try:
            return ef(texts)
        except Exception:
            return None

    def _keyword_search(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        query_words = set(query.lower().split())
        scores = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def ingest(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        if not documents:
            return []

        if ids is None:
            ids = [
                hashlib.sha256(f"{doc}:{i}".encode()).hexdigest()[:16]
                for i, doc in enumerate(documents)
            ]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        for meta in metadatas:
            meta.setdefault("source", "unknown")
            meta.setdefault(
                "ingested_at",
                str(__import__("datetime").datetime.now().isoformat()),
            )

        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self._compute_embeddings(documents)

        collection = self._get_collection()

        kwargs: dict[str, Any] = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
        }
        if embeddings:
            kwargs["embeddings"] = embeddings

        collection.add(**kwargs)
        return ids

    def _rust_top_k(
        self,
        query_embedding: list[float],
        embeddings: list[list[float]],
        top_k: int,
    ) -> list[tuple[int, float]] | None:
        try:
            from matsci_ext import top_k  # type: ignore[import-not-found]
        except Exception:
            return None
        if not embeddings:
            return None
        try:
            return top_k(query_embedding, embeddings, top_k)
        except Exception:
            return None

    def _matches_filter(self, metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        for key, value in filter_dict.items():
            if metadata.get(key) != value:
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self._get_collection()
        query_embedding = self._compute_embeddings([query])

        if query_embedding:
            # Try Rust-accelerated exact top-k over stored embeddings.
            try:
                all_data = collection.get(include=["documents", "metadatas", "embeddings"])
                docs = all_data.get("documents") or []
                metas = all_data.get("metadatas") or [{}] * len(docs)
                embs = all_data.get("embeddings") or []

                indices = list(range(len(docs)))
                if filter_dict:
                    indices = [i for i in indices if self._matches_filter(metas[i], filter_dict)]
                    docs = [docs[i] for i in indices]
                    metas = [metas[i] for i in indices]
                    embs = [embs[i] for i in indices]

                if embs:
                    ranked = self._rust_top_k(query_embedding[0], embs, top_k)
                    if ranked is not None:
                        output = []
                        for idx, score in ranked:
                            output.append({
                                "id": all_data["ids"][indices[idx]],
                                "document": docs[idx],
                                "metadata": metas[idx],
                                "distance": 1.0 - score,
                            })
                        return output
            except Exception:
                pass

            # Fallback to ChromaDB's native ANN query.
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
            return output

        else:
            all_docs = collection.get(include=["documents", "metadatas"])
            docs = all_docs.get("documents", [])
            if not docs:
                return []

            scored = self._keyword_search(query, docs, top_k)
            output = []
            for idx, score in scored:
                output.append({
                    "id": all_docs["ids"][idx],
                    "document": docs[idx],
                    "metadata": all_docs["metadatas"][idx]
                    if all_docs.get("metadatas")
                    else {},
                    "distance": 1.0 - score,
                })
            return output

    def delete(self, ids: list[str]) -> None:
        collection = self._get_collection()
        collection.delete(ids=ids)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        collection = self._get_collection()
        results = collection.get(include=["metadatas"], limit=limit)
        docs = []
        for i in range(len(results["ids"])):
            docs.append({
                "id": results["ids"][i],
                "metadata": results["metadatas"][i],
            })
        return docs

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a single document by ID."""
        collection = self._get_collection()
        result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def update_document(
        self,
        doc_id: str,
        document: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a document's content and/or metadata."""
        collection = self._get_collection()
        kwargs: dict[str, Any] = {"ids": [doc_id]}
        if document is not None:
            kwargs["documents"] = [document]
        if metadata is not None:
            kwargs["metadatas"] = [metadata]
        collection.update(**kwargs)

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._parse_file(path)
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        metadatas = [
            {
                "source": str(path),
                "filename": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        return self.ingest(chunks, metadatas=metadatas)

    def _parse_file(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                import PyPDF2
                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                return text
            except ImportError:
                raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")

        elif suffix in {".json"}:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif suffix in {".csv"}:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            return "\n".join(", ".join(row) for row in rows[:500])

        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                nl_pos = text.rfind("\n", start, end)
                if nl_pos > start + chunk_size // 2:
                    end = nl_pos + 1

            chunks.append(text[start:end].strip())
            if end >= text_len:
                break
            start = end - chunk_overlap
            if start <= 0:
                start = end

        return [c for c in chunks if c]


class EncryptedVectorStore:
    """Encrypted vector store wrapper.

    Documents and selected metadata fields are encrypted at rest.
    Embedding vectors remain in plaintext for similarity search.
    """

    ENCRYPTED_MARKER = "__ENC__"
    SENSITIVE_META_FIELDS = {"author", "email", "api_key", "password", "secret"}

    def __init__(
        self,
        vault: Any,  # CryptoVault or compatible
        persist_dir: str | None = None,
        collection_name: str | None = None,
        encrypt_documents: bool = True,
        encrypt_metadata: bool = True,
    ):
        self.vault = vault
        self.encrypt_documents = encrypt_documents
        self.encrypt_metadata = encrypt_metadata
        # Use a separate collection for encrypted data to avoid mixing
        coll = collection_name or VectorStore.DEFAULT_COLLECTION
        self._store = VectorStore(
            persist_dir=persist_dir,
            collection_name=f"{coll}_encrypted",
        )

    def _is_encrypted(self, text: str) -> bool:
        return isinstance(text, str) and text.startswith(self.ENCRYPTED_MARKER)

    def _encrypt(self, text: str) -> str:
        if not self.vault.is_unlocked():
            return text
        try:
            cipher = self.vault.encrypt(text)
            return f"{self.ENCRYPTED_MARKER}{cipher.decode('utf-8')}"
        except Exception:
            return text

    def _decrypt(self, text: str) -> str:
        if not self._is_encrypted(text):
            return text
        if not self.vault.is_unlocked():
            return text
        try:
            cipher = text[len(self.ENCRYPTED_MARKER) :].encode("utf-8")
            return self.vault.decrypt(cipher)
        except Exception:
            return text

    def _encrypt_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        if not self.encrypt_metadata or not self.vault.is_unlocked():
            return meta
        encrypted = {}
        for k, v in meta.items():
            if k in self.SENSITIVE_META_FIELDS and isinstance(v, str):
                encrypted[k] = self._encrypt(v)
            else:
                encrypted[k] = v
        return encrypted

    def _decrypt_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        if not self.encrypt_metadata or not self.vault.is_unlocked():
            return meta
        decrypted = {}
        for k, v in meta.items():
            if k in self.SENSITIVE_META_FIELDS and isinstance(v, str):
                decrypted[k] = self._decrypt(v)
            else:
                decrypted[k] = v
        return decrypted

    def ingest(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if not documents:
            return []

        if ids is None:
            ids = [hashlib.sha256(doc.encode()).hexdigest()[:16] for doc in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Compute embeddings on plaintext BEFORE encryption
        embeddings = self._store._compute_embeddings(documents)

        # Encrypt documents and sensitive metadata
        enc_docs = [self._encrypt(d) if self.encrypt_documents else d for d in documents]
        enc_metas = [self._encrypt_metadata(m) for m in metadatas]

        return self._store.ingest(
            documents=enc_docs,
            metadatas=enc_metas,
            ids=ids,
            embeddings=embeddings,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = self._store.search(query, top_k=top_k, filter_dict=filter_dict)

        # Decrypt returned documents and metadata
        for r in results:
            if self.encrypt_documents:
                r["document"] = self._decrypt(r["document"])
            if self.encrypt_metadata:
                r["metadata"] = self._decrypt_metadata(r["metadata"])
        return results

    def delete(self, ids: list[str]) -> None:
        self._store.delete(ids=ids)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        docs = self._store.list_documents(limit=limit)
        for d in docs:
            if self.encrypt_metadata:
                d["metadata"] = self._decrypt_metadata(d["metadata"])
        return docs

    def count(self) -> int:
        return self._store.count()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        doc = self._store.get_document(doc_id)
        if doc is None:
            return None
        if self.encrypt_documents:
            doc["document"] = self._decrypt(doc["document"])
        if self.encrypt_metadata:
            doc["metadata"] = self._decrypt_metadata(doc["metadata"])
        return doc

    def update_document(
        self,
        doc_id: str,
        document: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if document is not None and self.encrypt_documents:
            document = self._encrypt(document)
        if metadata is not None and self.encrypt_metadata:
            metadata = self._encrypt_metadata(metadata)
        self._store.update_document(doc_id, document=document, metadata=metadata)

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._store._parse_file(path)
        chunks = self._store._chunk_text(content, chunk_size, chunk_overlap)

        metadatas = [
            {
                "source": str(path),
                "filename": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        return self.ingest(chunks, metadatas=metadatas)
