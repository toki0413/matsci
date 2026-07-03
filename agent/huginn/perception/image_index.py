"""In-process image vector index for materials science images.

A tiny numpy-backed nearest-neighbour store: no external vector database,
just a list of (path, metadata, embedding) rows persisted to JSON. This is
deliberately simple — it is meant to give the agent a working visual memory
for a few hundred to a few thousand images without pulling in FAISS or
similar. If the index ever grows past that, swapping the ``_search``
implementation for an ANN library is a localized change.

The index degrades gracefully when no encoder backend is available: images
are still recorded with their path and metadata, but skipped during
similarity search (their embedding is stored as ``None``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from huginn.perception.visual_encoder import VisualEncoder, get_encoder


class ImageIndex:
    """Vector index for materials science images.

    Args:
        store_path: Where to persist the JSON index. If ``None`` the
            index is kept purely in-memory and never written to disk.
        encoder: Optional :class:`VisualEncoder`. When omitted the shared
            module-level encoder is used (built lazily on first use).
    """

    def __init__(
        self,
        store_path: str | Path | None = None,
        encoder: VisualEncoder | None = None,
    ) -> None:
        self.store_path = Path(store_path) if store_path else None
        self._encoder = encoder
        # Each row: {"path": str, "metadata": dict, "vector": list[float]|None,
        #            "added_at": float}
        self._entries: list[dict[str, Any]] = []

        if self.store_path is not None:
            self._load()

    # ── encoder access ──

    @property
    def encoder(self) -> VisualEncoder | None:
        """The encoder backing this index, built lazily."""
        if self._encoder is None:
            self._encoder = get_encoder()
        return self._encoder

    def set_encoder(self, encoder: VisualEncoder | None) -> None:
        """Override the encoder (handy for tests with a fake encoder)."""
        self._encoder = encoder

    # ── mutation ──

    def add_image(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode ``path`` and append it to the index.

        Returns the stored record. If the encoder is unavailable the record
        is still added (with ``vector=None``) so the path/metadata are not
        lost — it just won't be returned by :meth:`search`.
        """
        path = str(path)
        vec = self.encoder.encode_image(path) if self.encoder is not None else None

        record: dict[str, Any] = {
            "path": path,
            "metadata": dict(metadata or {}),
            "vector": vec.tolist() if vec is not None else None,
            "added_at": time.time(),
        }
        self._entries.append(record)
        return record

    def add_image_bytes(
        self,
        image_bytes: bytes,
        metadata: dict[str, Any] | None = None,
        path_label: str | None = None,
    ) -> dict[str, Any]:
        """Same as :meth:`add_image` but for in-memory bytes (e.g. uploads).

        ``path_label`` is stored as the record's ``path`` purely for
        bookkeeping — it does not need to be a real filesystem location.
        """
        vec = None
        if self.encoder is not None:
            vec = self.encoder.encode_image(image_bytes)

        record: dict[str, Any] = {
            "path": path_label or f"<bytes:{len(image_bytes)}>",
            "metadata": dict(metadata or {}),
            "vector": vec.tolist() if vec is not None else None,
            "added_at": time.time(),
        }
        self._entries.append(record)
        return record

    # ── search ──

    def search(
        self,
        query: str | Path | bytes,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar indexed images to ``query``.

        ``query`` is encoded with the index's encoder (path or bytes).
        Results are sorted by descending cosine similarity and each carry a
        ``similarity`` field in ``[0, 1]`` (vectors are L2-normalized, so
        cosine == dot product, clamped to be non-negative for display).
        """
        enc = self.encoder
        if enc is None or not enc.available:
            return []

        query_vec = enc.encode_image(query)
        if query_vec is None:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._entries:
            vec_list = entry.get("vector")
            if not vec_list:
                continue
            vec = np.asarray(vec_list, dtype=np.float32)
            sim = enc.similarity(query_vec, vec)
            scored.append((sim, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for sim, entry in scored[:top_k]:
            results.append({
                "path": entry["path"],
                "metadata": entry.get("metadata", {}),
                "similarity": round(float(max(sim, 0.0)), 6),
                "added_at": entry.get("added_at"),
            })
        return results

    # ── introspection ──

    def stats(self) -> dict[str, Any]:
        """Return a small summary used by the /visual/index/stats endpoint."""
        enc = self.encoder
        total = len(self._entries)
        indexed = sum(1 for e in self._entries if e.get("vector") is not None)
        return {
            "total_images": total,
            "indexed_with_vectors": indexed,
            "encoder_available": bool(enc and enc.available),
            "encoder_backend": enc.backend_name if enc else None,
            "embedding_dim": enc.dim if enc else 0,
            "store_path": str(self.store_path) if self.store_path else None,
        }

    def __len__(self) -> int:
        return len(self._entries)

    # ── persistence ──

    def save(self) -> None:
        """Write the index to ``store_path`` as JSON. No-op if unset."""
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": self._entries, "version": 1}, fh)
        os.replace(tmp, self.store_path)

    def _load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = list(data.get("entries", []))
        except (OSError, json.JSONDecodeError):
            # Corrupt or unreadable store — start fresh rather than crash.
            self._entries = []

    def clear(self) -> None:
        """Drop every entry (does not touch the on-disk file until save())."""
        self._entries = []
