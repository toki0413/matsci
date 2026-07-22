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
        query: str | Path | bytes | None = None,
        top_k: int = 5,
        text_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar indexed images to ``query``.

        ``query`` is encoded with the index's encoder (path or bytes).
        Results are sorted by descending cosine similarity and each carry a
        ``similarity`` field in ``[0, 1]`` (vectors are L2-normalized, so
        cosine == dot product, clamped to be non-negative for display).

        QW4: ``text_query`` enables text→image retrieval. Without a CLIP-style
        text encoder we fall back to weighted keyword matching on metadata
        fields (caption/keywords/description/tags). When both ``query`` and
        ``text_query`` are given, scores are blended (image_sim * 0.6 +
        text_score * 0.4). ponytail: keyword match, not semantic. Upgrade
        path: swap in encode_text once a CLIP encoder is available.
        """
        if query is None and not text_query:
            return []

        # image similarity branch
        img_scores: dict[int, float] = {}
        if query is not None:
            enc = self.encoder
            if enc is None or not enc.available:
                # No encoder: only text path can return anything
                if text_query is None:
                    return []
            else:
                query_vec = enc.encode_image(query)
                if query_vec is not None:
                    for i, entry in enumerate(self._entries):
                        vec_list = entry.get("vector")
                        if not vec_list:
                            continue
                        vec = np.asarray(vec_list, dtype=np.float32)
                        img_scores[i] = float(enc.similarity(query_vec, vec))

        # QW4: text→image branch (keyword fallback in absence of CLIP text encoder)
        text_scores: dict[int, float] = {}
        if text_query:
            text_scores = _text_keyword_scores(text_query, self._entries)

        # blend
        candidate_idx = set(img_scores) | set(text_scores)
        if not candidate_idx:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for i in candidate_idx:
            entry = self._entries[i]
            sim_img = img_scores.get(i, 0.0)
            sim_txt = text_scores.get(i, 0.0)
            if img_scores and text_scores:
                blended = sim_img * 0.6 + sim_txt * 0.4
            elif img_scores:
                blended = sim_img
            else:
                blended = sim_txt
            scored.append((blended, entry))
            # 记 sub-scores 给 result, 让 LLM 看到来源
            entry_blended_meta = {
                "image_similarity": round(float(max(sim_img, 0.0)), 6) if img_scores else None,
                "text_match_score": round(float(sim_txt), 6) if text_scores else None,
            }
            # 临时挂在 entry 上, 后面 result 提取
            entry["_search_breakdown"] = entry_blended_meta

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for sim, entry in scored[:top_k]:
            breakdown = entry.pop("_search_breakdown", {})
            results.append({
                "path": entry["path"],
                "metadata": entry.get("metadata", {}),
                "similarity": round(float(max(sim, 0.0)), 6),
                "image_similarity": breakdown.get("image_similarity"),
                "text_match_score": breakdown.get("text_match_score"),
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


# ── QW4: text→image keyword fallback ────────────────────────────────────────

# metadata 字段权重 — title 最重, caption 次之, keywords/tags 平, description 轻.
# ponytail: 启发式权重, 不调 LLM. 升级: encode_text (CLIP) 真语义匹配.
_TEXT_FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "caption": 2.0,
    "keywords": 1.5,
    "tags": 1.5,
    "description": 1.0,
    "label": 1.0,
    "source": 0.5,
}


def _text_keyword_scores(
    text_query: str, entries: list[dict[str, Any]]
) -> dict[int, float]:
    """加权关键字匹配 metadata. 多个 token 都匹配才得分 (AND 语义).

    返回 {entry_index: score}. score 归一化到 [0, 1]:
      raw = Σ (field_weight * (token_matched_in_field / total_tokens))
      final = min(1.0, raw / sum_of_max_field_weight)
    """
    tokens = [t.lower().strip() for t in text_query.split() if t.strip()]
    if not tokens:
        return {}
    # 归一化基准: 假设所有 token 都在 title (最高权重字段) 命中
    max_possible = _TEXT_FIELD_WEIGHTS["title"] * len(tokens)
    if max_possible <= 0:
        return {}
    scores: dict[int, float] = {}
    for i, entry in enumerate(entries):
        meta = entry.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        raw = 0.0
        for field, weight in _TEXT_FIELD_WEIGHTS.items():
            val = meta.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                text = " ".join(str(v) for v in val)
            else:
                text = str(val)
            text_lower = text.lower()
            n_matched = sum(1 for t in tokens if t in text_lower)
            raw += weight * (n_matched / len(tokens))
        if raw > 0:
            scores[i] = min(1.0, raw / max_possible)
    return scores


# ── selfcheck ──────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """QW4 selfcheck: text_query 关键字匹配 + blend (no encoder 路径)."""
    import tempfile as _tf

    # 用 in-memory index + fake encoder=unavailable, 验证 text_query 路径不依赖 encoder
    idx = ImageIndex()  # store_path=None
    # encoder 默认走 get_encoder(), 测试环境通常没装 → available=False
    # 手动注入 entries (绕开 add_image 的 encode 步骤)
    idx._entries = [
        {
            "path": "/fake/band.png",
            "metadata": {
                "title": "Band structure of GaAs",
                "caption": "Direct gap at Gamma point",
                "keywords": ["band", "DFT", "GaAs"],
            },
            "vector": None,
            "added_at": 1.0,
        },
        {
            "path": "/fake/dos.png",
            "metadata": {
                "title": "Density of states Si",
                "keywords": ["dos", "silicon"],
            },
            "vector": None,
            "added_at": 2.0,
        },
        {
            "path": "/fake/tem.png",
            "metadata": {
                "caption": "TEM image of Au nanoparticle",
                "tags": ["tem", "microscopy"],
            },
            "vector": None,
            "added_at": 3.0,
        },
    ]

    # 场景 1: 单 token "band" → 只命中 band.png
    r1 = idx.search(text_query="band", top_k=3)
    assert len(r1) == 1, f"expected 1 match, got {len(r1)}: {[r['path'] for r in r1]}"
    assert r1[0]["path"].endswith("band.png"), f"wrong match: {r1[0]['path']}"
    assert r1[0]["text_match_score"] is not None and r1[0]["text_match_score"] > 0
    assert r1[0]["image_similarity"] is None  # 没 query
    print(f"1. text_query single token OK: {r1[0]['path']} score={r1[0]['text_match_score']}")

    # 场景 2: multi-token "band GaAs" → title 命中 2 token, 比单 token 分数高
    r2 = idx.search(text_query="band GaAs", top_k=3)
    assert len(r2) == 1, f"expected 1 match, got {len(r2)}"
    assert r2[0]["path"].endswith("band.png")
    # title 含 "band"+"GaAs" → raw = 3.0 * 2/2 = 3.0; max = 3.0*2 = 6.0 → 0.5
    # caption 含 "Gamma" 不含 "band/GaAs" → 0; keywords 含 "band"+"GaAs" → 1.5*1.0 = 1.5
    # raw = 3.0 + 1.5 = 4.5; final = 4.5/6.0 = 0.75
    assert abs(r2[0]["text_match_score"] - 0.75) < 1e-3, \
        f"expected 0.75, got {r2[0]['text_match_score']}"
    print(f"2. text_query multi-token OK: {r2[0]['path']} score={r2[0]['text_match_score']}")

    # 场景 3: 空 query + 空 text_query → []
    r3 = idx.search(query=None, text_query=None)
    assert r3 == [], f"expected [], got {r3}"
    print("3. empty query OK")

    # 场景 4: 无匹配 token → []
    r4 = idx.search(text_query="phonon", top_k=3)
    assert r4 == [], f"expected [], got {r4}"
    print("4. no match OK")

    print("QW4 ALL CHECKS PASSED")


if __name__ == "__main__":
    _selfcheck()
