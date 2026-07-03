"""Reranker — two-stage retrieval with cross-encoder reranking.

Inspired by AstrBot's FaissVecDB.retrieve() which fetches fetch_k (> k)
candidates, then optionally reranks with a RerankProvider.

Stage 1: Vector search retrieves fetch_k candidates (default: top_k * 4)
Stage 2: Cross-encoder reranks and returns top_k results

The reranker gracefully degrades: if no cross-encoder model is available,
it returns the Stage 1 results as-is (just truncated to top_k).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("huginn.reranker")


@dataclass
class RerankResult:
    """A single reranked result."""

    document: str
    score: float
    metadata: dict[str, Any] | None = None
    original_rank: int = 0


class RerankProvider:
    """Abstract base for reranking providers."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[RerankResult]:
        ...

    def is_available(self) -> bool:
        ...


class CrossEncoderReranker(RerankProvider):
    """Cross-encoder reranker using sentence-transformers.

    Uses a cross-encoder model (e.g., bge-reranker-base) to re-score
    query-document pairs. Falls back gracefully if the model is unavailable.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._loaded = False

    def _load_model(self) -> None:
        # 懒加载: 第一次真正需要时才去拉模型, 避免启动时卡住
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            self._loaded = True
            logger.info("CrossEncoder reranker loaded: %s", self._model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed, reranker disabled")
        except Exception as e:
            logger.warning("Failed to load CrossEncoder %s: %s", self._model_name, e)

    def is_available(self) -> bool:
        self._load_model()
        return self._model is not None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[RerankResult]:
        import asyncio

        if not documents:
            return []

        self._load_model()
        if self._model is None:
            # 没有可用的 cross-encoder, 退回原顺序截断到 top_k
            return [
                RerankResult(document=doc, score=0.0, original_rank=i)
                for i, doc in enumerate(documents[:top_k])
            ]

        # 给所有 (query, doc) 对打分
        pairs = [(query, doc) for doc in documents]
        scores = await asyncio.to_thread(self._model.predict, pairs)

        # 按分数降序排
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for rank, (orig_idx, score) in enumerate(ranked[:top_k]):
            results.append(
                RerankResult(
                    document=documents[orig_idx],
                    score=float(score),
                    original_rank=orig_idx,
                )
            )

        return results


class NoopReranker(RerankProvider):
    """No-op reranker — just truncates to top_k. Used when reranking is disabled."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[RerankResult]:
        return [
            RerankResult(document=doc, score=0.0, original_rank=i)
            for i, doc in enumerate(documents[:top_k])
        ]

    def is_available(self) -> bool:
        return True


# 全局单例
_reranker: RerankProvider | None = None


def get_reranker(use_cross_encoder: bool = False) -> RerankProvider:
    """Get the global reranker instance.

    Args:
        use_cross_encoder: If True, try to load CrossEncoderReranker.
                          If False or loading fails, returns NoopReranker.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    if use_cross_encoder:
        ce = CrossEncoderReranker()
        if ce.is_available():
            _reranker = ce
        else:
            _reranker = NoopReranker()
    else:
        _reranker = NoopReranker()

    return _reranker


def reset_reranker() -> None:
    """Reset the global reranker (for testing)."""
    global _reranker
    _reranker = None


__all__ = [
    "RerankResult",
    "RerankProvider",
    "CrossEncoderReranker",
    "NoopReranker",
    "get_reranker",
    "reset_reranker",
]
