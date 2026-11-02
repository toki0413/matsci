"""共享的语义 embedding 懒加载.

DRY: ``tool_search_tool`` 与 ``agentic_search_tool`` 曾各复制一份逐字相同的
``_lazy_embed_fn``. 语义打分是可选增强, 两处 "模型没下载就安静降级" 的行为必须
一致, 否则工具发现与网页检索会在同一环境下走出不同的打分路径.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def lazy_embed_fn() -> Callable[[list[str]], list[list[float]]] | None:
    """复用 RAG 的 embedding 模型 (chroma DefaultEF ONNX), 未缓存则 None."""
    try:
        from huginn.rag.vector_store import _embedding_model_cached

        if not _embedding_model_cached():
            return None
        from chromadb.utils.embedding_functions import DefaultEF

        ef = DefaultEF()
        _ = ef(["probe"])  # 触发加载, 失败会抛异常

        def embed(texts: list[str]) -> list[list[float]]:
            return ef(texts)

        return embed
    except Exception:
        logger.debug("semantic embed unavailable, keyword scoring only", exc_info=True)
        return None
