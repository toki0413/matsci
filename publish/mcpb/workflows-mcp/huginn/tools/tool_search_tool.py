"""Tool search & invoke — 渐进式工具发现元工具.

借鉴 OpenAaaS 的渐进式能力发现: 不把 60+ 工具 schema 全塞进 context,
而是让 LLM 按需搜索, 找到后直接 proxy-call.
这样能大幅减少 system prompt 的 token 占用.

两个 action:
  - search: 关键词搜索, 返回 name + description + 参数摘要
  - invoke: 按名称调用工具, 代理执行

检索升级: 在子串匹配基础上叠加 embedding 语义相似度 (余弦) 与关键词命中
混合排序, 能召回到"换了说法但同义"的工具. embedding 模型未缓存时安静降级
为纯子串/关键词匹配, 不阻塞. 工具向量与查询结果都走 TTL 缓存避免重复计算.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.utils.cache import TimedLRUCache

logger = logging.getLogger(__name__)

# 工具向量 / 查询结果缓存: 语义打分贵 (embedding), 反复搜索不该重复烧模型
_TOOL_VEC_CACHE: TimedLRUCache[list[float]] = TimedLRUCache(max_size=256, ttl=3600.0)
_RESULT_CACHE: TimedLRUCache[ToolResult] = TimedLRUCache(max_size=128, ttl=300.0)

# 中→英同义词兜底: 工具描述都是英文, 无 embedding 模型时中文查询靠这个命中
_ZH_EN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "文献": ("literature", "paper", "arxiv", "publish"),
    "搜索": ("search", "query", "find", "retrieve"),
    "材料": ("material", "structure", "crystal"),
    "数据库": ("database", "db", "data"),
    "计算": ("calculation", "compute", "simulate", "dft"),
    "模拟": ("simulation", "simulate", "molecular", "md"),
    "带隙": ("band gap", "bandgap", "band"),
    "结构": ("structure", "crystal", "geometry"),
    "热力学": ("thermodynamic", "thermo", "free energy"),
    "力学": ("mechanical", "stress", "strain", "fracture", "fatigue"),
    "机器学习": ("machine learning", "sklearn", "neural", "pytorch", "gnn"),
    "图像": ("image", "visual", "cv", "microscopy", "sem", "tem"),
    "实验": ("experimental", "xrd", "data"),
    "动力学": ("dynamics", "reaction", "transition"),
    "验证": ("validate", "benchmark", "check", "verify"),
    "报告": ("report", "summary", "summarize"),
    "记忆": ("memory", "recall", "remember"),
    "计划": ("plan", "todo", "task", "workflow"),
    "代码": ("code", "python", "bash", "shell"),
    "数据库查询": ("database", "lookup", "query"),
}


def _synonym_tokens(query: str) -> list[str]:
    """query 命中的中→英同义词 token (不含原 query token). 未命中任何中文关键词返回空."""
    low = (query or "").lower()
    extra: list[str] = []
    for zh, en in _ZH_EN_SYNONYMS.items():
        if zh in low:
            for e in en:
                if e not in extra:
                    extra.append(e)
    return extra


def _expand_query_tokens(query: str) -> list[str]:
    """query token + 中→英同义词展开, 提升英文描述下的中文查询召回."""
    tokens = _query_tokens(query)
    extra = _synonym_tokens(query)
    if not extra:
        return tokens
    # 合并去重, 同义词排在原 token 后面 (加权较低)
    seen = set(tokens)
    for e in extra:
        if e not in seen:
            seen.add(e)
            tokens.append(e)
    return tokens


def _lazy_embed_fn() -> Callable[[list[str]], list[list[float]]] | None:
    """复用 RAG 的 embedding 模型 (chroma DefaultEF ONNX), 未缓存则 None.

    与 agentic_search_tool 同源: 语义打分是可选增强, embedding 模型没下载时
    安静降级为关键词打分, 不阻塞工具发现.
    """
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


def _query_tokens(query: str) -> list[str]:
    """query 拆 token: 英文按词 (>=2 字), 中文按单字 + 连续多字词组."""
    q = (query or "").lower().strip()
    tokens: list[str] = []
    for t in q.replace("_", " ").replace("-", " ").split():
        if t.isascii() and len(t) >= 2:
            tokens.append(t)
    for t in q:
        if "\u4e00" <= t <= "\u9fff":
            tokens.append(t)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _tool_search_text(name: str, tool: HuginnTool) -> str:
    """工具的可检索文本: name + description + category + 参数名."""
    parts = [name, tool.description or "", getattr(tool, "category", "")]
    schema = tool.input_json_schema
    if schema and "properties" in schema:
        parts.append(" ".join(schema["properties"].keys()))
    return " ".join(p for p in parts if p)


def _tool_vector(name: str, tool: HuginnTool, embed: Callable | None) -> list[float] | None:
    """工具的 embedding 向量, 带 TTL 缓存."""
    if embed is None:
        return None
    cached = _TOOL_VEC_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        vecs = embed([_tool_search_text(name, tool)])
        vec = vecs[0] if vecs else None
    except Exception:
        logger.debug("embed failed for %s", name, exc_info=True)
        return None
    if vec is not None:
        _TOOL_VEC_CACHE.set(name, vec)
    return vec


def _keyword_score(query: str, name: str, tool: HuginnTool) -> float:
    """关键词命中分: name 命中 ×2, description ×1. 无 token 返回 0."""
    tokens = _expand_query_tokens(query)
    if not tokens:
        return 0.0
    n_low = name.lower()
    d_low = (tool.description or "").lower()
    return 2 * sum(1 for t in tokens if t in n_low) + sum(
        1 for t in tokens if t in d_low
    )


class ToolSearchInput(BaseModel):
    action: Literal["search", "invoke"] = Field(
        ..., description="search=搜索工具; invoke=代理调用指定工具"
    )
    query: str | None = Field(
        default=None,
        description="search 时用: 关键词, 匹配工具名或描述",
    )
    tool_name: str | None = Field(
        default=None,
        description="invoke 时用: 要调用的工具名",
    )
    tool_args: dict[str, Any] | None = Field(
        default=None,
        description="invoke 时用: 传给目标工具的参数",
    )
    limit: int = Field(
        default=10, ge=1, le=30,
        description="search 时最多返回几条结果",
    )
    semantic: bool = Field(
        default=True,
        description="search 时是否启用 embedding 语义检索 (未缓存模型自动降级关键词)",
    )


class ToolSearchTool(HuginnTool):
    """搜索和调用已注册但未在当前 schema 中暴露的工具."""

    name = "tool_search"
    category = "meta"
    description = (
        "Search for and invoke tools not in the current schema. "
        "Use action='search' with a query to find tools by keyword, "
        "then action='invoke' with tool_name and tool_args to call them. "
        "This gives access to the full tool registry without bloating the context."
    )
    read_only = False
    input_schema = ToolSearchInput

    async def call(self, args: ToolSearchInput, context: ToolContext) -> ToolResult:

        if args.action == "search":
            return self._search(args.query or "", args.limit, args.semantic)
        if args.action == "invoke":
            return await self._invoke(args.tool_name or "", args.tool_args or {}, context)
        return ToolResult(data=None, success=False, error=f"unknown action: {args.action}")

    def _search(self, query: str, limit: int, semantic: bool = True) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        # 查询结果 TTL 缓存
        cache_key = (query.lower().strip(), limit, semantic)
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        q = query.lower().strip()
        candidates: list[tuple[str, HuginnTool]] = []
        # 1) 子串精确匹配保底 (原行为)
        for name, tool in ToolRegistry._tools.items():
            if name == self.name:
                continue
            desc = tool.description or ""
            if q and q not in name.lower() and q not in desc.lower():
                continue
            candidates.append((name, tool))

        # 1.5) 中文同义词召回: 与子串匹配合并 (而非互斥), 否则任何工具命中原始
        # 中文字符串就会让整段同义词召回被跳过, 漏掉真正的英文命名工具.
        # 只用英文同义词 token (单字 token 噪音大, 会把含该字的无关工具全拉进来).
        syn_tokens = _synonym_tokens(query)
        if q and syn_tokens:
            matched = {n for n, _ in candidates}
            for name, tool in ToolRegistry._tools.items():
                if name == self.name or name in matched:
                    continue
                n_low = name.lower()
                d_low = (tool.description or "").lower()
                if any(t in n_low or t in d_low for t in syn_tokens):
                    candidates.append((name, tool))

        # 空 query: 纯列表浏览, 跳过语义打分, 保持原"按注册顺序"语义
        if not q:
            results = self._to_result_rows(candidates, limit)
            result = ToolResult(
                data={"query": query, "count": len(results), "tools": results},
                success=True,
            )
            _RESULT_CACHE.set(cache_key, result)
            return result

        # 2) embedding 语义召回: 子串没命中但语义相近的工具
        embed = _lazy_embed_fn() if semantic else None
        query_vec: list[float] | None = None
        if embed is not None:
            try:
                query_vec = embed([q])[0]
            except Exception:
                logger.debug("query embed failed", exc_info=True)
                query_vec = None
        if embed is not None and query_vec is not None and not candidates:
            for name, tool in ToolRegistry._tools.items():
                if name == self.name:
                    continue
                vec = _tool_vector(name, tool, embed)
                if vec is not None and _cosine(query_vec, vec) >= 0.2:
                    candidates.append((name, tool))

        # 3) 混合排序: 关键词命中 + 语义相似度
        scored: list[dict[str, Any]] = []
        for name, tool in candidates:
            kw = _keyword_score(query, name, tool)
            sem = 0.0
            if query_vec is not None:
                vec = _tool_vector(name, tool, embed)
                if vec is not None:
                    sem = _cosine(query_vec, vec)
            # 语义分兜底: 子串已命中但没有语义分时给保底, 保持原文行为排前面
            relevance = kw + sem * 3.0
            scored.append((name, tool, relevance))

        scored.sort(key=lambda t: (t[2], t[0]), reverse=True)

        results = self._to_result_rows(
            [(n, t) for n, t, _ in scored[:limit]],
            limit,
        )
        # 附上实际 relevance 分
        rel_map = {n: r for n, t, r in scored}
        for row in results:
            row["relevance"] = round(rel_map.get(row["name"], 0.0), 4)

        result = ToolResult(
            data={
                "query": query,
                "count": len(results),
                "tools": results,
            },
            success=True,
        )
        _RESULT_CACHE.set(cache_key, result)
        return result

    @staticmethod
    def _to_result_rows(
        candidates: list[tuple[str, HuginnTool]],
        limit: int,
        relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """把 (name, tool) 候选转成结果行, 截断到 limit."""
        rows: list[dict[str, Any]] = []
        for name, tool in candidates[:limit]:
            params: list[str] = []
            schema = tool.input_json_schema
            if schema and "properties" in schema:
                for pname, pinfo in schema["properties"].items():
                    ptype = pinfo.get("type", "?")
                    params.append(f"{pname}:{ptype}")
            rows.append({
                "name": name,
                "description": (tool.description or "")[:200],
                "active": tool.active,
                "params": params[:8],
                "category": getattr(tool, "category", ""),
                "relevance": relevance,
            })
        return rows

    async def _invoke(
        self, tool_name: str, tool_args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get(tool_name)
        if tool is None:
            return ToolResult(
                data=None, success=False,
                error=f"tool '{tool_name}' not found in registry",
            )
        try:
            result = await tool.call(tool_args, context)
            return result
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
