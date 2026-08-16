"""锁定文献调研增强 (P0-1/P0-2/P1-4/P0-3).

这些断言把本次"更大范围"的文献增强变成可执行回归检查:
- P0-1: LLM query 拆分 -> 子查询 (``_expand_query``)
- P0-2: 相关度 rerank 层替代纯 citation 排序 (``_rerank``)
- P1-4: 检索结果 TTL 缓存 key 设计 (``_search_cache_key``) + 缓存命中
- P0-3: ingest 全文优先于 abstract (``_do_ingest_to_rag`` 喂全文)

全部本地可跑, 不打真实 API.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from huginn.core_types import ToolContext
from huginn.permissions import PermissionConfig
from huginn.tools.literature.search_sources import _rerank, _sort_papers
from huginn.tools.literature.tool import (
    _SEARCH_CACHE,
    _expand_query,
    _search_cache_key,
)


def _ctx() -> ToolContext:
    return ToolContext(
        session_id="test",
        workspace=".",
        config=PermissionConfig(auto_approve_all=True),
    )


# ── P0-1: query 拆分 ────────────────────────────────────────────────────────


class _FakeModel:
    """最小 LLM 假体: 直接返回预设 content, 支持 ainvoke/invoke."""

    def __init__(self, content: str):
        self._content = content
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return _Resp(self._content)

    async def invoke(self, messages):
        return await self.ainvoke(messages)


class _Resp:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_expand_query_returns_subqueries():
    model = _FakeModel('{"subqueries": ["硅的带隙是多少", "温度对带隙的影响"]}')
    subs = await _expand_query(model, "硅的带隙与温度依赖关系")
    assert len(subs) == 2
    assert "硅的带隙是多少" in subs
    assert model.calls, "should have invoked the model"


@pytest.mark.asyncio
async def test_expand_query_drops_duplicate_of_original():
    model = _FakeModel('{"subqueries": ["硅的带隙与温度依赖关系", "温度对带隙的影响"]}')
    subs = await _expand_query(model, "硅的带隙与温度依赖关系")
    # 与原始 query 归一化后重复的子查询被过滤
    assert "温度对带隙的影响" in subs
    assert all(s != "硅的带隙与温度依赖关系" for s in subs)


@pytest.mark.asyncio
async def test_expand_query_failure_returns_empty():
    class _Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("model down")

    subs = await _expand_query(_Boom(), "question")
    assert subs == []


@pytest.mark.asyncio
async def test_expand_query_respects_max_n():
    model = _FakeModel('{"subqueries": ["a", "b", "c", "d"]}')
    subs = await _expand_query(model, "q", max_n=2)
    assert len(subs) <= 2


# ── P0-2: rerank ────────────────────────────────────────────────────────────


def _paper(title: str, abstract: str = "", citations: int = 0, **kw) -> dict[str, Any]:
    d = {"title": title, "abstract": abstract, "citations": citations, "doi": "", "source": "t"}
    d.update(kw)
    return d


def test_rerank_boosts_relevant_over_highly_cited():
    papers = [
        # 高引但完全不沾 query
        _paper("General thermodynamics of materials", abstract="unrelated", citations=1000),
        # 低引但命中 query 关键词 (title)
        _paper("Band gap engineering in silicon alloys", abstract="band gap study", citations=5),
        # 命中 abstract 但不命中 title
        _paper("A review", abstract="we measure the silicon band gap", citations=50),
    ]
    ranked = _rerank("silicon band gap", list(papers))
    assert ranked[0]["title"] == "Band gap engineering in silicon alloys"
    # 相关度字段已标注
    assert ranked[0]["relevance"] >= ranked[1]["relevance"]


def test_rerank_adds_relevance_field():
    papers = [_paper("silicon band gap paper", abstract="a b c", citations=10)]
    ranked = _rerank("silicon band gap", papers)
    assert "relevance" in ranked[0]
    assert ranked[0]["relevance"] > 0


def test_rerank_falls_back_to_citation_when_no_keywords():
    # query 全是停用词 -> 无关键词 -> 走 _sort_papers (citation 排序)
    papers = [
        _paper("alpha beta gamma", abstract="", citations=3),
        _paper("delta epsilon", abstract="", citations=9),
    ]
    ranked = _rerank("the and of", list(papers))
    assert ranked[0]["title"] == "delta epsilon"


def test_rerank_top_n():
    papers = [
        _paper("silicon band gap x", abstract="a", citations=1),
        _paper("silicon band gap y", abstract="b", citations=2),
        _paper("silicon band gap z", abstract="c", citations=3),
    ]
    ranked = _rerank("silicon band gap", list(papers), top_n=2)
    assert len(ranked) == 2


def test_sort_papers_still_citation_ordered():
    # 降级路径保持原语义: 有 abstract 优先, 其次 citation 降序
    papers = [
        _paper("p1", abstract="x", citations=1),
        _paper("p2", abstract="", citations=2),
    ]
    out = _sort_papers(list(papers))
    assert out[0]["title"] == "p1"  # 有 abstract 优先
    papers2 = [
        _paper("p3", abstract="x", citations=1),
        _paper("p4", abstract="x", citations=9),
    ]
    out2 = _sort_papers(list(papers2))
    assert out2[0]["title"] == "p4"  # 同级下 citation 降序


# ── P1-4: 缓存 key ──────────────────────────────────────────────────────────


class _LitArgs:
    """LiteratureInput 的最小假体 — 只暴露缓存 key 用到的字段."""

    def __init__(self, **kw):
        self.query = kw.get("query", "q")
        self.sources = tuple(kw.get("sources", ["arxiv"]))
        self.year_from = kw.get("year_from")
        self.year_to = kw.get("year_to")
        self.max_results = kw.get("max_results", 10)
        self.expand_query = kw.get("expand_query", False)
        self.action = kw.get("action", "search")


def test_search_cache_key_excludes_action():
    """key 不含 action — summarize/ingest 的隐式 search 与显式 search 共享缓存."""
    a = _LitArgs(query="q", action="search")
    b = _LitArgs(query="q", action="summarize")
    assert _search_cache_key(a, ()) == _search_cache_key(b, ())


def test_search_cache_key_differs_on_query_or_sources():
    a = _LitArgs(query="q1")
    b = _LitArgs(query="q2")
    assert _search_cache_key(a, ()) != _search_cache_key(b, ())
    c = _LitArgs(query="q1", sources=["s2"])
    assert _search_cache_key(a, ()) != _search_cache_key(c, ())


def test_search_cache_key_includes_subqueries_and_expand():
    a = _LitArgs(query="q")
    b = _LitArgs(query="q")
    assert _search_cache_key(a, ("sub1",)) != _search_cache_key(b, ())
    c = _LitArgs(query="q", expand_query=True)
    assert _search_cache_key(c, ()) != _search_cache_key(a, ())


def test_search_cache_key_differs_on_year_and_count():
    a = _LitArgs(query="q", year_from=2020, year_to=2024, max_results=5)
    b = _LitArgs(query="q", year_from=2010, year_to=2015, max_results=10)
    assert _search_cache_key(a, ()) != _search_cache_key(b, ())


# ── P0-3: ingest 全文优先 ───────────────────────────────────────────────────


def test_ingest_uses_full_text_when_available(monkeypatch):
    """P0-3: 有 full_text 时 ingest 入库的是全文而不是 abstract."""
    from huginn.tools.literature.tool import LiteratureTool
    from huginn.tools.registry import ToolRegistry

    captured: list[str] = []

    class _FakeRag:
        async def call(self, inp, context):
            captured.append(getattr(inp, "document", ""))
            return type("R", (), {"success": True})()

    monkeypatch.setattr(
        ToolRegistry, "get",
        lambda name: _FakeRag() if name == "rag_tool" else None,
    )

    tool = LiteratureTool()
    args = type("A", (), {
        "papers": [{
            "title": "T",
            "authors": [],
            "year": 2024,
            "venue": "",
            "doi": "10.1/1",
            "abstract": "short abstract only",
            "full_text": "the full body of the paper with rich experimental detail",
        }],
        "query": None,
    })()
    result = asyncio.run(tool._do_ingest_to_rag(args, _ctx()))
    assert result.success
    assert len(captured) == 1
    assert "FullText: the full body" in captured[0]
    assert "Abstract" not in captured[0].split("FullText:")[0].split("\n")[-1]


def test_ingest_falls_back_to_abstract_without_full_text(monkeypatch):
    from huginn.tools.literature.tool import LiteratureTool
    from huginn.tools.registry import ToolRegistry

    captured: list[str] = []

    class _FakeRag:
        async def call(self, inp, context):
            captured.append(getattr(inp, "document", ""))
            return type("R", (), {"success": True})()

    monkeypatch.setattr(
        ToolRegistry, "get",
        lambda name: _FakeRag() if name == "rag_tool" else None,
    )

    tool = LiteratureTool()
    args = type("A", (), {
        "papers": [{
            "title": "T", "authors": [], "year": 2024, "venue": "",
            "doi": "10.1/2", "abstract": "only the abstract body",
            "full_text": "",
        }],
        "query": None,
    })()
    result = asyncio.run(tool._do_ingest_to_rag(args, _ctx()))
    assert result.success
    assert "Abstract: only the abstract body" in captured[0]
