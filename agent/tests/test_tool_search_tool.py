"""Unit tests for tool_search_tool.py — 语义检索升级的纯函数 + _search 行为.

覆盖: token 展开 (中→英同义词)、关键词打分、余弦相似度、空 query 列表浏览、
子串/同义词召回、relevance 标注、结果 TTL 缓存、invoke 找不到工具报错.
"""

from __future__ import annotations

import pytest

from huginn.tools.tool_search_tool import (
    _RESULT_CACHE,
    _TOOL_VEC_CACHE,
    ToolSearchInput,
    ToolSearchTool,
    _cosine,
    _expand_query_tokens,
    _keyword_score,
    _query_tokens,
)


class TestQueryTokens:
    def test_english_words_and_chinese_chars(self):
        toks = _query_tokens("find 带隙 of Li2O")
        assert "find" in toks
        assert "li2o" in toks
        # 中文按单字切分
        assert "带" in toks
        assert "隙" in toks

    def test_short_ascii_dropped(self):
        assert "a" not in _query_tokens("a")
        assert "ab" in _query_tokens("ab")

    def test_dedup_preserves_order(self):
        toks = _query_tokens("gap gap band")
        assert toks == ["gap", "band"]


class TestExpandQueryTokens:
    def test_chinese_synonym_expansion(self):
        toks = _expand_query_tokens("检索文献数据库")
        # 原 token (中文单字) 保底, 同义词附加在后面
        assert "literature" in toks
        assert "paper" in toks
        assert "database" in toks

    def test_no_synonym_keeps_original(self):
        toks = _expand_query_tokens("hello world")
        assert toks == ["hello", "world"]

    def test_empty_query(self):
        assert _expand_query_tokens("") == []

    def test_synonym_tokens_only_english(self):
        from huginn.tools.tool_search_tool import _synonym_tokens

        assert "literature" in _synonym_tokens("文献")
        # 不含原 query 的中文单字 token
        assert not any(t for t in _synonym_tokens("文献") if "\u4e00" <= t <= "\u9fff")
        assert _synonym_tokens("hello world") == []


class TestKeywordScore:
    def test_name_hit_counts_double(self):
        class FakeTool:
            description = "does arbitrary stuff"

        # '文献' → 'literature' 命中 name → 2 分 (无描述命中)
        assert _keyword_score("文献", "literature_tool", FakeTool()) == 2

    def test_desc_hit_single(self):
        class FakeTool:
            description = "searches arxiv papers"

        # 'arxiv' 命中 description → 1 分
        assert _keyword_score("arxiv", "other_tool", FakeTool()) == 1

    def test_no_tokens_zero(self):
        class FakeTool:
            description = "anything"

        assert _keyword_score("", "name", FakeTool()) == 0


class TestCosine:
    def test_identical(self):
        assert _cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_guard(self):
        assert _cosine([0, 0], [1, 0]) == pytest.approx(0.0)


class TestToolSearch:
    def setup_method(self):
        _RESULT_CACHE.clear()
        _TOOL_VEC_CACHE.clear()

    def test_empty_query_lists_in_registration_order(self):
        tool = ToolSearchTool()
        result = tool._search("", limit=10, semantic=False)
        assert result.success
        rows = result.data["tools"]
        assert rows  # 至少返回一些工具
        names = [r["name"] for r in rows]
        assert len(names) == len(set(names))  # 无重复
        # 空 query 跳过语义打分, relevance 全部为 0
        assert all(r["relevance"] == 0.0 for r in rows)

    def test_substring_search_finds_literature_tool(self):
        tool = ToolSearchTool()
        result = tool._search("literature", limit=10, semantic=False)
        names = [r["name"] for r in result.data["tools"]]
        assert any("literature" in n for n in names)

    def test_chinese_query_synonym_recall(self):
        """中文 '文献' 无 embedding 时靠同义词命中 literature 相关工具."""
        tool = ToolSearchTool()
        result = tool._search("文献", limit=10, semantic=False)
        names = [r["name"] for r in result.data["tools"]]
        # 即使有工具命中原始中文字符串, 同义词召回也合并进来 (回归 bug 修复)
        assert any("literature" in n for n in names), names

    def test_synonym_recall_merges_with_substring_matches(self):
        """回归: 子串命中非空时同义词召回不再被跳过 (paper_tool 命中原始'文献'时
        literature_tool 仍应被召回)."""
        tool = ToolSearchTool()
        result = tool._search("文献", limit=20, semantic=False)
        names = [r["name"] for r in result.data["tools"]]
        assert "paper_tool" in names  # 子串命中
        assert "literature_tool" in names  # 同义词召回合并

    def test_semantic_off_no_embedding_graceful(self):
        """semantic=False 且无 embedding 模型 → 静默走关键词, 不抛异常."""
        tool = ToolSearchTool()
        result = tool._search("band gap 材料", limit=5, semantic=False)
        assert result.success
        assert "relevance" in result.data["tools"][0]

    def test_result_ttl_cache(self):
        tool = ToolSearchTool()
        r1 = tool._search("literature", limit=5, semantic=False)
        key = ("literature", 5, False)
        cached = _RESULT_CACHE.get(key)
        assert cached is not None
        assert cached is r1  # 同对象缓存命中

    def test_search_includes_relevance(self):
        tool = ToolSearchTool()
        result = tool._search("literature", limit=5, semantic=False)
        row = result.data["tools"][0]
        assert "relevance" in row
        assert row["relevance"] >= 0

    def test_invoke_unknown_tool_returns_error(self):
        import asyncio

        tool = ToolSearchTool()
        result = asyncio.run(
            tool.call(
                ToolSearchInput(action="invoke", tool_name="no_such_tool_xyz"),
                context=None,
            )
        )
        assert not result.success
        assert "not found" in result.error

    def test_unknown_action(self):
        tool = ToolSearchTool()
        # Literal 约束下非法 action 走 pydantic 校验
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolSearchInput(action="bogus")
