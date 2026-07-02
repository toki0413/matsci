"""Tests for AgenticSearchTool (W2 A2).

The search and fetch steps are injectable, so every test runs hermetically
without touching the network. Covers:
- schema validation (Literal action, bounds, required question)
- helper functions (html_to_text, sentences, keywords, extract_relevant,
  derive_followups, synthesize)
- quick action: single search -> snippets as findings, no fetch
- research action: multi-hop, fetch + extract, dedupe visited URLs,
  follow-up query derivation, fetch-failure snippet fallback
- HUGINN_DISABLE_WEB_SEARCH short-circuit
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from pydantic import ValidationError

from huginn.permissions import PermissionConfig
from huginn.tools.agentic_search_tool import (
    AgenticSearchInput,
    AgenticSearchTool,
    _derive_followups,
    _extract_relevant,
    _html_to_text,
    _keywords,
    _score_passage,
    _sentences,
    _synthesize,
)
from huginn.types import ToolContext


# ── helpers ──────────────────────────────────────────────────────────────────


def _ctx() -> ToolContext:
    return ToolContext(
        session_id="test",
        workspace=".",
        config=PermissionConfig(auto_approve_all=True),
    )


def _make_searcher(results_by_query: dict[str, list[dict[str, Any]]]):
    """Build an injectable searcher that returns canned results per query."""
    calls: list[tuple[str, int]] = []

    async def _search(query: str, max_results: int) -> list[dict[str, Any]]:
        calls.append((query, max_results))
        return results_by_query.get(query, [])[:max_results]

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


def _make_fetcher(content_by_url: dict[str, str | None], *, delay: float = 0.0):
    """Build an injectable fetcher. None value = fetch failure."""
    calls: list[str] = []

    async def _fetch(url: str, max_chars: int) -> str | None:
        calls.append(url)
        content = content_by_url.get(url)
        if content is None:
            return None
        return content[:max_chars]

    _fetch.calls = calls  # type: ignore[attr-defined]
    return _fetch


# ── schema ───────────────────────────────────────────────────────────────────


class TestSchema:
    def test_rejects_bad_action(self):
        with pytest.raises(ValidationError):
            AgenticSearchInput(action="bogus", question="q")

    def test_rejects_missing_question(self):
        with pytest.raises(ValidationError):
            AgenticSearchInput()

    def test_rejects_max_hops_below_min(self):
        with pytest.raises(ValidationError):
            AgenticSearchInput(question="q", max_hops=0)

    def test_rejects_max_hops_above_max(self):
        with pytest.raises(ValidationError):
            AgenticSearchInput(question="q", max_hops=5)

    def test_defaults(self):
        inp = AgenticSearchInput(question="q")
        assert inp.action == "research"
        assert inp.max_hops == 2
        assert inp.max_results_per_hop == 3
        assert inp.max_content_chars == 4000


# ── helper functions ─────────────────────────────────────────────────────────


class TestHtmlToText:
    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        assert _html_to_text(html) == "Hello world"

    def test_removes_script_and_style(self):
        html = "<style>.x{}</style><script>alert(1)</script><p>visible</p>"
        assert _html_to_text(html) == "visible"

    def test_collapses_whitespace(self):
        html = "<p>a\n\n   b</p>"
        assert _html_to_text(html) == "a b"

    def test_empty_input(self):
        assert _html_to_text("") == ""


class TestSentences:
    def test_splits_on_punctuation(self):
        text = "First sentence. Second one! Third? Fourth."
        s = _sentences(text)
        assert len(s) == 4
        assert s[0] == "First sentence"

    def test_filters_short_fragments(self):
        text = "ok. This is a long enough sentence. a."
        s = _sentences(text)
        assert len(s) == 1
        assert "long enough" in s[0]

    def test_chinese_period(self):
        text = "第一句足够长的话。第二句也是够长的哦。"
        s = _sentences(text)
        assert len(s) == 2


class TestKeywords:
    def test_removes_english_stopwords(self):
        kws = _keywords("what is the band gap of silicon")
        assert "band" in kws
        assert "silicon" in kws
        assert "the" not in kws
        assert "what" not in kws

    def test_removes_chinese_stopwords(self):
        kws = _keywords("硅的带隙是什么")
        assert "硅" in kws
        assert "带隙" in kws or "带" in kws
        assert "的" not in kws
        assert "什么" not in kws

    def test_empty_question(self):
        assert _keywords("the of a") == set()


class TestScorePassage:
    def test_counts_keyword_hits(self):
        kws = {"silicon", "band", "gap"}
        assert _score_passage("silicon has a band gap", kws) == 3
        assert _score_passage("silicon is a semiconductor", kws) == 1
        assert _score_passage("unrelated text", kws) == 0


class TestExtractRelevant:
    def test_returns_top_scoring_sentences(self):
        text = (
            "Silicon has a band gap of 1.12 eV. "
            "The weather is nice today. "
            "Germanium has a smaller band gap than silicon. "
            "Random unrelated filler text about nothing."
        )
        passages = _extract_relevant(text, "band gap of silicon", max_passages=2)
        assert len(passages) <= 2
        assert any("1.12" in p for p in passages)
        # the weather sentence should not make the cut
        assert not any("weather" in p for p in passages)

    def test_returns_empty_when_no_relevant_sentences(self):
        text = "Completely unrelated content about cooking and sports."
        passages = _extract_relevant(text, "silicon band gap")
        assert passages == []

    def test_falls_back_to_first_sentences_when_no_keywords(self):
        # question is all stopwords -> no keywords -> return first sentences
        text = "First sentence is long enough. Second sentence also qualifies."
        passages = _extract_relevant(text, "the of a", max_passages=1)
        assert len(passages) == 1


class TestDeriveFollowups:
    def test_generates_queries_from_titles(self):
        findings = [
            {"title": "Silicon band gap temperature dependence", "url": "u1"},
            {"title": "Germanium electronic properties review", "url": "u2"},
        ]
        followups = _derive_followups(findings, "band gap of silicon", max_n=3)
        assert len(followups) >= 1
        # follow-ups should include new keywords from titles
        assert any("temperature" in f or "dependence" in f for f in followups)

    def test_empty_findings_returns_empty(self):
        assert _derive_followups([], "q") == []

    def test_dedupes_keywords_across_findings(self):
        findings = [
            {"title": "Silicon properties", "url": "u1"},
            {"title": "Silicon properties again", "url": "u2"},
        ]
        followups = _derive_followups(findings, "band gap", max_n=3)
        # "silicon" + "properties" only produce one unique follow-up
        assert len(followups) <= 1


class TestSynthesize:
    def test_formats_findings_with_citations(self):
        findings = [
            {
                "url": "http://example.com/a",
                "title": "Paper A",
                "passages": ["key fact one"],
                "hop": 0,
            }
        ]
        out = _synthesize(findings, "question")
        assert "Agentic research on: question" in out
        assert "Paper A" in out
        assert "http://example.com/a" in out
        assert "key fact one" in out
        assert "[1]" in out

    def test_empty_findings_message(self):
        out = _synthesize([], "my question")
        assert "No relevant findings" in out
        assert "my question" in out


# ── tool: quick action ───────────────────────────────────────────────────────


class TestQuickAction:
    def test_returns_snippets_as_findings_without_fetching(self):
        searcher = _make_searcher({
            "band gap": [
                {"url": "u1", "title": "t1", "snippet": "snip1"},
                {"url": "u2", "title": "t2", "snippet": "snip2"},
            ]
        })
        fetcher = _make_fetcher({})
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"action": "quick", "question": "band gap"}, _ctx()
        ))
        assert result.success
        data = result.data
        assert data["action"] == "quick"
        assert data["n_findings"] == 2
        assert data["findings"][0]["passages"] == ["snip1"]
        assert data["findings"][0]["hop"] == 0
        # quick must not fetch page bodies
        assert fetcher.calls == []

    def test_quick_handles_empty_results(self):
        searcher = _make_searcher({"q": []})
        tool = AgenticSearchTool(searcher=searcher, fetcher=_make_fetcher({}))
        result = asyncio.run(tool.call({"action": "quick", "question": "q"}, _ctx()))
        assert result.success
        assert result.data["n_findings"] == 0


# ── tool: research action ────────────────────────────────────────────────────


class TestResearchAction:
    def test_fetches_and_extracts_passages(self):
        searcher = _make_searcher({
            "silicon band gap": [
                {"url": "u1", "title": "Silicon properties", "snippet": "snip"},
            ]
        })
        fetcher = _make_fetcher({
            "u1": "Silicon has a band gap of 1.12 eV at room temperature. "
                  "Unrelated text about weather."
        })
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"question": "silicon band gap", "max_hops": 1}, _ctx()
        ))
        assert result.success
        data = result.data
        assert data["action"] == "research"
        assert data["n_findings"] == 1
        finding = data["findings"][0]
        assert finding["url"] == "u1"
        assert finding["hop"] == 0
        # extracted passage should mention 1.12, not weather
        joined = " ".join(finding["passages"])
        assert "1.12" in joined
        assert "weather" not in joined

    def test_multi_hop_follows_up_queries(self):
        # hop 1: returns u1 with a title that drives a follow-up query
        searcher = _make_searcher({
            "band gap": [
                {"url": "u1", "title": "Silicon temperature dependence study", "snippet": "s"},
            ],
            # follow-up query derived from title keywords (orig + new words)
            "band gap silicon temperature dependence": [
                {"url": "u2", "title": "Temperature effects on semiconductors", "snippet": "s2"},
            ],
        })
        fetcher = _make_fetcher({
            "u1": "Silicon band gap has temperature dependence. Varshni equation applies.",
            "u2": "Temperature lowers the band gap in silicon semiconductors significantly.",
        })
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"question": "band gap", "max_hops": 2, "max_results_per_hop": 2}, _ctx()
        ))
        assert result.success
        data = result.data
        # both hops produced findings
        urls = {f["url"] for f in data["findings"]}
        assert "u1" in urls
        assert "u2" in urls
        hops = {f["hop"] for f in data["findings"]}
        assert 0 in hops
        assert 1 in hops

    def test_dedupes_visited_urls_across_hops(self):
        # both hops' searches return the same URL -> fetched once
        searcher = _make_searcher({
            "q": [{"url": "u1", "title": "same title", "snippet": "s"}],
            "q followup": [{"url": "u1", "title": "same title", "snippet": "s"}],
        })
        fetcher = _make_fetcher({"u1": "Relevant content about the query topic here."})
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"question": "q", "max_hops": 2, "max_results_per_hop": 2}, _ctx()
        ))
        assert result.success
        # u1 fetched only once despite appearing in both hops
        assert fetcher.calls.count("u1") == 1
        # only one finding (the duplicate URL is skipped on hop 2)
        assert result.data["n_findings"] == 1

    def test_fetch_failure_falls_back_to_snippet(self):
        searcher = _make_searcher({
            "q": [{"url": "u1", "title": "t1", "snippet": "fallback snippet text"}]
        })
        # fetcher returns None (network failure) -> fall back to snippet
        fetcher = _make_fetcher({"u1": None})
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"question": "q", "max_hops": 1}, _ctx()
        ))
        assert result.success
        finding = result.data["findings"][0]
        assert finding["passages"] == ["fallback snippet text"]

    def test_no_results_returns_empty_findings(self):
        searcher = _make_searcher({"q": []})
        tool = AgenticSearchTool(searcher=searcher, fetcher=_make_fetcher({}))
        result = asyncio.run(tool.call(
            {"question": "q", "max_hops": 2}, _ctx()
        ))
        assert result.success
        assert result.data["n_findings"] == 0
        assert "No relevant findings" in result.data["answer"]

    def test_stops_early_when_no_followups(self):
        # hop 1 finding title has no new keywords -> no follow-up -> stop
        searcher = _make_searcher({
            "band gap": [{"url": "u1", "title": "band gap", "snippet": "s"}],
        })
        fetcher = _make_fetcher({
            "u1": "The band gap is a property. Band gap values vary."
        })
        tool = AgenticSearchTool(searcher=searcher, fetcher=fetcher)
        result = asyncio.run(tool.call(
            {"question": "band gap", "max_hops": 3}, _ctx()
        ))
        assert result.success
        # only one hop ran because no follow-up queries could be derived
        assert result.data["n_hops"] == 1


# ── disable short-circuit ────────────────────────────────────────────────────


class TestDisableShortCircuit:
    def test_disabled_returns_error_without_searching(self, monkeypatch):
        monkeypatch.setenv("HUGINN_DISABLE_WEB_SEARCH", "1")
        searcher = _make_searcher({"q": [{"url": "u", "title": "t", "snippet": "s"}]})
        tool = AgenticSearchTool(searcher=searcher, fetcher=_make_fetcher({}))
        result = asyncio.run(tool.call({"question": "q"}, _ctx()))
        assert not result.success
        assert "disabled" in result.error.lower()
        # searcher must not have been called
        assert searcher.calls == []


# ── default backends (mocked) ────────────────────────────────────────────────


class TestDefaultBackends:
    def test_default_search_delegates_to_web_search_tool(self, monkeypatch):
        # Patch WebSearchTool.call so no real network happens.
        from huginn.tools import web_search_tool as wst_mod

        async def _fake_call(self, args, ctx):
            from huginn.types import ToolResult
            return ToolResult(
                data={"results": [{"url": "x", "title": "y", "snippet": "z"}]},
                success=True,
            )

        monkeypatch.setattr(wst_mod.WebSearchTool, "call", _fake_call)
        tool = AgenticSearchTool()  # defaults
        results = asyncio.run(tool._default_search("query", 5))
        assert results == [{"url": "x", "title": "y", "snippet": "z"}]

    def test_default_fetch_returns_text_for_html(self, monkeypatch):
        # Patch urllib.request.urlopen so no real network happens.
        from huginn.tools import agentic_search_tool as ast_mod

        class _FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b"<p>hello world from a page</p>"

        def _fake_urlopen(req, timeout=None):
            return _FakeResp()

        monkeypatch.setattr(ast_mod.urllib.request, "urlopen", _fake_urlopen)
        tool = AgenticSearchTool()
        text = asyncio.run(tool._default_fetch("http://example.com", 1000))
        assert text == "hello world from a page"

    def test_default_fetch_returns_none_on_error(self, monkeypatch):
        from huginn.tools import agentic_search_tool as ast_mod

        def _boom(req, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr(ast_mod.urllib.request, "urlopen", _boom)
        tool = AgenticSearchTool()
        text = asyncio.run(tool._default_fetch("http://example.com", 1000))
        assert text is None
