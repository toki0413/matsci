"""BS1: web_search_tool 多后端健康分路由 —— 新后端激活门控 + cooldown + JSON 解析。

不发真网络请求: monckeypatch urllib.request.urlopen 返回固定 JSON,
验证 bing/brave/searxng 的解析结果与 search_engine 标注, 以及无 key 时静默跳过、
连续失败进入 cooldown、成功整体复位.
"""

from __future__ import annotations

import json

import pytest

from huginn.tools.web_search_tool import WebSearchTool


class _FakeResp:
    def __init__(self, payload: str, ctype: str = "application/json") -> None:
        self._payload = payload.encode("utf-8")
        self.headers = {"Content-Type": ctype}

    def read(self, n: int | None = None) -> bytes:
        return self._payload if n is None else self._payload[:n]

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_keys(monkeypatch):
    """测试间清掉会污染激活判定的各后端 key/env."""
    for k in (
        "TAVILY_API_KEY", "HUGINN_BING_API_KEY", "HUGINN_BRAVE_API_KEY",
        "HUGINN_SEARXNG_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_health():
    WebSearchTool._backend_health = {}
    WebSearchTool._backend_cooldown = set()
    WebSearchTool._consecutive_failures = 0
    WebSearchTool._circuit_broken = False
    yield


class TestActiveGating:
    def test_bing_requires_key(self, monkeypatch):
        assert WebSearchTool._backend_active("bing") is False
        monkeypatch.setenv("HUGINN_BING_API_KEY", "k")
        assert WebSearchTool._backend_active("bing") is True

    def test_brave_requires_key(self, monkeypatch):
        assert WebSearchTool._backend_active("brave") is False
        monkeypatch.setenv("HUGINN_BRAVE_API_KEY", "k")
        assert WebSearchTool._backend_active("brave") is True

    def test_searxng_requires_url(self, monkeypatch):
        assert WebSearchTool._backend_active("searxng") is False
        monkeypatch.setenv("HUGINN_SEARXNG_URL", "https://my.searxng.local")
        assert WebSearchTool._backend_active("searxng") is True

    def test_keyless_backends_active_by_default(self):
        assert WebSearchTool._backend_active("arxiv") is True
        assert WebSearchTool._backend_active("ddgs") is True


class TestHealthRouting:
    def test_failures_push_into_cooldown_then_reset(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        for _ in range(WebSearchTool._BACKEND_COOLDOWN_THRESHOLD):
            WebSearchTool._mark_backend_failure("tavily")
        assert "tavily" in WebSearchTool._backend_cooldown
        # cooldown 的后端不再出现在活跃路由里
        assert "tavily" not in WebSearchTool._ordered_active_backends()
        # 任一成功整体复位
        WebSearchTool._on_success()
        assert WebSearchTool._backend_cooldown == set()
        assert WebSearchTool._backend_health == {}

    def test_only_active_and_not_cooldown_are_ordered(self, monkeypatch):
        monkeypatch.setenv("HUGINN_BING_API_KEY", "k")
        monkeypatch.setenv("HUGINN_BRAVE_API_KEY", "k")
        monkeypatch.setenv("HUGINN_SEARXNG_URL", "https://s.example")
        # 没有 tavily key → tavily 不在活跃列表; bing/brave/searxng/arxiv/ddgs 在
        active = WebSearchTool._ordered_active_backends()
        assert "tavily" not in active
        for name in ("bing", "brave", "searxng", "arxiv", "ddgs"):
            assert name in active, f"{name} should be active"


class TestNewBackendsParsing:
    def test_bing_parses_webpages(self, monkeypatch):
        monkeypatch.setenv("HUGINN_BING_API_KEY", "k")
        payload = {
            "webPages": {
                "value": [
                    {"name": "B1", "url": "https://b.example/1",
                     "snippet": "one"},
                    {"name": "B2", "url": "https://b.example/2",
                     "snippet": "two"},
                ]
            }
        }
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _FakeResp(json.dumps(payload))
        )
        tool = WebSearchTool()
        res = tool._search_bing("q", 5)
        assert res is not None
        assert res.data["search_engine"] == "bing"
        assert res.data["results"][0]["title"] == "B1"
        assert res.data["results"][0]["url"] == "https://b.example/1"
        assert res.data["results"][1]["snippet"] == "two"

    def test_bing_without_key_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network"))
        )
        tool = WebSearchTool()
        assert tool._search_bing("q", 5) is None

    def test_brave_parses_web_results(self, monkeypatch):
        monkeypatch.setenv("HUGINN_BRAVE_API_KEY", "k")
        payload = {
            "web": {
                "results": [
                    {"title": "Br", "url": "https://brave.example/x",
                     "description": "desc"}
                ]
            }
        }
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _FakeResp(json.dumps(payload))
        )
        tool = WebSearchTool()
        res = tool._search_brave("q", 5)
        assert res is not None and res.data["search_engine"] == "brave"
        assert res.data["results"][0]["title"] == "Br"
        assert res.data["results"][0]["snippet"] == "desc"

    def test_searxng_parses_results(self, monkeypatch):
        monkeypatch.setenv("HUGINN_SEARXNG_URL", "https://my.searxng.local")
        payload = {"results": [{"title": "Sx", "url": "https://sx.example/y",
                                "content": "c"}]}
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _FakeResp(json.dumps(payload))
        )
        tool = WebSearchTool()
        res = tool._search_searxng("q", 5)
        assert res is not None and res.data["search_engine"] == "searxng"
        assert res.data["results"][0]["title"] == "Sx"
        assert res.data["results"][0]["snippet"] == "c"

    def test_searxng_without_url_returns_none(self):
        tool = WebSearchTool()
        assert tool._search_searxng("q", 5) is None
