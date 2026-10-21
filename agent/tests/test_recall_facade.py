"""P3 召回门面契约门 (红蓝双轨).

蓝队: 一次 ``recall`` 调用按意图确定性路由到 memory / context / observation,
      三个后端各自被正确传参, 统一返回结构 (intent/source/available/count/results).
红队: 后端缺失或失败必须 available=False 并带 error, 绝不伪装成"查到了空";
      未知 mode 不得静默降级; 无 memory_manager 时 context/observation 仍可达.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from huginn.tools.memory_tool import RecallInput, RecallTool
from huginn.tools.recall_context_tool import recall_context
from huginn.tools.recall_facade import (
    AUTO,
    INTENT_CONTEXT,
    INTENT_MEMORY,
    INTENT_OBSERVATION,
    detect_intent,
    route_recall,
)


class FakeMemory:
    def __init__(self, results=None, raises=None):
        self.results = results if results is not None else [{"content": "old"}]
        self.raises = raises
        self.calls: list[dict] = []

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self.results


class FakeArchive:
    calls: list[tuple] = []

    def __init__(self, session_id):
        self.session_id = session_id

    def recall(self, handle, *, page=0, page_size=4000):
        FakeArchive.calls.append((handle, page, page_size))
        return {
            "found": True,
            "handle": handle,
            "page": page,
            "page_size": page_size,
            "text": "archived body",
            "more": False,
        }


# ── 意图判定 ────────────────────────────────────────────────


class TestIntentDetection:
    def test_blue_handle_beats_category(self):
        assert detect_intent(handle="@obs:abc", category="failure") == INTENT_OBSERVATION

    def test_blue_category_selects_context(self):
        assert detect_intent(category="failure") == INTENT_CONTEXT

    def test_blue_free_text_defaults_to_memory(self):
        assert detect_intent() == INTENT_MEMORY

    def test_blue_explicit_mode_overrides_heuristics(self):
        assert detect_intent(handle="@obs:abc", mode="context") == INTENT_CONTEXT
        assert detect_intent(category="failure", mode=AUTO) == INTENT_CONTEXT

    def test_red_unknown_mode_does_not_silently_fall_back(self):
        with pytest.raises(ValueError, match="unknown recall mode"):
            detect_intent(mode="semantic-ish")


# ── 路由: memory ───────────────────────────────────────────


class TestMemoryRoute:
    def test_blue_memory_route_passes_args_and_clamps_top_k(self):
        mem = FakeMemory()
        out = route_recall(query="q", top_k=999, tier="long", memory_manager=mem)
        assert out["intent"] == INTENT_MEMORY
        assert out["source"] == "memory_manager"
        assert out["available"] is True
        assert out["count"] == 1
        assert mem.calls[0] == {"query": "q", "category": None, "tier": "long", "top_k": 50}

    def test_red_missing_manager_is_unavailable_with_guidance(self):
        out = route_recall(query="q")
        assert out["available"] is False
        assert out["count"] == 0 and out["results"] == []
        assert "mode='context'" in out["error"]
        assert "handle='@obs:" in out["error"]

    def test_red_backend_exception_is_unavailable_not_empty(self):
        out = route_recall(query="q", memory_manager=FakeMemory(raises=RuntimeError("db locked")))
        assert out["available"] is False
        assert "db locked" in out["error"]


# ── 路由: context ──────────────────────────────────────────


class TestContextRoute:
    def test_blue_context_route_delegates_and_reports_count(self, monkeypatch):
        seen: list[dict] = []

        def fake_recall_context(category, query="", top_k=3, *, raise_on_error=False):
            seen.append(
                {"category": category, "query": query, "top_k": top_k,
                 "raise_on_error": raise_on_error}
            )
            return [{"content": "prior failure", "category": category}]

        monkeypatch.setattr(
            "huginn.metacog.audit_context.recall_context", fake_recall_context
        )
        out = route_recall(category="failure", query="gap", top_k=2)
        assert out["intent"] == INTENT_CONTEXT
        assert out["source"] == "metacog.recall_context"
        assert out["available"] is True and out["count"] == 1
        # 必须走 strict 模式, 否则"查询失败"会被后端吞成空 list
        assert seen[0]["raise_on_error"] is True
        assert seen[0]["category"] == "failure"

    def test_red_context_backend_failure_is_unavailable(self, monkeypatch):
        def boom(category, query="", top_k=3, *, raise_on_error=False):
            raise RuntimeError("memory db missing")

        monkeypatch.setattr("huginn.metacog.audit_context.recall_context", boom)
        out = route_recall(category="failure")
        assert out["available"] is False
        assert "memory db missing" in out["error"]

    def test_red_blank_category_is_unavailable(self):
        out = route_recall(category="   ", mode="context")
        assert out["available"] is False
        assert "requires a category" in out["error"]


# ── 路由: observation ─────────────────────────────────────


class TestObservationRoute:
    def test_blue_observation_route_pages_by_handle(self, monkeypatch):
        FakeArchive.calls.clear()
        monkeypatch.setattr(
            "huginn.tools.observation_pack.ObservationArchive", FakeArchive
        )
        out = route_recall(handle="@obs:11111111-2222-3333-4444-555555555555",
                           page=2, page_size=100)
        assert out["intent"] == INTENT_OBSERVATION
        assert out["source"] == "observation_pack"
        assert out["available"] is True and out["found"] is True
        assert FakeArchive.calls[0][1:] == (2, 100)

    def test_red_missing_or_unfound_handle_is_unavailable(self, monkeypatch):
        class MissingArchive:
            def __init__(self, session_id):
                pass

            def recall(self, handle, *, page=0, page_size=4000):
                return {"found": False, "error": "observation not found"}

        monkeypatch.setattr(
            "huginn.tools.observation_pack.ObservationArchive", MissingArchive
        )
        out = route_recall(mode="observation")
        assert out["available"] is False
        assert "requires a handle" in out["error"]

        out = route_recall(handle="@obs:deadbeef")
        assert out["available"] is False
        assert out["found"] is False
        assert "not found" in out["error"]


# ── RecallTool 接线 ────────────────────────────────────────


class TestRecallToolWiring:
    async def test_blue_recall_tool_routes_memory(self):
        mem = FakeMemory(results=[{"content": "old"}])
        tool = RecallTool()
        result = await tool.call(
            RecallInput(query="C-S-H", top_k=10, tier="long"),
            context=SimpleNamespace(session_id="t", memory_manager=mem),
        )
        assert result.success is True
        assert result.data["results"] == [{"content": "old"}]
        assert result.data["intent"] == INTENT_MEMORY
        assert result.data["count"] == 1

    async def test_blue_recall_tool_reaches_context_without_manager(self, monkeypatch):
        monkeypatch.setattr(
            "huginn.metacog.audit_context.recall_context",
            lambda category, query="", top_k=3, *, raise_on_error=False: [
                {"content": "ctx", "category": category}
            ],
        )
        tool = RecallTool()
        result = await tool.call(
            RecallInput(category="failure"),
            context=SimpleNamespace(session_id="t", memory_manager=None),
        )
        assert result.success is True
        assert result.data["intent"] == INTENT_CONTEXT
        assert result.data["results"][0]["content"] == "ctx"

    async def test_red_recall_tool_is_honest_when_nothing_available(self):
        tool = RecallTool()
        result = await tool.call(
            RecallInput(query="anything"),
            context=SimpleNamespace(session_id="t", memory_manager=None),
        )
        assert result.success is False
        assert result.error
        assert result.data["available"] is False
        assert result.data["count"] == 0


# ── 窄门委托 (消除重复路由逻辑) ─────────────────────────────


def test_blue_recall_context_shim_delegates_to_facade(monkeypatch):
    seen: list[dict] = []

    def fake_route_recall(**kwargs):
        seen.append(kwargs)
        return {
            "intent": INTENT_CONTEXT,
            "source": "metacog.recall_context",
            "available": True,
            "count": 1,
            "results": [{"content": "x"}],
            "error": None,
        }

    monkeypatch.setattr("huginn.tools.recall_facade.route_recall", fake_route_recall)
    out = recall_context(category="failure", query="q", top_k=2)
    assert seen[0]["mode"] == "context"
    assert out["category"] == "failure"
    assert out["count"] == 1
    assert out["available"] is True
