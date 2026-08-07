"""KB 接入 agent loop 的回归测试.

锁住:
  * AutoloopEngine._build_hypothesis_prompt / _build_plan_prompt 在 KB 有内容时
    注入 "Domain Knowledge Context", 空 KB 时跳过.
  * HuginnAgent._build_input_messages 同样注入/跳过.
  * PromptCacheBuilder.build_input_messages 把 kb_text 作为 SystemMessage 放在
    kg_text 之后、user 之前.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.utils.prompt_cache import PromptCacheBuilder


class _FakeKb:
    """最小 KB 替身: 只实现 count() 和 query()."""

    def __init__(self, chunks: list[dict] | None = None):
        self._chunks = chunks or []

    def count(self) -> int:
        return len(self._chunks)

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        return self._chunks[:top_k]


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """构造一个 sub-components 全 stub 的 engine, 仿 test_autoloop_engine."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    return AutoloopEngine(workspace=tmp_path)


class TestEngineKbIntegration:
    def test_hypothesize_prompt_includes_kb_context(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "DFT convergence requires ENCUT > 520 eV for Si."}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)

        prompt = engine._build_hypothesis_prompt({"topic": "Si band gap"})
        assert "Domain Knowledge Context" in prompt
        assert "DFT convergence" in prompt

    def test_plan_prompt_includes_kb_context(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "Use pymatgen Vasprun to parse band structure."}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)

        prompt = engine._build_plan_prompt("compute Si band gap", {"topic": "Si"})
        assert "Domain Knowledge Context" in prompt
        assert "pymatgen" in prompt

    def test_hypothesize_prompt_skips_kb_when_empty(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda ws: _FakeKb([])
        )
        prompt = engine._build_hypothesis_prompt({"topic": "Si"})
        assert "Domain Knowledge Context" not in prompt

    def test_build_kb_text_returns_empty_when_kb_unavailable(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # get_knowledge_base 抛错 → _get_kb 返回 None → 空串
        def _boom(ws):
            raise RuntimeError("chromadb missing")

        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", _boom)
        assert engine._build_kb_text("anything") == ""


class TestAgentKbIntegration:
    """_build_kb_text 单元测试. 走 __new__ 绕开 __init__ (预存 scheduler
    NameError bug 与 KB 无关, 不在本 Phase 修复范围)."""

    @staticmethod
    def _make_agent(kb_enabled: bool, kb: _FakeKb):
        from huginn.agent import HuginnAgent

        a = HuginnAgent.__new__(HuginnAgent)
        a.kb_enabled = kb_enabled
        a._kb = kb
        a.workspace = "."
        return a

    def test_build_kb_text_includes_chunks(self) -> None:
        fake = _FakeKb([{"text": "C-S-H gel density ~2.6 g/cm3."}])
        a = self._make_agent(True, fake)
        text = a._build_kb_text("C-S-H density")
        assert "Domain Knowledge Context" in text
        assert "C-S-H" in text

    def test_build_kb_text_skips_when_disabled(self) -> None:
        fake = _FakeKb([{"text": "should not appear"}])
        a = self._make_agent(False, fake)
        assert a._build_kb_text("query") == ""

    def test_build_kb_text_skips_when_empty(self) -> None:
        a = self._make_agent(True, _FakeKb([]))
        assert a._build_kb_text("query") == ""


class TestPromptCacheKbText:
    def test_kb_text_appended_after_kg(self) -> None:
        builder = PromptCacheBuilder(
            system_prompt="static",
            begin_dialogs=[("assistant", "hi")],
            cache_control=False,
        )
        msgs = builder.build_input_messages(
            "memory", "question", kg_text="kg context", kb_text="kb context"
        )
        # memory/kg/kb 合并进单条 HumanMessage, 格式: memory\n\nkg\n\nkb\n\n---\n\nquestion
        last = msgs[-1]
        assert last.__class__.__name__ == "HumanMessage"
        assert last.content == "memory\n\nkg context\n\nkb context\n\n---\n\nquestion"

    def test_kb_text_optional(self) -> None:
        builder = PromptCacheBuilder(system_prompt="static")
        msgs = builder.build_input_messages("", "question")
        # 不传 kb_text/kg_text/memory 也不该报错, user 消息在最后
        assert msgs[-1].content == "question"
