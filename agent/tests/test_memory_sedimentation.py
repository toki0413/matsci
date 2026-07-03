"""Tests for memory sedimentation — distillation, LLM insight, and maintainer.

Covers:
  - MemoryManager accepts / sets an optional LLM
  - _extract_llm_insight returns text with a mock LLM, None without
  - promote_session_summary triggers KnowledgeDistiller and stores results
  - promote_session_summary stores LLM-extracted insights
  - MemoryMaintainer run_once / start+stop lifecycle
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from huginn.evolution.knowledge_distiller import DistilledKnowledge
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.memory.maintainer import MemoryMaintainer
from huginn.types import ToolResult


# ── helpers ──────────────────────────────────────────────────────


class _MockLLMResponse:
    """Mimics the response object returned by langchain-style LLMs."""

    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Minimal LLM stub for testing insight extraction."""

    def __init__(self, response_text: str = "Insight: GaN bandgap is 3.4eV"):
        self._response = _MockLLMResponse(response_text)

    def invoke(self, prompt: str):
        return self._response


def _make_mock_distiller(knowledge_items=None):
    """Build a MagicMock distiller that returns canned knowledge."""
    mock = MagicMock()
    mock.distill_error_lessons.return_value = knowledge_items or []
    mock.distill_success_patterns.return_value = []
    mock.distill_tool_tips.return_value = []
    mock.knowledge_base = knowledge_items or []
    return mock


# ── LLM parameter tests ──────────────────────────────────────────


class TestMemoryManagerLLM:
    def test_memory_manager_accepts_llm(self, tmp_path):
        """MemoryManager(llm=mock_llm) stores _llm."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mock_llm = MockLLM()
        mgr = MemoryManager(longterm=mem, llm=mock_llm)
        assert mgr._llm is mock_llm

    def test_set_llm_method(self, tmp_path):
        """mm.set_llm(mock_llm) sets _llm."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mgr = MemoryManager(longterm=mem)
        assert mgr._llm is None
        mock_llm = MockLLM()
        mgr.set_llm(mock_llm)
        assert mgr._llm is mock_llm

    def test_extract_llm_insight(self, tmp_path):
        """With a mock LLM, _extract_llm_insight returns the insight text."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mock_llm = MockLLM("Insight: GaN bandgap is 3.4eV")
        mgr = MemoryManager(longterm=mem, llm=mock_llm)
        mgr.add_message("user", "What is the bandgap of GaN?")
        mgr.add_message("assistant", "The bandgap of GaN is approximately 3.4 eV.")
        insight = mgr._extract_llm_insight()
        assert insight is not None
        assert "GaN" in insight

    def test_extract_llm_insight_no_llm(self, tmp_path):
        """Without an LLM, _extract_llm_insight returns None."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mgr = MemoryManager(longterm=mem)
        mgr.add_message("user", "What is the bandgap of GaN?")
        assert mgr._extract_llm_insight() is None


# ── promote_session_summary tests ────────────────────────────────


class TestPromoteSessionSummary:
    def test_promote_session_summary_with_distillation(self, tmp_path):
        """promote_session_summary triggers distillation and stores results."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mgr = MemoryManager(longterm=mem)

        # Add a failed tool call so there's something to distill
        mgr.add_tool_call(
            "vasp_tool",
            {"action": "scf"},
            result=ToolResult(data=None, success=False, error="SCF convergence failed"),
        )

        mock_dk = DistilledKnowledge(
            knowledge_id="test_err_001",
            content="SCF convergence failures often indicate poor initial geometry",
            source_type="error_lesson",
            source_evidence=["test_session"],
            confidence=0.6,
            tags=["error", "lesson", "vasp"],
        )
        mock_distiller = _make_mock_distiller([mock_dk])

        with patch(
            "huginn.evolution.knowledge_distiller.KnowledgeDistiller",
            return_value=mock_distiller,
        ):
            mgr.promote_session_summary()

        # The distilled knowledge should have been ingested into long-term memory
        distilled = mem.list_by_category("distilled_knowledge")
        assert len(distilled) > 0
        assert "SCF convergence" in distilled[0]["content"]

    def test_promote_session_summary_with_llm_insight(self, tmp_path):
        """With a mock LLM, promote_session_summary stores the LLM insight."""
        mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
        mock_llm = MockLLM("Insight: GaN bandgap is 3.4eV")
        mgr = MemoryManager(longterm=mem, llm=mock_llm)

        mgr.add_message("user", "What is the bandgap of GaN?")
        mgr.add_message("assistant", "The bandgap of GaN is approximately 3.4 eV.")

        # Patch KnowledgeDistiller so it doesn't touch the filesystem
        mock_distiller = _make_mock_distiller([])
        with patch(
            "huginn.evolution.knowledge_distiller.KnowledgeDistiller",
            return_value=mock_distiller,
        ):
            mgr.promote_session_summary()

        # The LLM insight should be stored in long-term memory
        insights = mem.list_by_category("insight")
        llm_insights = [
            e for e in insights if "llm_extracted" in (e.get("tags") or "")
        ]
        assert len(llm_insights) > 0
        assert "GaN" in llm_insights[0]["content"]


# ── MemoryMaintainer tests ───────────────────────────────────────


class TestMemoryMaintainer:
    def test_maintainer_run_once(self):
        """run_once() calls maintenance on the memory manager."""
        mock_mm = MagicMock()
        mock_mm.maintenance.return_value = {
            "pruned": 2,
            "deduplicated": 1,
            "expired": 0,
        }
        maintainer = MemoryMaintainer(memory_manager=mock_mm)
        result = maintainer.run_once()
        assert mock_mm.maintenance.called
        assert result["pruned"] == 2
        assert result["deduplicated"] == 1

    def test_maintainer_start_stop(self):
        """Start and stop the maintainer thread cleanly."""
        mock_mm = MagicMock()
        mock_mm.maintenance.return_value = {
            "pruned": 0,
            "deduplicated": 0,
            "expired": 0,
        }
        maintainer = MemoryMaintainer(memory_manager=mock_mm, interval=0.05)
        maintainer.start()
        assert maintainer._thread is not None
        # Let the thread run at least one maintenance cycle
        time.sleep(0.3)
        maintainer.stop()
        assert mock_mm.maintenance.called
