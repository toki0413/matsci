"""Tests for LingBot-World inspired improvements.

LW-1: Director/Pilot dual-layer split
LW-2: Proactive pipeline event injection
LW-3: Value-aware context retention
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── LW-1: Director / Pilot ─────────────────────────────────────


class TestDirective:
    def test_directive_fields(self):
        from huginn.agents.director import Directive

        d = Directive(
            objective="Calculate band structure",
            tool_hint="vasp_tool",
            parameters={"action": "run", "icharg": 11},
            rationale="SCF converged, ready for band calculation",
            expected_outcome="Band gap near 1.1 eV for Si",
            hypothesis_id="h_abc123",
        )
        assert d.objective == "Calculate band structure"
        assert d.tool_hint == "vasp_tool"
        assert d.hypothesis_id == "h_abc123"

    def test_to_plan_dict(self):
        from huginn.agents.director import Directive

        d = Directive(objective="Optimize structure", tool_hint="vasp_tool")
        plan = d.to_plan_dict()
        assert plan["mode"] == "vasp_tool"
        assert plan["description"] == "Optimize structure"
        assert "rationale" in plan


class TestDirectorAgent:
    def test_propose_returns_none_when_no_data(self):
        from huginn.agents.director import DirectorAgent

        director = DirectorAgent(model=None)
        # No pipeline, no provenance, no model → None
        with patch.object(director, "_read_pipeline_state", return_value=[]):
            with patch.object(director, "_read_provenance", return_value=[]):
                with patch.object(director, "_read_hypothesis_graph", return_value={}):
                    result = director.propose()
                    assert result is None

    def test_propose_from_pipeline_hints(self):
        from huginn.agents.director import DirectorAgent, Directive

        director = DirectorAgent(model=None)
        hints = [{
            "tool_hint": "vasp_tool",
            "description": "Run SCF calculation",
            "parameters": {"encut": 520},
            "reason": "Structure ready for SCF",
            "stage": "scf",
            "prerequisite_met": True,
        }]
        with patch.object(director, "_read_pipeline_state", return_value=hints):
            with patch.object(director, "_read_provenance", return_value=[]):
                result = director.propose()
                assert result is not None
                assert isinstance(result, Directive)
                assert result.tool_hint == "vasp_tool"
                assert result.objective == "Run SCF calculation"

    def test_propose_from_provenance_fallback(self):
        from huginn.agents.director import DirectorAgent

        director = DirectorAgent(model=None)
        prov = [{"produced_by": "structure_tool", "file_format": "cif"}]
        with patch.object(director, "_read_pipeline_state", return_value=[]):
            with patch.object(director, "_read_provenance", return_value=prov):
                result = director.propose()
                assert result is not None
                assert "structure_tool" in result.objective


class TestPilotAgent:
    def test_execute_no_adapter(self):
        from huginn.agents.director import PilotAgent, Directive

        pilot = PilotAgent(tool_adapter=None)
        directive = Directive(objective="test", tool_hint="explore")
        result = asyncio.run(pilot.execute(directive))
        assert not result.success
        assert "no tool adapter" in result.error

    def test_execute_success(self):
        from huginn.agents.director import PilotAgent, Directive

        async def _mock_call(**kw):
            return {"result": {"energy": -10.5, "converged": True}}

        mock_adapter = MagicMock()
        mock_adapter.call = MagicMock(side_effect=_mock_call)
        pilot = PilotAgent(tool_adapter=mock_adapter)
        directive = Directive(
            objective="Run SCF",
            tool_hint="vasp_tool",
            parameters={"action": "run"},
        )
        result = asyncio.run(pilot.execute(directive))
        assert result.success
        assert result.key_properties.get("energy") == -10.5

    def test_execution_count_increments(self):
        from huginn.agents.director import PilotAgent, Directive

        async def _mock_call(**kw):
            return {"result": {}}
        mock_adapter = MagicMock()
        mock_adapter.call = MagicMock(side_effect=_mock_call)
        pilot = PilotAgent(tool_adapter=mock_adapter)
        directive = Directive(objective="test")
        asyncio.run(pilot.execute(directive))
        asyncio.run(pilot.execute(directive))
        assert pilot._execution_count == 2


class TestEngineDirectorPilotIntegration:
    def test_run_accepts_use_director_pilot(self):
        """AutoloopEngine.run() should accept use_director_pilot parameter."""
        import inspect
        from huginn.autoloop.engine import AutoloopEngine

        sig = inspect.signature(AutoloopEngine.run)
        assert "use_director_pilot" in sig.parameters
        assert sig.parameters["use_director_pilot"].default is False


# ── LW-2: Proactive Suggestion ─────────────────────────────────


class TestProactiveSuggestion:
    def test_method_exists(self):
        from huginn.agent import HuginnAgent

        assert hasattr(HuginnAgent, "_maybe_inject_proactive_suggestion")

    async def test_no_crash_empty_pipeline(self):
        """Method should not crash when pipeline has no suggestions."""
        from huginn.agent import HuginnAgent

        agent = HuginnAgent.__new__(HuginnAgent)
        agent._pending_synthetic_messages = []

        with patch("huginn.provenance.pipeline.get_pipeline") as mock:
            mock.return_value._latest = []
            mock.return_value._latest_entry.return_value = None
            await agent._maybe_inject_proactive_suggestion()

        assert len(agent._pending_synthetic_messages) == 0

    async def test_injects_when_ready(self):
        """Should inject message when pipeline has ready suggestions."""
        from huginn.agent import HuginnAgent

        agent = HuginnAgent.__new__(HuginnAgent)
        agent._pending_synthetic_messages = []

        # Mock pipeline with a ready suggestion
        suggestion = MagicMock()
        suggestion.prerequisite_met = True
        suggestion.stage.value = "scf"
        suggestion.tool_hint = "vasp_tool"
        suggestion.description = "Run SCF calculation"

        with patch("huginn.provenance.pipeline.get_pipeline") as mock:
            mock.return_value._latest = [suggestion]
            await agent._maybe_inject_proactive_suggestion()

        assert len(agent._pending_synthetic_messages) == 1
        content = agent._pending_synthetic_messages[0].content
        assert "Pipeline Suggestion" in content
        assert "vasp_tool" in content


# ── LW-3: Value-Aware Context ───────────────────────────────────


class TestMessageValueScore:
    def test_energy_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("The total energy is -10.5 eV") > 0

    def test_bandgap_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("Band gap = 1.12 eV (indirect)") > 0

    def test_converged_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("SCF converged after 12 iterations") > 0

    def test_lattice_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("a = 5.43, b = 5.43, c = 5.43") > 0

    def test_debug_negative(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("debug: loading model checkpoint") < 0

    def test_traceback_negative(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("Traceback (most recent call last):") < 0

    def test_neutral_zero(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("Hello, how are you?") == 0

    def test_dict_message(self):
        from huginn.utils.context import _message_value_score

        msg = {"role": "assistant", "content": "The energy converged to -10.5 eV"}
        assert _message_value_score(msg) > 0

    def test_long_tool_message_bonus(self):
        from huginn.utils.context import _message_value_score

        long_content = "x" * 200  # > 100 chars
        msg = {"role": "tool", "content": long_content}
        assert _message_value_score(msg) > 0


class TestSummarizePreservesHighValue:
    async def test_high_value_messages_survive(self):
        """Messages with energy/bandgap should be preserved in keep_zone."""
        from huginn.utils.context import summarize_compact_messages

        messages = [
            {"role": "user", "content": "Start calculation"},
            {"role": "assistant", "content": "The energy is -10.5 eV and bandgap is 1.1 eV"},
            {"role": "assistant", "content": "Debug: loading modules"},
            {"role": "assistant", "content": "Debug: importing packages"},
            {"role": "user", "content": "What are the results?"},
        ]
        # Force compaction with small budget and keep_last_n
        result, _ = await summarize_compact_messages(
            messages, budget_tokens=100, keep_last_n=2, summarizer=None,
        )
        # The energy message should survive (not be dropped)
        all_content = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in result
        )
        assert "energy" in all_content.lower()

    def test_low_value_messages_summarized(self):
        """Debug messages should be prioritized for summarization."""
        from huginn.utils.context import _message_value_score

        debug_score = _message_value_score("debug: loading model")
        energy_score = _message_value_score("energy = -10.5 eV")
        assert debug_score < energy_score
