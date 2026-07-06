"""Tests for research mode behavioral differences.

Covers mode switching, system prompt enhancement, tool filtering,
and verification model gating. Tests use unbound-method binding to
avoid the heavyweight agent __init__ — we only need a few attributes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from huginn.agent import HuginnAgent


def _make_mock_tool(name: str) -> SimpleNamespace:
    """A fake LangChain-style tool with just a .name attribute."""
    return SimpleNamespace(name=name)


def _make_agent_stub(
    mode: str = "chat",
    tools: list | None = None,
    system_prompt: str = "You are a materials science agent.",
) -> SimpleNamespace:
    """Build a lightweight stand-in with the attributes the methods read.

    We bind the real HuginnAgent methods to this stub so we're testing
    the actual implementation, not a copy. SimpleNamespace is used
    instead of MagicMock so class-variable lookups (like
    _EXPENSIVE_TOOL_NAMES) fall through to HuginnAgent, not auto-mock.
    """
    stub = SimpleNamespace()

    # Core attributes the methods touch.
    stub._mode = mode
    stub.langchain_tools = tools or []
    stub.system_prompt = system_prompt
    stub.workspace = None  # _effective_system_prompt handles None gracefully
    # Copy the class variable so the unbound method finds it on the stub.
    stub._EXPENSIVE_TOOL_NAMES = HuginnAgent._EXPENSIVE_TOOL_NAMES

    # Phase manager — no filtering by default (tool_filter returns None).
    stub._phase_manager = MagicMock()
    stub._phase_manager.tool_filter.return_value = None
    stub._phase_manager.prompt_prefix.return_value = ""

    return stub


# ── set_mode / get_mode / is_research_mode ──────────────────────


class TestModeSwitching:
    def test_default_mode_is_chat(self):
        agent = _make_agent_stub()
        assert agent._mode == "chat"

    def test_set_mode_to_research(self):
        agent = _make_agent_stub(mode="chat")
        HuginnAgent.set_mode(agent, "research")
        assert agent._mode == "research"

    def test_set_mode_to_chat(self):
        agent = _make_agent_stub(mode="research")
        HuginnAgent.set_mode(agent, "chat")
        assert agent._mode == "chat"

    def test_set_mode_invalid_raises(self):
        agent = _make_agent_stub()
        with pytest.raises(ValueError, match="未知模式"):
            HuginnAgent.set_mode(agent, "bogus")

    def test_is_research_mode(self):
        agent = _make_agent_stub(mode="research")
        assert HuginnAgent.is_research_mode(agent) is True

    def test_is_not_research_mode_in_chat(self):
        agent = _make_agent_stub(mode="chat")
        assert HuginnAgent.is_research_mode(agent) is False


# ── Verification model gating ─────────────────────────────────────


class TestVerificationGating:
    def test_should_verify_in_research(self):
        agent = _make_agent_stub(mode="research")
        assert HuginnAgent.should_use_verification_model(agent) is True

    def test_should_not_verify_in_chat(self):
        agent = _make_agent_stub(mode="chat")
        assert HuginnAgent.should_use_verification_model(agent) is False


# ── System prompt enhancement ────────────────────────────────────


class TestSystemPrompt:
    def test_research_mode_prompt_contains_research_mode(self):
        agent = _make_agent_stub(mode="research")
        prompt = HuginnAgent._effective_system_prompt(agent)
        assert "RESEARCH MODE" in prompt
        assert "cite literature" in prompt.lower() or "cite" in prompt.lower()

    def test_chat_mode_prompt_has_no_research_prefix(self):
        agent = _make_agent_stub(mode="chat", system_prompt="Base prompt here.")
        prompt = HuginnAgent._effective_system_prompt(agent)
        assert "RESEARCH MODE" not in prompt
        # The base prompt should still be present
        assert "Base prompt here." in prompt

    def test_research_prompt_mentions_uncertainty(self):
        agent = _make_agent_stub(mode="research")
        prompt = HuginnAgent._effective_system_prompt(agent)
        assert "uncertainty" in prompt.lower()


# ── Tool filtering ───────────────────────────────────────────────


class TestToolFiltering:
    def _make_tools(self):
        return [
            _make_mock_tool("web_search_tool"),
            _make_mock_tool("vasp_tool"),
            _make_mock_tool("lammps_tool"),
            _make_mock_tool("cp2k_tool"),
            _make_mock_tool("transolver_tool"),
            _make_mock_tool("image_analysis_tool"),
        ]

    def test_chat_mode_filters_expensive_tools(self):
        tools = self._make_tools()
        agent = _make_agent_stub(mode="chat", tools=tools)
        effective = HuginnAgent._effective_tools(agent)
        names = {t.name for t in effective}
        assert "vasp_tool" not in names
        assert "lammps_tool" not in names
        assert "cp2k_tool" not in names
        assert "transolver_tool" not in names
        # Quick tools should still be there
        assert "web_search_tool" in names
        assert "image_analysis_tool" in names

    def test_research_mode_includes_all_tools(self):
        tools = self._make_tools()
        agent = _make_agent_stub(mode="research", tools=tools)
        effective = HuginnAgent._effective_tools(agent)
        names = {t.name for t in effective}
        assert "vasp_tool" in names
        assert "lammps_tool" in names
        assert "cp2k_tool" in names
        assert "transolver_tool" in names
        assert len(names) == len(tools)

    def test_chat_mode_tool_count_reduced(self):
        tools = self._make_tools()
        agent = _make_agent_stub(mode="chat", tools=tools)
        effective = HuginnAgent._effective_tools(agent)
        assert len(effective) == len(tools) - 4  # 4 expensive tools removed

    def test_research_mode_tool_count_unchanged(self):
        tools = self._make_tools()
        agent = _make_agent_stub(mode="research", tools=tools)
        effective = HuginnAgent._effective_tools(agent)
        assert len(effective) == len(tools)
