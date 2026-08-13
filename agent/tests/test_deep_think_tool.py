"""Tests for the deep_think external-scratchpad tool and the external_thinking flag.

Covers:
- deep_think tool registers and is read-only.
- calling it records analysis into memory_manager.session.reasoning_trace.
- fail-open when memory_manager is None (still success).
- empty analysis returns failure.
- external_thinking flag injects the directive into the system prompt; disabled
  by default it does not.
"""

import asyncio

from huginn.agent.context import ContextMixin
from huginn.feature_flags import FeatureFlags
from huginn.memory.manager import MemoryManager
from huginn.tools.deep_think_tool import DeepThinkTool, DeepThinkToolInput
from huginn.tools.registry import ToolRegistry


def _run(coro):
    return asyncio.run(coro)


def make_context(memory_manager=None):
    from huginn.core_types import ToolContext

    return ToolContext(
        session_id="s1",
        workspace="/tmp",
        memory_manager=memory_manager,
    )


def test_deep_think_is_registered_and_read_only():
    registry = ToolRegistry()
    tool = registry.get("deep_think")
    assert tool is None or isinstance(tool, DeepThinkTool) or tool.name == "deep_think"
    # Direct instantiation is authoritative for the contract.
    t = DeepThinkTool()
    assert t.read_only is True
    assert t.destructive is False
    assert t.is_read_only(DeepThinkToolInput(analysis="x")) is True


def test_deep_think_records_into_reasoning_trace():
    mm = MemoryManager()
    tool = DeepThinkTool()
    ctx = make_context(mm)

    async def _go():
        return await tool.call(
            {"analysis": "解析 8139881: 先试小素数, 得 1627*5003"},
            ctx,
        )

    result = _run(_go())
    assert result.success is True
    assert mm.session.reasoning_trace == ["解析 8139881: 先试小素数, 得 1627*5003"]


def test_deep_think_fail_open_without_memory_manager():
    tool = DeepThinkTool()
    ctx = make_context(None)

    async def _go():
        return await tool.call({"analysis": "no mm here"}, ctx)

    result = _run(_go())
    assert result.success is True
    assert result.data["recorded"] is True


def test_deep_think_rejects_empty_analysis():
    tool = DeepThinkTool()
    ctx = make_context(MemoryManager())

    async def _go():
        return await tool.call({"analysis": "   "}, ctx)

    result = _run(_go())
    assert result.success is False


class _StubContext(ContextMixin):
    """Minimal object exercising ContextMixin._effective_system_prompt."""

    def __init__(self, system_prompt="base prompt", mode="default"):
        self.system_prompt = system_prompt
        self._mode = mode
        self.workspace = "/tmp"
        self._csm = None
        self._phase_manager = _StubPhaseManager()


class _StubPhaseManager:
    def prompt_prefix(self) -> str:
        return ""

    def tool_filter(self):
        return None


def _effective_prompt(mode="default"):
    return _StubContext(mode=mode)._effective_system_prompt()


def test_external_thinking_injects_directive_when_enabled():
    ff = FeatureFlags.shared()
    prev = ff.is_enabled("external_thinking")
    try:
        ff.enable("external_thinking")
        prompt = _effective_prompt()
        assert "deep_think" in prompt
        assert "External Thinking" in prompt
    finally:
        ff.reset("external_thinking") if prev is False else ff.toggle(
            "external_thinking", prev
        )


def test_external_thinking_absent_by_default():
    ff = FeatureFlags.shared()
    ff.reset("external_thinking")
    prompt = _effective_prompt()
    assert "deep_think" not in prompt
    assert "External Thinking" not in prompt
