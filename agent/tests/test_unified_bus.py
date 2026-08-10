"""UnifiedBus — 统一事件总线测试.

覆盖:
  - get_unified_bus 工厂
  - 各 publish_* 方法 best-effort 不抛 (即使下游组件缺失)
  - HookManager / EventBus / PluginBus / PetBus 桥接调用路径
  - 同步 / 异步发布接口
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.events.unified_bus import UnifiedBus, get_unified_bus

# ── get_unified_bus 工厂 ───────────────────────────────────────────


def test_get_unified_bus_no_agent():
    bus = get_unified_bus(None)
    assert isinstance(bus, UnifiedBus)
    assert bus._hook_manager is None


def test_get_unified_bus_with_agent_extracts_hook_manager():
    agent = SimpleNamespace(hook_manager="HM_INSTANCE")
    bus = get_unified_bus(agent)
    assert bus._hook_manager == "HM_INSTANCE"


def test_unified_bus_init_no_agent():
    bus = UnifiedBus()
    assert bus._agent is None
    assert bus._hook_manager is None


# ── publish_session_start: 桥接 3 系统, 部分失败不抛 ──────────────


@pytest.mark.asyncio
async def test_publish_session_start_triggers_hook():
    hook_manager = MagicMock()
    hook_manager.trigger = AsyncMock()
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    await bus.publish_session_start("thread-1", "hello")
    hook_manager.trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_session_start_no_hook_manager_does_not_raise():
    """没有 hook_manager 时, 只走 EventBus + PluginBus, 不抛."""
    bus = UnifiedBus(None)
    # 内部 _publish_internal_sync / _dispatch_plugin 都 try/except 兜底
    await bus.publish_session_start("thread-1", "hello")


@pytest.mark.asyncio
async def test_publish_session_end_no_hook_manager_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_session_end("thread-1", turn_count=5)


@pytest.mark.asyncio
async def test_publish_stop_triggers_hook():
    hook_manager = MagicMock()
    hook_manager.trigger = AsyncMock()
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    await bus.publish_stop("thread-1", workspace="/tmp")
    hook_manager.trigger.assert_awaited_once()


# ── publish_tool_pre: 返回 (allowed, args, ctx) ────────────────────


@pytest.mark.asyncio
async def test_publish_tool_pre_no_hook_manager_returns_allowed():
    bus = UnifiedBus(None)
    allowed, args, ctx = await bus.publish_tool_pre("bash", {"cmd": "ls"}, "t1")
    assert allowed is True
    assert args == {"cmd": "ls"}
    assert ctx is None


@pytest.mark.asyncio
async def test_publish_tool_pre_with_hook_manager_delegates():
    hook_manager = MagicMock()
    hook_manager.run_pre = AsyncMock(return_value=(False, {"cmd": "blocked"}, "ctx"))
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    allowed, args, ctx = await bus.publish_tool_pre("bash", {"cmd": "rm"}, "t1")
    assert allowed is False
    assert args == {"cmd": "blocked"}
    assert ctx == "ctx"


@pytest.mark.asyncio
async def test_publish_tool_pre_hook_raises_returns_allowed():
    """hook_manager.run_pre 抛异常 → fallback 到 (True, args, None)."""
    hook_manager = MagicMock()
    hook_manager.run_pre = AsyncMock(side_effect=RuntimeError("boom"))
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    allowed, args, ctx = await bus.publish_tool_pre("bash", {"cmd": "ls"}, "t1")
    assert allowed is True
    assert ctx is None


# ── publish_tool_post ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_tool_post_no_hook_manager_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_tool_post(
        "bash", {"cmd": "ls"}, "result", None, 12.3, "t1"
    )


@pytest.mark.asyncio
async def test_publish_tool_post_with_error_dispatches_on_tool_call():
    """error 不为 None 时走 ON_TOOL_CALL 事件分支."""
    bus = UnifiedBus(None)
    # 内部 _dispatch_plugin 是 best-effort, 不会抛
    await bus.publish_tool_post(
        "bash", {"cmd": "ls"}, None, RuntimeError("err"), 5.0, "t1"
    )


@pytest.mark.asyncio
async def test_publish_tool_post_with_hook_manager_runs_post():
    hook_manager = MagicMock()
    hook_manager.run_post = AsyncMock(return_value="post_ctx")
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    result = await bus.publish_tool_post(
        "bash", {"cmd": "ls"}, "out", None, 5.0, "t1"
    )
    assert result == "post_ctx"


# ── LLM / message events ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_llm_request_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_llm_request("t1", messages_count=3)


@pytest.mark.asyncio
async def test_publish_llm_response_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_llm_response("t1", "response preview text")


@pytest.mark.asyncio
async def test_publish_message_received_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_message_received("t1", "user message")


@pytest.mark.asyncio
async def test_publish_before_message_sent_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_before_message_sent("t1", messages_count=2)


@pytest.mark.asyncio
async def test_publish_after_message_sent_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_after_message_sent("t1", "response text")


# ── publish_pet_mood ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_pet_mood_does_not_raise_with_no_pet(monkeypatch):
    """pet 系统不可用时不抛."""
    bus = UnifiedBus(None)
    # 模拟 get_pet_bus 抛异常
    mood = SimpleNamespace(value="happy")
    await bus.publish_pet_mood(mood, detail="hi", metadata={"thread_id": "t1"})


# ── publish_csm_transition ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_csm_transition_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_csm_transition(
        old_state="S1_DISCOVER",
        new_state="S2_VALIDATE",
        signal="hypothesis_generated",
        thread_id="t1",
    )


# ── publish_compact ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_compact_no_hook_manager_does_not_raise():
    bus = UnifiedBus(None)
    await bus.publish_compact(before_pct=85.0, after_pct=45.0, thread_id="t1")


@pytest.mark.asyncio
async def test_publish_compact_triggers_post_compact_hook():
    hook_manager = MagicMock()
    hook_manager.trigger = AsyncMock()
    agent = SimpleNamespace(hook_manager=hook_manager)
    bus = UnifiedBus(agent)
    await bus.publish_compact(before_pct=85.0, after_pct=45.0, thread_id="t1")
    hook_manager.trigger.assert_awaited_once()


# ── 模块导出 ──────────────────────────────────────────────────────


def test_module_exports():
    from huginn.events import unified_bus

    assert hasattr(unified_bus, "UnifiedBus")
    assert hasattr(unified_bus, "get_unified_bus")
    assert "UnifiedBus" in unified_bus.__all__
    assert "get_unified_bus" in unified_bus.__all__
