"""统一事件总线 — 4 套事件系统的单一入口.

之前 huginn 有 5 套独立事件系统互不互通:
  1. HookManager  (hooks/__init__.py, 10 个字符串常量)
  2. 内部 EventBus (events/event_bus.py, 19 个 dotted 字符串)
  3. 插件 EventBus (plugins/event_bus.py, 17 个 EventType enum)
  4. PetBus       (pet/__init__.py, PetMood 枚举)
  5. CSMListener  (cognitive_engine.py, Protocol, 声明了但零注册)

v23 Round 9: CSMListener 已删除 (零注册死代码), CSM 状态转移通过
UnifiedBus 的 'cognitive.csm.transition' 事件发布. 现剩 4 套系统.

streaming.py 主循环里 30+ 处显式三发/四发到不同总线, 维护成本高且容易遗漏.

UnifiedBus 收敛为单一 publish 入口, 内部桥接到各子系统:

    bus = UnifiedBus(agent)
    await bus.publish_session_start(thread_id, message)
    await bus.publish_session_end(thread_id, turn_count)
    await bus.publish_tool_pre(tool_name, args, thread_id)
    await bus.publish_tool_post(tool_name, args, result, error, duration_ms, thread_id)
    await bus.publish_pet_mood(mood, detail, metadata)
    await bus.publish_csm_transition(old_state, new_state, signal)

各子系统仍可独立使用 (测试/调试), 但生产代码应走 UnifiedBus.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.hooks import HookManager
    from huginn.pet import PetMood

logger = logging.getLogger(__name__)


class UnifiedBus:
    """统一事件总线 — 桥接 HookManager / EventBus / PluginBus / PetBus.

    不是新总线, 是现有 5 套系统的桥接层. 每次 publish 内部扇出到
    各子系统, 但调用方只需一次调用.

    使用方式:
        bus = UnifiedBus(agent)
        await bus.publish_session_start(thread_id, message)

    或者直接用模块级便捷函数:
        from huginn.events.unified_bus import publish_session_start
        await publish_session_start(agent, thread_id, message)
    """

    def __init__(self, agent: Any = None) -> None:
        """绑定 agent 以访问其 hook_manager / pet 等组件."""
        self._agent = agent
        self._hook_manager: HookManager | None = getattr(agent, "hook_manager", None) if agent else None

    # ── 内部桥接方法 (不对外) ──────────────────────────────────────

    async def _trigger_hook(self, event: str, ctx: Any) -> None:
        """触发 HookManager 事件 (best-effort)."""
        if self._hook_manager is None:
            return
        try:
            await self._hook_manager.trigger(event, ctx)
        except Exception:
            logger.warning("%s hook raised", event, exc_info=True)

    async def _publish_internal(self, event_type: str, data: dict, thread_id: str = "", source: str = "") -> None:
        """发布到内部 EventBus (events/event_bus.py, best-effort)."""
        try:
            from huginn.events.event_bus import AgentEvent, EventBus
            await EventBus.shared().publish(AgentEvent(
                type=event_type,
                timestamp=time.time(),
                data=data,
                thread_id=thread_id,
                source=source,
            ))
        except Exception:
            logger.debug("internal event publish failed: %s", event_type, exc_info=True)

    def _publish_internal_sync(self, event_type: str, data: dict, thread_id: str = "", source: str = "") -> None:
        """同步发布到内部 EventBus (best-effort)."""
        try:
            from huginn.events.integration import _publish, _schedule_sync
            _schedule_sync(_publish(event_type, data, thread_id, source))
        except Exception:
            logger.debug("internal sync event publish failed: %s", event_type, exc_info=True)

    async def _dispatch_plugin(self, event_type: Any, data: dict, thread_id: str = "") -> None:
        """分发到插件 EventBus (plugins/event_bus.py, best-effort)."""
        try:
            from huginn.api.event import Event
            from huginn.plugins.event_bus import EventBus as PluginBus
            bus = PluginBus()
            await bus.dispatch(Event(
                type=event_type,
                plugin_name="",
                data={"thread_id": thread_id, **data},
            ))
        except Exception:
            logger.debug("plugin dispatch failed for %s (non-fatal)", event_type, exc_info=True)

    def _publish_pet(self, mood: Any, detail: str = "", metadata: dict | None = None) -> None:
        """发布到 PetBus (best-effort)."""
        try:
            from huginn.pet import get_pet_bus
            pet = get_pet_bus()
            pet.publish(mood, detail, metadata or {})
        except Exception:
            logger.debug("pet publish failed (non-fatal)", exc_info=True)

    # ── 统一发布接口 ──────────────────────────────────────────────

    async def publish_session_start(
        self,
        thread_id: str,
        user_message: str = "",
    ) -> None:
        """会话开始 — 统一发布到 3 套事件系统.

        替代 streaming.py 里的三段式:
          1. hook_manager.trigger(SESSION_START, ctx)
          2. publish_session_event_sync("start", ...)
          3. _PluginBus().dispatch(ON_AGENT_BEGIN, ...)
        """
        from huginn.hooks import SESSION_START, HookContext

        # 1. HookManager
        ctx = HookContext(
            tool_name="session",
            metadata={"thread_id": thread_id, "user_message": user_message},
        )
        await self._trigger_hook(SESSION_START, ctx)

        # 2. 内部 EventBus
        self._publish_internal_sync(
            "session.start",
            {"user_message": user_message},
            thread_id=thread_id,
            source="session",
        )

        # 3. 插件 EventBus
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_AGENT_BEGIN,
            {"user_message": user_message[:200]},
            thread_id=thread_id,
        )

    async def publish_session_end(
        self,
        thread_id: str,
        turn_count: int = 0,
    ) -> None:
        """会话结束 — 统一发布到 3 套事件系统."""
        from huginn.hooks import SESSION_END, HookContext

        # 1. HookManager
        ctx = HookContext(
            tool_name="session",
            metadata={"thread_id": thread_id, "turn_count": turn_count},
        )
        await self._trigger_hook(SESSION_END, ctx)

        # 2. 内部 EventBus
        self._publish_internal_sync(
            "session.end",
            {"turn_count": turn_count},
            thread_id=thread_id,
            source="session",
        )

        # 3. 插件 EventBus
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_AGENT_DONE,
            {"turn_count": turn_count},
            thread_id=thread_id,
        )

    async def publish_stop(
        self,
        thread_id: str,
        workspace: str = "",
    ) -> None:
        """Agent 完成一轮回复 — 触发 STOP hook."""
        from huginn.hooks import STOP, HookContext

        ctx = HookContext(
            tool_name="agent_turn",
            metadata={"thread_id": thread_id, "workspace": workspace},
        )
        await self._trigger_hook(STOP, ctx)

    async def publish_tool_pre(
        self,
        tool_name: str,
        args: Any,
        thread_id: str = "",
    ) -> tuple[bool, Any, Any]:
        """工具调用前 — 触发 PRE_TOOL_USE hook, 返回 (allowed, args, ctx)."""

        if self._hook_manager is None:
            return True, args, None
        try:
            return await self._hook_manager.run_pre(tool_name, args, thread_id)
        except Exception:
            logger.warning("pre_tool_use hook raised for %s", tool_name, exc_info=True)
            return True, args, None

    async def publish_tool_post(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        error: BaseException | None,
        duration_ms: float,
        thread_id: str = "",
        user_message: str | None = None,
    ) -> Any:
        """工具调用后 — 触发 POST_TOOL_USE hook + 内部事件总线."""
        from huginn.events.integration import publish_tool_event_sync

        # 1. 内部 EventBus (tool.call + tool.result/error)
        publish_tool_event_sync(
            tool_name, args, result, thread_id,
            error=str(error) if error else None,
        )

        # 2. 插件 EventBus
        from huginn.api.event import EventType
        if error is not None:
            await self._dispatch_plugin(
                EventType.ON_TOOL_CALL,
                {"tool": tool_name, "error": str(error)},
                thread_id=thread_id,
            )
        else:
            await self._dispatch_plugin(
                EventType.ON_TOOL_RESPOND,
                {"tool": tool_name, "result": str(result)[:500]},
                thread_id=thread_id,
            )

        # 3. HookManager (POST_TOOL_USE)
        if self._hook_manager is None:
            return None
        try:
            return await self._hook_manager.run_post(
                tool_name, args, result, error, duration_ms, thread_id, user_message,
            )
        except Exception:
            logger.warning("post_tool_use hook raised for %s", tool_name, exc_info=True)
            return None

    async def publish_llm_request(self, thread_id: str, messages_count: int = 0) -> None:
        """LLM 请求前 — 分发到插件 EventBus."""
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_LLM_REQUEST,
            {"messages_count": messages_count},
            thread_id=thread_id,
        )

    async def publish_llm_response(self, thread_id: str, response_preview: str = "") -> None:
        """LLM 响应后 — 分发到插件 EventBus + 内部事件总线."""
        from huginn.api.event import EventType

        await self._dispatch_plugin(
            EventType.ON_LLM_RESPONSE,
            {"response_preview": response_preview[:200]},
            thread_id=thread_id,
        )
        self._publish_internal_sync(
            "llm.response",
            {"response_preview": response_preview[:200]},
            thread_id=thread_id,
            source="agent",
        )

    async def publish_message_received(self, thread_id: str, message: str = "") -> None:
        """收到用户消息 — 分发到插件 EventBus."""
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_MESSAGE_RECEIVED,
            {"message": message[:200]},
            thread_id=thread_id,
        )

    async def publish_before_message_sent(self, thread_id: str, messages_count: int = 0) -> None:
        """消息发送前 — 分发到插件 EventBus."""
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_BEFORE_MESSAGE_SENT,
            {"messages_count": messages_count},
            thread_id=thread_id,
        )

    async def publish_after_message_sent(self, thread_id: str, response: str = "") -> None:
        """消息发送后 — 分发到插件 EventBus."""
        from huginn.api.event import EventType
        await self._dispatch_plugin(
            EventType.ON_AFTER_MESSAGE_SENT,
            {"response": response[:200]},
            thread_id=thread_id,
        )

    async def publish_pet_mood(
        self,
        mood: PetMood,
        detail: str = "",
        metadata: dict | None = None,
    ) -> None:
        """宠物心情变化 — 发布到 PetBus + 内部事件总线."""
        self._publish_pet(mood, detail, metadata)
        self._publish_internal_sync(
            "pet.mood",
            {"mood": mood.value if hasattr(mood, "value") else str(mood), "detail": detail},
            thread_id=(metadata or {}).get("thread_id", ""),
            source="pet",
        )

    async def publish_csm_transition(
        self,
        old_state: str,
        new_state: str,
        signal: str = "",
        thread_id: str = "",
    ) -> None:
        """CSM 状态转移 — 发布到内部 EventBus (v23: CSMListener 已删除)."""
        self._publish_internal_sync(
            "cognitive.csm.transition",
            {"old_state": old_state, "new_state": new_state, "signal": signal},
            thread_id=thread_id,
            source="csm",
        )

    async def publish_compact(
        self,
        before_pct: float,
        after_pct: float,
        thread_id: str = "",
    ) -> None:
        """上下文压缩完成 — 发布到内部 EventBus + 触发 POST_COMPACT hook.

        PRE_COMPACT hook 在压缩前由调用方自行触发 (时序要求不同).
        """
        from huginn.events.integration import publish_compact_event_sync
        from huginn.hooks import POST_COMPACT, HookContext

        # 1. 内部 EventBus (compact.start + compact.end)
        publish_compact_event_sync(before_pct, after_pct, thread_id)

        # 2. HookManager: POST_COMPACT
        post_ctx = HookContext(
            tool_name="compact",
            metadata={"thread_id": thread_id, "before_pct": before_pct, "after_pct": after_pct},
        )
        await self._trigger_hook(POST_COMPACT, post_ctx)

    async def publish_step_retry(
        self,
        thread_id: str,
        attempt: int,
        max_attempts: int,
        error_type: str = "",
        error_message: str = "",
        wait_ms: int = 0,
        states_yielded: int = 0,
    ) -> None:
        """Agent 一步重试 — 发布到内部 EventBus (STEP_RETRY).

        统一入口, 替代 streaming.py 里直发 ``EventBus.shared().publish(STEP_RETRY)``,
        让 retry 信号与其它事件走同一 publish 通道, 对 SSE / audit / transcript 可见.
        """
        from huginn.events.event_types import STEP_RETRY

        self._publish_internal_sync(
            STEP_RETRY,
            {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error_type": error_type,
                "error_message": error_message[:200],
                "wait_ms": wait_ms,
                "states_yielded": states_yielded,
            },
            thread_id=thread_id,
            source="agent.streaming",
        )


# ── 模块级便捷函数 ──────────────────────────────────────────────

def get_unified_bus(agent: Any = None) -> UnifiedBus:
    """获取 UnifiedBus 实例.

    agent 参数可选 — 如果传了 agent, 会从其获取 hook_manager.
    不传则只桥接 EventBus + PluginBus + PetBus (无 hook).
    """
    return UnifiedBus(agent)


__all__ = ["UnifiedBus", "get_unified_bus"]
