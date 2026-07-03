"""AutoFixLoop ↔ EventBus 桥接 —— 事件驱动的工具失败自修复。

AstrBot 里 OnPluginErrorEvent 会触发错误恢复。本模块把 Huginn 的
AutoFixLoop 接到事件系统上: 工具调用失败时 (ON_PLUGIN_ERROR 或带工具
上下文的 ToolErrorEvent), AutoFixLoop 分析错误并给出修复后的参数。

链路:
  工具失败 → EventBus.dispatch(ToolErrorEvent) → AutoFixHandler
    → AutoFixLoop.apply_fix() → 命中规则 → dispatch ON_PLUGIN_ERROR (retry)

handler 以高优先级注册, 保证在其它错误处理之前先尝试自动修复。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.api.event import Event, EventType
from huginn.api.filter import StarHandlerMetadata
from huginn.execution.autofix import AutoFixLoop

logger = logging.getLogger("huginn.autofix_hook")

# 不往 EventType 枚举里加新类型 (避免改 api/event.py), 复用 ON_PLUGIN_ERROR,
# 用 ToolErrorEvent / ToolRetryEvent 这两个子类区分语义。


@dataclass
class ToolErrorEvent(Event):
    """工具执行失败事件。

    带够 AutoFixLoop 诊断 + 给修复参数所需的上下文。
    """

    # 复用 ON_PLUGIN_ERROR, 这类事件默认就归到这里
    type: EventType = EventType.ON_PLUGIN_ERROR
    tool_name: str = ""
    error_message: str = ""
    current_params: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)


@dataclass
class ToolRetryEvent(Event):
    """AutoFixLoop 给出修复参数后, 通知重试的事件。"""

    type: EventType = EventType.ON_PLUGIN_ERROR  # 复用, 专用类型更好但先不动枚举
    tool_name: str = ""
    fixed_params: dict[str, Any] = field(default_factory=dict)
    matched_patterns: list[str] = field(default_factory=list)
    attempt: int = 1  # 第几次重试


class AutoFixHandler:
    """把工具错误事件桥接到 AutoFixLoop 的事件 handler。

    监听工具错误事件, 跑 AutoFixLoop.apply_fix(), 命中就发一个 retry 事件。
    高优先级 (100) 注册, 抢在其它错误 handler 之前先尝试自动修复。

    用法:
        handler = AutoFixHandler(event_bus, autofix=AutoFixLoop())
        handler.register()  # 注册到事件系统
    """

    PRIORITY = 100  # 高优先级, 跑在其它错误 handler 之前
    MAX_RETRIES = 3  # 重试上限, 对齐 AstrBot 的 max_agent_step 思路

    def __init__(
        self,
        event_bus: Any,  # EventBus, 鸭子类型
        autofix: AutoFixLoop | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._autofix = autofix or AutoFixLoop()
        self._retry_counts: dict[str, int] = {}  # tool_name -> 已重试次数

    def register(self) -> None:
        """把自己注册到 StarHandlerRegistry。"""
        registry = self._event_bus.registry
        if registry is None:
            logger.warning("EventBus has no registry, AutoFixHandler not registered")
            return

        meta = StarHandlerMetadata(
            handler=self._handle_tool_error,
            event_type=EventType.ON_PLUGIN_ERROR,
            priority=self.PRIORITY,
            plugin_name="__autofix__",  # 内置, 不是真实插件
            permissions=[],
            name="autofix_on_tool_error",
        )
        registry.register(meta)
        logger.info("AutoFixHandler registered (priority=%d)", self.PRIORITY)

    async def _handle_tool_error(self, event: Event) -> None:
        """处理工具错误事件, 由 EventBus.dispatch() 调用。

        事件是 ToolErrorEvent (或带 tool_name) 时, 跑 AutoFixLoop.apply_fix(),
        命中修复规则就 dispatch 一个 retry 事件。
        """
        tool_name = getattr(event, "tool_name", "")
        error_message = getattr(event, "error_message", "")
        current_params = getattr(event, "current_params", {})

        if not tool_name or not error_message:
            return  # 不是工具错误, 跳过

        # 重试次数上限
        attempts = self._retry_counts.get(tool_name, 0)
        if attempts >= self.MAX_RETRIES:
            logger.warning(
                "AutoFixHandler: tool '%s' hit max retries (%d), giving up",
                tool_name, self.MAX_RETRIES,
            )
            return

        fixed = self._autofix.apply_fix(tool_name, error_message, current_params)
        if fixed is None:
            logger.debug("AutoFixLoop: no fix found for tool '%s'", tool_name)
            return

        # AutoFixLoop 把诊断信息塞进两个内部 key:
        #   __auto_fix                  -> 命中规则的描述 (str)
        #   __auto_fix_patterns_matched -> 命中模式计数 (int)
        # dispatch 前剥掉这些内部标记, 免得污染下游 retry 的参数。
        clean_params = {
            k: v for k, v in fixed.items()
            if not k.startswith("__auto_fix")
        }
        description = fixed.get("__auto_fix", "")
        matched = [description] if description else []

        self._retry_counts[tool_name] = attempts + 1

        retry_event = ToolRetryEvent(
            tool_name=tool_name,
            fixed_params=clean_params,
            matched_patterns=matched,
            attempt=attempts + 1,
        )
        await self._event_bus.dispatch(retry_event)
        logger.info(
            "AutoFixHandler: suggested retry for '%s' (attempt %d), patterns: %s",
            tool_name, attempts + 1, matched,
        )

    def reset_retries(self, tool_name: str | None = None) -> None:
        """重置某工具 (或全部工具) 的重试计数。"""
        if tool_name:
            self._retry_counts.pop(tool_name, None)
        else:
            self._retry_counts.clear()


__all__ = ["ToolErrorEvent", "ToolRetryEvent", "AutoFixHandler"]
