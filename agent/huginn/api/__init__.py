"""huginn.api —— 插件公共 API 门面。

参考 AstrBot 的 `astrbot/api/` 设计: 插件只依赖本包导出的稳定接口,
内部实现 (huginn/core, huginn/plugins 内部) 可以随便重构。

插件作者只需要:
    from huginn.api import Star, filter, EventType, PluginContext, Event

不要从 huginn.plugins.* 或 huginn.tools.* 直接 import —— 那些是内部 API,
后续版本可能改签名。需要新能力请走 filter 装饰器或 PluginContext 接口扩展。
"""

from __future__ import annotations

from huginn.api.context import (
    EventBusLike,
    LLMClient,
    PluginContext,
    Storage,
    ToolRegistryLike,
)
from huginn.api.event import (
    Event,
    EventType,
    LLMRequestEvent,
    LLMResponseEvent,
    MessageEvent,
    ToolCallEvent,
    ToolRespondEvent,
    WorkflowStageEvent,
)
from huginn.api.filter import (
    FilterMarker,
    HandlerT,
    StarHandlerMetadata,
    command,
    command_group,
    event_message_type,
    filter,
    on_llm_request,
    on_llm_response,
    on_tool_call,
    on_tool_respond,
    on_workflow_stage_done,
    permission_type,
)
from huginn.api.star import Star

__all__ = [
    # 基类
    "Star",
    # 上下文
    "PluginContext",
    "LLMClient",
    "ToolRegistryLike",
    "Storage",
    "EventBusLike",
    # 事件
    "EventType",
    "Event",
    "LLMRequestEvent",
    "LLMResponseEvent",
    "ToolCallEvent",
    "ToolRespondEvent",
    "WorkflowStageEvent",
    "MessageEvent",
    # 装饰器
    "filter",
    "command",
    "command_group",
    "event_message_type",
    "permission_type",
    "on_llm_request",
    "on_llm_response",
    "on_tool_call",
    "on_tool_respond",
    "on_workflow_stage_done",
    # 元数据 (插件作者一般用不到, 但导出方便高级用法)
    "StarHandlerMetadata",
    "FilterMarker",
    "HandlerT",
]

__version__ = "0.1.0"
