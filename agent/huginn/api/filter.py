"""装饰器 + handler 元数据 —— 借鉴 AstrBot 的 @filter.* 体系。

设计:
  - 装饰器只往函数对象上挂标记 (_huginn_filters), 不包 wrapper,
    保持 handler 签名 (async def handler(event: Event) -> AsyncGenerator)。
  - 多个装饰器叠加 = AND 逻辑, 所有 filter 都满足才触发。
  - Star 子类被 registry 收集时, 扫描成员函数上的标记, 生成 StarHandlerMetadata。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from huginn.api.event import (
    Event,
    EventType,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolCallEvent,
    ToolRespondEvent,
    WorkflowStageEvent,
)

# handler 签名: async generator (yield 流式) 或 async func (return)
# 两者都允许, 由 event_bus 统一适配
HandlerT = Callable[..., Any]

# 标记函数上的 key
_FILTER_ATTR = "_huginn_filters"


@dataclass
class FilterMarker:
    """单个装饰器条件。matcher 返回 True 表示该条件通过。"""

    name: str                                  # 装饰器名, 调试用
    event_type: EventType                      # 绑定的事件类型
    matcher: Callable[[Event], bool]           # 额外匹配 (command 名 / tool 名 等)
    permission: str | None = None              # 需要的权限标识, 如 "tool_call:vasp_tool"


@dataclass
class StarHandlerMetadata:
    """registry 注册的单元。一个 handler 函数对应一条。"""

    handler: HandlerT
    event_type: EventType
    plugin_name: str = ""
    priority: int = 0                          # 越大越先执行
    matchers: list[Callable[[Event], bool]] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    name: str = ""                             # handler 函数名

    def matches(self, event: Event) -> bool:
        """所有 matcher 都通过才算匹配 (AND 逻辑)。"""
        return all(m(event) for m in self.matchers)


# ── 内部工具 ─────────────────────────────────────────────────────────

def _attach(func: HandlerT, marker: FilterMarker) -> HandlerT:
    """把 marker 追加到 func 的 filter 列表。多个装饰器叠加即追加多次。"""
    existing: list[FilterMarker] = getattr(func, _FILTER_ATTR, None) or []
    existing.append(marker)
    setattr(func, _FILTER_ATTR, existing)
    return func


def _get_markers(func: HandlerT) -> list[FilterMarker]:
    return list(getattr(func, _FILTER_ATTR, None) or [])


def markers_to_metadata(
    func: HandlerT,
    plugin_name: str,
    priority: int,
) -> list[StarHandlerMetadata]:
    """把函数上的 marker 列表转成 metadata 列表。

    一个函数被多个事件装饰器装饰时, 每个 event_type 生成一条 metadata,
    但所有 matcher (含别的 event_type 的 matcher) 都会塞进 matchers,
    这样链式组合 (AND) 仍然生效。
    """
    markers = _get_markers(func)
    if not markers:
        return []

    # 同 event_type 的 marker 合并到一条
    by_event: dict[EventType, list[FilterMarker]] = {}
    for m in markers:
        by_event.setdefault(m.event_type, []).append(m)

    result: list[StarHandlerMetadata] = []
    for evt_type, group in by_event.items():
        # AND 逻辑: 该 event_type 下, 该函数所有 marker 都通过才触发。
        # 跨 event_type 的 marker 也会被塞进 matchers, 但因为分发时
        # event.type 已经是 evt_type, 跨类型 matcher 一般不会误判
        # (例如 command matcher 只看消息文本, on_tool_call 事件不会命中)。
        matchers = [m.matcher for m in group]
        perms = [m.permission for m in group if m.permission]
        result.append(StarHandlerMetadata(
            handler=func,
            event_type=evt_type,
            plugin_name=plugin_name,
            priority=priority,
            matchers=matchers,
            permissions=perms,
            name=getattr(func, "__name__", ""),
        ))
    return result


# ── 装饰器实现 ───────────────────────────────────────────────────────

def command(name: str, *, priority: int = 0):
    """匹配消息文本以 `name` 开头的指令。常用于 `/hello` 这种。

    注意: 这是个事件无关装饰器, 必须跟一个事件装饰器 (如 on_message_received)
    叠加使用, 单独用不会触发。
    """
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            text = getattr(event, "text", "") or ""
            return text.strip().startswith(name)
        return _attach(func, FilterMarker(
            name=f"command({name!r})",
            event_type=EventType.ON_MESSAGE_RECEIVED,
            matcher=matcher,
        ))
    return deco


def command_group(prefix: str, *, priority: int = 0):
    """一组指令的前缀匹配。跟 command 类似, 留给后续 alias/子命令扩展。"""
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            text = getattr(event, "text", "") or ""
            return text.strip().startswith(prefix)
        return _attach(func, FilterMarker(
            name=f"command_group({prefix!r})",
            event_type=EventType.ON_MESSAGE_RECEIVED,
            matcher=matcher,
        ))
    return deco


def event_message_type(*types: EventType, priority: int = 0):
    """按 EventType 过滤。一般用于一个 handler 想监听多个事件的场景。"""
    types_set = set(types)

    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            return event.type in types_set
        # event_type 取第一个, 真正的多事件靠 matcher
        primary = types[0] if types else EventType.ON_MESSAGE_RECEIVED
        return _attach(func, FilterMarker(
            name=f"event_message_type({types!r})",
            event_type=primary,
            matcher=matcher,
        ))
    return deco


def permission_type(perm: str, *, priority: int = 0):
    """声明该 handler 需要的权限。perm 形如 `llm_call` / `tool_call:vasp_tool`。

    实际强制由 PermissionChecker 在 dispatch 时检查, 装饰器只负责声明。
    """
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            return True  # 权限不是 matcher 的事, 这里恒 True
        return _attach(func, FilterMarker(
            name=f"permission_type({perm!r})",
            event_type=EventType.ON_MESSAGE_RECEIVED,  # 占位, 会被别的装饰器覆盖
            matcher=matcher,
            permission=perm,
        ))
    return deco


def on_llm_request(*, priority: int = 0):
    """LLM 请求前钩子。handler 可改 event.system_prompt / event.context。"""
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            return isinstance(event, LLMRequestEvent)
        return _attach(func, FilterMarker(
            name="on_llm_request",
            event_type=EventType.ON_LLM_REQUEST,
            matcher=matcher,
        ))
    return deco


def on_llm_response(*, priority: int = 0):
    """LLM 响应后钩子。handler 可改 event.reply。"""
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            return isinstance(event, LLMResponseEvent)
        return _attach(func, FilterMarker(
            name="on_llm_response",
            event_type=EventType.ON_LLM_RESPONSE,
            matcher=matcher,
        ))
    return deco


def on_tool_call(tool_name: str | None = None, *, priority: int = 0):
    """工具调用前钩子。tool_name 为 None 表示监听所有工具。

    handler 可改 event.args 或调 event.stop() 拦截。
    """
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            if not isinstance(event, ToolCallEvent):
                return False
            if tool_name is None:
                return True
            return event.tool_name == tool_name
        perm = f"tool_call:{tool_name}" if tool_name else "tool_call:*"
        return _attach(func, FilterMarker(
            name=f"on_tool_call({tool_name!r})",
            event_type=EventType.ON_TOOL_CALL,
            matcher=matcher,
            permission=perm,
        ))
    return deco


def on_tool_respond(tool_name: str | None = None, *, priority: int = 0):
    """工具返回后钩子。handler 可改 event.result。"""
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            if not isinstance(event, ToolRespondEvent):
                return False
            if tool_name is None:
                return True
            return event.tool_name == tool_name
        return _attach(func, FilterMarker(
            name=f"on_tool_respond({tool_name!r})",
            event_type=EventType.ON_TOOL_RESPOND,
            matcher=matcher,
        ))
    return deco


def on_workflow_stage_done(*, priority: int = 0):
    """workflow stage 完成钩子。handler 可读 stage_name / duration_sec。"""
    def deco(func: HandlerT) -> HandlerT:
        def matcher(event: Event) -> bool:
            return isinstance(event, WorkflowStageEvent)
        return _attach(func, FilterMarker(
            name="on_workflow_stage_done",
            event_type=EventType.ON_WORKFLOW_STAGE_DONE,
            matcher=matcher,
        ))
    return deco


# ── filter 门面对象 ─────────────────────────────────────────────────
# 让插件作者写 `from huginn.api import filter` 然后 `@filter.command("x")`,
# 跟 AstrBot 的导入风格保持一致, 降低迁移成本。

class _FilterNamespace:
    """装饰器门面。`from huginn.api import filter` 后用 `@filter.xxx`。"""

    command = staticmethod(command)
    command_group = staticmethod(command_group)
    event_message_type = staticmethod(event_message_type)
    permission_type = staticmethod(permission_type)
    on_llm_request = staticmethod(on_llm_request)
    on_llm_response = staticmethod(on_llm_response)
    on_tool_call = staticmethod(on_tool_call)
    on_tool_respond = staticmethod(on_tool_respond)
    on_workflow_stage_done = staticmethod(on_workflow_stage_done)


filter = _FilterNamespace

__all__ = [
    "FilterMarker",
    "StarHandlerMetadata",
    "HandlerT",
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
    "markers_to_metadata",
]
