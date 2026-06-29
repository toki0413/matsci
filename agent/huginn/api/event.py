"""事件类型与事件对象 —— 插件系统的生命周期信号。

参考 AstrBot 的 EventType 全生命周期钩子, 但裁剪掉用不到的,
并补上材料科研 workflow 相关的几个 stage 事件。
所有 handler 都靠这些事件拿到上下文, 不直接摸内部组件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """插件能监听的事件全集。

    分组:
      - 生命周期: 引擎/插件自身的加载卸载
      - Agent 流水线: 一次 agent run 的开始结束
      - LLM 调用: 请求前可改 prompt, 响应后可改 reply
      - 工具调用: 三段式 (call / execute / respond), 中间一段预留给执行器
      - Workflow: 材料科研多阶段流程的钩子, 这是 AstrBot 没有的
      - 消息: 收到消息 / 发送前后
    """

    # 生命周期
    ON_HUGINN_LOADED = auto()
    ON_PLUGIN_LOADED = auto()
    ON_PLUGIN_UNLOADED = auto()
    ON_PLUGIN_ERROR = auto()

    # Agent 流水线
    ON_AGENT_BEGIN = auto()
    ON_AGENT_DONE = auto()

    # LLM 调用
    ON_LLM_REQUEST = auto()      # 可改 system_prompt / context
    ON_LLM_RESPONSE = auto()     # 可改 reply

    # 工具调用三段式
    ON_TOOL_CALL = auto()        # 可拦截 / 改参数
    ON_TOOL_EXECUTE = auto()     # 工具执行中 (一般只读, 用来打日志)
    ON_TOOL_RESPOND = auto()     # 可改返回值

    # Workflow 钩子 (材料科研特色)
    ON_WORKFLOW_BEGIN = auto()
    ON_WORKFLOW_STAGE_START = auto()
    ON_WORKFLOW_STAGE_DONE = auto()
    ON_WORKFLOW_DONE = auto()
    ON_WORKFLOW_FAILED = auto()

    # 消息
    ON_MESSAGE_RECEIVED = auto()
    ON_BEFORE_MESSAGE_SENT = auto()
    ON_AFTER_MESSAGE_SENT = auto()


@dataclass
class Event:
    """所有事件的基类。

    handler 可以设 stop_propagation=True 阻断后续低优先级 handler,
    但不能阻断已经执行的 —— 优先级由 registry 决定, 不是事件自己决定。
    """

    type: EventType
    plugin_name: str = ""        # 触发来源 (空表示核心引擎)
    timestamp: datetime = field(default_factory=datetime.now)
    stop_propagation: bool = False

    def stop(self) -> None:
        """让事件分发跳过后续低优先级 handler。"""
        self.stop_propagation = True


# ── 具体事件类型 ────────────────────────────────────────────────────
# 不为每个 EventType 都造子类, 只为高频/需要强类型的造。
# 其他事件直接用 base Event + data 字段塞东西。
# 这样能控制文件膨胀, 又让常用 handler 拿到 typed object。

@dataclass
class LLMRequestEvent(Event):
    """LLM 请求前。handler 可改 system_prompt / messages / context。"""

    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = EventType.ON_LLM_REQUEST


@dataclass
class LLMResponseEvent(Event):
    """LLM 响应后。handler 可改 reply。"""

    reply: str = ""
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = EventType.ON_LLM_RESPONSE


@dataclass
class ToolCallEvent(Event):
    """工具调用前。handler 可改参数或拦截。

    设 stop() 后引擎应跳过该工具调用 (具体行为由调用点决定)。
    """

    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.ON_TOOL_CALL


@dataclass
class ToolRespondEvent(Event):
    """工具返回后。handler 可改 result。"""

    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True

    def __post_init__(self) -> None:
        self.type = EventType.ON_TOOL_RESPOND


@dataclass
class WorkflowStageEvent(Event):
    """workflow stage 开始 / 结束 / 失败复用同一个结构。"""

    workflow_name: str = ""
    stage_name: str = ""
    stage_index: int = 0
    duration_sec: float = 0.0
    error: str | None = None


@dataclass
class MessageEvent(Event):
    """消息收发。"""

    text: str = ""
    session_id: str = ""
    user_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
