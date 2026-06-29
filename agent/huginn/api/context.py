"""PluginContext —— 注入给插件 Star 的聚焦接口集合。

这是反 AstrBot "Context 上帝对象" 的关键:
  - AstrBot 把 LLM / tool / storage / platform / config / persona 全塞 Context,
    插件能摸到一切, 导致耦合爆炸, 重构时一动百动。
  - 这里只暴露 4 个聚焦接口, 每个都是 Protocol (鸭子类型),
    不绑死实现, 也不让插件直接摸 agent / config / persona / memory。
插件要更多能力, 走 EventBus 事件或申请权限, 不走 Context 扩字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端最小接口。插件只能 chat, 不能改模型配置。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        model: str | None = None,
        **kwargs: Any,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class ToolRegistryLike(Protocol):
    """工具注册表接口。插件只能查 / 调, 不能注册新工具 (走权限申请)。"""

    def get(self, name: str) -> Any | None: ...
    def list_tools(self) -> list[str]: ...


@runtime_checkable
class Storage(Protocol):
    """插件级 KV 存储。每个插件名空间隔离, 不互相污染。"""

    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> bool: ...
    def keys(self) -> list[str]: ...


@runtime_checkable
class EventBusLike(Protocol):
    """事件总线接口。插件可以主动发事件 (少用) 或订阅 (一般由装饰器自动注册)。"""

    async def dispatch(self, event: Any) -> None: ...


@dataclass
class PluginContext:
    """注入给 Star 子类的聚焦上下文。

    故意只放 4 个字段。新增能力时, 优先扩接口 Protocol,
    而不是往这里堆字段 —— 这是避免上帝对象复发的硬约束。
    """

    llm_client: LLMClient | None = None
    tool_registry: ToolRegistryLike | None = None
    storage: Storage | None = None
    event_bus: EventBusLike | None = None
    # 插件自己的名空间标识, 由 loader 注入, 插件不该自己改
    plugin_name: str = ""
    # 只读的元信息 (调试用), 不放 config 对象本身
    extra: dict[str, Any] = field(default_factory=dict)

    def require_llm(self) -> LLMClient:
        """强约束: 用之前必须检查。没注入就抛, 不静默返回 None 坑插件。"""
        if self.llm_client is None:
            raise RuntimeError(
                f"plugin {self.plugin_name!r} needs llm_client but it's not injected; "
                "check metadata.yaml permissions and loader wiring"
            )
        return self.llm_client

    def require_storage(self) -> Storage:
        if self.storage is None:
            raise RuntimeError(
                f"plugin {self.plugin_name!r} needs storage but it's not injected"
            )
        return self.storage


__all__ = [
    "LLMClient",
    "ToolRegistryLike",
    "Storage",
    "EventBusLike",
    "PluginContext",
]
