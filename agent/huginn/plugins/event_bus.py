"""EventBus —— 异步事件分发, 按 EventType 路由到 handler。

参考 AstrBot 的 yield 流式响应: handler 可以是
  - async def handler(event) -> None          (普通异步)
  - def handler(event) -> None                (同步, 自动包装)
  - async def handler(event) -> AsyncGenerator (yield 流式)

设计:
  - 按 priority 降序执行 (从 registry 取)
  - handler 抛异常不阻断其他 handler, 但记录日志
  - handler 调 event.stop() 阻断后续低优先级 handler
  - 权限检查在 handler 执行前, 不通过跳过该 handler (不抛, 记录)
  - async generator 的 yield 值收集到列表返回, 给调用方拼流式响应
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.api.event import Event, EventType
from huginn.api.filter import StarHandlerMetadata
from huginn.plugins.permissions import PermissionChecker

logger = logging.getLogger("huginn.event_bus")


@dataclass
class DispatchResult:
    """一次 dispatch 的结果汇总。"""

    # 所有 async generator handler yield 出来的片段, 按执行顺序
    streamed: list[Any] = field(default_factory=list)
    # 实际执行的 handler 数
    executed: int = 0
    # 因权限/匹配被跳过的 handler 数
    skipped: int = 0
    # 抛异常的 handler 数
    failed: int = 0
    # 是否被某 handler stop 了
    stopped: bool = False


@dataclass
class EventBus:
    """事件总线。

    依赖:
      - registry: StarHandlerRegistry, 提供 handler 查询
      - permission_checker: PermissionChecker | None, None 表示不做权限检查
        (仅用于测试 / 信任的内置 handler, 插件加载场景必须传)
    """

    registry: Any = None  # StarHandlerRegistry, 但鸭子类型, 避免硬依赖
    permission_checker: PermissionChecker | None = None
    # 默认 True: handler 抛异常时 catch + log, 不冒泡。
    # 设 False 让异常冒泡 (调用方自己接), 用于测试。
    swallow_exceptions: bool = True

    def __post_init__(self) -> None:
        if self.registry is None:
            # 延迟导入避免循环
            from huginn.plugins.registry import StarHandlerRegistry
            self.registry = StarHandlerRegistry()

    async def dispatch(self, event: Event) -> DispatchResult:
        """分发事件到所有匹配 handler。

        顺序: priority 降序。每个 handler 拿到同一个 event 对象,
        可以直接改字段 (如 event.system_prompt)。handler 调 event.stop()
        后, 后续低优先级 handler 不再执行。
        """
        result = DispatchResult()
        if self.registry is None:
            return result

        handlers = self.registry.get_handlers_for(event)
        for meta in handlers:
            if event.stop_propagation:
                result.stopped = True
                break

            # 权限检查
            if self.permission_checker is not None and meta.permissions:
                if not self.permission_checker.check_handler(meta.plugin_name, meta):
                    logger.warning(
                        "handler %s.%s skipped: missing permission (declared %s)",
                        meta.plugin_name, meta.name, meta.permissions,
                    )
                    result.skipped += 1
                    continue

            try:
                await self._invoke(meta, event, result)
                result.executed += 1
            except Exception as e:
                result.failed += 1
                if self.swallow_exceptions:
                    logger.exception(
                        "handler %s.%s raised: %s",
                        meta.plugin_name, meta.name, e,
                    )
                else:
                    raise
        return result

    async def _invoke(
        self,
        meta: StarHandlerMetadata,
        event: Event,
        result: DispatchResult,
    ) -> None:
        """调用单个 handler, 适配 sync / async / async-gen 三种签名。

        handler 签名约定: handler(event: Event) -> ...
        我们不传 self, 因为 meta.handler 已经是 bound method (Star 实例方法)。
        """
        handler = meta.handler
        # async generator: 调用得到 async generator, 迭代收集 yield 值
        if asyncio.iscoroutinefunction(handler):
            # 普通协程 —— await 一下, 返回值忽略 (handler 改 event 生效)
            ret = await handler(event)
            # 如果返回的是 async generator (协程返回 generator), 也消费
            if inspect.isasyncgen(ret):
                async for chunk in ret:
                    result.streamed.append(chunk)
            return

        # async generator function: iscoroutinefunction 检测不到, 单独判断
        if inspect.isasyncgenfunction(handler):
            async for chunk in handler(event):
                result.streamed.append(chunk)
            return

        # 同步函数: 在线程里跑避免阻塞 loop? 这里直接调, 因为 handler 一般很快
        if callable(handler):
            ret = handler(event)
            # 同步 generator: 消费 yield 值
            if inspect.isgenerator(ret):
                for chunk in ret:
                    result.streamed.append(chunk)
            return

        logger.warning("handler %s is not callable, skipped", meta.name)

    async def dispatch_simple(self, event_type: EventType, **payload: Any) -> DispatchResult:
        """便捷构造一个 base Event 并 dispatch。

        只用于没有专门子类的事件类型。复杂事件请构造子类再 dispatch。
        """
        event = Event(type=event_type)
        # 把 payload 塞到 event 上 (dataclass 动态字段不支持, 用 __dict__)
        for k, v in payload.items():
            setattr(event, k, v)
        return await self.dispatch(event)


__all__ = ["EventBus", "DispatchResult"]
