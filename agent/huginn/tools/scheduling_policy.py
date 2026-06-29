"""只读门面 —— 统一工具调度元数据查询入口.

4 个检查点 (phase / router / permission / constraint) 各有不同生命周期,
不应合并到一个类。这个门面只解决"外部消费者想查某工具的调度属性时,
不用跨层 import 三个模块"的问题。全部读 live ToolRegistry, 不缓存。

用法::

    from huginn.tools.scheduling_policy import ToolSchedulingPolicy

    if ToolSchedulingPolicy.is_heavy(name):
        alts = ToolSchedulingPolicy.alternatives_for(name)
    heavy_names = sorted(ToolSchedulingPolicy.heavy_tool_names())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huginn.phases import ResearchPhase


class ToolSchedulingPolicy:
    """工具调度元数据的只读视图, 背后是 ToolRegistry。

    不做任何检查/拦截, 只把 ToolProfile 上的字段暴露成统一查询接口。
    检查逻辑留在 PhaseManager / ToolCallRouter / ToolAdapter 各自模块。
    """

    @staticmethod
    def is_heavy(name: str) -> bool:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t is not None and t.cost_tier == "heavy"

    @staticmethod
    def is_light(name: str) -> bool:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t is not None and t.cost_tier == "light"

    @staticmethod
    def scope_of(name: str) -> str | None:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t.constraint_scope if t is not None else None

    @staticmethod
    def phases_of(name: str) -> frozenset["ResearchPhase"] | None:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t.phases if t is not None else None

    @staticmethod
    def alternatives_for(name: str) -> tuple[str, ...]:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t.light_alternatives if t is not None else ()

    @staticmethod
    def heavy_actions_of(name: str) -> frozenset[str]:
        from huginn.tools.registry import ToolRegistry

        t = ToolRegistry.get(name)
        return t.heavy_actions if t is not None and t.heavy_actions else frozenset()

    @staticmethod
    def heavy_tool_names() -> frozenset[str]:
        from huginn.tools.registry import ToolRegistry

        return frozenset(
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "heavy"
        )

    @staticmethod
    def light_tool_names() -> frozenset[str]:
        from huginn.tools.registry import ToolRegistry

        return frozenset(
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "light"
        )
