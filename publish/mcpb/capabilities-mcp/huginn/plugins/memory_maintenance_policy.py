"""记忆整理策略注册表 —— "Everything is a Plugin" 形态 B 在 memory maintainer 的落地.

记忆 mantenance (decay / prune / dedupe) 有多个策略分歧点: 每日衰减率、裁剪阈值、
是否去重。过去想换一套阈值 ("按重要性排序的淘汰策略") 得改核心。这里把整组阈值
声明成 MemoryMaintenancePolicy 注册进来, ``resolve_memory_policy()`` 按
StrategyRegistry 的 resolve 语义选出**单个**生效策略 (最高 priority, 否则默认),
maintenance() 对未显式传入的旋钮取策略值, 显式传参优先。

形态说明: 记忆整理是后台 housekeeping, 非热路径, 但同样用同步注册表 (形态 B) —
单策略选择用 resolve (非 union), 不做事件分发。默认策略与内置字面量一致, 保证
无第三方接入时行为零变化。
"""

from __future__ import annotations

from dataclasses import dataclass

from huginn.plugins.strategy import StrategyRegistry


# 内置默认策略 (与原 longterm.maintenance 字面量一致).
@dataclass(frozen=True)
class MemoryMaintenancePolicy:
    """一次记忆整理的一组阈值.

    Attributes:
        decay_per_day: 每日记忆强度衰减因子 (0.97 = 每天乘 0.97).
        prune_threshold: 记忆强度低于该阈值的条目被裁剪.
        deduplicate: 是否跑 near-identical 去重.
    """

    decay_per_day: float = 0.97
    prune_threshold: float = 0.15
    deduplicate: bool = True


_DEFAULT = MemoryMaintenancePolicy()
_registry: StrategyRegistry[MemoryMaintenancePolicy] = StrategyRegistry(
    fallback=_DEFAULT
)


def register_memory_maintenance_policy(
    name: str,
    policy: MemoryMaintenancePolicy,
    priority: int = 0,
) -> None:
    """注册一个记忆整理策略.

    注册后成为下一轮 maintenance 的生效策略。同名覆盖; 不同名时 resolve 按
    priority 仲裁取 priority 最高者 (见 StrategyRegistry.resolve 语义)。
    """
    _registry.register(name, policy, priority=priority)


def unregister_memory_maintenance_policy(name: str) -> int:
    """移除策略, 返回移除条数."""
    return _registry.unregister(name)


def resolve_memory_policy() -> MemoryMaintenancePolicy:
    """返回当前生效策略: 已注册策略中 priority 最高者, 否则内置默认.

    记忆整理是"整组阈值"的单策略选择, 不按 key 匹配 — 取最高优先级注册项即可
    (同 priority 后注册者优先, 由 ordered() 稳定升序保证).
    """
    entries = _registry.ordered()  # priority 升序
    if entries:
        return entries[-1][2]
    return _DEFAULT


def clear_policies() -> None:
    """清空注册表 (测试用)."""
    _registry.clear()


__all__ = [
    "MemoryMaintenancePolicy",
    "clear_policies",
    "register_memory_maintenance_policy",
    "resolve_memory_policy",
    "unregister_memory_maintenance_policy",
]
