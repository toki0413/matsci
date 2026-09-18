"""@capability 装饰器 + CapabilityMetadata —— 插件以"能力维度"接入 harness.

对齐 DeepSeek Harness 的 Cordis "everything is a plugin" 但保持现有体系:
huginn 已有事件驱动的 Star handler (@filter.*); 这里补一个**平行**的维度声明,
让插件能声明"我提供某种能力": loop / session / storage / sysprompt.

与 @filter 的关系:
  - @filter.*   → 事件 handler (响应事件, 由 EventBus 分发)
  - @capability → 能力单元 (可被 AgentSession 查询/选择/装配), 由 CapabilityRegistry 持有
  两者正交不冲突: 一个插件可以既响应事件, 又提供能力.

装饰器只往方法上挂 marker (不包 wrapper), 保持原方法签名 —— 与 filter.py 同套路.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# 已纳入 mount 的能力维度 (边界见 P1 设计: storage 已有内置, scheduling 本轮不做)
DIMENSIONS: frozenset[str] = frozenset({"loop", "session", "storage", "sysprompt"})

_CAP_ATTR = "_huginn_capabilities"


@dataclass
class CapabilityMetadata:
    """registry 持有的能力单元."""

    dimension: str                       # DIMENSIONS 之一
    name: str                            # 该维度下的能力名 (如 loop=code_act)
    impl: Callable[..., Any]             # 能力实现 (签名随维度而定)
    plugin_name: str = ""                # 提供方 (卸载时按此撤销)
    priority: int = 0                    # 同维度装配顺序 (越大越先)


def capability(dimension: str, name: str | None = None, *, priority: int = 0):
    """声明一个方法提供某维度能力.

    Example:
        @capability("loop", "code_act")
        async def my_loop(self, agent, message, thread_id="default"): ...
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown capability dimension {dimension!r}; allowed {sorted(DIMENSIONS)}")

    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        existing: list[tuple[str, str, int]] = list(getattr(func, _CAP_ATTR, None) or [])
        existing.append((dimension, name or func.__name__, priority))
        setattr(func, _CAP_ATTR, existing)
        return func

    return deco


def get_capabilities(func: Callable[..., Any]) -> list[tuple[str, str, int]]:
    """读方法上挂的 capability marker."""
    return list(getattr(func, _CAP_ATTR, None) or [])


def capabilities_to_metadata(
    func: Callable[..., Any], plugin_name: str
) -> list[CapabilityMetadata]:
    """把一个方法上的 capability markers 转成元数据列表."""
    out: list[CapabilityMetadata] = []
    for dimension, name, priority in get_capabilities(func):
        out.append(CapabilityMetadata(
            dimension=dimension, name=name, impl=func,
            plugin_name=plugin_name, priority=priority,
        ))
    return out


__all__ = [
    "DIMENSIONS", "CapabilityMetadata",
    "capability", "get_capabilities", "capabilities_to_metadata",
]
