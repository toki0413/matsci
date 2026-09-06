"""System object — deprecated alias for ServerContext.

v23 状态树合并: HuginnSystem 已折叠为 ServerContext 的子类.
所有字段 / 方法 (is_configured / get_component / list_components /
active_threads / edit_tools) 都从 ServerContext 继承, 不再独立维护.

新代码应直接用 ServerContext (huginn.server_context.ServerContext).
本模块保留 get_system / set_system 全局单例访问, 仅为向后兼容.
"""
from __future__ import annotations

from huginn.server_context import ServerContext


class HuginnSystem(ServerContext):
    """Deprecated alias for :class:`huginn.server_context.ServerContext`.

    v23 状态树合并后, HuginnSystem 不再持有独立字段 — 它就是 ServerContext.
    保留此类仅为向后兼容已有 import (tests/test_system_object.py, routes/system.py).
    新代码请直接用 ServerContext.

    .. deprecated:: v23
       Use :class:`huginn.server_context.ServerContext` instead.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # 不抛 DeprecationWarning 在 __init__ — 测试会刷屏. 仅在文档里标注.
        # 生产代码如需提醒, 用 warnings.warn(..., DeprecationWarning, stacklevel=2).
        super().__init__(*args, **kwargs)


_system: HuginnSystem | None = None


def get_system() -> HuginnSystem:
    """Return the global HuginnSystem, creating one lazily if needed.

    Note: 生产路径应优先用 huginn.server_context.get_server_context().
    本函数返回的 HuginnSystem 实例在 server_core.get_system_snapshot()
    每次调用时被刷新为最新 ServerContext 快照.
    """
    global _system
    if _system is None:
        _system = HuginnSystem()
    return _system


def set_system(system: HuginnSystem | ServerContext) -> None:
    """Replace the global HuginnSystem instance.

    Accepts either HuginnSystem or ServerContext (since HuginnSystem is now
    a subclass, both are valid).
    """
    global _system
    if isinstance(system, HuginnSystem):
        _system = system
    else:
        # ServerContext 实例: 包装成 HuginnSystem (无副作用, 字段已对齐).
        # 用 dataclasses.replace 保持字段值, 转型为 HuginnSystem.
        import dataclasses
        _system = HuginnSystem(**{
            f.name: getattr(system, f.name)
            for f in dataclasses.fields(ServerContext)
            if hasattr(system, f.name)
        })


__all__ = ["HuginnSystem", "get_system", "set_system"]
