"""PluginHost 统一挂载点测试 — Cordis 时空可组合的最小落地.

验证三面**同生同灭**:
  - 时间: 卸载回滚该插件的逆 (且不误伤其他插件);
  - 事件: 卸载撤掉该插件的 handler 订阅;
  - 空间: 卸载注销依赖声明 → 依赖者级联停用.
外加 ``StarHandlerRegistry.unregister`` 的定向撤销 (单条订阅的逆).
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.api.event import Event, EventType
from huginn.plugins.host import Mount, PluginHost
from huginn.plugins.registry import StarHandlerRegistry


def _host() -> PluginHost:
    # 每测试独立 registry, 避免进程级共享单例互相污染.
    return PluginHost(registry=StarHandlerRegistry())


def _dispatch(host: PluginHost, event_type: EventType) -> None:
    asyncio.run(host.bus.dispatch(Event(type=event_type)))


# ── 生命周期: 挂载 / 重复挂载 / 幂等卸载 ──────────────────────────


def test_mount_and_unmount_lifecycle():
    host = _host()
    m = host.mount("p1")
    assert isinstance(m, Mount)
    assert host.mounted() == ["p1"]
    assert host.get("p1") is m

    host.unmount("p1")
    assert host.mounted() == []
    assert host.get("p1") is None
    assert m.unmounted is True

    # 幂等: 再卸载不抛.
    host.unmount("p1")


def test_duplicate_mount_rejected_then_ok_after_unmount():
    host = _host()
    host.mount("p1")
    with pytest.raises(ValueError):
        host.mount("p1")
    host.unmount("p1")
    host.mount("p1")  # 卸载后可重新挂载.
    assert host.mounted() == ["p1"]


# ── 事件面: 订阅即效应, 卸载即撤销 ────────────────────────────────


def test_subscription_lives_and_dies_with_mount():
    host = _host()
    seen: list[str] = []

    m = host.mount("p1")
    m.on(EventType.ON_PLUGIN_LOADED, lambda e: seen.append("hit"))

    _dispatch(host, EventType.ON_PLUGIN_LOADED)
    assert seen == ["hit"]

    host.unmount("p1")
    _dispatch(host, EventType.ON_PLUGIN_LOADED)
    # 卸载后 handler 已随逆栈撤销 — 不再触发.
    assert seen == ["hit"]
    assert host.registry.get_handlers(EventType.ON_PLUGIN_LOADED) == []


def test_on_returns_targeted_disposer():
    host = _host()
    seen: list[str] = []
    m = host.mount("p1")
    h1 = lambda e: seen.append("h1")  # noqa: E731
    h2 = lambda e: seen.append("h2")  # noqa: E731
    d1 = m.on(EventType.ON_PLUGIN_LOADED, h1)
    m.on(EventType.ON_PLUGIN_LOADED, h2)

    d1()  # 只撤 h1, 不影响 h2.
    _dispatch(host, EventType.ON_PLUGIN_LOADED)
    assert seen == ["h2"]
    # 插件仍挂载, h2 订阅仍在.
    assert len(host.registry.get_handlers(EventType.ON_PLUGIN_LOADED)) == 1
    assert host.mounted() == ["p1"]


def test_registry_unregister_targeted():
    reg = StarHandlerRegistry()
    host = PluginHost(registry=reg)
    m1, m2 = host.mount("a"), host.mount("b")
    # 直接经 registry 注册两条不同插件的 handler.
    from huginn.api.filter import StarHandlerMetadata

    meta = StarHandlerMetadata(
        handler=lambda e: None,
        event_type=EventType.ON_AGENT_BEGIN,
        plugin_name="a",
    )
    reg.register(meta)
    assert reg.unregister(meta) == 1
    assert reg.unregister(meta) == 0  # 重复撤销安全.
    del m1, m2


# ── 时间面: 卸载回滚本插件逆, 且严格隔离 ──────────────────────────


def test_unmount_reverts_own_effects():
    host = _host()
    env: dict[str, str] = {"K": "orig"}
    m = host.mount("p1")
    m.revertible.set_env("K", "changed", env)
    assert env["K"] == "changed"

    host.unmount("p1")
    assert env["K"] == "orig"


def test_unmount_isolation_between_plugins():
    host = _host()
    env: dict[str, str] = {"A": "a0", "B": "b0"}
    ma = host.mount("pa")
    mb = host.mount("pb")
    ma.revertible.set_env("A", "a1", env)
    mb.revertible.set_env("B", "b1", env)

    host.unmount("pa")
    # 只回滚 pa 的逆; pb 的改动原样保留.
    assert env == {"A": "a0", "B": "b1"}

    host.unmount("pb")
    assert env == {"A": "a0", "B": "b0"}


def test_effects_are_lifo():
    host = _host()
    order: list[str] = []
    m = host.mount("p1")
    m.track(lambda: order.append("first"))
    m.track(lambda: order.append("second"))
    host.unmount("p1")
    assert order == ["second", "first"]


# ── 空间面: 依赖门控 + 级联停用 ──────────────────────────────────


def test_dependent_deactivated_when_provider_unmounts():
    host = _host()
    host.mount("db", provides={"db"})
    events: list[tuple[str, str]] = []
    host.mount("app", requires={"db"}, on_dependency_change=lambda k, a: events.append((k, a)))

    assert host.is_active("app") is True
    host.unmount("db")
    assert host.is_active("app") is False
    assert ("db", "deactivate") in events


def test_requires_gating_activates_on_cascade():
    host = _host()
    events: list[tuple[str, str]] = []
    host.mount("app", requires={"db"}, on_dependency_change=lambda k, a: events.append((k, a)))

    # db 未提供 → app 未激活.
    assert host.is_active("app") is False
    assert events == []

    # db 上线 → app 级联激活.
    host.mount("db", provides={"db"})
    assert host.is_active("app") is True
    assert events == [("db", "activate")]


def test_requires_chain_deactivates_transitively():
    host = _host()
    host.mount("db", provides={"db"})
    host.mount("cache", provides={"cache"}, requires={"db"})
    host.mount("app", requires={"cache"})
    assert host.is_active("app") is True

    # 掀掉链头 → cache 停用 → cache 的 provides 下线 → app 随之停用.
    host.unmount("db")
    assert host.is_active("cache") is False
    assert host.is_active("app") is False


def test_set_available_drives_dependents():
    host = _host()
    host.mount("extern")
    events: list[tuple[str, str]] = []
    host.mount("app", requires={"ext"}, on_dependency_change=lambda k, a: events.append((k, a)))
    assert host.is_active("app") is False

    host.set_available("ext", True)
    assert host.is_active("app") is True
    assert events == [("ext", "activate")]
