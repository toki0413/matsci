"""P1 能力维度 mount 的测试.

覆盖:
  - @capability 声明 + collect_capabilities 收集 (经 Star)
  - CapabilityRegistry 注册/按 plugin 卸载 (可逆)
  - CapabilityRegistry 与 StarHandlerRegistry/ToolRegistry 平行 (不互相侵入)
  - AgentSession.set_loop 热切换 astream 走向
  - AgentSession.sysprompt 持有段 + trajectory 来源审计 (贡献的 section 落注入事件)
"""
from __future__ import annotations

import pytest

from huginn.agent.agent_session import AgentSession
from huginn.capabilities.capability import capability
from huginn.capabilities.registry import (
    CapabilityMountRegistry,
    get_shared_capability_registry,
)


@pytest.fixture(autouse=True)
def _clean_caps():
    """每用例清空共享 CapabilityMountRegistry, 隔离."""
    snap = dict(CapabilityMountRegistry._caps)  # type: ignore[attr-defined]
    CapabilityMountRegistry.clear()
    yield
    CapabilityMountRegistry._caps = snap  # type: ignore[attr-defined]


class _Stub:
    def __init__(self) -> None:
        self.thread_id = "stub"
        self.messages = []

    def chat(self, message, thread_id="default"):
        async def _g():
            yield {"type": "message", "content": message, "thread_id": thread_id}
        return _g()

    def invoke(self, message, thread_id="default"):
        return {"final": True, "message": message}


# ── @capability 收集 ───────────────────────────────────────────────

def test_capability_decorator_collected_via_star():
    from huginn.api.context import PluginContext
    from huginn.api.star import Star

    class P(Star):
        name = "ptest"

        @capability("loop", "mine")
        def my_loop(self, agent, message, thread_id="default"):
            pass

    p = P(context=PluginContext(plugin_name="ptest"))
    caps = p.collect_capabilities()
    assert [(c.dimension, c.name, c.plugin_name) for c in caps] == [
        ("loop", "mine", "ptest")
    ]


# ── registry 注册 / 可逆卸载 ───────────────────────────────────────

def test_capability_registry_register_and_unmount():
    from huginn.capabilities.capability import CapabilityMetadata

    reg = get_shared_capability_registry()
    reg.register(
        CapabilityMetadata(dimension="loop", name="a", impl=lambda: None, plugin_name="pa"),
        CapabilityMetadata(dimension="loop", name="b", impl=lambda: None, plugin_name="pb"),
        CapabilityMetadata(dimension="sysprompt", name="x", impl=lambda: None, plugin_name="pa"),
    )
    assert set(reg.list_names()) == {("loop", "a"), ("loop", "b"), ("sysprompt", "x")}
    # 按 plugin 卸载 pa → a 和 x 消失, b 保留
    removed = reg.unregister_plugin("pa")
    assert removed == 2
    assert reg.get("loop", "a") is None
    assert reg.get("sysprompt", "x") is None
    assert reg.get("loop", "b") is not None


def test_capability_registry_unknown_dimension_ignored():
    from huginn.capabilities.capability import CapabilityMetadata

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(dimension="bogus", name="z", impl=lambda: None))
    assert reg.list() == []


# ── AgentSession.set_loop 热切换 ───────────────────────────────────

async def _collect(agen):
    return [ev async for ev in agen]


def test_set_loop_switches_astream():
    from huginn.capabilities.capability import CapabilityMetadata

    async def custom_loop(agent, message, thread_id="default"):
        yield {"type": "looped", "content": message, "thread_id": thread_id}

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(dimension="loop", name="mine", impl=custom_loop, plugin_name="t"))

    s = AgentSession.attach(_Stub())
    assert s.set_loop("mine") == {"loop": "mine", "plugin": "t"}
    import asyncio
    evs = asyncio.run(_collect(s.astream("hi")))
    assert evs == [{"type": "looped", "content": "hi", "thread_id": "stub"}]


def test_set_loop_default_is_builtin():
    s = AgentSession.attach(_Stub())
    out = s.set_loop("default")
    assert out["loop"] == "default"
    assert out["plugin"] == "builtin"
    assert ("loop", "default") in s.loops()
    assert ("loop", "code_act") in s.loops()


def test_set_loop_unknown_raises():
    from huginn.capabilities.capability import CapabilityMetadata

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(dimension="loop", name="only", impl=lambda *_: None))
    s = AgentSession.attach(_Stub())
    with pytest.raises(ValueError):
        s.set_loop("nope")


def test_sysprompt_sections_assembly_and_injection():
    """插件贡献的 prompt section 被收集, 并以 sysprompt:<name> 来源落注入审计."""
    from huginn.capabilities.capability import CapabilityMetadata

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(
        dimension="sysprompt", name="lab_rules",
        impl=lambda ctx: "## 实验规范: 文献须可复现",
        plugin_name="t",
    ))

    s = AgentSession.attach(_Stub())
    injected: list[str] = []
    s.inject = lambda source, payload=None: injected.append(source)  # monkeypatch 防写盘

    sections = s.sysprompt_sections()
    lab = [x for x in sections if x["capability"] == "lab_rules"]
    assert len(lab) == 1
    assert lab[0]["source"] == "sysprompt:lab_rules"
    assert "可复现" in lab[0]["text"]
    assert "sysprompt:lab_rules" in injected  # 来源审计已埋


def test_sysprompt_sections_skip_none_contributors():
    from huginn.capabilities.capability import CapabilityMetadata

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(
        dimension="sysprompt", name="empty", impl=lambda ctx: None, plugin_name="t",
    ))
    s = AgentSession.attach(_Stub())
    sources = {sec["source"] for sec in s.sysprompt_sections()}
    assert "sysprompt:empty" not in sources  # 返回 None 的贡献者被跳过
