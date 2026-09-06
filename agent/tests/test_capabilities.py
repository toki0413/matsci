"""能力集装箱化抽象层测试 — 统一契约 / 三类能力 / 堆场 / 组合流程.

全部本地可跑, 不打真实 API. 用假 HuginnTool / 假原子能力隔离网络.
覆盖:
- CapabilityResult / Capability.run 契约 (校验 + 异常收敛)
- AtomicCapability 包装真/假 tool
- CompositeCapability 顺序编排 + step state 传递 + fail-fast + best_effort
- ExternalCapability 包装外部 backend
- CapabilityRegistry 注册/发现/manifest/run + 从 ToolRegistry 自动装箱
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from huginn.capabilities.base import (
    Capability,
    CapabilityResult,
)
from huginn.capabilities.intents import (
    AtomicCapability,
    CompositeCapability,
    ExternalCapability,
)
from huginn.capabilities.presets import register_capability_presets
from huginn.capabilities.registry import (
    CapabilityRegistry,
)
from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


# ── 假 HuginnTool ───────────────────────────────────────────────────────────


class _FakeInput(BaseModel):
    x: int = 1


class _EchoTool(HuginnTool):
    name = "echo_tool"
    description = "returns input back"
    input_schema = _FakeInput
    read_only = True

    async def call(self, args, context):
        return ToolResult(data={"x": args.x if hasattr(args, "x") else args}, success=True)


class _FailingTool(HuginnTool):
    name = "fail_tool"
    description = "always fails"
    input_schema = _FakeInput

    async def call(self, args, context):
        return ToolResult(data=None, success=False, error="boom")


class _UnavailableTool(HuginnTool):
    name = "unavail_tool"
    description = "unavailable"
    input_schema = _FakeInput

    def is_available(self) -> bool:
        return False


# ── Capability.run 契约 ────────────────────────────────────────────────────


class _AddOneCap(Capability[_FakeInput]):
    name = "add_one_cap"
    input_schema = _FakeInput

    async def _run(self, args, context):
        return CapabilityResult.ok({"result": args.x + 1})


class TestCapabilityContract:
    def test_run_validates_dict_into_schema(self):
        cap = _AddOneCap()
        res = asyncio.run(cap.run({"x": 5}, _ctx()))
        assert res.success
        assert res.data == {"result": 6}

    def test_run_accepts_model_instance(self):
        cap = _AddOneCap()
        res = asyncio.run(cap.run(_FakeInput(x=9), _ctx()))
        assert res.data == {"result": 10}

    def test_run_tolerates_fixed_exception(self):
        class _BadCap(Capability):
            name = "bad_cap"
            input_schema = _FakeInput

            async def _run(self, args, context):
                raise RuntimeError("kaboom")

        res = asyncio.run(_BadCap().run(_FakeInput(), _ctx()))
        assert not res.success
        assert "kaboom" in (res.error or "")

    def test_run_validation_error_is_fail_not_raise(self):
        cap = _AddOneCap()
        res = asyncio.run(cap.run({"x": "not-an-int"}, _ctx()))  # 类型不匹配, 校验失败
        assert not res.success
        assert res.code == "invalid_input"


# ── AtomicCapability ────────────────────────────────────────────────────────


class TestAtomicCapability:
    def test_wraps_tool_and_proxies(self):
        cap = AtomicCapability(_EchoTool())
        assert cap.name == "echo_tool"
        assert cap.is_available() is True
        res = asyncio.run(cap.run(_FakeInput(x=7), _ctx()))
        assert res.success
        assert res.data == {"x": 7}

    def test_forward_degradation_chain_from_profile(self):
        tool = _EchoTool()
        tool.profile = None  # 无 profile 时 degradation_chain 应为空
        cap = AtomicCapability(tool)
        assert cap.degradation_chain == ()

    def test_unavailable_proxied(self):
        cap = AtomicCapability(_UnavailableTool())
        assert cap.is_available() is False

    def test_tool_failure_maps_to_fail(self):
        cap = AtomicCapability(_FailingTool())
        res = asyncio.run(cap.run(_FakeInput(), _ctx()))
        assert not res.success
        assert res.code == "tool_error"

    def test_capability_from_tool_factory(self):
        from huginn.capabilities.intents import capability_from_tool

        cap = capability_from_tool(_EchoTool())
        assert isinstance(cap, AtomicCapability)


# ── CompositeCapability ─────────────────────────────────────────────────────


class _DoubleCap(Capability[_FakeInput]):
    name = "double_cap"
    input_schema = _FakeInput

    async def _run(self, args, context):
        return CapabilityResult.ok({"value": args.x * 2})


class TestCompositeCapability:
    def test_chain_steps_and_pass_state(self):
        # 步骤1: add_one(x) -> x+1; 步骤2: 用 state["s1"].data["result"] 再翻倍
        s1 = _AddOneCap()
        comp = CompositeCapability(
            name="chain_cap",
            steps=[
                ("s1", s1, lambda prev, inp: inp),
                # 从 prev 里取 s1 的结果拼下一步输入
                ("s2", _DoubleCap(),
                 lambda prev, inp: _FakeInput(x=prev["s1"].data["result"])),
            ],
            input_schema=_FakeInput,
        )
        res = asyncio.run(comp.run(_FakeInput(x=2), _ctx()))
        assert res.success
        # s1: 2+1=3; s2: 3*2=6
        assert res.data["state"]["s2"].data["value"] == 6
        assert len(res.data["steps"]) == 2

    def test_fail_fast_stops_on_failure(self):
        comp = CompositeCapability(
            name="fail_fast_cap",
            steps=[
                ("s1", _AddOneCap(), None),
                ("s2", AtomicCapability(_FailingTool()), None),
                ("s3", _DoubleCap(), None),
            ],
            input_schema=_FakeInput,
        )
        res = asyncio.run(comp.run(_FakeInput(x=1), _ctx()))
        assert not res.success
        assert res.code == "step_failed"
        assert "s2" in (res.error or "")
        # s3 不应执行
        assert all(s.get("capability") != "double_cap" for s in res.metadata.get("steps", []))

    def test_best_effort_step_does_not_abort(self):
        comp = CompositeCapability(
            name="best_effort_cap",
            steps=[
                ("s1", _AddOneCap(), None),
                ("s2", AtomicCapability(_FailingTool()), None),
            ],
            best_effort={"s2"},
            input_schema=_FakeInput,
        )
        res = asyncio.run(comp.run(_FakeInput(x=1), _ctx()))
        assert res.success  # 失败被 best_effort 吞掉, 整体成功


# ── ExternalCapability ──────────────────────────────────────────────────────


class _FakeBackend:
    """模拟外部服务 (如 MCP adapter)."""

    def __init__(self, ok: bool = True):
        self._ok = ok

    async def call(self, args, context):
        if self._ok:
            return ToolResult(data={"from_external": True}, success=True)
        return ToolResult(data=None, success=False, error="external down")


class TestExternalCapability:
    def test_wraps_external_backend(self):
        cap = ExternalCapability(
            "ext_cap", description="外部服务", backend=_FakeBackend(ok=True)
        )
        res = asyncio.run(cap.run({"q": 1}, _ctx()))
        assert res.success
        assert res.data == {"from_external": True}

    def test_external_failure(self):
        cap = ExternalCapability(
            "ext_cap2", description="外部服务", backend=_FakeBackend(ok=False)
        )
        res = asyncio.run(cap.run({"q": 1}, _ctx()))
        assert not res.success
        assert res.code == "external_error"


# ── CapabilityRegistry ──────────────────────────────────────────────────────


class TestCapabilityRegistry:
    def setup_method(self):
        CapabilityRegistry.clear()

    def test_register_and_get(self):
        CapabilityRegistry.register(_AddOneCap())
        cap = CapabilityRegistry.get("add_one_cap")
        assert cap is not None

    def test_manifest_includes_composite_deps(self):
        comp = CompositeCapability(
            name="mcap",
            steps=[("a", _AddOneCap(), None)],
            input_schema=_FakeInput,
            degradation_chain=("alt_cap",),
        )
        CapabilityRegistry.register(comp)
        manifest = CapabilityRegistry.manifest()
        entry = next(m for m in manifest if m["name"] == "mcap")
        assert entry["sub_capabilities"] == ["add_one_cap"]
        assert "alt_cap" in entry["degradation_chain"]

    def test_run_by_name_not_found(self):
        res = asyncio.run(CapabilityRegistry.run("nope", {}))
        assert not res.success
        assert res.code == "not_found"

    def test_run_by_name_success(self):
        CapabilityRegistry.register(_AddOneCap())
        res = asyncio.run(
            CapabilityRegistry.run("add_one_cap", {"x": 3}, _ctx())
        )
        assert res.success
        assert res.data == {"result": 4}

    def test_run_respects_is_available(self):
        unavailable = AtomicCapability(_UnavailableTool())
        CapabilityRegistry.register(unavailable)
        res = asyncio.run(
            CapabilityRegistry.run("unavail_tool", _FakeInput(), _ctx())
        )
        assert not res.success
        assert res.code == "unavailable"


# ── 自动装箱 (从 ToolRegistry scan) ─────────────────────────────────────────


class TestAutoScan:
    def setup_method(self):
        CapabilityRegistry.clear()
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.register(_EchoTool())

    def teardown_method(self):
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.unregister("echo_tool")
        CapabilityRegistry.clear()

    def test_scans_registered_tools_as_atomics(self):
        names = CapabilityRegistry.scan_tool_registry(["echo_tool"])
        assert "echo_tool" in names
        cap = CapabilityRegistry.get("echo_tool")
        assert cap is not None
        assert cap.name == "echo_tool"

    def test_manifest_lists_auto_atomic(self):
        CapabilityRegistry.scan_tool_registry(["echo_tool"])
        manifest = CapabilityRegistry.manifest()
        assert any(m["name"] == "echo_tool" for m in manifest)

    def test_run_auto_atomic(self):
        CapabilityRegistry.scan_tool_registry(["echo_tool"])
        res = asyncio.run(
            CapabilityRegistry.run("echo_tool", _FakeInput(x=11), _ctx())
        )
        assert res.success
        assert res.data == {"x": 11}


# ── 组合能力预设 (literature_research_cap) ───────────────────────────────────


class _LitInput(BaseModel):
    query: str = "band gap"
    system: str | None = None
    property: str | None = None
    max_results: int = 5
    min_citations: int | None = None
    oa_only: bool = False


class _MockLitTool(HuginnTool):
    """模拟 literature_tool, 只验证预设能注册出 composite 与 manifest."""

    name = "literature_tool"
    description = "mock literature tool"
    input_schema = _LitInput

    async def call(self, args, context):
        return ToolResult(
            data={
                "papers": [{"title": t, "doi": f"10.x/{i}"} for i, t in enumerate(
                    ["Paper A", "Paper B", "Paper C", "Paper D", "Paper E"]
                )][: (getattr(args, "max_results", 5) or 5)],
            },
            success=True,
        )


class TestPresetRegistration:
    def setup_method(self):
        self._tsnap = ToolRegistry.snapshot()
        CapabilityRegistry.clear()
        # 确保 literature_tool 存在, 让预设能构建出 composite
        if ToolRegistry.get("literature_tool") is None:
            ToolRegistry.register(_MockLitTool())

    def teardown_method(self):
        ToolRegistry.restore(self._tsnap)
        CapabilityRegistry.clear()

    def test_literature_research_composite_registers(self):
        names = register_capability_presets()
        assert "literature_research_cap" in names
        cap = CapabilityRegistry.get("literature_research_cap")
        assert cap is not None
        assert isinstance(cap, CompositeCapability)
        # 三个子能力 (search/verify/review) 都已声明
        assert "literature_search_cap" in cap.sub_capabilities
        assert "literature_benchmark_lookup_cap" in cap.sub_capabilities
        assert "literature_summarize_cap" in cap.sub_capabilities
        # 降级链透声
        assert "web_search_cap" in cap.degradation_chain

    def test_composite_steps_use_three_atomics(self):
        register_capability_presets()
        cap = CapabilityRegistry.get("literature_research_cap")
        step_caps = [s[1].name for s in cap._steps]
        assert step_caps == [
            "literature_search_cap",
            "literature_benchmark_lookup_cap",
            "literature_summarize_cap",
        ]

    def test_manifest_contains_composite(self):
        register_capability_presets()
        manifest = CapabilityRegistry.manifest()
        entry = next((m for m in manifest if m["name"] == "literature_research_cap"), None)
        assert entry is not None
        assert entry["category"] == "literature"
        assert entry["available"] is True
