"""Action Fusion 机制测试: tool→verifier 契约 + 同调用融合验证."""

from __future__ import annotations

import types

import pytest

from huginn.core_types import ToolResult
from huginn.tools.action_fusion import (
    enable_action_fusion,
    fused_verify,
    register_solpi_default_verifiers,
    register_verifier,
    verifier_for,
)
from huginn.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _restore_flag():
    from huginn.tools import action_fusion as af

    before = af.is_action_fusion_enabled()
    yield
    # 复位总开关与契约
    af._action_fusion_enabled = before
    af._VERIFIERS._items.clear()


@pytest.fixture()
def clean_registry():
    snap = ToolRegistry.snapshot()
    from huginn.tools.action_fusion import _VERIFIERS

    async def call(self, args, ctx):
        return ToolResult(success=True, data={"validated": args})

    fake = types.SimpleNamespace()
    fake.name = "fake_validate"
    fake.call = types.MethodType(call, fake)
    ToolRegistry.register(fake)
    yield fake
    ToolRegistry.restore(snap)
    _VERIFIERS._items.clear()


async def test_no_contract_returns_none(clean_registry) -> None:
    from huginn.core_types import ToolContext

    enable_action_fusion()
    ctx = ToolContext(session_id="t", workspace="/tmp", config=None)
    assert verifier_for("ghost") is None
    assert await fused_verify("ghost", {}, {}, ctx) is None


async def test_disabled_flag_returns_none(clean_registry) -> None:
    from huginn.core_types import ToolContext

    # 默认关: 有契约也不融合
    register_verifier("file_edit_tool", "fake_validate", {"path": "$args.file_path"})
    ctx = ToolContext(session_id="t", workspace="/tmp", config=None)
    assert await fused_verify("file_edit_tool", {"file_path": "/tmp/m.py"}, {}, ctx) is None


async def test_exact_and_prefix_resolution(clean_registry) -> None:
    register_verifier("file_edit_tool", "fake_validate", {"path": "$args.file_path"})
    assert verifier_for("file_edit_tool") is not None

    register_verifier("vasp.", "fake_validate", {"x": 1})
    assert verifier_for("vasp_run") is not None
    assert verifier_for("vasp_parse") is not None


async def test_fused_verify_enabled_replaces_args_from_input(clean_registry) -> None:
    from huginn.core_types import ToolContext

    enable_action_fusion()
    register_verifier("file_edit_tool", "fake_validate", {"path": "$args.file_path"})
    ctx = ToolContext(session_id="t", workspace="/tmp", config=None)
    r = await fused_verify("file_edit_tool", {"file_path": "/tmp/m.py"}, {"ok": 1}, ctx)
    assert r is not None and r["verified"] is True, r
    assert r["detail"]["validated"]["path"] == "/tmp/m.py"


async def test_default_verifiers_registered(clean_registry) -> None:
    register_solpi_default_verifiers()
    assert verifier_for("file_write_tool") is not None
    assert verifier_for("file_edit_tool").verifier_tool == "bash_tool"


async def test_missing_verifier_tool_is_none(clean_registry) -> None:
    from huginn.core_types import ToolContext

    enable_action_fusion()
    register_verifier("only_reg", "missing_verifier_tool", {})
    ctx = ToolContext(session_id="t", workspace="/tmp", config=None)
    assert await fused_verify("only_reg", {}, {}, ctx) is None
