"""Pi 化: make_tool 自扩展原语 + pi 极简内核模式的测试.

conftest 的 ``_restore_tool_registry`` 卫生门禁要求任何改动全局 ToolRegistry 的
测试用 ``ToolRegistry.snapshot()/restore()`` 恢复 —— 这里用统一 fixture 处理.
make_tool 落盘用 tmp_path 隔离, 不写 ~/.huginn.
"""
from __future__ import annotations

import asyncio

import pytest

from huginn.core_types import PermissionMode, ToolContext
from huginn.tools.extensions import build_extension, list_extensions
from huginn.tools.make_tool import MakeTool
from huginn.tools.registry import ToolRegistry

_SIMPLE_SRC = '''
from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool

class CitySimTool(HuginnTool):
    name = 'parsec_flux'
    category = 'self'
    description = 'compute a fake flux'

    def call(self, args, context: ToolContext | None = None):
        return ToolResult(data={"flux_wm2": 1366.0})
'''


@pytest.fixture
def _reg():
    """全局 registry 卫生: 改动前快照, teardown 恢复."""
    snap = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(snap)


# ── make_tool: extensions 基础设施 ────────────────────────────────

def test_build_extension_registers_and_is_callable(tmp_path, _reg):
    r = build_extension("parsec_flux", _SIMPLE_SRC, category="self", base=tmp_path)
    assert r["registered"] is True
    assert r["name"] == "parsec_flux"
    tool = ToolRegistry.get("parsec_flux")
    assert tool is not None
    assert tool.active is True
    assert any(e["name"] == "parsec_flux" for e in list_extensions())


def test_build_extension_rejects_bad_name(tmp_path, _reg):
    with pytest.raises(ValueError):
        build_extension("Bad Name", _SIMPLE_SRC, base=tmp_path)


def test_build_extension_rejects_non_compilable(tmp_path, _reg):
    with pytest.raises(ValueError):
        build_extension("badcode", "def broken(:", base=tmp_path)


def test_build_extension_rejects_no_huginn_tool(tmp_path, _reg):
    with pytest.raises(ValueError):
        build_extension("notatool", "x = 1", base=tmp_path)


def test_build_extension_rejects_name_mismatch(tmp_path, _reg):
    wrong = _SIMPLE_SRC.replace("parsec_flux", "other_name")
    with pytest.raises(ValueError):
        build_extension("parsec_flux", wrong, base=tmp_path)


# ── make_tool: 模型侧工具 ─────────────────────────────────────────

def test_make_tool_call_registers(_reg):
    async def _run():
        tool = MakeTool()
        return await tool.call(
            {"action": "make", "name": "parsec_flux", "code": _SIMPLE_SRC}, None
        )

    r = asyncio.run(_run())
    assert r.success is True
    assert r.data["action"] == "make"
    assert r.data["registered"] is True
    assert r.data["name"] == "parsec_flux"
    assert ToolRegistry.get("parsec_flux") is not None


def test_make_tool_call_requires_name_and_code(_reg):
    async def _run():
        t = MakeTool()
        return await t.call({"action": "make", "name": "", "code": ""}, None)

    r = asyncio.run(_run())
    assert r.success is False
    assert "name" in r.error


def test_make_tool_default_permission_is_ask():
    """可执行代码原语默认走权限门禁 (ASK), 需授权才放行."""

    async def _run():
        t = MakeTool()
        return await t.check_permissions(
            {}, ToolContext(session_id="s", workspace=".")
        )

    p = asyncio.run(_run())
    assert p.mode == PermissionMode.ASK


def test_make_tool_list_and_info(tmp_path, _reg):
    build_extension("parsec_flux", _SIMPLE_SRC, base=tmp_path)

    async def _run():
        t = MakeTool()
        return await t.call({"action": "list"}, None)

    r = asyncio.run(_run())
    assert r.success is True
    assert any(e["name"] == "parsec_flux" for e in r.data["extensions"])


# ── pi 极简内核模式 ──────────────────────────────────────────────

def test_pi_apply_hides_non_primitives_and_exit_restores(tmp_path, _reg):
    from huginn.modes.pi import PRIMITIVES, apply_pi_mode, exit_pi_mode, pi_active

    # 确保有个非原语工具在场可被隐藏 (落盘到 tmp, 不进 ~/.huginn)
    build_extension("parsec_flux", _SIMPLE_SRC, category="self", base=tmp_path)
    info = apply_pi_mode()
    assert info["mode"] == "pi"
    assert info["hidden"] >= 1
    assert pi_active() is True
    # 原语必须可见
    for prim in PRIMITIVES:
        t = ToolRegistry.get(prim)
        if t is not None:
            assert t.active is True
    resume = exit_pi_mode()
    assert resume["mode"] == "default"
    assert pi_active() is False


def test_pi_system_prompt_is_small():
    from huginn.modes.pi import pi_system_prompt

    words = len(pi_system_prompt().split())
    assert words < 200
