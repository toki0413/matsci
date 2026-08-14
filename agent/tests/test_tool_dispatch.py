"""Tests for H5-b unified tool dispatch — 让 H4 的 tool_whitelist 强制生效.

覆盖 huginn/harness/tool_dispatch.py:
  * is_tool_allowed — 无 phase / 空白名单 / 在白名单 / 不在白名单
  * _phase_whitelist — phase 存在 / 不存在 / 无白名单
  * dispatch_tool — 白名单拦截不执行 / 工具不存在 / 无 phase 放行 / 白名单放行执行
  * dynamic_workflow 接入 — orchestrator.run(phase=...) 透传白名单强制
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.core_types import ToolContext
from huginn.harness import tool_dispatch as td
from huginn.harness.phase_spec import PhaseRegistry
from huginn.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _isolate_registry_and_phase():
    """隔离全局注册表 + PhaseRegistry 单例, 避免测试间串扰."""
    reg_snap = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(reg_snap)
    PhaseRegistry._instance = None


def _register_core():
    import huginn.tools as _tools
    _tools.register_core_tools()


def _with_validate_whitelist(*tools: str) -> None:
    """给 _validate phase 设 tool_whitelist override."""
    PhaseRegistry._instance = None
    PhaseRegistry.get_instance().register_phase_override(
        "_validate", {"tool_whitelist": list(tools)}
    )
    PhaseRegistry._instance = None


def _ctx(tmp_path) -> ToolContext:
    return ToolContext(
        session_id="test-dispatch", workspace=str(tmp_path), config=None
    )


# ── is_tool_allowed ────────────────────────────────────────────────

def test_allowed_no_phase():
    assert td.is_tool_allowed("any_tool", None) is True


def test_allowed_phase_without_whitelist():
    # _report baseline 无 tool_whitelist → 不限制
    assert td.is_tool_allowed("any_tool", "_report") is True


def test_allowed_whitelisted_tool():
    _with_validate_whitelist("bash_tool", "file_read_tool")
    assert td.is_tool_allowed("bash_tool", "_validate") is True
    assert td.is_tool_allowed("file_read_tool", "_validate") is True


def test_blocked_non_whitelisted_tool():
    _with_validate_whitelist("bash_tool")
    assert td.is_tool_allowed("file_read_tool", "_validate") is False


def test_allowed_unknown_phase_returns_true():
    # 未知 phase (PhaseRegistry 无该 phase) → 无白名单 → 允许
    assert td.is_tool_allowed("any_tool", "nonexistent_phase") is True


# ── dispatch_tool ──────────────────────────────────────────────────

def test_dispatch_blocks_non_whitelisted(tmp_path):
    _register_core()
    _with_validate_whitelist("bash_tool")
    res = asyncio.run(
        td.dispatch_tool("file_read_tool", {}, _ctx(tmp_path), phase="_validate")
    )
    assert res.success is False
    assert "not allowed" in res.error


def test_dispatch_not_found(tmp_path):
    _register_core()
    res = asyncio.run(
        td.dispatch_tool("no_such_tool", {}, _ctx(tmp_path))
    )
    assert res.success is False
    assert "not registered" in res.error


def test_dispatch_no_phase_passes_through(tmp_path):
    _register_core()
    # 无 phase → 不校验白名单 → 工具被调用
    res = asyncio.run(
        td.dispatch_tool("file_read_tool", {"file_path": str(tmp_path)}, _ctx(tmp_path))
    )
    # file_read_tool 需有效 path; 空 path 可能报错但 success 字段由工具决定
    assert res is not None


def test_dispatch_whitelisted_executes(tmp_path):
    _register_core()
    _with_validate_whitelist("file_read_tool")
    res = asyncio.run(
        td.dispatch_tool("file_read_tool", {"file_path": str(tmp_path)}, _ctx(tmp_path), phase="_validate")
    )
    # 在白名单内 → 不被拦截, 进入工具执行 (成功与否取决于工具输入)
    assert "not allowed" not in (res.error or "")


# ── dynamic_workflow 接入 ──────────────────────────────────────────

def test_workflow_orchestrator_phase_whitelist(tmp_path):
    """run(phase=...) 时白名单强制: 不在白名单的 subtask 被标记 failed."""
    from huginn.autoloop.dynamic_workflow import (
        Subtask,
        WorkflowOrchestrator,
        WorkflowScript,
    )

    _register_core()
    _with_validate_whitelist("bash_tool")

    script = WorkflowScript(
        id="wf-test",
        objective="test",
        subtasks=[
            Subtask(id="s1", tool_name="file_read_tool", args={"path": "."}),
        ],
    )
    orch = WorkflowOrchestrator()
    result = asyncio.run(orch.run(script, _ctx(tmp_path), phase="_execute"))
    # file_read_tool 在 _execute 白名单? 我们设的是 _validate 的白名单,
    # _execute 无白名单 → 放行. 这里验证 phase 透传不破坏正常执行.
    assert result.status == "completed"


def test_workflow_orchestrator_blocks_via_phase(tmp_path):
    """在 _validate 白名单下, 用 phase="_validate" 跑含 file_read_tool 的 workflow →
    file_read_tool 不在白名单 → 该 subtask failed."""
    from huginn.autoloop.dynamic_workflow import (
        Subtask,
        WorkflowOrchestrator,
        WorkflowScript,
    )

    _register_core()
    _with_validate_whitelist("bash_tool")

    script = WorkflowScript(
        id="wf-block",
        objective="test",
        subtasks=[
            Subtask(id="s1", tool_name="file_read_tool", args={"path": "."}),
        ],
    )
    orch = WorkflowOrchestrator()
    result = asyncio.run(orch.run(script, _ctx(tmp_path), phase="_validate"))
    sr = result.subtask_results["s1"]
    assert sr.status == "failed"
    assert "not allowed" in (sr.error or "")
