"""授权强度与 observe 影子档 — 移植自 dsh 的 authorization strength / accept mode.

dsh 的原则:
- 授权不只是布尔, 还要区分 ``explicit`` (人明确批) 与 ``inferred`` (策略推断);
- 新策略先跑 ``observe``: 记录 would-accept, 但不执行, 观察期过后再放权;
- 证据必须可追溯, 且日志只记键名不记值.

对应到 huginn:
- ``PermissionResult.strength`` 三档 IGNORE / INFERRED / EXPLICIT;
- ``HUGINN_APPROVE_MODE=observe`` 只拦 ``auto_approve_all`` 把 ASK 工具变成
  自动放行的那一类推断授权; 工具显式声明的 AUTO 与只读工具不受影响;
- ``record_authorization`` 结构化记录 tool / strength / session / reason / arg_keys.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import BaseModel

from huginn.core_types import AuthorizationStrength, PermissionMode, ToolResult
from huginn.permissions import (
    PermissionChecker,
    PermissionConfig,
    approval_mode,
    autoloop_auto_approve,
    observe_mode_enabled,
    record_authorization,
)
from huginn.tools.adapter import ToolAdapter
from huginn.tools.base import HuginnTool, ToolProfile


class _FakeDftInput(BaseModel):
    command: str = "run"


class _FakeDftTool(HuginnTool):
    # 用探针专用名, 不占用真实工具名: ToolAdapter 按 name 全局登记, 复用真实名会
    # 污染其它测试文件 (同一 name 的第二个实例不会被真正调用).
    name = "probe_ask_tool"
    description = "Fake ASK-level tool"
    input_schema = _FakeDftInput
    profile = ToolProfile(constraint_scope="dft")

    async def call(self, args, context):
        return ToolResult(data={"ok": True})


class _FakeAnalyticsTool(HuginnTool):
    name = "probe_auto_tool"
    description = "Fake AUTO tool"
    input_schema = _FakeDftInput
    read_only = True

    async def call(self, args, context):
        return ToolResult(data={"ok": True})


def _check(tool: str, mode: PermissionMode):
    config = PermissionConfig()
    config.set_mode(tool, mode)
    return asyncio.run(PermissionChecker(config).check(tool))


# ── 1. 强度与 mode 正交 ─────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,expected",
    [
        (PermissionMode.AUTO, AuthorizationStrength.INFERRED),
        (PermissionMode.ASK, AuthorizationStrength.IGNORE),
        (PermissionMode.DENY, AuthorizationStrength.IGNORE),
    ],
)
def test_strength_follows_mode(mode, expected):
    assert _check("probe_tool", mode).strength == expected


def test_dangerous_ask_is_ignore_not_inferred():
    """危险命令即使被 auto_approve_all 放行, 决策本身仍是未授权 (IGNORE)."""
    config = PermissionConfig(auto_approve_all=True)
    result = asyncio.run(
        PermissionChecker(config).check("bash_tool", args={"command": "rm -rf /"})
    )
    assert result.mode == PermissionMode.ASK
    assert result.strength == AuthorizationStrength.IGNORE


# ── 2. observe 档位语义 ─────────────────────────────────────────


def test_observe_is_a_valid_mode_but_does_not_grant(monkeypatch):
    monkeypatch.setenv("HUGINN_APPROVE_MODE", "observe")
    monkeypatch.delenv("HUGINN_AUTOLOOP_AUTO_APPROVE", raising=False)
    assert approval_mode() == "observe"
    assert observe_mode_enabled() is True
    # observe 绝不能被当成"已放权": autoloop 不应自动批准写/执行.
    assert autoloop_auto_approve() is False


def test_observe_is_not_the_default(monkeypatch):
    monkeypatch.delenv("HUGINN_APPROVE_MODE", raising=False)
    monkeypatch.delenv("HUGINN_AUTOLOOP_AUTO_APPROVE", raising=False)
    assert approval_mode() == "workspace"
    assert observe_mode_enabled() is False


# ── 3. adapter: 只拦推断出来的授权 ─────────────────────────────


def _invoke(tool, *, auto_approve: bool, base_mode: PermissionMode):
    config = PermissionConfig(auto_approve_all=auto_approve)
    config.set_mode(tool.name, base_mode)
    return ToolAdapter().adapt(tool, permission_config=config).invoke({"command": "run"})


def test_observe_blocks_inferred_auto_approval(monkeypatch, caplog):
    monkeypatch.setenv("HUGINN_APPROVE_MODE", "observe")
    caplog.set_level(logging.INFO, logger="huginn.permissions")
    out = _invoke(
        _FakeDftTool(), auto_approve=True, base_mode=PermissionMode.ASK
    )
    assert out.get("error") and "observe" in out["error"]
    decisions = [r for r in caplog.records if r.getMessage().startswith("permission.decision")]
    assert decisions, "observe 拒绝必须留下可追溯的决策记录"
    assert "strength=inferred" in decisions[-1].getMessage()


def test_observe_does_not_block_explicitly_auto_tools(monkeypatch):
    """工具自己声明 AUTO (无需确认) 不属于推断授权, observe 不拦."""
    monkeypatch.setenv("HUGINN_APPROVE_MODE", "observe")
    out = _invoke(
        _FakeAnalyticsTool(), auto_approve=True, base_mode=PermissionMode.AUTO
    )
    assert not out.get("error")


def test_workspace_mode_still_auto_approves_ask_tools(monkeypatch):
    """observe 是显式选择的影子档, 默认 workspace 行为不得被它改变."""
    monkeypatch.setenv("HUGINN_APPROVE_MODE", "workspace")
    out = _invoke(
        _FakeDftTool(), auto_approve=True, base_mode=PermissionMode.ASK
    )
    assert not out.get("error")


# ── 4. 证据日志不泄漏参数值 ─────────────────────────────────────


def test_record_authorization_logs_keys_never_values(caplog):
    caplog.set_level(logging.INFO, logger="huginn.permissions")
    record_authorization(
        "bash_tool",
        AuthorizationStrength.INFERRED,
        "sess-1",
        reason="would-approve",
        arg_keys=["command", "cwd"],
    )
    msg = caplog.records[-1].getMessage()
    assert "strength=inferred" in msg
    assert "sess-1" in msg
    assert "command" in msg and "cwd" in msg
    # 只记键名: 不出现任何值 / 路径分隔符.
    assert "/" not in msg
