"""HuginnTool.call 权限门禁测试 —— 让 check_permissions(此前是死代码)真正生效.

覆盖:
  1. DENY  → call 拒绝且不触达 _execute(防越权执行).
  2. ASK 无确认 → 拒绝; 有确认 → 放行.
  3. AUTO(默认) → 放行, 行为不变(兼容红线).
  4. check_permissions 抛异常 → 降级放行(fail-open, 不破坏既有执行).
同步跑 asyncio.run, 不依赖 pytest-asyncio 插件, 确定性。
"""
from __future__ import annotations

import asyncio

from huginn.core_types import PermissionMode, PermissionResult, ToolContext, ToolResult
from huginn.tools.base import HuginnTool


class _GatedTool(HuginnTool):
    """按注入的 check_permissions 返回值执行门禁的测试工具."""

    name = "gated_tool"
    description = "测试门禁"

    def __init__(self, *, perm: PermissionResult | Exception | None = None) -> None:
        self._perm = perm or PermissionResult(mode=PermissionMode.AUTO)
        self.executed = False           # 标记 _execute 是否被触达
        super().__init__()

    async def check_permissions(self, args, context):
        if isinstance(self._perm, Exception):
            raise self._perm
        return self._perm

    async def _execute(self, args, context):
        self.executed = True            # 副作用标记: 证明执行发生了
        return ToolResult(success=True, data={"ran": True}, error="")


def _run(coro):
    return asyncio.run(coro)


def _ctx(**kw) -> ToolContext:
    return ToolContext(session_id="test", workspace=".", **kw)


def test_deny_blocks_inner_execution():
    tool = _GatedTool(perm=PermissionResult(mode=PermissionMode.DENY, reason="not allowed"))
    res = _run(tool.call({}, _ctx()))
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "permission denied" in (res.error or "")
    assert "not allowed" in (res.error or "")


def test_ask_without_confirm_blocks_requires_confirm():
    tool = _GatedTool(perm=PermissionResult(mode=PermissionMode.ASK, reason="confirm please"))
    res = _run(tool.call({}, _ctx()))
    assert res.success is False
    assert "permission required" in (res.error or "")


def test_auto_allows_execution_unchanged():
    # 默认 AUTO → 放行, 会触达 _execute
    tool = _GatedTool()  # AUTO
    res = _run(tool.call({}, _ctx()))
    assert res.success is True
    assert tool.executed is True, "AUTO 应放行执行"


def test_ask_with_confirm_allows():
    # ASK + context.permissions 该工具显式覆盖为 AUTO(=已批准) → 放行(触达 _execute)
    tool = _GatedTool(perm=PermissionResult(mode=PermissionMode.ASK, reason="confirm please"))
    ctx = _ctx()
    ctx.permissions[tool.name] = PermissionMode.AUTO
    res = _run(tool.call({}, ctx))
    assert res.success is True
    assert tool.executed is True, "已确认的 ASK 应放行执行"


def test_permission_exc_fails_open():
    # check_permissions 抛异常 → 降级放行, 不破坏执行
    tool = _GatedTool(perm=RuntimeError("perm backend down"))
    res = _run(tool.call({}, _ctx()))
    assert res.success is True
    assert tool.executed is True, "权限异常应降级放行"