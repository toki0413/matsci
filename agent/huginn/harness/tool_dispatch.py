"""H5-b: 统一工具分派入口 — 让 H4 的 tool_whitelist 在 autoloop 主循环内强制生效.

背景: H4 (phase_spec.py) 把 tool_whitelist 抽成 PhaseSpec 可演化字段, 但 phase
方法体内 9+ 处直接 ``tool.call()`` 绕过白名单, 白名单实际是 advisory 不强制.
H5-b 提供一个统一分派入口 ``dispatch_tool``, 从 ToolRegistry 取工具 + 按 phase
白名单校验, 让 tool_whitelist 真正强制生效.

设计:
- ``dispatch_tool(name, args, ctx, phase=None, registry=None)``:
  - 从 registry (默认 ToolRegistry) 取工具; 不存在返回 not_found 结果.
  - phase 非空时, 从 PhaseRegistry 取该 phase 的 tool_whitelist; 若白名单非空且
    工具名不在其中, 返回 blocked 结果 (不执行).
  - 校验通过则调用 tool.call(args, ctx), 返回其结果.
- 兼容性: phase 为 None, 或 phase 无白名单 (空列表), 或白名单包含该工具 → 不限制.
  toggle: cfg.feature_flags.harness_tool_dispatch (默认受 harness 总开关约束).
- 与 routes/tools.py 的 call_tool (HTTP 端点, 无 phase 概念) 互补: 本模块是
  agent 内部 phase 感知的分派, 供 autoloop 主循环调用.

数学: 白名单强制 = 在工具调用点前加一道集合包含检查
  is_allowed(t) = (whitelist(phase) == []) or (t ∈ whitelist(phase)).
  未命中时短路返回, 工具不执行 — 语义上等价于 agent 在 subagent dispatch 路径
  (subagent.py:332-333 tool_filter) 已有的强制, 只是作用到主循环 phase 方法体.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from huginn.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from huginn.core_types import ToolContext

logger = logging.getLogger(__name__)


def _phase_whitelist(phase: str | None) -> list[str] | None:
    """取 phase 的 tool_whitelist. 返回 None 表示无白名单 (不限制).

    phase 不存在或 PhaseRegistry 无该 phase 时返回 None (caller 不校验场合).
    白名单直接从 PhaseRegistry 的 baseline/override 读取, 不依赖 phase_evolve
    toggle — H5-b 的目的正是让 tool_whitelist 无条件强制生效.
    """
    if not phase:
        return None
    try:
        from huginn.harness.phase_spec import PhaseRegistry
        spec = PhaseRegistry.get_instance().get_phase_spec(phase)
        if spec is None:
            return None
        return spec.tool_whitelist or None
    except Exception:
        logger.debug("tool_dispatch: phase whitelist lookup failed", exc_info=True)
        return None


def is_tool_allowed(name: str, phase: str | None = None) -> bool:
    """判断工具在指定 phase 下是否被允许. 无白名单时恒为 True."""
    whitelist = _phase_whitelist(phase)
    if not whitelist:
        return True
    return name in whitelist


async def dispatch_tool(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    phase: str | None = None,
    registry: Any = None,
) -> Any:
    """统一分派: 从 registry 取工具 + 按 phase 白名单校验 + 调用.

    Args:
        name: 工具名 (ToolRegistry 注册名).
        args: 工具输入 (dict, 工具内部会做 schema 校验).
        ctx: ToolContext.
        phase: 可选的 phase 名, 用于白名单校验. None 或 phase 无白名单则不限制.
        registry: 可选注册表, 默认 ToolRegistry.

    Returns:
        ToolResult (或等价对象). 工具不存在时返回含 not_found 的失败结果;
        白名单拦截时返回含 blocked 的失败结果 (不执行工具).
    """
    reg = registry if registry is not None else ToolRegistry
    tool = reg.get(name)
    if tool is None:
        from huginn.core_types import ToolResult
        return ToolResult(
            data=None,
            success=False,
            error=f"Tool '{name}' not registered",
        )

    if not is_tool_allowed(name, phase):
        from huginn.core_types import ToolResult
        return ToolResult(
            data=None,
            success=False,
            error=f"Tool '{name}' not allowed in phase '{phase}' (tool_whitelist)",
        )

    return await tool.call(args, ctx)


def _selfcheck() -> None:
    """H5-b selfcheck: 白名单校验 + 兼容性 + 工具不存在."""
    import os
    import tempfile

    from huginn.core_types import ToolContext, ToolResult

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    try:
        import huginn.tools as _tools
        _tools.register_core_tools()  # 注册核心工具, 供 dispatch_tool 分派
        import huginn.harness.phase_spec as ps
        import huginn.harness.tool_dispatch as td
        from huginn.harness.phase_spec import PhaseRegistry

        # 1. 无 phase → 无白名单 → 允许
        assert td.is_tool_allowed("any_tool", None) is True
        print("1. no phase → allowed OK")

        # 2. phase 无白名单 (空列表) → 允许
        assert td.is_tool_allowed("any_tool", "_report") is True
        print("2. phase empty whitelist → allowed OK")

        # 3. 白名单包含 → 允许
        ps.PhaseRegistry._instance = None
        reg = PhaseRegistry.get_instance()
        reg.register_phase_override("_validate", {
            "tool_whitelist": ["file_read_tool", "bash_tool"],
        })
        ps.PhaseRegistry._instance = None
        assert td.is_tool_allowed("file_read_tool", "_validate") is True
        assert td.is_tool_allowed("bash_tool", "_validate") is True
        print("3. whitelisted tool → allowed OK")

        # 4. 白名单不包含 → 拒绝
        assert td.is_tool_allowed("vasp_tool", "_validate") is False
        print("4. non-whitelisted tool → blocked OK")

        # 5. dispatch_tool: 白名单拦截返回 blocked 结果, 不执行
        #    用已注册的工具 (file_read_tool) 作为"不在白名单"对象 — 未注册
        #    工具会先命中 not_found 分支, 无法观测白名单拦截.
        ps.PhaseRegistry._instance = None
        reg2 = PhaseRegistry.get_instance()
        reg2.register_phase_override("_validate", {
            "tool_whitelist": ["bash_tool"],
        })
        ps.PhaseRegistry._instance = None
        ctx = ToolContext(session_id="selftest", workspace=tmp, config=None)
        res = None
        import asyncio
        res = asyncio.run(td.dispatch_tool("file_read_tool", {}, ctx, phase="_validate"))
        assert isinstance(res, ToolResult)
        assert res.success is False
        assert "not allowed" in res.error
        print("5. dispatch blocked OK")

        # 6. dispatch_tool: 工具不存在 → not_found 结果
        reg2_ctx = ToolContext(session_id="selftest", workspace=tmp, config=None)
        res2 = asyncio.run(td.dispatch_tool("nonexistent_tool", {}, reg2_ctx))
        assert isinstance(res2, ToolResult)
        assert res2.success is False
        assert "not registered" in res2.error
        print("6. dispatch not found OK")

        # 7. dispatch_tool: 无 phase → 正常放行到工具 (工具不存在与否由工具层决定)
        #    这里用 None phase, 工具不存在直接返回 not_found (不涉及白名单)
        res3 = asyncio.run(td.dispatch_tool("nonexistent_tool", {}, reg2_ctx))
        assert "not registered" in res3.error
        print("7. dispatch no phase OK")

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        del os.environ["HUGINN_CACHE_DIR"]
        # restore singleton
        with contextlib.suppress(Exception):
            ps.PhaseRegistry._instance = None

    print("H5-b tool_dispatch selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
