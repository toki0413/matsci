"""Catalog reconcile —— 把 control-plane 的启停标记落到各注册表.

骨架期 set_enabled 只翻转追踪视图; 这里把它补完整:
- mcp 下线: 记录工具名 → 断开会话 → 把该 server 的工具从 ToolRegistry 摘除.
- mcp 上线: reconnect (用存储的 config) → 重新注册工具.
- tool 下线/上线: 用模块级 stash 暂存 HuginnTool, 下线 unregister, 上线 register 复原.

`ponytail:` 只覆盖 mcp 与 tool 两类; prompt/skill/model 的"禁用"语义未达成共识,
直接 noop (返回 "unsupported"), 不强行套用导致注册表错乱.
"""
from __future__ import annotations

import logging

from huginn.catalog.models import CatalogEntry

logger = logging.getLogger(__name__)

# 非 MCP 工具 stash, 供 re-enable 复原 (MCP 工具 re-enable 走 reconnect 重新生成适配器).
_tool_stash: dict[str, "object"] = {}


def _mcp_tool_names(mcp_manager: object, server: str) -> list[str]:
    """采集某 server 当前的工具名, 断电前调用以拿到摘除名单."""
    try:
        return [
            t.name for t in mcp_manager.list_tools()
            if getattr(t, "server_name", None) == server
        ]
    except Exception:  # pragma: no cover - 读不到就读不到, unregister 幂等
        return []


async def _apply_mcp(name: str, enabled: bool, mcp_manager: object) -> str:
    from huginn.tools.mcp_adapter import register_mcp_tools
    from huginn.tools.registry import ToolRegistry

    if enabled:
        # 上线: 用存储的 config 重连并重新注册该 server 的工具.
        if mcp_manager is None or name not in mcp_manager._configs:
            return "unsupported"
        await mcp_manager.reconnect(name)
        register_mcp_tools(mcp_manager, server_name=name)
        return "ok"

    # 下线: 先拿工具名单, 再断会话并摘除工具.
    if mcp_manager is None:
        return "unsupported"
    names = _mcp_tool_names(mcp_manager, name)
    await mcp_manager.disconnect(name)
    for tname in names:
        ToolRegistry.unregister(tname)
    return "ok"


async def apply_enabled(entry: CatalogEntry, enabled: bool, mcp_manager: object) -> str:
    """将 entry 的上线/下线落实到注册表. 返回 'ok' | 'unsupported' | 'noop'."""
    from huginn.tools.registry import ToolRegistry

    kind = entry.kind

    if kind == "mcp":
        return await _apply_mcp(entry.name, enabled, mcp_manager)

    if kind == "tool":
        if not enabled:
            stashed = ToolRegistry.get(entry.name)
            if stashed is not None:
                _tool_stash[entry.name] = stashed
            ToolRegistry.unregister(entry.name)
            return "ok"
        stashed = _tool_stash.pop(entry.name, None)
        if stashed is None:
            logger.warning(f"[Catalog] No stash for tool '{entry.name}', cannot re-enable")
            return "unsupported"
        ToolRegistry.register(stashed)
        return "ok"

    # prompt / skill / model: 暂无禁用语义, 不强行落地.
    return "noop"