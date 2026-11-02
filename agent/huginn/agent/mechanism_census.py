"""机制普查段 —— 把 Catalog 的已启用接入项投影成模型可见的一段文本.

披露等级按 kind 声明 (`KIND_DISCLOSURE`): 小集合列名 (always), 大集合只给数量
与取用入口 (on_demand), 框架内部段声明 hidden 并写理由. 未声明的 kind 由门判红,
避免「新接入的机制默认不可见」.

Catalog 在函数内懒 import: 本模块导入时不拉起注册表重栈, 门可离线跑.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# kind -> 等级. 与 huginn.catalog.models.KINDS 一一对应; hidden 必须带理由.
KIND_DISCLOSURE: dict[str, str] = {
    "tool": "on_demand",
    "skill": "always",
    "plugin": "always",
    "workflow": "always",
    "mcp": "on_demand",
    "model": "on_demand",
    "prompt": "hidden: prompt 段已在 system prompt 正文里生效, 再列一遍是重复",
}

# on_demand 级别的取用入口说明: 模型据此知道「看不见 ≠ 不存在」.
_ON_DEMAND_HINT: dict[str, str] = {
    "tool": "call `tool_search` when a needed tool is not in your visible list",
    "mcp": "call `tool_search`; MCP servers expose their tools the same way",
    "model": "selectable through model routing, not a callable tool",
}

# 列名上限. 超过则截断, 避免把 prompt 变成清单.
_MAX_NAMES = 20


def disclosure_level(kind: str) -> str:
    """返回 kind 的披露等级 ('always' / 'on_demand' / 'hidden')."""
    raw = KIND_DISCLOSURE.get(kind)
    if raw is None:
        raise ValueError(f"catalog kind {kind!r} has no disclosure declaration")
    return raw.split(":", 1)[0].strip()


def disclosure_reason(kind: str) -> str:
    """hidden 等级的理由; 非 hidden 返回空串."""
    if disclosure_level(kind) != "hidden":
        return ""
    raw = KIND_DISCLOSURE.get(kind, "")
    return raw.split(":", 1)[1].strip() if ":" in raw else ""


def _discover_entries() -> list[Any]:
    """取本进程可见的 Catalog 条目.

    优先级: 复用 `server_context` 里**已初始化**的共享 Catalog 视图 (它由 lifespan
    带 mcp_manager/model_registry/plugins_dir 采集过, 因而是 mcp/model/plugin 三类
    唯一可见的来源); 视图为空时才用它补采一次. 没有共享上下文时不创建上下文
    (那会拉起整套服务装配), 退化为本地一次性采集 —— 缺 handle 的 kind 如实缺席,
    不伪造条目.
    """
    module = sys.modules.get("huginn.server_context")
    ctx = getattr(module, "_server_context", None) if module is not None else None
    if ctx is not None:
        catalog = getattr(ctx, "catalog", None)
        if catalog is not None:
            entries = catalog.list()
            if entries:
                return entries
            from huginn.catalog.manager import discover_handles

            return catalog.discover_all(**discover_handles(ctx))

    from huginn.catalog.manager import CatalogManager

    return CatalogManager().discover_all()


def census_text(entries: Iterable[Any] | None = None) -> str:
    """生成 `## MECHANISMS` 段.

    `entries` 省略时从本进程 Catalog 取 (见 _discover_entries); 传入时是纯投影,
    便于门禁以合成条目离线判定披露规则.

    永不返回空串: 「没有可披露的能力」本身也是必须告知模型的事实, 返回空段会让
    「声明该段生效」与「实际渲染为空」对不上, 把披露退化成静默.
    """
    if entries is None:
        try:
            entries = _discover_entries()
        except Exception:  # 非致命: Catalog 不可用时如实告知, prompt 构建不崩
            logger.debug("mechanism census unavailable", exc_info=True)
            return "## MECHANISMS\nCapability catalog unavailable this run."

    return _render_census(entries)


def _render_census(entries: Iterable[Any]) -> str:
    """把 Catalog 条目投影成文本 (纯函数, 无 I/O)."""
    lines: list[str] = []
    for kind in sorted({e.kind for e in entries}):
        try:
            level = disclosure_level(kind)
        except ValueError:
            logger.warning("catalog kind %s has no disclosure declaration", kind)
            continue
        enabled = sorted(e.name for e in entries if e.kind == kind and e.enabled)
        if not enabled or level == "hidden":
            continue
        if level == "always":
            shown = ", ".join(enabled[:_MAX_NAMES])
            more = (
                ""
                if len(enabled) <= _MAX_NAMES
                else f" (+{len(enabled) - _MAX_NAMES} more; use `tool_search` to look one up)"
            )
            lines.append(f"- {kind} ({len(enabled)}): {shown}{more}")
        else:
            hint = _ON_DEMAND_HINT.get(kind, "discoverable on demand")
            lines.append(f"- {kind} ({len(enabled)}): {hint}")

    if not lines:
        return "## MECHANISMS\nNo capabilities registered."
    return "## MECHANISMS\nRegistered capabilities (source: catalog).\n" + "\n".join(lines)


__all__ = ["KIND_DISCLOSURE", "census_text", "disclosure_level", "disclosure_reason"]
