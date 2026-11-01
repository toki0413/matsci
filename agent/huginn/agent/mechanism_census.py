"""机制普查段 —— 把 Catalog 的已启用接入项投影成模型可见的一段文本.

披露等级按 kind 声明 (`KIND_DISCLOSURE`): 小集合列名 (always), 大集合只给数量
与取用入口 (on_demand), 框架内部段声明 hidden 并写理由. 未声明的 kind 由门判红,
避免「新接入的机制默认不可见」.

Catalog 在函数内懒 import: 本模块导入时不拉起注册表重栈, 门可离线跑.
"""

from __future__ import annotations

import logging

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


def census_text() -> str:
    """生成 `## MECHANISMS` 段; Catalog 不可用或无条目时返回空串 (该段缺席)."""
    try:
        from huginn.catalog.manager import CatalogManager

        entries = CatalogManager().discover_all()
    except Exception:  # 非致命: Catalog 不可用则该段缺席, prompt 构建不崩
        logger.debug("mechanism census unavailable", exc_info=True)
        return ""

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
            more = "" if len(enabled) <= _MAX_NAMES else f" (+{len(enabled) - _MAX_NAMES})"
            lines.append(f"- {kind} ({len(enabled)}): {shown}{more}")
        else:
            hint = _ON_DEMAND_HINT.get(kind, "discoverable on demand")
            lines.append(f"- {kind} ({len(enabled)}): {hint}")

    if not lines:
        return ""
    return "## MECHANISMS\nRegistered capabilities (source: catalog).\n" + "\n".join(lines)


__all__ = ["KIND_DISCLOSURE", "census_text", "disclosure_level", "disclosure_reason"]
