"""Catalog 只读检索门面 —— 统一目录的"读"侧, 与 CatalogManager(控制面)分离.

P1 孤岛收口的单一入口: LLM 不必分别猜 capability_tool / skill / workflow_tool,
一次 ``search_entries`` 就能跨 tool/capability/skill/workflow 找到名字 + 描述 + 来源.
**不新增注册表**: 复用 manager 的采集器, 各注册表仍是唯一事实源.
"""
from __future__ import annotations

from typing import Any

from huginn.catalog.manager import _kind_collectors
from huginn.catalog.models import ORIGIN_PRIORITY, CatalogEntry

# LLM 一次检索默认返回条数上限 —— 防止目录快照淹掉上下文
DEFAULT_LIMIT = 20


def collect_entries(**handles: Any) -> list[CatalogEntry]:
    """实时采集并按 (kind, name) 去重; 单项采集失败不拖垮整张目录."""
    raw: dict[str, CatalogEntry] = {}
    for entries in _kind_collectors(handles).values():
        for entry in entries:
            prev = raw.get(entry.id)
            if prev is None or ORIGIN_PRIORITY[entry.origin] > ORIGIN_PRIORITY[prev.origin]:
                raw[entry.id] = entry
    return sorted(raw.values(), key=lambda e: (e.kind, e.name))


def _haystack(entry: CatalogEntry) -> str:
    meta = entry.meta or {}
    parts = [
        entry.name,
        str(meta.get("description", "")),
        str(meta.get("category", "")),
        " ".join(str(t) for t in (meta.get("tags") or [])),
    ]
    return " ".join(parts).lower()


def search_entries(query: str = "", kind: str | None = None,
                   limit: int = DEFAULT_LIMIT, **handles: Any) -> list[CatalogEntry]:
    """按关键词 + kind 过滤统一目录; 空 query 返回排序后的前 ``limit`` 条.

    ``kind`` 支持单值或逗号分隔多值 (如 ``"skill,workflow"``); 未知 kind 返回空.
    """
    entries = collect_entries(**handles)
    if kind:
        wanted = {k.strip() for k in kind.split(",") if k.strip()}
        entries = [e for e in entries if e.kind in wanted]
    q = (query or "").strip().lower()
    if q:
        entries = [e for e in entries if q in _haystack(e)]
    cap = max(1, int(limit))
    return entries[:cap]


def summarize(entries: list[CatalogEntry]) -> dict[str, int]:
    """按 kind 计数 (无条目时也给出 0, 便于调用方判断"这一类整体缺失")."""
    out = {"tool": 0, "capability": 0, "skill": 0, "workflow": 0}
    for e in entries:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out
