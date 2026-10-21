"""Consolidator 的显式 apply / rollback — 只做可逆修复.

七类修复都有专用执行器；每类都保存回滚句柄，且**不删除任何 direct 证据**：

- ``soft_decay``: importance 乘 0.8；句柄保存原值；
- ``quarantine``: archived=1；句柄保存原 archived 状态；
- ``merge_duplicate``: 给重复条目标 ``duplicate_of:<original>`` 并归档，
  保留原件与重复件两份；句柄保存 tags/archived；
- ``add_link``: 给源记忆追加 ``link:<target>`` 标签（索引层链接，不删内容）；
- ``summarize``: 新写一条 cluster_summary 记忆并归档源条目；句柄保存 summary id
  与源条目 archived 状态（回滚只归档摘要、恢复源条目，不物理删除）；
- ``split_cluster``: 摘掉 cluster 标签并恢复被 cluster 归档的条目；
- ``re_anchor``: 对**带可验证锚**的陈旧记忆延长保留层级到 long；无锚则不执行
  （无锚的陈旧记忆必须重新验证，不能靠改 TTL 冒充可信）。

缺少必要参数/目标不存在时返回 ``supported=True, applied=False`` 并说明原因；
未知动作才返回 ``supported=False``（保持提案，不假装执行）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huginn.memory.anchor import parse_anchor
from huginn.memory.dream import RepairAction, apply_soft_decay


@dataclass(frozen=True)
class ApplyResult:
    repair_id: str
    action: str
    applied: bool
    supported: bool
    reason: str = ""
    handle: dict[str, Any] = field(default_factory=dict)


_SUPPORTED = frozenset({
    "soft_decay",
    "quarantine",
    "merge_duplicate",
    "add_link",
    "summarize",
    "split_cluster",
    "re_anchor",
})
_DECAY_FACTOR = 0.8
_LINK_PREFIX = "link:"
_DUPLICATE_PREFIX = "duplicate_of:"


def _as_tags(entry: dict[str, Any] | None) -> list[str]:
    """把 memories.tags（JSON 串或 list）归一成 list[str]."""
    raw = (entry or {}).get("tags")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return [t.strip() for t in raw.split(",") if t.strip()]
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
    return []


def _supported(action: str) -> bool:
    return action in _SUPPORTED


def _fail(reason: str) -> tuple[bool, str, dict[str, Any], list[str]]:
    return False, reason, {}, []


# ── 各动作的 apply ───────────────────────────────────────────────────────


def _apply_soft_decay(
    longterm: Any, target_ids: tuple[str, ...], *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    handle: dict[str, Any] = {}
    applied: list[str] = []
    for memory_id in target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        before = float(entry.get("importance", 0.5) or 0.5)
        after, _ = apply_soft_decay(before, _DECAY_FACTOR)
        if not dry_run:
            longterm.update(memory_id, importance=after)
        handle[memory_id] = before
        applied.append(memory_id)
    return bool(applied), "" if applied else "no live target", handle, applied


def _apply_quarantine(
    longterm: Any, target_ids: tuple[str, ...], *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    handle: dict[str, Any] = {}
    applied: list[str] = []
    for memory_id in target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        handle[memory_id] = bool(entry.get("archived", 0))
        if not dry_run:
            longterm.update_archived(memory_id, True)
        applied.append(memory_id)
    return bool(applied), "" if applied else "no live target", handle, applied


def _apply_merge_duplicate(
    longterm: Any, target_ids: tuple[str, ...], *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    if len(target_ids) < 2:
        return _fail("merge_duplicate requires (duplicate_id, original_id)")
    duplicate_id, original_id = target_ids[0], target_ids[1]
    duplicate = longterm.get_by_id(duplicate_id)
    original = longterm.get_by_id(original_id)
    if not duplicate or not original:
        return _fail("duplicate or original memory not found")
    before_tags = _as_tags(duplicate)
    before_archived = bool(duplicate.get("archived", 0))
    tag = f"{_DUPLICATE_PREFIX}{original_id}"
    new_tags = before_tags if tag in before_tags else [*before_tags, tag]
    if not dry_run:
        longterm.update(duplicate_id, tags=new_tags)
        longterm.update_archived(duplicate_id, True)
    handle = {
        "duplicate_id": duplicate_id,
        "original_id": original_id,
        "before_archived": before_archived,
        "before_tags": before_tags,
    }
    return True, "", handle, [duplicate_id]


def _apply_add_link(
    longterm: Any, repair: RepairAction, *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    if not repair.target_ids:
        return _fail("add_link requires a source memory")
    source_id = repair.target_ids[0]
    link_to = str(repair.evidence.get("link_to") or "").strip()
    if not link_to and len(repair.target_ids) > 1:
        link_to = repair.target_ids[1]
    if not link_to:
        return _fail("add_link requires link_to (evidence or second target)")
    if link_to == source_id:
        return _fail("cannot link a memory to itself")
    source = longterm.get_by_id(source_id)
    if not source:
        return _fail("source memory not found")
    before_tags = _as_tags(source)
    tag = f"{_LINK_PREFIX}{link_to}"
    new_tags = before_tags if tag in before_tags else [*before_tags, tag]
    if not dry_run:
        longterm.update(source_id, tags=new_tags)
    handle = {"memory_id": source_id, "link_to": link_to, "before_tags": before_tags}
    return True, "", handle, [source_id]


def _apply_summarize(
    longterm: Any, repair: RepairAction, repair_id: str, *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    summary = str(repair.evidence.get("summary") or "").strip()
    if not summary:
        return _fail("summarize requires evidence.summary (no LLM call in executor)")
    if not callable(getattr(longterm, "store", None)):
        return _fail("longterm.store() unavailable; kept as proposal")
    sources: dict[str, bool] = {}
    source_rows: list[dict[str, Any]] = []
    for memory_id in repair.target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        sources[memory_id] = bool(entry.get("archived", 0))
        source_rows.append(entry)
    if not sources:
        return _fail("no live source memory")
    importance = max(
        (float(r.get("importance", 0.5) or 0.5) for r in source_rows), default=0.5
    )
    if dry_run:
        return True, "", {"summary_id": "<dry_run>", "sources": sources}, list(sources)
    try:
        summary_id = longterm.store(
            content=summary,
            category=str(repair.evidence.get("category", "fact")),
            tags=["cluster_summary", *sources, f"dream_summary:{repair_id}"],
            source=str(repair.evidence.get("source", "")),
            importance=importance,
            tier="long",
        )
    except Exception as exc:  # 存储失败保持提案，不假装执行
        return _fail(f"summarize store failed: {exc}")
    if not summary_id:
        return _fail("summarize store returned no id")
    for memory_id in sources:
        longterm.update_archived(memory_id, True)
    handle = {"summary_id": str(summary_id), "sources": sources}
    return True, "", handle, list(sources)


def _apply_split_cluster(
    longterm: Any, target_ids: tuple[str, ...], *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    handle: dict[str, Any] = {}
    applied: list[str] = []
    for memory_id in target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        tags = _as_tags(entry)
        kept = [t for t in tags if not (
            t == "cluster_summary" or t.startswith("cluster_summary_")
        )]
        if kept == tags:
            continue  # 不在任何 cluster，无法 split
        handle[memory_id] = {
            "tags": tags,
            "archived": bool(entry.get("archived", 0)),
        }
        if not dry_run:
            longterm.update(memory_id, tags=kept)
            if entry.get("archived", 0):
                longterm.update_archived(memory_id, False)
        applied.append(memory_id)
    if not applied:
        return _fail("no cluster membership tag to split")
    return True, "", handle, applied


def _apply_re_anchor(
    longterm: Any, target_ids: tuple[str, ...], *, dry_run: bool
) -> tuple[bool, str, dict[str, Any], list[str]]:
    handle: dict[str, Any] = {}
    applied: list[str] = []
    refused: list[str] = []
    for memory_id in target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        if parse_anchor(entry) is None:
            refused.append(memory_id)
            continue  # 无锚 → 必须重新验证，不能延长保留
        before_tier = str(entry.get("tier") or "mid")
        if before_tier == "long":
            continue
        handle[memory_id] = {"tier": before_tier}
        if not dry_run:
            longterm.update(memory_id, tier="long")
        applied.append(memory_id)
    if not applied:
        if refused:
            return _fail("no verifiable anchor; revalidation required")
        return _fail("no memory needs re-anchoring")
    return True, "", handle, applied


# ── 公共入口 ─────────────────────────────────────────────────────────────


def apply_repair(
    repair: RepairAction,
    *,
    repair_id: str,
    longterm: Any,
    store: Any,
    dry_run: bool = False,
) -> ApplyResult:
    """执行一条可逆修复；返回实际变更与回滚句柄."""
    if not repair.reversible or not _supported(repair.action):
        return ApplyResult(
            repair_id=repair_id,
            action=repair.action,
            applied=False,
            supported=False,
            reason="action requires a dedicated executor; kept as proposal",
        )

    if repair.action in {"soft_decay", "quarantine"}:
        fn = _apply_soft_decay if repair.action == "soft_decay" else _apply_quarantine
        applied, reason, handle, targets = fn(
            longterm, repair.target_ids, dry_run=dry_run
        )
    elif repair.action == "merge_duplicate":
        applied, reason, handle, targets = _apply_merge_duplicate(
            longterm, repair.target_ids, dry_run=dry_run
        )
    elif repair.action == "add_link":
        applied, reason, handle, targets = _apply_add_link(
            longterm, repair, dry_run=dry_run
        )
    elif repair.action == "summarize":
        applied, reason, handle, targets = _apply_summarize(
            longterm, repair, repair_id, dry_run=dry_run
        )
    elif repair.action == "split_cluster":
        applied, reason, handle, targets = _apply_split_cluster(
            longterm, repair.target_ids, dry_run=dry_run
        )
    else:  # re_anchor
        applied, reason, handle, targets = _apply_re_anchor(
            longterm, repair.target_ids, dry_run=dry_run
        )

    if applied and not dry_run:
        store.set_repair_result(
            repair_id,
            applied=True,
            evidence_patch={"rollback": handle},
        )
    return ApplyResult(
        repair_id=repair_id,
        action=repair.action,
        applied=applied and not dry_run,
        supported=True,
        reason=reason,
        handle=handle,
    )


def _rollback_entry(longterm: Any, action: str, handle: dict[str, Any]) -> None:
    if action == "soft_decay":
        for memory_id, before in handle.items():
            longterm.update(memory_id, importance=float(before))
        return
    if action == "quarantine":
        for memory_id, before in handle.items():
            longterm.update_archived(memory_id, bool(before))
        return
    if action == "merge_duplicate":
        duplicate_id = handle.get("duplicate_id")
        if duplicate_id:
            longterm.update(
                duplicate_id, tags=list(handle.get("before_tags") or [])
            )
            longterm.update_archived(
                duplicate_id, bool(handle.get("before_archived", False))
            )
        return
    if action == "add_link":
        memory_id = handle.get("memory_id")
        if memory_id:
            longterm.update(memory_id, tags=list(handle.get("before_tags") or []))
        return
    if action == "summarize":
        summary_id = handle.get("summary_id")
        if summary_id:
            longterm.update_archived(str(summary_id), True)
        for memory_id, before in (handle.get("sources") or {}).items():
            longterm.update_archived(memory_id, bool(before))
        return
    if action == "split_cluster":
        for memory_id, before in handle.items():
            if not isinstance(before, dict):
                continue
            longterm.update(memory_id, tags=list(before.get("tags") or []))
            longterm.update_archived(memory_id, bool(before.get("archived", False)))
        return
    if action == "re_anchor":
        for memory_id, before in handle.items():
            tier = str((before or {}).get("tier") or "short")
            longterm.update(memory_id, tier=tier)


def rollback_repair(
    repair_id: str,
    *,
    longterm: Any,
    store: Any,
) -> ApplyResult:
    """按 rollback 句柄恢复；只对已 applied 的修复生效."""
    record = store.get_repair(repair_id)
    if record is None:
        return ApplyResult(repair_id, "unknown", False, False, "repair not found")
    action = record.get("action", "")
    if not record.get("applied"):
        return ApplyResult(repair_id, action, False, True, "not applied")
    handle = (record.get("evidence_json") or {}).get("rollback") or {}
    if not isinstance(handle, dict):
        return ApplyResult(repair_id, action, False, False, "no rollback handle")

    _rollback_entry(longterm, action, handle)
    store.set_repair_result(
        repair_id,
        applied=False,
        evidence_patch={"rolled_back": True},
    )
    return ApplyResult(repair_id, action, False, True, "rolled_back", dict(handle))


__all__ = ["ApplyResult", "apply_repair", "rollback_repair"]
