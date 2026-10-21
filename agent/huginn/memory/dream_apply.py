"""Consolidator 的显式 apply / rollback — 只支持可逆动作.

- soft_decay: importance 乘 0.8，回滚句柄保存原值；
- quarantine: archived=1，回滚恢复原 archived 状态；
- add_link / merge_duplicate / summarize / split_cluster / re_anchor 仍需专门实现，
  本模块返回 unsupported（保持提案，不假装执行）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huginn.memory.dream import RepairAction, apply_soft_decay


@dataclass(frozen=True)
class ApplyResult:
    repair_id: str
    action: str
    applied: bool
    supported: bool
    reason: str = ""
    handle: dict[str, Any] = field(default_factory=dict)


_SUPPORTED = frozenset({"soft_decay", "quarantine"})
_DECAY_FACTOR = 0.8


def apply_repair(
    repair: RepairAction,
    *,
    repair_id: str,
    longterm: Any,
    store: Any,
    dry_run: bool = False,
) -> ApplyResult:
    """执行一条可逆修复；不支持的动作保持提案状态."""
    if not repair.reversible or repair.action not in _SUPPORTED:
        return ApplyResult(
            repair_id=repair_id,
            action=repair.action,
            applied=False,
            supported=False,
            reason="action requires a dedicated executor; kept as proposal",
        )

    handle: dict[str, Any] = {}
    applied_targets: list[str] = []
    for memory_id in repair.target_ids:
        entry = longterm.get_by_id(memory_id)
        if not entry:
            continue
        applied_targets.append(memory_id)
        if repair.action == "soft_decay":
            before = float(entry.get("importance", 0.5) or 0.5)
            after, _ = apply_soft_decay(before, _DECAY_FACTOR)
            if not dry_run:
                longterm.update(memory_id, importance=after)
            handle[memory_id] = before
        else:  # quarantine
            before_archived = bool(entry.get("archived", 0))
            if not dry_run:
                longterm.update_archived(memory_id, True)
            handle[memory_id] = before_archived

    if not dry_run and applied_targets:
        store.set_repair_result(
            repair_id,
            applied=True,
            evidence_patch={"rollback": handle},
        )
    return ApplyResult(
        repair_id=repair_id,
        action=repair.action,
        applied=bool(applied_targets) and not dry_run,
        supported=True,
        reason="" if applied_targets else "no live target",
        handle=handle,
    )


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

    for memory_id, before in handle.items():
        if action == "soft_decay":
            longterm.update(memory_id, importance=float(before))
        elif action == "quarantine":
            longterm.update_archived(memory_id, bool(before))
    store.set_repair_result(
        repair_id,
        applied=False,
        evidence_patch={"rolled_back": True},
    )
    return ApplyResult(repair_id, action, False, True, "rolled_back", dict(handle))


__all__ = ["ApplyResult", "apply_repair", "rollback_repair"]
