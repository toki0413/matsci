"""rewind 数据面 —— 以"用户消息"为锚点的回滚规划与执行.

已有三层能力各自独立:
  - 会话级: ``SessionEventLog.branch`` / ``rollback_to`` —— 只挪叶指针, 不动文件.
  - 文件级: ``SnapshotManager.revert(step_id)``          —— 只回滚单步, 不认锚点.
  - 溯源级: ``ProvenanceRegistry.rollback_to(event_id)``  —— 按 provenance event id 回滚.

缺的是**把三者按"用户消息"这一个用户真正能理解的锚点串起来**, 并在动手之前
给出 dry-run 影响清单 (哪些文件会被删掉 / 还原, 涉及哪些步骤). 本模块补这一层.

语义::

    anchor   = 会话事件日志里的一条 user message (kind=message, payload.role=user)
    作用域   = 锚点之后发生的全部文件快照步骤
    相关性   = 优先按 ``snapshot.session_id == session_id`` 精确匹配;
               老记录没有 session_id → 退化为按 timestamp 匹配, 结果里显式标
               ``correlation="timestamp"``, 让调用方知道这条是"猜"的
    plan     = 只读, 不改任何东西 (dry-run)
    apply    = 按**时间倒序**逐步 revert, 再把会话叶指针挪回锚点之前

为什么必须倒序: 设步骤 S1 → S2, S1 前状态 A, S2 前状态 B. 要回到 A:
  先 revert(S2) 得 B, 再 revert(S1) 得 A —— 正确.
  正序则先 revert(S1) 得 A, 但 S2 新建的文件还在, 再 revert(S2) 又把 B 的内容
  写回来 —— 错.

用法::

    from huginn.snapshot.rewind import list_anchors, plan_rewind, apply_rewind

    anchors = list_anchors("thread-1", limit=5)
    impact = plan_rewind("thread-1", anchors[0])      # dry-run, 给用户看
    print(impact.to_dict())
    if user_confirms:
        apply_rewind("thread-1", anchors[0])          # 真回滚
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.events.session_log import (
    EVENT_BRANCH_SUMMARY,
    EVENT_MESSAGE,
    SessionEventLog,
)
from huginn.snapshot.file_snapshot import FilePatch, FileSnapshot, SnapshotManager

logger = logging.getLogger(__name__)

# 作用域内步骤上限: 超了就只保留最近的 (回滚最关心最近发生的事), 并置 truncated.
_MAX_STEPS_DEFAULT = 200
# 锚点消息预览截断长度.
_PREVIEW_LEN = 200


@dataclass(frozen=True)
class RewindAnchor:
    """一条可回滚到的用户消息锚点."""

    session_id: str
    event_id: str
    seq: int
    ts: float
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "seq": self.seq,
            "ts": self.ts,
            "preview": self.preview,
        }


@dataclass
class FileImpact:
    """回滚对单个文件的净影响 (跨作用域内所有步骤合并后)."""

    path: str                  # 相对工作区路径
    effect: str                # "remove" (锚点时尚不存在, 回滚会删掉) | "restore" (还原)
    at_anchor: str             # "absent" | "present" —— 锚点时刻该文件是否存在
    currently_present: bool    # 现在是否还在 (effect=restore 且不在 → 回滚后重新出现)
    steps: list[str] = field(default_factory=list)   # 触碰过它的 step_id, 时间升序
    preview_at_anchor: str = ""    # 锚点时刻内容前 500 字符
    preview_now: str = ""          # 当前内容前 500 字符

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "effect": self.effect,
            "at_anchor": self.at_anchor,
            "currently_present": self.currently_present,
            "steps": list(self.steps),
            "preview_at_anchor": self.preview_at_anchor,
            "preview_now": self.preview_now,
        }


@dataclass
class RewindImpact:
    """一次 rewind 的作用域与净影响. ``plan_rewind`` 返回 dry-run 版 (applied=False)."""

    session_id: str
    anchor: RewindAnchor
    steps: list[dict[str, Any]] = field(default_factory=list)   # 时间升序
    files: list[FileImpact] = field(default_factory=list)
    truncated: bool = False
    applied: bool = False
    reverted_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)      # 已回滚过, 跳过
    failed_steps: list[str] = field(default_factory=list)
    leaf_moved_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "anchor": self.anchor.to_dict(),
            "steps": self.steps,
            "files": [f.to_dict() for f in self.files],
            "summary": self.summary(),
            "truncated": self.truncated,
            "applied": self.applied,
            "reverted_steps": list(self.reverted_steps),
            "skipped_steps": list(self.skipped_steps),
            "failed_steps": list(self.failed_steps),
            "leaf_moved_to": self.leaf_moved_to,
        }

    def summary(self) -> dict[str, int]:
        """给 UI/日志用的一行计数."""
        return {
            "steps": len(self.steps),
            "files": len(self.files),
            "removed": sum(1 for f in self.files if f.effect == "remove"),
            "restored": sum(1 for f in self.files if f.effect == "restore"),
        }


# ── 锚点 ─────────────────────────────────────────────────────────


def list_anchors(
    session_id: str,
    *,
    limit: int = 20,
    log: SessionEventLog | None = None,
) -> list[RewindAnchor]:
    """列出会话里可回滚到的用户消息锚点, **最新在前**.

    只收 ``kind=message`` 且 ``payload.role == "user"`` 的事件 (见
    ``events/session_writer.record_user_message``). ``log`` 可注入 (测试用);
    缺省按 session_id 打开事件日志, 打不开则返回空 (fail-open).
    """
    if log is None:
        try:
            log = SessionEventLog.open(session_id)
        except Exception:  # noqa: BLE001 — 日志不可用 → 无锚点, 不抛
            logger.debug("rewind: open session log %s failed", session_id, exc_info=True)
            return []

    anchors: list[RewindAnchor] = []
    for ev in log:
        if ev.kind != EVENT_MESSAGE:
            continue
        payload = ev.payload or {}
        if payload.get("role") != "user":
            continue
        anchors.append(
            RewindAnchor(
                session_id=session_id,
                event_id=ev.id,
                seq=ev.seq,
                ts=ev.ts,
                preview=str(payload.get("content", ""))[:_PREVIEW_LEN],
            )
        )
    anchors.reverse()   # 最新在前
    return anchors[: max(0, limit)]


# ── 规划 (dry-run) ───────────────────────────────────────────────


def _correlation(
    snap: FileSnapshot, session_id: str, anchor_ts: float
) -> str | None:
    """这条快照是否落在锚点作用域内? 是则返回相关性标签, 否则 None.

    - 早于锚点的步骤一律出局 (rewind 只管锚点之后).
    - 有 session_id 的按会话精确匹配.
    - 没有 session_id 的老记录退化为按时间戳, 标签标成 "timestamp" 以便调用方
      区分"确定的"和"猜的".
    """
    if snap.timestamp < anchor_ts:
        return None
    if snap.session_id:
        return "session" if snap.session_id == session_id else None
    return "timestamp"


def plan_rewind(
    session_id: str,
    anchor: RewindAnchor,
    *,
    manager: SnapshotManager | None = None,
    max_steps: int = _MAX_STEPS_DEFAULT,
) -> RewindImpact:
    """算一次 rewind 的作用域与净影响. **只读**, 不动文件也不动会话叶指针."""
    mgr = manager or SnapshotManager()

    scoped: list[tuple[FileSnapshot, str]] = []
    for snap in mgr.get_history():
        label = _correlation(snap, session_id, anchor.ts)
        if label is not None:
            scoped.append((snap, label))
    scoped.sort(key=lambda t: t[0].timestamp)

    truncated = len(scoped) > max_steps
    if truncated:
        scoped = scoped[-max_steps:]

    steps: list[dict[str, Any]] = []
    # 每个文件**最早**进入作用域的那一步决定了锚点时刻的状态:
    # old_hash is None → 锚点时尚不存在 → 回滚会删掉它; 否则 → 还原内容.
    first: dict[str, tuple[FileSnapshot, FilePatch]] = {}
    touched: dict[str, list[str]] = {}
    latest_preview: dict[str, str] = {}

    for snap, label in scoped:
        steps.append(
            {
                "step_id": snap.step_id,
                "tool_name": snap.tool_name,
                "timestamp": snap.timestamp,
                "workspace": snap.workspace,
                "patches": len(snap.patches),
                "reverted": snap.reverted,
                "correlation": label,
            }
        )
        for p in snap.patches:
            touched.setdefault(p.file_path, []).append(snap.step_id)
            latest_preview[p.file_path] = p.new_content_preview
            first.setdefault(p.file_path, (snap, p))

    files: list[FileImpact] = []
    for path, (snap, p) in first.items():
        at_anchor = "absent" if p.old_hash is None else "present"
        effect = "remove" if p.old_hash is None else "restore"
        files.append(
            FileImpact(
                path=path,
                effect=effect,
                at_anchor=at_anchor,
                currently_present=(Path(snap.workspace) / path).is_file()
                if snap.workspace
                else False,
                steps=touched.get(path, []),
                preview_at_anchor=p.old_content_preview,
                preview_now=latest_preview.get(path, ""),
            )
        )
    files.sort(key=lambda f: f.path)

    return RewindImpact(
        session_id=session_id,
        anchor=anchor,
        steps=steps,
        files=files,
        truncated=truncated,
    )


# ── 执行 ─────────────────────────────────────────────────────────


def _move_leaf(
    log: SessionEventLog, anchor: RewindAnchor, keep_anchor: bool, summary: str
) -> str | None:
    """把会话叶指针挪回锚点之前, 并落一条 ``branch_summary`` 事件.

    必须落事件: ``branch()`` 只改内存里的叶指针, 重开日志时叶指针会被重置成
    "最后一条事件", 回滚就白做了. ``branch_with_summary`` 追加的事件本身成为
    新叶, 重启后仍是它.
    """
    target: str | None
    if keep_anchor:
        target = anchor.event_id
    else:
        ev = log.get(anchor.event_id)
        target = ev.parent_id if ev is not None else None

    if target:
        return log.branch_with_summary(target, summary)
    # 锚点是根事件且不保留 → 叶指针归零, 后续 append 从新的根开始
    log.reset_leaf()
    return log.append(EVENT_BRANCH_SUMMARY, {"summary": summary}).id


def apply_rewind(
    session_id: str,
    anchor: RewindAnchor,
    *,
    manager: SnapshotManager | None = None,
    log: SessionEventLog | None = None,
    keep_anchor: bool = False,
    move_leaf: bool = True,
    max_steps: int = _MAX_STEPS_DEFAULT,
) -> RewindImpact:
    """真正执行 rewind: 倒序 revert 作用域内所有未回滚的步骤, 再挪会话叶指针.

    Args:
        keep_anchor: True 时保留锚点那条用户消息 (叶指针停在它上面), 默认 False
            —— 连锚点消息一起丢掉, 等价"回到发这条消息之前".
        move_leaf: False 时只回滚文件, 不动会话 (纯文件级 undo).

    单个步骤 revert 失败不中断其余步骤, 失败 step_id 收进 ``failed_steps``;
    已回滚过的收进 ``skipped_steps``. 返回的 impact 里 ``applied=True``.
    """
    mgr = manager or SnapshotManager()
    impact = plan_rewind(session_id, anchor, manager=mgr, max_steps=max_steps)

    # 倒序: 先撤最近的步骤 (见模块 docstring 里的 S1/S2 说明)
    for step in reversed(impact.steps):
        sid = step["step_id"]
        if step["reverted"]:
            impact.skipped_steps.append(sid)
            continue
        ws = Path(step["workspace"]) if step["workspace"] else Path.cwd()
        try:
            mgr.revert(sid, ws)
            impact.reverted_steps.append(sid)
        except Exception:  # noqa: BLE001 — 单步失败不拖垮整次 rewind
            logger.warning("rewind: revert %s failed (non-fatal)", sid, exc_info=True)
            impact.failed_steps.append(sid)

    if move_leaf:
        if log is None:
            try:
                log = SessionEventLog.open(session_id)
            except Exception:  # noqa: BLE001 — 挪不动会话也不影响文件已回滚
                logger.debug(
                    "rewind: open session log %s failed, leaf untouched",
                    session_id,
                    exc_info=True,
                )
                log = None
        if log is not None:
            summary = (
                f"rewind to message seq={anchor.seq}: "
                f"{len(impact.reverted_steps)} steps reverted, "
                f"{len(impact.files)} files affected"
            )
            impact.leaf_moved_to = _move_leaf(log, anchor, keep_anchor, summary)

    impact.applied = True
    logger.info(
        "rewind %s @seq=%d: %d reverted, %d skipped, %d failed, %d files",
        session_id,
        anchor.seq,
        len(impact.reverted_steps),
        len(impact.skipped_steps),
        len(impact.failed_steps),
        len(impact.files),
    )
    return impact


__all__ = [
    "FileImpact",
    "RewindAnchor",
    "RewindImpact",
    "apply_rewind",
    "list_anchors",
    "plan_rewind",
]
