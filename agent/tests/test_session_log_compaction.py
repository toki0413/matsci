"""T-BCSE-10/11: SessionEventLog compaction 边界 + 分支/回滚/快照.

验证:
  * build_state(boundary=True) 只返回最后 compaction 之后的事件 (LLM 可见窗口),
    全历史仍保留 (read_after / len 不减).
  * compaction_entries 返回当前路径上的边界清单.
  * branch_with_summary 移动叶 + 追加 branch_summary, 历史不删.
  * rollback_to 移动叶指针, 后续 append 从新叶继续.
  * snapshot / restore_from_snapshot 物化 + 精确恢复 (叶指针 + next_seq).
"""

from __future__ import annotations

from pathlib import Path

from huginn.events.session_log import (
    EVENT_BRANCH_SUMMARY,
    EVENT_COMPACTION,
    EVENT_MESSAGE,
    SessionEventLog,
)


def _log(tmp_path: Path, name: str = "s1") -> SessionEventLog:
    return SessionEventLog(name, Path(tmp_path) / f"{name}.jsonl", load=False)


# ── T-BCSE-10: compaction 边界 ────────────────────────────────────
def test_build_state_truncates_to_last_compaction(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    log.append(EVENT_MESSAGE, {"role": "assistant", "content": "a1"})
    log.append(EVENT_COMPACTION, {"summary": "compressed m1/a1", "first_kept_seq": 3})
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    log.append(EVENT_MESSAGE, {"role": "assistant", "content": "a2"})

    # boundary=True (LLM 窗口): 只含 compaction 之后的消息
    llm_view = log.build_state(boundary=True)
    contents = [ev.payload.get("content") for ev in llm_view]
    assert contents == ["m2", "a2"], f"LLM 应只见 compaction 后, got {contents}"

    # 全历史保留: 日志长度不变, boundary=False 返回完整路径
    assert len(log) == 5, "全历史必须保留"
    full = log.build_state(boundary=False)
    assert len(full) == 5
    assert [ev.payload.get("content") for ev in full][:2] == ["m1", "a1"]


def test_build_state_no_compaction_returns_all(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    log.append(EVENT_MESSAGE, {"role": "assistant", "content": "a1"})
    assert len(log.build_state(boundary=True)) == 2


def test_compaction_entries_lists_boundaries(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    log.append(EVENT_COMPACTION, {"summary": "s1"})
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    log.append(EVENT_COMPACTION, {"summary": "s2"})
    entries = log.compaction_entries()
    assert [e.summary for e in entries] == ["s1", "s2"]
    assert entries[0].boundary_seq == 1
    assert entries[0].first_kept_seq == 2


# ── T-BCSE-11: 分支 / 回滚 / 快照 ────────────────────────────────
def test_branch_with_summary_moves_leaf_and_appends(tmp_path: Path) -> None:
    log = _log(tmp_path)
    e1 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    e2 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    e3 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m3"})

    leaf = log.branch_with_summary(e1.seq, "abandoned m2/m3 path")
    assert log.leaf_id == leaf
    # 追加了 branch_summary 事件, 历史保留
    assert len(log) == 4
    tail = log.events_on_path()[-1]
    assert tail.kind == EVENT_BRANCH_SUMMARY
    assert tail.payload["summary"] == "abandoned m2/m3 path"
    # 路径只含 e1 + branch_summary
    assert [ev.id for ev in log.events_on_path()] == [e1.id, tail.id]
    assert log.get(e2.id) is not None and log.get(e3.id) is not None


def test_rollback_to_moves_leaf_history_kept(tmp_path: Path) -> None:
    log = _log(tmp_path)
    e1 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    e2 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    e3 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m3"})

    log.rollback_to(e1.seq)
    assert log.leaf_id == e1.id
    assert len(log) == 3, "rollback 不删历史"
    # 后续 append 从新叶继续 → 形成回滚分支
    e4 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m4"})
    assert e4.parent_id == e1.id
    assert [ev.id for ev in log.events_on_path()] == [e1.id, e4.id]


def test_snapshot_restore_roundtrip(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    e2 = log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m3"})

    snap = log.snapshot()
    assert snap["leaf_id"] == log.leaf_id
    assert snap["next_seq"] == 3
    assert len(snap["events"]) == 3

    # 从快照恢复出等价状态
    restored = _log(tmp_path, name="restored")
    restored.restore_from_snapshot(snap)
    assert restored.leaf_id == log.leaf_id
    assert restored.seq == log.seq
    assert [ev.payload["content"] for ev in restored.events_on_path()] == [
        "m1", "m2", "m3",
    ]


def test_snapshot_restore_then_incremental_replay(tmp_path: Path) -> None:
    """快照 + read_after 增量重放 = 快速恢复 (T-BCSE-11 加速路径)."""
    log = _log(tmp_path)
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m1"})
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m2"})
    snap = log.snapshot()
    # 快照后又追加
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m3"})
    log.append(EVENT_MESSAGE, {"role": "user", "content": "m4"})

    restored = _log(tmp_path, name="restored2")
    restored.restore_from_snapshot(snap)
    # 增量补齐 snap 之后的事件
    for ev in log.read_after(snap["next_seq"] - 1):
        restored.append(ev.kind, ev.payload)
    assert [ev.payload["content"] for ev in restored.events_on_path()] == [
        "m1", "m2", "m3", "m4",
    ]
