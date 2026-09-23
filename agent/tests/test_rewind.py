"""rewind 数据面测试 —— 以"用户消息"为锚点的回滚规划与执行.

覆盖:
  - list_anchors 只收用户消息, 最新在前
  - plan_rewind 是纯 dry-run (不动文件, 不动会话)
  - 影响清单的净效果判定 (created→remove / modified→restore)
  - apply_rewind 倒序回滚 (同一文件多步修改后回到最初内容)
  - 会话叶指针挪动 (keep_anchor 两种语义)
  - 作用域: 按 session_id 精确匹配, 老记录退化为时间戳并标 correlation
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from pathlib import Path

import pytest

from huginn.events.session_log import SessionEventLog
from huginn.snapshot.file_snapshot import SnapshotManager
from huginn.snapshot.rewind import (
    RewindAnchor,
    apply_rewind,
    list_anchors,
    plan_rewind,
)

SESSION = "s1"


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tmp_root():
    """独立 temp 目录, 测试完后台删 (AV 慢)."""
    root = Path(tempfile.mkdtemp(prefix="rewind_test_"))
    yield root
    import threading

    threading.Thread(
        target=shutil.rmtree, args=(root,), kwargs={"ignore_errors": True},
        daemon=True,
    ).start()


@pytest.fixture
def ws(tmp_root):
    d = tmp_root / "ws"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mgr(tmp_root):
    """独立 SnapshotManager, 不碰全局 ~/.huginn."""
    return SnapshotManager(root=tmp_root / "snapshots")


@pytest.fixture
def slog(tmp_root):
    return SessionEventLog.create(SESSION, tmp_root / "s1.jsonl")


# ── 场景构造 ──────────────────────────────────────────────────


def _scenario(mgr: SnapshotManager, slog: SessionEventLog, ws: Path):
    """一条用户消息 → 两步文件改动.

    锚点后:
      step1: a.dat  a1 → a2
      step2: b.dat  b1 → b2 ; c.dat 新建
    返回 (anchor, sid1, sid2).
    """
    slog.append("message", {"role": "user", "content": "改两个文件"})
    anchor = list_anchors(SESSION, log=slog)[0]

    (ws / "a.dat").write_text("a1\n", encoding="utf-8")
    (ws / "b.dat").write_text("b1\n", encoding="utf-8")

    sid1 = mgr.track("file_write_tool", ws, session_id=SESSION)
    (ws / "a.dat").write_text("a2\n", encoding="utf-8")
    mgr.patch(sid1, ws)

    sid2 = mgr.track("file_write_tool", ws, session_id=SESSION)
    (ws / "b.dat").write_text("b2\n", encoding="utf-8")
    (ws / "c.dat").write_text("c1\n", encoding="utf-8")
    mgr.patch(sid2, ws)

    return anchor, sid1, sid2


# ── 锚点 ─────────────────────────────────────────────────────


class TestListAnchors:
    def test_only_user_messages(self, slog):
        slog.append("message", {"role": "user", "content": "第一条"})
        slog.append("message", {"role": "assistant", "content": "回复"})
        slog.append("message", {"role": "user", "content": "第二条"})
        slog.append("tool_call", {"name": "file_write_tool", "args": {}})

        anchors = list_anchors(SESSION, log=slog)

        assert len(anchors) == 2
        assert [a.preview for a in anchors] == ["第二条", "第一条"]

    def test_newest_first_and_limit(self, slog):
        for i in range(5):
            slog.append("message", {"role": "user", "content": f"msg-{i}"})

        anchors = list_anchors(SESSION, log=slog, limit=3)

        assert [a.preview for a in anchors] == ["msg-4", "msg-3", "msg-2"]

    def test_anchor_carries_seq_and_id(self, slog):
        slog.append("message", {"role": "user", "content": "只有一条"})
        a = list_anchors(SESSION, log=slog)[0]
        assert a.session_id == SESSION
        assert a.seq == 0
        assert a.event_id

    def test_empty_log_returns_empty(self, slog):
        assert list_anchors(SESSION, log=slog) == []


# ── dry-run ───────────────────────────────────────────────────


class TestPlanRewind:
    def test_plan_does_not_touch_files(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)

        assert impact.applied is False
        assert impact.reverted_steps == []
        # 文件原封不动
        assert (ws / "a.dat").read_text() == "a2\n"
        assert (ws / "b.dat").read_text() == "b2\n"
        assert (ws / "c.dat").read_text() == "c1\n"

    def test_scope_covers_all_steps_after_anchor(self, mgr, slog, ws):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)

        assert [s["step_id"] for s in impact.steps] == [sid1, sid2]
        assert all(s["correlation"] == "session" for s in impact.steps)

    def test_created_file_effect_is_remove(self, mgr, slog, ws):
        anchor, _sid1, sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)
        by_path = {f.path: f for f in impact.files}

        assert by_path["c.dat"].effect == "remove"
        assert by_path["c.dat"].at_anchor == "absent"
        assert by_path["c.dat"].steps == [sid2]

    def test_modified_file_effect_is_restore(self, mgr, slog, ws):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)
        by_path = {f.path: f for f in impact.files}

        assert by_path["a.dat"].effect == "restore"
        assert by_path["a.dat"].at_anchor == "present"
        assert by_path["a.dat"].preview_at_anchor.startswith("a1")
        assert by_path["a.dat"].preview_now.startswith("a2")
        assert by_path["a.dat"].steps == [sid1]

        # b.dat 只在 step2 被改
        assert by_path["b.dat"].steps == [sid2]

    def test_steps_before_anchor_excluded(self, mgr, slog, ws):
        # 锚点之前先改一次文件
        (ws / "old.dat").write_text("v1\n", encoding="utf-8")
        pre = mgr.track("file_write_tool", ws, session_id=SESSION)
        (ws / "old.dat").write_text("v2\n", encoding="utf-8")
        mgr.patch(pre, ws)

        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)

        assert pre not in [s["step_id"] for s in impact.steps]
        assert "old.dat" not in {f.path for f in impact.files}

    def test_summary_counts(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        summary = plan_rewind(SESSION, anchor, manager=mgr).summary()

        assert summary == {"steps": 2, "files": 3, "removed": 1, "restored": 2}

    def test_truncation_keeps_most_recent(self, mgr, slog, ws):
        anchor, _sid1, sid2 = _scenario(mgr, slog, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr, max_steps=1)

        assert impact.truncated is True
        assert [s["step_id"] for s in impact.steps] == [sid2]

    def test_to_dict_is_json_shaped(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        d = plan_rewind(SESSION, anchor, manager=mgr).to_dict()

        assert d["anchor"]["session_id"] == SESSION
        assert d["summary"]["files"] == 3
        assert d["applied"] is False
        assert {f["path"] for f in d["files"]} == {"a.dat", "b.dat", "c.dat"}


# ── 作用域相关性 ───────────────────────────────────────────────


class TestScopeCorrelation:
    def test_other_session_excluded(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        # 另一会话的步骤 (时间在锚点之后), 不该被卷进来
        other = mgr.track("file_write_tool", ws, session_id="other")
        (ws / "d.dat").write_text("d1\n", encoding="utf-8")
        mgr.patch(other, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)

        assert other not in [s["step_id"] for s in impact.steps]
        assert "d.dat" not in {f.path for f in impact.files}

    def test_legacy_record_falls_back_to_timestamp(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        # 老记录没有 session_id
        legacy = mgr.track("file_write_tool", ws)
        (ws / "e.dat").write_text("e1\n", encoding="utf-8")
        mgr.patch(legacy, ws)

        impact = plan_rewind(SESSION, anchor, manager=mgr)
        by_id = {s["step_id"]: s for s in impact.steps}

        assert legacy in by_id
        assert by_id[legacy]["correlation"] == "timestamp"


# ── 执行 ─────────────────────────────────────────────────────


class TestApplyRewind:
    def test_restores_modified_and_removes_created(self, mgr, slog, ws):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)

        impact = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert impact.applied is True
        assert sorted(impact.reverted_steps) == sorted([sid1, sid2])
        assert impact.failed_steps == []
        assert (ws / "a.dat").read_text() == "a1\n"
        assert (ws / "b.dat").read_text() == "b1\n"
        assert not (ws / "c.dat").exists()

    def test_reverse_order_two_writes_same_file(self, mgr, slog, ws):
        """同一文件连改两次: 回滚后应回到最初内容, 而不是中间态."""
        (ws / "x.dat").write_text("v1\n", encoding="utf-8")
        slog.append("message", {"role": "user", "content": "改两次"})
        anchor = list_anchors(SESSION, log=slog)[0]

        s1 = mgr.track("file_write_tool", ws, session_id=SESSION)
        (ws / "x.dat").write_text("v2\n", encoding="utf-8")
        mgr.patch(s1, ws)

        s2 = mgr.track("file_write_tool", ws, session_id=SESSION)
        (ws / "x.dat").write_text("v3\n", encoding="utf-8")
        mgr.patch(s2, ws)

        assert (ws / "x.dat").read_text() == "v3\n"

        apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert (ws / "x.dat").read_text() == "v1\n"

    def test_moves_leaf_dropping_anchor_message(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        impact = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        # 锚点是根事件且不保留 → 新根是那条 branch_summary
        assert impact.leaf_moved_to is not None
        path = slog.events_on_path()
        assert len(path) == 1
        assert path[0].kind == "branch_summary"
        assert path[0].id == impact.leaf_moved_to

    def test_keep_anchor_message(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)

        apply_rewind(
            SESSION, anchor, manager=mgr, log=slog, keep_anchor=True
        )

        path = slog.events_on_path()
        assert [e.kind for e in path] == ["message", "branch_summary"]
        assert path[0].id == anchor.event_id

    def test_move_leaf_false_leaves_session_alone(self, mgr, slog, ws):
        anchor, _sid1, _sid2 = _scenario(mgr, slog, ws)
        before = slog.leaf_id

        impact = apply_rewind(
            SESSION, anchor, manager=mgr, log=slog, move_leaf=False
        )

        assert impact.leaf_moved_to is None
        assert slog.leaf_id == before
        # 文件照样回滚了
        assert (ws / "a.dat").read_text() == "a1\n"

    def test_already_reverted_step_is_skipped(self, mgr, slog, ws):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)
        mgr.revert(sid1, ws)

        impact = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert impact.skipped_steps == [sid1]
        assert impact.reverted_steps == [sid2]

    def test_second_apply_skips_everything(self, mgr, slog, ws):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)

        apply_rewind(SESSION, anchor, manager=mgr, log=slog)
        second = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert sorted(second.skipped_steps) == sorted([sid1, sid2])
        assert second.reverted_steps == []

    def test_revert_failure_is_isolated(self, mgr, slog, ws, monkeypatch):
        anchor, sid1, sid2 = _scenario(mgr, slog, ws)

        real_revert = mgr.revert

        def flaky(sid, workspace):
            if sid == sid2:
                raise RuntimeError("simulated revert failure")
            return real_revert(sid, workspace)

        monkeypatch.setattr(mgr, "revert", flaky)

        impact = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert impact.failed_steps == [sid2]
        assert impact.reverted_steps == [sid1]
        # sid1 的回滚照常生效
        assert (ws / "a.dat").read_text() == "a1\n"

    def test_empty_scope_still_reports_applied(self, mgr, slog, ws):
        # 锚点之后什么都没干
        slog.append("message", {"role": "user", "content": "闲聊"})
        anchor = list_anchors(SESSION, log=slog)[0]

        impact = apply_rewind(SESSION, anchor, manager=mgr, log=slog)

        assert impact.applied is True
        assert impact.steps == []
        assert impact.files == []
        assert impact.leaf_moved_to is not None


# ── 数据类 ───────────────────────────────────────────────────


class TestDataclasses:
    def test_anchor_frozen(self):
        a = RewindAnchor(
            session_id="s", event_id="e", seq=1, ts=0.0, preview="p"
        )
        assert a.to_dict()["event_id"] == "e"
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.seq = 2  # type: ignore[misc]
