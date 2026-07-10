"""VC-1 + VC-2: version control integration tests.

验证 provenance↔snapshot 双向链 + 统一版本时钟 + 乐观并发.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from huginn.provenance.registry import (
    ProvenanceEntry,
    ProvenanceRegistry,
    VersionConflict,
    _ProvenanceStore,
)
from huginn.snapshot.file_snapshot import SnapshotManager


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tmp_root():
    """独立 temp 目录, 测试完后台删 (AV 慢)."""
    root = Path(tempfile.mkdtemp(prefix="vc_test_"))
    yield root
    import threading
    threading.Thread(
        target=shutil.rmtree, args=(root,), kwargs={"ignore_errors": True},
        daemon=True,
    ).start()


@pytest.fixture
def isolated_registry(tmp_root):
    """带独立 SQLite 的 ProvenanceRegistry, 不碰全局单例."""
    db_path = str(tmp_root / "test_provenance.db")
    store = _ProvenanceStore(db_path)
    reg = ProvenanceRegistry.__new__(ProvenanceRegistry)
    reg._entries = []
    reg._by_path = {}
    reg._by_tool = {}
    reg._store = store
    return reg


@pytest.fixture
def isolated_snapshot_mgr(tmp_root):
    """独立 SnapshotManager, 不碰全局 ~/.huginn.
    把 _instance 指向它, 这样 rollback_to 里的 SnapshotManager() 能找到.
    """
    mgr = SnapshotManager(root=tmp_root / "snapshots")
    old = SnapshotManager._instance
    SnapshotManager._instance = mgr
    yield mgr
    SnapshotManager._instance = old


# ── VC-1: ProvenanceEntry 带 snapshot_step_id ────────────────


class TestSnapshotStepIdField:
    def test_default_none(self):
        entry = ProvenanceEntry(
            file_path="/a/b.cif", produced_by="test", produced_at=0.0
        )
        assert entry.snapshot_step_id is None
        assert entry.reverted is False

    def test_set_step_id(self):
        entry = ProvenanceEntry(
            file_path="/a/b.cif", produced_by="test", produced_at=0.0,
            snapshot_step_id="abc123",
        )
        assert entry.snapshot_step_id == "abc123"

    def test_to_dict_includes_new_fields(self):
        entry = ProvenanceEntry(
            file_path="/a/b.cif", produced_by="test", produced_at=0.0,
            snapshot_step_id="abc123", reverted=True,
        )
        d = entry.to_dict()
        assert d["snapshot_step_id"] == "abc123"
        assert d["reverted"] is True


# ── VC-1: SQLite schema migration ────────────────────────────


class TestSchemaMigration:
    def test_new_db_has_columns(self, tmp_root):
        db_path = str(tmp_root / "fresh.db")
        store = _ProvenanceStore(db_path)
        cur = store._conn.execute("PRAGMA table_info(entries)")
        cols = {r[1] for r in cur.fetchall()}
        assert "snapshot_step_id" in cols
        assert "reverted" in cols
        store.close()

    def test_old_db_migrates(self, tmp_root):
        """模拟老库: 先建没新列的表, 再初始化 store, 验证自动补列."""
        db_path = str(tmp_root / "old.db")
        # 手动建一个旧 schema
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                produced_by TEXT NOT NULL,
                produced_at REAL NOT NULL,
                input_files TEXT,
                parameters TEXT,
                file_format TEXT,
                key_properties TEXT
            );
        """)
        conn.execute(
            "INSERT INTO entries (file_path, produced_by, produced_at) VALUES (?, ?, ?)",
            ("old.cif", "old_tool", 0.0),
        )
        conn.commit()
        conn.close()

        # 现在初始化 store, 应该自动补列
        store = _ProvenanceStore(db_path)
        cur = store._conn.execute("PRAGMA table_info(entries)")
        cols = {r[1] for r in cur.fetchall()}
        assert "snapshot_step_id" in cols
        assert "reverted" in cols

        # 老数据应该能读出来, 新字段是默认值
        entries = store.get_events_since(0, 10)
        assert len(entries) == 1
        assert entries[0].snapshot_step_id is None
        assert entries[0].reverted is False
        store.close()


# ── VC-1: register 带 snapshot_step_id ──────────────────────


class TestRegisterWithSnapshot:
    def test_register_with_step_id(self, isolated_registry):
        entry = isolated_registry.register(
            file_path="/ws/OUTCAR",
            produced_by="vasp_tool",
            snapshot_step_id="snap_001",
        )
        assert entry.snapshot_step_id == "snap_001"

    def test_register_without_step_id(self, isolated_registry):
        entry = isolated_registry.register(
            file_path="/ws/OUTCAR",
            produced_by="vasp_tool",
        )
        assert entry.snapshot_step_id is None

    def test_step_id_persisted_to_sqlite(self, isolated_registry):
        isolated_registry.register(
            file_path="/ws/OUTCAR",
            produced_by="vasp_tool",
            snapshot_step_id="snap_002",
        )
        # 从 SQLite 查回
        events = isolated_registry.get_events()
        assert len(events) == 1
        assert events[0].snapshot_step_id == "snap_002"


# ── VC-1: rollback_to 调 SnapshotManager.revert ──────────────


class TestRollbackIntegration:
    def test_rollback_reverts_files(
        self, isolated_registry, isolated_snapshot_mgr
    ):
        """端到端: register 带 step_id → rollback_to → 文件被回滚."""
        ws = Path(isolated_snapshot_mgr._root) / "ws"
        ws.mkdir(parents=True, exist_ok=True)

        # 初始文件
        (ws / "POSCAR").write_text("Cu\n 1.0\n", encoding="utf-8")

        # snapshot track
        step_id = isolated_snapshot_mgr.track("vasp_tool", ws)
        isolated_snapshot_mgr.patch(step_id, ws)

        # 模拟工具修改文件
        (ws / "POSCAR").write_text("Cu\n 2.0\n", encoding="utf-8")

        # 注册 provenance event 带 snapshot_step_id
        isolated_registry.register(
            file_path=str(ws / "POSCAR"),
            produced_by="vasp_tool",
            snapshot_step_id=step_id,
        )

        # 验证文件被改了
        assert (ws / "POSCAR").read_text() == "Cu\n 2.0\n"

        # rollback_to(0): 回滚所有事件
        paths = isolated_registry.rollback_to(0)

        # 文件应该回到执行前
        assert (ws / "POSCAR").read_text() == "Cu\n 1.0\n"
        assert len(paths) > 0

    def test_rollback_marks_events_reverted(self, isolated_registry, isolated_snapshot_mgr):
        ws = Path(isolated_snapshot_mgr._root) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "data.dat").write_text("old\n", encoding="utf-8")

        step_id = isolated_snapshot_mgr.track("vasp_tool", ws)
        isolated_snapshot_mgr.patch(step_id, ws)
        (ws / "data.dat").write_text("new\n", encoding="utf-8")

        isolated_registry.register(
            file_path=str(ws / "data.dat"),
            produced_by="vasp_tool",
            snapshot_step_id=step_id,
        )

        isolated_registry.rollback_to(0)

        # 查回事件, 验证 reverted=True
        events = isolated_registry.get_events()
        assert len(events) == 1
        assert events[0].reverted is True

    def test_rollback_only_after_target(self, isolated_registry, isolated_snapshot_mgr):
        """rollback_to(1) 只回滚 id > 1 的事件."""
        ws = Path(isolated_snapshot_mgr._root) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "a.dat").write_text("a1\n", encoding="utf-8")
        (ws / "b.dat").write_text("b1\n", encoding="utf-8")

        # 第一步
        sid1 = isolated_snapshot_mgr.track("vasp_tool", ws)
        isolated_snapshot_mgr.patch(sid1, ws)
        (ws / "a.dat").write_text("a2\n", encoding="utf-8")
        isolated_registry.register(
            file_path=str(ws / "a.dat"),
            produced_by="vasp_tool",
            snapshot_step_id=sid1,
        )

        # 第二步
        sid2 = isolated_snapshot_mgr.track("vasp_tool", ws)
        isolated_snapshot_mgr.patch(sid2, ws)
        (ws / "b.dat").write_text("b2\n", encoding="utf-8")
        isolated_registry.register(
            file_path=str(ws / "b.dat"),
            produced_by="vasp_tool",
            snapshot_step_id=sid2,
        )

        # 当前版本
        version = isolated_registry.current_version()
        assert version >= 2

        # rollback 到第一步之后 (只回第二步)
        first_event_id = 1  # 第一个 event id
        isolated_registry.rollback_to(first_event_id)

        # b.dat 应该被回滚
        assert (ws / "b.dat").read_text() == "b1\n"
        # a.dat 不受影响 (第一步的 revert 在 rollback_to(1) 时不会触发, 因为 event id=1 不 > 1)
        assert (ws / "a.dat").read_text() == "a2\n"


# ── VC-2: 乐观并发控制 ──────────────────────────────────────


class TestOptimisticConcurrency:
    def test_expected_version_match(self, isolated_registry):
        """版本号匹配时正常注册."""
        v = isolated_registry.current_version()
        entry = isolated_registry.register(
            file_path="/ws/test.cif",
            produced_by="structure_tool",
            expected_version=v,
        )
        assert entry is not None

    def test_expected_version_mismatch(self, isolated_registry):
        """版本号不匹配时抛 VersionConflict."""
        # 先注册一条, 版本号变
        isolated_registry.register(
            file_path="/ws/a.cif", produced_by="tool_a"
        )
        current = isolated_registry.current_version()

        # 用过期的版本号注册
        with pytest.raises(VersionConflict):
            isolated_registry.register(
                file_path="/ws/b.cif",
                produced_by="tool_b",
                expected_version=current - 1,
            )

    def test_expected_version_none_skips_check(self, isolated_registry):
        """expected_version=None 时不做并发检查."""
        entry = isolated_registry.register(
            file_path="/ws/c.cif",
            produced_by="tool_c",
            expected_version=None,
        )
        assert entry is not None

    def test_version_increments(self, isolated_registry):
        """每条注册后版本号递增."""
        v0 = isolated_registry.current_version()
        isolated_registry.register(file_path="/ws/1.cif", produced_by="t")
        v1 = isolated_registry.current_version()
        assert v1 == v0 + 1

        isolated_registry.register(file_path="/ws/2.cif", produced_by="t")
        v2 = isolated_registry.current_version()
        assert v2 == v1 + 1


# ── VC-2: revert_to_version 统一入口 ─────────────────────────


class TestRevertToVersion:
    def test_revert_to_version_calls_rollback(
        self, isolated_registry, isolated_snapshot_mgr
    ):
        ws = Path(isolated_snapshot_mgr._root) / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "x.dat").write_text("v1\n", encoding="utf-8")

        sid = isolated_snapshot_mgr.track("tool", ws)
        isolated_snapshot_mgr.patch(sid, ws)
        (ws / "x.dat").write_text("v2\n", encoding="utf-8")

        isolated_registry.register(
            file_path=str(ws / "x.dat"),
            produced_by="tool",
            snapshot_step_id=sid,
        )

        v = isolated_registry.current_version()
        # revert_to_version(v - 1) = 回滚最后一步
        isolated_registry.revert_to_version(v - 1)

        assert (ws / "x.dat").read_text() == "v1\n"


# ── VC-1: consume_last_snapshot_step_id ──────────────────────


class TestConsumeSnapshotStepId:
    def test_returns_none_when_empty(self):
        from huginn.snapshot.integration import consume_last_snapshot_step_id
        import huginn.snapshot.integration as integ
        integ._last_step_id = None
        assert consume_last_snapshot_step_id() is None

    def test_returns_and_clears(self):
        from huginn.snapshot.integration import consume_last_snapshot_step_id
        import huginn.snapshot.integration as integ
        integ._last_step_id = "test_step_123"
        assert consume_last_snapshot_step_id() == "test_step_123"
        assert consume_last_snapshot_step_id() is None


# ── mark_reverted store method ───────────────────────────────


class TestMarkReverted:
    def test_mark_and_query(self, isolated_registry):
        isolated_registry.register(
            file_path="/ws/a.cif", produced_by="tool_a"
        )
        isolated_registry.register(
            file_path="/ws/b.cif", produced_by="tool_b"
        )

        # 标记第一条为 reverted
        isolated_registry._store.mark_reverted([1], reverted=True)

        events = isolated_registry.get_events()
        assert events[0].reverted is True
        assert events[1].reverted is False

    def test_mark_empty_list_noop(self, isolated_registry):
        result = isolated_registry._store.mark_reverted([])
        assert result == 0
