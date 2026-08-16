"""P2 session 持久化升级测试: 按 session_id 精确读 + 版本化.

覆盖:
  * save_session_snapshot 自动递增 version
  * load_session_snapshot(session_id) 精确读该 session 最新一代
  * 多 session 并存时互不串读 (不再读"任意最新一条")
  * session_id 为空回退读任意最新一条 (向后兼容)
"""

from __future__ import annotations

from pathlib import Path

from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager


def _make_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))


def test_save_increments_version(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr.save_session_snapshot({"session_id": "s1", "mode": "chat"})
    mgr.save_session_snapshot({"session_id": "s1", "mode": "research"})
    snap = mgr.load_session_snapshot(session_id="s1")
    assert snap is not None
    assert snap["version"] == 2
    assert snap["mode"] == "research"


def test_precise_read_by_session_id(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr.save_session_snapshot({"session_id": "s1", "mode": "chat"})
    mgr.save_session_snapshot({"session_id": "s2", "mode": "plan"})
    # 精确读 s1 不应串到 s2
    s1 = mgr.load_session_snapshot(session_id="s1")
    s2 = mgr.load_session_snapshot(session_id="s2")
    assert s1["mode"] == "chat"
    assert s2["mode"] == "plan"


def test_empty_session_id_falls_back(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr.save_session_snapshot({"session_id": "s1", "mode": "chat"})
    # 无 session_id 仍能读到任意一条 (向后兼容)
    snap = mgr.load_session_snapshot()
    assert snap is not None
    assert snap["mode"] == "chat"


def test_unknown_session_returns_none(tmp_path):
    mgr = _make_manager(tmp_path)
    mgr.save_session_snapshot({"session_id": "s1", "mode": "chat"})
    assert mgr.load_session_snapshot(session_id="nope") is None
