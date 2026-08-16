"""P2①: checkpointer auto_vacuum=INCREMENTAL — 长周期/多线程空洞回收.

验证新建 checkpointer 库会预置 ``auto_vacuum=INCREMENTAL`` (建表前设置才对
已存在数据生效), 且删行后 ``incremental_vacuum`` 能真正回收 freepages:
1. ``_preset_incremental_vacuum`` 只对全新文件生效
2. 预置后建库 → auto_vacuum 为 2 (INCREMENTAL)
3. 预置库删数据 → incremental_vacuum 回收 freepages
4. 无 conn / :memory: 静默降级不崩
5. create_checkpointer 端到端: 新建库带 INCREMENTAL
"""

from __future__ import annotations

import sqlite3

from huginn.checkpointer import (
    _enable_incremental_vacuum,
    create_checkpointer,
)


class _Saver:
    def __init__(self, conn):
        self.conn = conn


def _preset_and_open(db):
    """模拟真实流程: 先 preset, 再由 SqliteSaver 打开同一文件建表."""
    from huginn.checkpointer import _preset_incremental_vacuum

    _preset_incremental_vacuum(db)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE checkpoints (id TEXT PRIMARY KEY)")
    return conn


def test_preset_only_on_new_file(tmp_path):
    from huginn.checkpointer import _preset_incremental_vacuum

    # 已存在的文件 → preset 不应改动 (避免对在用库做迁移)
    db = tmp_path / "existing.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    _preset_incremental_vacuum(db)
    conn = sqlite3.connect(str(db))
    mode = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
    assert mode == 0, f"已存在库不应被改动, got mode={mode}"
    conn.close()


def test_preset_then_create_is_incremental(tmp_path):
    db = tmp_path / "fresh.sqlite"
    conn = _preset_and_open(db)
    mode = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
    assert mode == 2, f"应为 INCREMENTAL(2), got {mode}"
    conn.close()


def test_incremental_vacuum_reclaims_freepages(tmp_path):
    # 预置 INCREMENTAL 的库, 写入后删除, incremental_vacuum 回收 freepages
    db = tmp_path / "reclaim.sqlite"
    conn = _preset_and_open(db)
    conn.execute("CREATE TABLE t (x BLOB)")
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")  # 建表后仍需再确认
    chunk = b"x" * (256 * 1024)
    for _ in range(40):
        conn.execute("INSERT INTO t VALUES (?)", (chunk,))
    conn.commit()
    conn.execute("DELETE FROM t WHERE rowid % 2 = 0")
    conn.commit()
    before = conn.execute("PRAGMA freelist_count").fetchone()[0]
    _enable_incremental_vacuum(_Saver(conn))
    after = conn.execute("PRAGMA freelist_count").fetchone()[0]
    conn.close()
    assert after < before, (
        f"incremental_vacuum 应回收 freepages: {before} -> {after}"
    )


def test_no_conn_silent():
    class _NoConn:
        conn = None
        _conn = None

    _enable_incremental_vacuum(_NoConn())


def test_create_checkpointer_sets_incremental(tmp_path):
    path = tmp_path / "ckpt.sqlite"
    saver = create_checkpointer(path=path)
    try:
        conn = getattr(saver, "conn", None) or getattr(saver, "_conn", None)
        assert conn is not None
        mode = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
        assert mode == 2, f"create_checkpointer 新建库应带 INCREMENTAL, got {mode}"
    finally:
        close = getattr(saver, "close", None)
        if callable(close):
            close()