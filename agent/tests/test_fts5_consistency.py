"""G35 FTS5 索引一致性回归测试 (audit_20260717/14 P1-FTS5).

守护 LongTermMemory 的 FTS5 全文索引在以下场景不丢/不脏:
  1. _rebuild_fts_index / _validate_fts_consistency 方法存在
  2. 全量重建: delete-all 清空 token 后 MATCH 丢失, 重建恢复
  3. 启动时自动愈合: 新实例 _init_db 后 token 丢失能自检重建
  4. bulk DELETE 路径 (prune_expired/prune_low_importance/dedup) 后 MATCH 不命中已删记忆

外部内容表的 SELECT count(*) 会优化成查源表, 永远相等, 检测不了真正的
索引损坏. 改用 MATCH 抽查才是真验证.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from huginn.memory.longterm import LongTermMemory


def _fresh_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _match_count(mem: LongTermMemory, term: str) -> int:
    with mem._connect() as conn:
        return conn.execute(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?",
            (term,),
        ).fetchone()[0]


def _clear_fts_tokens(mem: LongTermMemory) -> None:
    """模拟 FTS5 token 损坏: delete-all 清空 token 但保留主表数据."""
    with mem._connect() as conn:
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('delete-all')")
        conn.commit()


class TestFts5IndexConsistency:
    """G35: FTS5 索引一致性 — 守护 bulk DELETE 路径不留下脏 token."""

    def test_methods_exist(self):
        mem = LongTermMemory(_fresh_db())
        assert hasattr(mem, "_rebuild_fts_index")
        assert hasattr(mem, "_validate_fts_consistency")

    def test_rebuild_restores_match_after_delete_all(self):
        mem = LongTermMemory(_fresh_db())
        mem.store("graphene band structure", category="fact", importance=0.8)
        assert _match_count(mem, "graphene") >= 1

        _clear_fts_tokens(mem)
        assert _match_count(mem, "graphene") == 0

        count = mem._rebuild_fts_index()
        assert count >= 1
        assert _match_count(mem, "graphene") >= 1

    def test_validate_detects_stale_and_auto_rebuilds(self):
        mem = LongTermMemory(_fresh_db())
        mem.store("perovskite stability", category="fact", importance=0.7)

        _clear_fts_tokens(mem)
        # 默认 HUGINN_FTS_AUTO_REBUILD=1, 校验应检测到 token 丢失并重建
        os.environ.pop("HUGINN_FTS_AUTO_REBUILD", None)
        ok = mem._validate_fts_consistency()
        assert ok is True
        assert _match_count(mem, "perovskite") >= 1

    def test_validate_respects_auto_rebuild_disabled(self):
        mem = LongTermMemory(_fresh_db())
        mem.store("zeolite framework", category="fact", importance=0.6)

        _clear_fts_tokens(mem)
        os.environ["HUGINN_FTS_AUTO_REBUILD"] = "0"
        try:
            ok = mem._validate_fts_consistency()
            assert ok is False  # 只告警不重建
        finally:
            os.environ.pop("HUGINN_FTS_AUTO_REBUILD", None)

    def test_prune_expired_keeps_fts_in_sync(self):
        """prune_expired 用 bulk DELETE 绕过 FTS5 'delete' 命令,
        G35 修复后应触发 _rebuild_fts_index 保持同步."""
        mem = LongTermMemory(_fresh_db())
        mem.store(
            "expired memory entry",
            category="fact",
            importance=0.3,
            ttl_hours=-1.0,  # 立即过期
        )
        assert _match_count(mem, "expired") >= 1

        deleted = mem.prune_expired()
        assert deleted >= 1
        # 已删记忆在 FTS5 不应再命中
        assert _match_count(mem, "expired") == 0

    def test_dedup_keeps_fts_in_sync(self):
        """MemoryDeduplicator 用 per-row DELETE 绕过 FTS5 'delete',
        G35 修复后应触发 _rebuild_fts_index 保持同步."""
        from huginn.memory.decay import MemoryDeduplicator

        mem = LongTermMemory(_fresh_db())
        mem.store("duplicate content xyz", category="fact", importance=0.5)
        mem.store("duplicate content xyz", category="fact", importance=0.6)

        removed = MemoryDeduplicator().run(mem)
        assert removed >= 1
        # 重复被删, 只留一份; FTS5 应只命中一次的内容对应一条主表记录
        # (MATCH 命中数 == 主表剩余记录数, 不多不少)
        with mem._connect() as conn:
            main_count = conn.execute(
                "SELECT count(*) FROM memories WHERE content LIKE '%duplicate content xyz%'"
            ).fetchone()[0]
        assert main_count == 1
        assert _match_count(mem, "duplicate") == main_count
