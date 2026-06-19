"""Tests for memory decay, pruning, and deduplication."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from huginn.memory.longterm import LongTermMemory


class TestMemoryDecayPolicy:
    def test_decay_reduces_importance_of_idle_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("old fact", importance=0.8, tier="mid")

            # Manually age the memory.
            old = (datetime.now() - timedelta(days=30)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                    (old, old, eid),
                )
                conn.commit()

            summary = mem.apply_decay_policy(decay_per_day=0.95, prune_threshold=0.1)
            assert summary["decayed"] >= 1

            entry = mem.get_by_id(eid)
            assert entry["importance"] < 0.8

    def test_pruning_removes_low_importance_old_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("forgettable", importance=0.05, tier="mid")

            old = (datetime.now() - timedelta(days=30)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                    (old, old, eid),
                )
                conn.commit()

            summary = mem.apply_decay_policy(
                decay_per_day=1.0, prune_threshold=0.1, min_age_days=1
            )
            assert summary["pruned"] >= 1
            assert mem.get_by_id(eid) is None

    def test_access_boost_protects_important_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("popular", importance=0.2, tier="mid")

            old = (datetime.now() - timedelta(days=30)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ?, access_count = ? WHERE id = ?",
                    (old, old, 20, eid),
                )
                conn.commit()

            mem.apply_decay_policy(
                decay_per_day=0.9, access_boost=0.05, prune_threshold=0.5
            )
            entry = mem.get_by_id(eid)
            assert entry["importance"] >= 0.5


class TestMemoryDeduplicator:
    def test_deduplicate_keeps_most_important(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            mem.store("duplicate fact", importance=0.3)
            mem.store("duplicate fact", importance=0.9)
            mem.store("duplicate fact", importance=0.4)

            removed = mem.deduplicate()
            assert removed == 2
            remaining = mem.list_by_category("fact")
            assert len(remaining) == 1
            assert remaining[0]["importance"] == pytest.approx(0.9, abs=0.01)


class TestMemoryMaintenance:
    def test_maintenance_runs_all_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            mem.store("dup", importance=0.5)
            mem.store("dup", importance=0.5)
            old_id = mem.store("old low", importance=0.05, tier="mid")
            old = (datetime.now() - timedelta(days=60)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                    (old, old, old_id),
                )
                conn.commit()

            summary = mem.maintenance(decay_per_day=1.0, prune_threshold=0.1)
            assert summary["pruned"] >= 1
            assert summary["deduplicated"] >= 1
