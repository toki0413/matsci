"""Tests for memory maintenance policy plugin (Everything is a Plugin, 形态 B)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from huginn.memory.longterm import LongTermMemory
from huginn.plugins.memory_maintenance_policy import (
    MemoryMaintenancePolicy,
    clear_policies,
    register_memory_maintenance_policy,
    resolve_memory_policy,
    unregister_memory_maintenance_policy,
)


@pytest.fixture(autouse=True)
def _reset_policies():
    yield
    clear_policies()


class TestPolicyRegistry:
    def test_default_matches_builtin_literals(self):
        p = resolve_memory_policy()
        assert p.decay_per_day == 0.97
        assert p.prune_threshold == 0.15
        assert p.deduplicate is True

    def test_register_changes_resolved_policy(self):
        register_memory_maintenance_policy(
            "strict", MemoryMaintenancePolicy(decay_per_day=0.9, prune_threshold=0.5)
        )
        p = resolve_memory_policy()
        assert p.decay_per_day == 0.9
        assert p.prune_threshold == 0.5

    def test_highest_priority_wins(self):
        register_memory_maintenance_policy(
            "low", MemoryMaintenancePolicy(prune_threshold=0.1), priority=1
        )
        register_memory_maintenance_policy(
            "high", MemoryMaintenancePolicy(prune_threshold=0.9), priority=10
        )
        assert resolve_memory_policy().prune_threshold == 0.9

    def test_unregister_falls_back_to_default(self):
        register_memory_maintenance_policy(
            "strict", MemoryMaintenancePolicy(prune_threshold=0.9)
        )
        assert resolve_memory_policy().prune_threshold == 0.9
        unregister_memory_maintenance_policy("strict")
        assert resolve_memory_policy().prune_threshold == 0.15


class TestPolicyDrivesMaintenance:
    def test_policy_prune_threshold_applied_by_maintenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("borderline fact", importance=0.5, tier="mid")
            aged = (datetime.now() - timedelta(days=30)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                    (aged, aged, eid),
                )
                conn.commit()

            # 默认阈值 0.15 不会裁掉 0.5 强记忆; 注册 0.9 后应被裁.
            register_memory_maintenance_policy(
                "aggressive", MemoryMaintenancePolicy(prune_threshold=0.9)
            )
            summary = mem.maintenance()
            assert summary["pruned"] >= 1
            assert mem.get_by_id(eid) is None

    def test_explicit_args_override_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("kept fact", importance=0.5, tier="mid")
            aged = (datetime.now() - timedelta(days=30)).isoformat()
            with mem._connect() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                    (aged, aged, eid),
                )
                conn.commit()

            register_memory_maintenance_policy(
                "aggressive", MemoryMaintenancePolicy(prune_threshold=0.9)
            )
            # 显式传 prune_threshold=0.1 → 不裁 0.5 强记忆 (显式优先于策略).
            mem.maintenance(prune_threshold=0.1)
            assert mem.get_by_id(eid) is not None
