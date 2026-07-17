"""Memory decay and maintenance policies.

Long-term memory grows indefinitely. These policies keep it useful by:

1. Decaying the importance of rarely-accessed memories over time.
2. Boosting memories that are frequently accessed.
3. Pruning memories that fall below an importance threshold.
4. Removing expired entries and deduplicating near-duplicate facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huginn.memory.longterm import LongTermMemory


@dataclass
class MemoryDecayPolicy:
    """Configurable policy for aging and pruning long-term memories."""

    # Importance is multiplied by this factor for each day since last access.
    decay_per_day: float = 0.97
    # Minimum importance a memory can decay to before it is eligible for pruning.
    prune_threshold: float = 0.15
    # Importance is boosted by this amount on each access (up to 1.0).
    access_boost: float = 0.05
    # Only prune non-long memories younger than max_age_days.
    max_age_days: int = 90
    # Minimum age (days) before a memory can be pruned.
    min_age_days: int = 7
    # 高频访问的 mid 记忆自动晋升到 long: 超过此 access_count 且 importance
    # 高于 promote_importance 时触发. 以前衰减只做降级/删除, 不做升级.
    promote_access_threshold: int = 10
    promote_importance: float = 0.5

    def apply(self, memory: LongTermMemory) -> dict[str, int]:
        """Apply decay, boost, and pruning to ``memory``.

        Returns a summary of how many entries were decayed and pruned.
        """
        decayed = 0
        pruned = 0
        now = datetime.now()

        with memory._connect() as conn:
            rows = conn.execute(
                "SELECT id, importance, access_count, created_at, last_accessed, tier, "
                "last_decay_access_count FROM memories"
            ).fetchall()

            promoted = 0
            for row in rows:
                entry_id = row["id"]
                importance = row["importance"]
                access_count = row["access_count"]
                tier = row["tier"] if "tier" in row.keys() else "mid"
                # 旧库没这列时 row["last_decay_access_count"] 取不到, 默认 0
                last_decay_count = (
                    row["last_decay_access_count"]
                    if "last_decay_access_count" in row.keys()
                    else 0
                ) or 0
                created_at = self._parse_dt(row["created_at"]) or now
                last_accessed = self._parse_dt(row["last_accessed"]) or created_at

                age_days = (now - created_at).total_seconds() / 86400
                idle_days = (now - last_accessed).total_seconds() / 86400

                # Decay importance based on idle time.
                new_importance = importance * (self.decay_per_day**idle_days)
                # Boost based on visits since last decay — not cumulative
                # access_count, otherwise each decay re-adds the whole history
                # and importance gets pinned to 1.0 forever.
                new_visits = max(0, access_count - last_decay_count)
                new_importance = min(
                    1.0, new_importance + self.access_boost * new_visits
                )

                if abs(new_importance - importance) > 0.001 or new_visits > 0:
                    conn.execute(
                        "UPDATE memories SET importance = ?, last_decay_access_count = ? WHERE id = ?",
                        (round(new_importance, 4), access_count, entry_id),
                    )
                    decayed += 1

                # 频繁访问的 mid 记忆自动晋升到 long — 以前衰减只做降级,
                # 不做升级. 这是 access_count → tier 的反馈环.
                if (
                    tier == "mid"
                    and access_count >= self.promote_access_threshold
                    and new_importance >= self.promote_importance
                ):
                    conn.execute(
                        "UPDATE memories SET tier = 'long' WHERE id = ?",
                        (entry_id,),
                    )
                    promoted += 1

                # Prune low-importance, non-permanent memories that are old enough.
                # long tier 是"永久保留"的 (直觉/关键洞察/蒸馏知识), 不参与 prune.
                # 之前不检查 tier, 科研长周期项目里早期直觉 memory 会因 importance
                # 衰减被误删 — 研究线索断裂.
                if (
                    tier != "long"
                    and new_importance < self.prune_threshold
                    and age_days >= self.min_age_days
                    and age_days <= self.max_age_days
                ):
                    conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
                    pruned += 1

            conn.commit()

        # G35: 上面的 per-row DELETE 绕过了 FTS5 'delete' 命令, 有删就重建
        if pruned > 0:
            memory._rebuild_fts_index()

        # Clean up expired entries via the existing helper.
        expired = memory.prune_expired()

        return {
            "decayed": decayed,
            "pruned": pruned,
            "promoted": promoted,
            "expired": expired,
        }

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None


class MemoryDeduplicator:
    """Simple exact-content deduplication for long-term memory."""

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def run(self, memory: LongTermMemory) -> int:
        """Delete duplicate memories, keeping the most important/recent one.

        Returns the number of duplicates removed.
        """
        removed = 0
        seen: dict[str, str] = {}  # normalized content -> kept id

        with memory._connect() as conn:
            rows = conn.execute("""
                SELECT id, content, importance, last_accessed FROM memories
                ORDER BY importance DESC, last_accessed DESC
                """).fetchall()

            for row in rows:
                normalized = (
                    row["content"] if self.case_sensitive else row["content"].lower()
                )
                if normalized in seen:
                    conn.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                    removed += 1
                else:
                    seen[normalized] = row["id"]

            conn.commit()
        # G35: per-row DELETE 绕过 FTS5 'delete', 有删就重建
        if removed > 0:
            memory._rebuild_fts_index()
        return removed
