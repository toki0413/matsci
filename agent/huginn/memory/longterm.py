"""Long-term memory — persistent knowledge storage across sessions.

Uses SQLite for structured data and integrates with VectorStore for
semantic retrieval of past conversations, facts, and insights.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from huginn.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Imported lazily to avoid circular imports at module load time.
_decay_module: Any | None = None


TIER_TTL_HOURS = {
    "short": 6.0,
    "mid": 168.0,  # 7 days
    "long": None,  # permanent
}

MATERIAL_CATEGORIES = {
    "structure",
    "property",
    "synthesis",
    "characterization",
    "simulation",
}


def _migrate_memories_v1(conn: sqlite3.Connection) -> None:
    """Add tier, expires_at, formula, user_id columns to the memories table.

    These were previously scattered as ALTER TABLE + suppress(OperationalError)
    in _init_db. Consolidating into a versioned migration gives us a proper
    schema version via PRAGMA user_version and stops swallowing errors
    silently. Each column is checked individually so databases at different
    old states all converge to the same schema.
    """
    from huginn.utils.migrations import column_exists

    if not column_exists(conn, "memories", "tier"):
        conn.execute("ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT 'mid'")
    if not column_exists(conn, "memories", "expires_at"):
        conn.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")
    if not column_exists(conn, "memories", "formula"):
        conn.execute("ALTER TABLE memories ADD COLUMN formula TEXT")
    # Multi-tenant isolation: tag every memory with the owning user.
    # Old rows stay NULL and are treated as shared/global when no user_id
    # filter is supplied (backward compat).
    if not column_exists(conn, "memories", "user_id"):
        conn.execute("ALTER TABLE memories ADD COLUMN user_id TEXT")


def _run_memory_migrations(db_path: str) -> None:
    """Run pending memory schema migrations via MigrationManager."""
    from huginn.utils.migrations import MigrationManager

    mgr = MigrationManager(db_path)
    try:
        mgr.run_migrations([(1, _migrate_memories_v1)])
    finally:
        mgr.close()


@dataclass
class MemoryEntry:
    """A single long-term memory entry."""

    id: str
    category: str  # "fact", "insight", "conversation", "calculation", "error"
    content: str
    tags: list[str]
    source: str  # e.g., "session:abc123", "vasp_calc:TiO2", "user_input"
    importance: float  # 0.0 - 1.0
    tier: str  # "short", "mid", "long"
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime | None
    access_count: int = 0


class LongTermMemory:
    """SQLite-backed long-term memory with optional vector semantic search and tiered TTL."""

    def __init__(
        self,
        db_path: str | None = None,
        vector_store: VectorStore | None = None,
        enable_semantic: bool = True,
    ):
        self.db_path = (
            Path(db_path)
            if db_path
            else Path(os.environ.get("HUGINN_CACHE_DIR", Path.home() / ".huginn")) / "memory.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_store = vector_store
        self._enable_semantic = enable_semantic and vector_store is not None
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # 每次开连接都设 WAL, 写入不卡读, 并发场景少踩 "database is locked"
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        # Base table + indexes that only reference original columns
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    tier TEXT DEFAULT 'mid',
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags)
            """)
            conn.commit()

        # Run versioned migrations -- replaces the old scattered
        # ALTER TABLE + suppress(OperationalError) approach
        _run_memory_migrations(str(self.db_path))

        # Indexes on migrated columns + FTS (depend on columns added above)
        with self._connect() as conn:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_formula ON memories(formula)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON memories(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content, tags, source,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            conn.commit()

    def store(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        source: str = "",
        importance: float = 0.5,
        tier: str = "mid",
        ttl_hours: float | None = None,
        formula: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store a new memory entry. Returns the entry ID.

        tier: short (6h), mid (7d), long (permanent). ttl_hours overrides default TTL.
        formula: optional material formula (e.g. "GaN") for material entries.
        user_id: optional owner. When set the memory is private to that user;
            when omitted the memory is shared (backward compatible).
        """
        if tier not in TIER_TTL_HOURS:
            raise ValueError(f"Invalid tier {tier}; choose from {list(TIER_TTL_HOURS)}")
        entry_id = (
            f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        tags = tags or []
        now = datetime.now()
        expires = None
        if ttl_hours is None:
            ttl_hours = TIER_TTL_HOURS[tier]
        if ttl_hours is not None:
            expires = (now + timedelta(hours=ttl_hours)).isoformat()

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO memories
                    (id, category, content, tags, source, importance, tier, created_at, last_accessed, expires_at, formula, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        category,
                        content,
                        json.dumps(tags),
                        source,
                        importance,
                        tier,
                        now.isoformat(),
                        now.isoformat(),
                        expires,
                        formula,
                        user_id,
                    ),
                )
                # formula 进 tags 让 FTS5 也能搜到
                fts_tags = " ".join(tags + ([formula] if formula else []))
                conn.execute(
                    "INSERT INTO memory_fts (rowid, content, tags, source) VALUES (?, ?, ?, ?)",
                    (
                        conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                        content,
                        fts_tags,
                        source,
                    ),
                )
                conn.commit()
            except sqlite3.OperationalError as e:
                if "disk" in str(e).lower() or "full" in str(e).lower():
                    logger.error("disk full, cannot write to long-term memory: %s", e)
                    # 紧急清理：删低重要度的非 long 记忆腾空间
                    conn.execute("DELETE FROM memories WHERE tier != 'long' ORDER BY importance ASC LIMIT 100")
                    conn.commit()
                    raise
                raise

        if self._enable_semantic:
            self._vector_store.ingest(
                [content],
                metadatas=[
                    {
                        "memory_id": entry_id,
                        "category": category,
                        "tags": fts_tags,
                        "source": source,
                        "importance": str(importance),
                        "tier": tier,
                        "formula": formula or "",
                        "user_id": user_id or "",
                    }
                ],
                ids=[entry_id],
            )

        return entry_id

    def store_material(
        self,
        formula: str,
        category: str,
        payload: dict[str, Any],
        tier: str = "long",
        source: str = "",
        importance: float = 0.7,
        user_id: str | None = None,
    ) -> str:
        """存一条材料记忆. category 必须在 MATERIAL_CATEGORIES 里.

        formula: 化学式, 如 "GaN"
        category: structure | property | synthesis | characterization | simulation
        payload: 任意 dict, json 序列化后存 content
        user_id: 可选, 绑定到具体用户做多租户隔离
        """
        if category not in MATERIAL_CATEGORIES:
            raise ValueError(
                f"Invalid material category {category}; choose from {sorted(MATERIAL_CATEGORIES)}"
            )
        return self.store(
            content=json.dumps(payload, ensure_ascii=False, default=str),
            category=f"material_{category}",
            tags=[formula, category],
            source=source or f"material:{formula}",
            importance=importance,
            tier=tier,
            formula=formula,
            user_id=user_id,
        )

    def _where_alive(self, alias: str = "m") -> tuple[str, tuple]:
        """Return WHERE clause and params filtering out expired short/mid memories."""
        return (
            f"({alias}.expires_at IS NULL OR {alias}.expires_at > ?)",
            (datetime.now().isoformat(),),
        )

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Convert a natural-language query into a safe FTS5 MATCH string.

        FTS5 treats unquoted tokens as implicit AND, which is what we want for
        multi-word recall. Special characters are stripped to avoid syntax errors.
        Each token gets a '*' suffix for prefix matching (e.g. "silic*" matches
        "silicon").
        """
        import re

        # Strip FTS5 special characters that could break MATCH syntax
        clean = re.sub(r'["\'\-\*\(\)\:]', " ", query)
        tokens = [t for t in clean.split() if t]
        if not tokens:
            return ""
        # Quote each token and add prefix wildcard for flexible matching
        return " ".join(f'"{t}"*' for t in tokens)

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
        semantic: bool = True,
        formula: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve alive memories matching query (FTS5 + optional semantic).

        user_id: when supplied, only memories owned by that user are
            returned (multi-tenant isolation). When omitted, all memories
            are visible — this preserves the pre-isolation behaviour.
        """
        results = []
        alive_where, alive_params = self._where_alive()

        # FTS5 tokenized search — handles multi-word queries that LIKE misses.
        # Falls back to LIKE substring match if FTS5 query syntax errors out.
        with self._connect() as conn:
            sql = "SELECT * FROM memories AS m WHERE " + alive_where
            params: list[Any] = list(alive_params)
            # Scope to a single tenant before any other filter. Omitting the
            # clause entirely keeps the old shared-memory behaviour intact.
            if user_id is not None:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            if category:
                sql += " AND category = ?"
                params.append(category)
            if tier:
                sql += " AND tier = ?"
                params.append(tier)
            if formula:
                sql += " AND m.formula = ?"
                params.append(formula)
            if query:
                fts_matched = False
                # Try FTS5 first for proper tokenized matching
                fts_query = self._build_fts_query(query)
                if fts_query:
                    try:
                        fts_sql = (
                            sql
                            + " AND m.rowid IN (SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?)"
                        )
                        fts_params = params + [fts_query]
                        fts_sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
                        fts_params.append(top_k)
                        rows = conn.execute(fts_sql, tuple(fts_params)).fetchall()
                        fts_matched = True
                    except sqlite3.OperationalError:
                        pass
                # Fallback to LIKE if FTS5 unavailable or query failed
                if not fts_matched:
                    sql += " AND content LIKE ?"
                    params.append(f"%{query}%")
                    sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
                    params.append(top_k)
                    rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
                params.append(top_k)
                rows = conn.execute(sql, tuple(params)).fetchall()

            now = datetime.now().isoformat()
            for row in rows:
                results.append(dict(row))
                # Update access stats and rejuvenate short/mid TTL
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (now, row["id"]),
                )
            conn.commit()
            self._rejuvenate([r["id"] for r in rows])

        # Semantic search via vector store
        if semantic and self._enable_semantic:
            try:
                vec_results = self._vector_store.search(query, top_k=top_k)
            except Exception:
                logger.warning("vector search failed, falling back to FTS-only", exc_info=True)
                vec_results = []
            seen_ids = {r["id"] for r in results}
            for vr in vec_results:
                if vr["id"] in seen_ids:
                    continue
                with self._connect() as conn:
                    # Re-apply the tenant filter on the metadata fetch so a
                    # vector hit from another user can't leak across tenants.
                    if user_id is not None:
                        row = conn.execute(
                            "SELECT * FROM memories WHERE id = ? AND (expires_at IS NULL OR expires_at > ?) AND user_id = ?",
                            (vr["id"], datetime.now().isoformat(), user_id),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            "SELECT * FROM memories WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)",
                            (vr["id"], datetime.now().isoformat()),
                        ).fetchone()
                    if row:
                        results.append(dict(row))

        return results[:top_k]

    def _rejuvenate(self, entry_ids: list[str]) -> None:
        """Extend expiry on access for tiered memories (short/mid refresh their TTL)."""
        if not entry_ids:
            return
        now = datetime.now()
        with self._connect() as conn:
            for tier, ttl in TIER_TTL_HOURS.items():
                if ttl is None:
                    continue
                placeholders = ",".join("?" * len(entry_ids))
                conn.execute(
                    f"""
                    UPDATE memories
                    SET expires_at = ?
                    WHERE tier = ? AND id IN ({placeholders}) AND (expires_at IS NOT NULL AND expires_at > ?)
                    """,
                    (
                        (now + timedelta(hours=ttl)).isoformat(),
                        tier,
                        *entry_ids,
                        now.isoformat(),
                    ),
                )
            conn.commit()

    def get_by_id(self, entry_id: str) -> dict[str, Any] | None:
        alive_where, alive_params = self._where_alive()
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM memories AS m WHERE id = ? AND {alive_where}",
                (entry_id, *alive_params),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (datetime.now().isoformat(), entry_id),
                )
                conn.commit()
                self._rejuvenate([entry_id])
                return dict(row)
            return None

    def touch(self, entry_id: str) -> bool:
        """Touch a memory entry to rejuvenate its TTL and increment access count.

        Used by the distilled knowledge verification loop: when a tool
        succeeds, related distilled knowledge entries are touched to
        signal they were validated by real use.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (datetime.now().isoformat(), entry_id),
            )
            conn.commit()
            if cur.rowcount > 0:
                self._rejuvenate([entry_id])
                return True
            return False

    def update(
        self,
        entry_id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        tier: str | None = None,
        ttl_hours: float | None = None,
    ) -> bool:
        with self._connect() as conn:
            if content is not None:
                conn.execute(
                    "UPDATE memories SET content = ? WHERE id = ?", (content, entry_id)
                )
                if self._enable_semantic:
                    self._vector_store.ingest(
                        [content],
                        metadatas=[{"memory_id": entry_id}],
                        ids=[entry_id],
                    )
            if tags is not None:
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE id = ?",
                    (json.dumps(tags), entry_id),
                )
            if importance is not None:
                conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (importance, entry_id),
                )
            if tier is not None:
                if tier not in TIER_TTL_HOURS:
                    raise ValueError(f"Invalid tier {tier}")
                ttl = ttl_hours if ttl_hours is not None else TIER_TTL_HOURS[tier]
                expires = (
                    (datetime.now() + timedelta(hours=ttl)).isoformat() if ttl else None
                )
                conn.execute(
                    "UPDATE memories SET tier = ?, expires_at = ? WHERE id = ?",
                    (tier, expires, entry_id),
                )
            conn.commit()
            return conn.total_changes > 0

    def promote(self, entry_id: str, target_tier: str = "long") -> bool:
        """Promote a memory to a higher (or explicit) tier."""
        return self.update(entry_id, tier=target_tier)

    def delete(self, entry_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            conn.commit()
            if self._enable_semantic:
                self._vector_store.delete([entry_id])
            return conn.total_changes > 0

    def list_by_category(
        self,
        category: str,
        limit: int = 50,
        alive_only: bool = True,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        alive_where, alive_params = self._where_alive()
        with self._connect() as conn:
            sql = f"SELECT * FROM memories AS m WHERE category = ? AND {alive_where}"
            params: list[Any] = [category, *alive_params]
            if user_id is not None:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            sql += " ORDER BY last_accessed DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def list_all(
        self,
        limit: int = 200,
        alive_only: bool = True,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if alive_only:
            alive_where, alive_params = self._where_alive()
            sql = f"SELECT * FROM memories AS m WHERE {alive_where}"
            params: list[Any] = [*alive_params]
        else:
            sql = "SELECT * FROM memories AS m WHERE 1=1"
            params = []
        if user_id is not None:
            sql += " AND m.user_id = ?"
            params.append(user_id)
        sql += " ORDER BY last_accessed DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def prune_expired(self) -> int:
        """Remove all expired memories. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (datetime.now().isoformat(),),
            )
            conn.commit()
            return cursor.rowcount

    def prune_low_importance(
        self, threshold: float = 0.2, older_than_days: int = 30
    ) -> int:
        """Remove old, low-importance non-long memories. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE tier != 'long'
                  AND importance < ?
                  AND julianday('now') - julianday(created_at) > ?
                """,
                (threshold, older_than_days),
            )
            conn.commit()
            return cursor.rowcount

    def export(self, path: str | Path) -> None:
        """Export all memories to JSON."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            data = [dict(r) for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_(self, path: str | Path) -> int:
        """Import memories from JSON. Returns count imported."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data:
            try:
                tier = item.get("tier", "mid")
                ttl_hours = None
                if "expires_at" in item and item["expires_at"]:
                    try:
                        expires = datetime.fromisoformat(item["expires_at"])
                        ttl_hours = max(
                            0, (expires - datetime.now()).total_seconds() / 3600
                        )
                    except Exception:
                        pass
                self.store(
                    content=item["content"],
                    category=item.get("category", "fact"),
                    tags=json.loads(item.get("tags", "[]")),
                    source=item.get("source", ""),
                    importance=item.get("importance", 0.5),
                    tier=tier,
                    ttl_hours=ttl_hours,
                    formula=item.get("formula"),
                    user_id=item.get("user_id"),
                )
                count += 1
            except Exception:
                continue
        return count

    def apply_decay_policy(
        self,
        decay_per_day: float = 0.97,
        prune_threshold: float = 0.15,
        access_boost: float = 0.05,
        max_age_days: int = 90,
        min_age_days: int = 7,
    ) -> dict[str, int]:
        """Apply importance decay, access boost, and pruning."""
        global _decay_module
        if _decay_module is None:
            from huginn.memory import decay as _decay_module
        policy = _decay_module.MemoryDecayPolicy(
            decay_per_day=decay_per_day,
            prune_threshold=prune_threshold,
            access_boost=access_boost,
            max_age_days=max_age_days,
            min_age_days=min_age_days,
        )
        return policy.apply(self)

    def deduplicate(self, case_sensitive: bool = False) -> int:
        """Remove exact-duplicate memories, keeping the most important one."""
        global _decay_module
        if _decay_module is None:
            from huginn.memory import decay as _decay_module
        dedup = _decay_module.MemoryDeduplicator(case_sensitive=case_sensitive)
        return dedup.run(self)

    def maintenance(
        self,
        decay_per_day: float = 0.97,
        prune_threshold: float = 0.15,
        deduplicate: bool = True,
    ) -> dict[str, int]:
        """Run a full maintenance pass: decay, prune, dedupe, expire."""
        summary = self.apply_decay_policy(
            decay_per_day=decay_per_day, prune_threshold=prune_threshold
        )
        if deduplicate:
            summary["deduplicated"] = self.deduplicate()
        return summary

    def lint(self, limit: int = 100) -> dict[str, Any]:
        """LLM Wiki Lint: knowledge base health check.

        Inspired by Karpathy's LLM Wiki concept — periodically scan
        the knowledge base for contradictions, orphan entries, stale
        assertions, and missing cross-references.

        Returns a report dict with issues found.
        """
        import re as _re

        report: dict[str, Any] = {
            "total_entries": 0,
            "contradictions": [],
            "orphans": [],
            "stale": [],
            "low_confidence": [],
            "cross_ref_candidates": [],
            "summary": "",
        }

        with self._connect() as conn:
            alive_where, alive_params = self._where_alive()
            rows = conn.execute(
                f"""SELECT * FROM memories WHERE {alive_where}
                    ORDER BY importance DESC, access_count DESC
                    LIMIT ?""",
                (*alive_params, limit),
            ).fetchall()

        report["total_entries"] = len(rows)
        if not rows:
            report["summary"] = "No entries to lint."
            return report

        # Collect entries by category for cross-reference analysis
        entries_by_category: dict[str, list[dict]] = {}
        all_entries = []
        for row in rows:
            entry = dict(row)
            all_entries.append(entry)
            cat = entry.get("category", "unknown")
            entries_by_category.setdefault(cat, []).append(entry)

        # 1. Find contradictions: entries in same category with
        # conflicting numeric values (e.g., "band_gap = 1.12" vs
        # "band_gap = 1.15" for the same material)
        for cat, entries in entries_by_category.items():
            if len(entries) < 2:
                continue
            for i, e1 in enumerate(entries):
                for e2 in entries[i + 1:]:
                    # Check if entries reference same material/formula
                    f1 = (e1.get("formula") or "").lower()
                    f2 = (e2.get("formula") or "").lower()
                    if f1 and f2 and f1 == f2:
                        # Same formula — check for numeric conflicts
                        nums1 = set(
                            _re.findall(
                                r"(\d+\.?\d*)\s*(?:eV|eV/atom|eV/\u00c5|GPa|K|THz|\u00c5)",
                                e1.get("content", ""),
                            )
                        )
                        nums2 = set(
                            _re.findall(
                                r"(\d+\.?\d*)\s*(?:eV|eV/atom|eV/\u00c5|GPa|K|THz|\u00c5)",
                                e2.get("content", ""),
                            )
                        )
                        if nums1 and nums2 and nums1 != nums2:
                            report["contradictions"].append({
                                "formula": f1,
                                "entry1_id": e1["id"],
                                "entry1_nums": list(nums1)[:5],
                                "entry2_id": e2["id"],
                                "entry2_nums": list(nums2)[:5],
                                "category": cat,
                            })

        # 2. Find orphans: entries never accessed (access_count = 0)
        # and older than 7 days
        for entry in all_entries:
            if entry.get("access_count", 0) == 0:
                created = entry.get("created_at", "")
                try:
                    age_days = (
                        datetime.now() - datetime.fromisoformat(created)
                    ).days
                    if age_days > 7:
                        report["orphans"].append({
                            "id": entry["id"],
                            "category": entry.get("category"),
                            "age_days": age_days,
                            "content_preview": (entry.get("content") or "")[:80],
                        })
                except Exception:
                    pass

        # 3. Find stale: long-tier entries not accessed in 30+ days
        for entry in all_entries:
            if entry.get("tier") == "long":
                last_accessed = entry.get("last_accessed") or entry.get(
                    "created_at", ""
                )
                try:
                    age_days = (
                        datetime.now() - datetime.fromisoformat(last_accessed)
                    ).days
                    if age_days > 30:
                        report["stale"].append({
                            "id": entry["id"],
                            "category": entry.get("category"),
                            "days_since_access": age_days,
                            "content_preview": (entry.get("content") or "")[:80],
                        })
                except Exception:
                    pass

        # 4. Find low confidence distilled knowledge
        for entry in all_entries:
            if entry.get("category") == "distilled_knowledge":
                imp = entry.get("importance", 0)
                if imp < 0.3:
                    report["low_confidence"].append({
                        "id": entry["id"],
                        "importance": imp,
                        "content_preview": (entry.get("content") or "")[:80],
                    })

        # 5. Cross-reference candidates: entries mentioning the same
        # material/formula but in different categories (e.g., a
        # calculation result and a distilled lesson about the same
        # material should be cross-linked)
        formula_map: dict[str, list[str]] = {}
        for entry in all_entries:
            formula = (entry.get("formula") or "").lower()
            if formula:
                formula_map.setdefault(formula, []).append(
                    f"{entry['category']}:{entry['id']}"
                )
        for formula, refs in formula_map.items():
            if len(refs) > 1:
                report["cross_ref_candidates"].append({
                    "formula": formula,
                    "entries": refs,
                })

        # Build summary
        issues = (
            len(report["contradictions"])
            + len(report["orphans"])
            + len(report["stale"])
            + len(report["low_confidence"])
        )
        report["summary"] = (
            f"Linted {report['total_entries']} entries: "
            f"{len(report['contradictions'])} contradictions, "
            f"{len(report['orphans'])} orphans, "
            f"{len(report['stale'])} stale, "
            f"{len(report['low_confidence'])} low-confidence, "
            f"{len(report['cross_ref_candidates'])} cross-ref candidates."
        )

        return report
