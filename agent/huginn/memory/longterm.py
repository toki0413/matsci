"""Long-term memory — persistent knowledge storage across sessions.

Uses SQLite for structured data and integrates with VectorStore for
semantic retrieval of past conversations, facts, and insights.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from huginn.rag.vector_store import VectorStore

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
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
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
            # Migrate old databases without tier/expires_at/formula before indexing them
            for _col, ddl in [
                ("tier", "ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT 'mid'"),
                ("expires_at", "ALTER TABLE memories ADD COLUMN expires_at TEXT"),
                ("formula", "ALTER TABLE memories ADD COLUMN formula TEXT"),
            ]:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(ddl)
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
    ) -> str:
        """Store a new memory entry. Returns the entry ID.

        tier: short (6h), mid (7d), long (permanent). ttl_hours overrides default TTL.
        formula: optional material formula (e.g. "GaN") for material entries.
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
            conn.execute(
                """
                INSERT INTO memories
                (id, category, content, tags, source, importance, tier, created_at, last_accessed, expires_at, formula)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> str:
        """存一条材料记忆. category 必须在 MATERIAL_CATEGORIES 里.

        formula: 化学式, 如 "GaN"
        category: structure | property | synthesis | characterization | simulation
        payload: 任意 dict, json 序列化后存 content
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
    ) -> list[dict[str, Any]]:
        """Retrieve alive memories matching query (FTS5 + optional semantic)."""
        results = []
        alive_where, alive_params = self._where_alive()

        # FTS5 tokenized search — handles multi-word queries that LIKE misses.
        # Falls back to LIKE substring match if FTS5 query syntax errors out.
        with self._connect() as conn:
            sql = "SELECT * FROM memories AS m WHERE " + alive_where
            params: list[Any] = list(alive_params)
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
            vec_results = self._vector_store.search(query, top_k=top_k)
            seen_ids = {r["id"] for r in results}
            for vr in vec_results:
                if vr["id"] in seen_ids:
                    continue
                with self._connect() as conn:
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
        self, category: str, limit: int = 50, alive_only: bool = True
    ) -> list[dict[str, Any]]:
        alive_where, alive_params = self._where_alive()
        with self._connect() as conn:
            sql = f"SELECT * FROM memories AS m WHERE category = ? AND {alive_where} ORDER BY last_accessed DESC LIMIT ?"
            rows = conn.execute(sql, (category, *alive_params, limit)).fetchall()
            return [dict(r) for r in rows]

    def list_all(
        self, limit: int = 200, alive_only: bool = True
    ) -> list[dict[str, Any]]:
        if alive_only:
            alive_where, alive_params = self._where_alive()
            sql = f"SELECT * FROM memories AS m WHERE {alive_where} ORDER BY last_accessed DESC LIMIT ?"
            params = (*alive_params, limit)
        else:
            sql = "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
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
