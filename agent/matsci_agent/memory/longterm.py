"""Long-term memory — persistent knowledge storage across sessions.

Uses SQLite for structured data and integrates with VectorStore for
semantic retrieval of past conversations, facts, and insights.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matsci_agent.rag.vector_store import VectorStore


@dataclass
class MemoryEntry:
    """A single long-term memory entry."""
    id: str
    category: str  # "fact", "insight", "conversation", "calculation", "error"
    content: str
    tags: list[str]
    source: str  # e.g., "session:abc123", "vasp_calc:TiO2", "user_input"
    importance: float  # 0.0 - 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


class LongTermMemory:
    """SQLite-backed long-term memory with optional vector semantic search."""

    def __init__(
        self,
        db_path: str | None = None,
        vector_store: VectorStore | None = None,
        enable_semantic: bool = True,
    ):
        self.db_path = Path(db_path) if db_path else Path.home() / ".matsci" / "memory.db"
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
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags)
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
    ) -> str:
        """Store a new memory entry. Returns the entry ID."""
        entry_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(content) % 10000:04d}"
        tags = tags or []
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, category, content, tags, source, importance, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, category, content, json.dumps(tags), source, importance, now, now),
            )
            conn.commit()

        # Also index in vector store for semantic search
        if self._enable_semantic:
            self._vector_store.ingest(
                [content],
                metadatas=[{
                    "memory_id": entry_id,
                    "category": category,
                    "tags": ",".join(tags),
                    "source": source,
                    "importance": str(importance),
                }],
                ids=[entry_id],
            )

        return entry_id

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve memories matching query (keyword + optional semantic)."""
        results = []

        # Keyword search via FTS
        with self._connect() as conn:
            if query:
                like = f"%{query}%"
                if category:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE category = ? AND content LIKE ? ORDER BY importance DESC LIMIT ?",
                        (category, like, top_k),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                        (like, top_k),
                    ).fetchall()
            else:
                if category:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE category = ? ORDER BY importance DESC LIMIT ?",
                        (category, top_k),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memories ORDER BY importance DESC LIMIT ?",
                        (top_k,),
                    ).fetchall()

            for row in rows:
                results.append(dict(row))
                # Update access stats
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
            conn.commit()

        # Semantic search via vector store
        if semantic and self._enable_semantic:
            vec_results = self._vector_store.search(query, top_k=top_k)
            # Merge with keyword results, avoiding duplicates
            seen_ids = {r["id"] for r in results}
            for vr in vec_results:
                if vr["id"] not in seen_ids:
                    # Fetch full record from SQLite
                    with self._connect() as conn:
                        row = conn.execute(
                            "SELECT * FROM memories WHERE id = ?", (vr["id"],)
                        ).fetchone()
                        if row:
                            results.append(dict(row))

        return results[:top_k]

    def get_by_id(self, entry_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (datetime.now().isoformat(), entry_id),
                )
                conn.commit()
                return dict(row)
            return None

    def update(self, entry_id: str, content: str | None = None, importance: float | None = None) -> bool:
        with self._connect() as conn:
            if content is not None:
                conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, entry_id))
            if importance is not None:
                conn.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, entry_id))
            conn.commit()
            return conn.total_changes > 0

    def delete(self, entry_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            conn.commit()
            if self._enable_semantic:
                self._vector_store.delete([entry_id])
            return conn.total_changes > 0

    def list_by_category(self, category: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY last_accessed DESC LIMIT ?",
                (category, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def prune_low_importance(self, threshold: float = 0.2, older_than_days: int = 30) -> int:
        """Remove old, low-importance memories. Returns count deleted."""
        cutoff = datetime.now().timestamp() - older_than_days * 86400
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE importance < ? AND julianday('now') - julianday(created_at) > ?",
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
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data:
            try:
                self.store(
                    content=item["content"],
                    category=item.get("category", "fact"),
                    tags=json.loads(item.get("tags", "[]")),
                    source=item.get("source", ""),
                    importance=item.get("importance", 0.5),
                )
                count += 1
            except Exception:
                continue
        return count
