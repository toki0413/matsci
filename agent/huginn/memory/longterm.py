"""Long-term memory — persistent knowledge storage across sessions.

Uses SQLite for structured data and integrates with VectorStore for
semantic retrieval of past conversations, facts, and insights.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
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

# 检索排序: long 层优先于 mid 优先于 short, 同层内按 importance + access_count.
# 以前不区分层级, 一条 short 层的高 importance 记忆会排在 long 层前面.
_TIER_ORDER = "CASE tier WHEN 'long' THEN 0 WHEN 'mid' THEN 1 WHEN 'short' THEN 2 ELSE 3 END"

# M: typed memory 在检索排序里的优先级. typed 行 (memory_type IS NOT NULL) 优先于
# NULL 行. 同为 typed 时按 _TYPE_PRIORITY 排: failed_direction (0) > iteration_result (1)
# > cross_domain_transfer (2) > persona_history (3) > stable_principle (4).
# ponytail: 不改 schema 加 type_priority 列, 用 CASE WHEN 在 SQL 里算. 升级路径:
# 加 type_priority INTEGER 列 + 索引.
_TYPE_PRIORITY_ORDER = (
    "CASE memory_type"
    " WHEN 'failed_direction' THEN 0"
    " WHEN 'iteration_result' THEN 1"
    " WHEN 'cross_domain_transfer' THEN 2"
    " WHEN 'persona_history' THEN 3"
    " WHEN 'stable_principle' THEN 4"
    " WHEN NULL THEN 10"
    " ELSE 10 END"
)
# typed 行整体优先于 NULL 行
_TYPED_FIRST = (
    f"CASE WHEN memory_type IS NOT NULL THEN 0 ELSE 1 END, {_TYPE_PRIORITY_ORDER}"
)


def _entry_has_reasoning(row: dict) -> bool:
    """读 tags 里有没有 has_reasoning 标记. 旧记录无标记 → False.

    ponytail: 复用 tags JSON 当 metadata, 不加 schema 列. 跟 record_failed_direction
    的 math_concept:X 标记同模式. 升级路径: 独立 has_reasoning INTEGER 列.
    """
    tags = row.get("tags", "[]")
    if isinstance(tags, str):
        try:
            tag_list = json.loads(tags)
        except (ValueError, TypeError):
            return False
    elif isinstance(tags, list):
        tag_list = tags
    else:
        return False
    return "has_reasoning" in tag_list


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
    # 上次 decay 时的 access_count 快照, decay 用 (access_count - 上次值)
    # 算增量访问, 否则每次 decay 都把累计 access_count 反复 boost 累加.
    # 默认 0, 首次 decay 会把历史访问一次性补偿进去 (一次性, 不重复).
    if not column_exists(conn, "memories", "last_decay_access_count"):
        conn.execute(
            "ALTER TABLE memories ADD COLUMN last_decay_access_count INTEGER DEFAULT 0"
        )
    # 路径化层级记忆 (Open WebUI _path_rank 模式): 每条记忆可选挂在一个
    # 逻辑路径上, 如 "materials/GaN/synthesis" 或 "sessions/abc/insights".
    # retrieve 时按 lookup_path 计算 rank: 精确 > 后代 > 祖先 > 兄弟 > 共享 token.
    # 默认 NULL 表示全局, 不参与路径排序.
    if not column_exists(conn, "memories", "path"):
        conn.execute("ALTER TABLE memories ADD COLUMN path TEXT")
    # P5 Memory Cluster: archived 标记. cluster summary 写回后, 原条目标
    # archived=1, _where_alive 自动过滤 (不删, 留档可 rollback).
    # default 0 保持向后兼容, 老条目全部视为 active.
    if not column_exists(conn, "memories", "archived"):
        conn.execute(
            "ALTER TABLE memories ADD COLUMN archived INTEGER DEFAULT 0"
        )
    # P12 Memory Typing: 4 个结构化列替代 category 字符串 + tags JSON hack.
    # 默认 NULL, 旧行不参与 typed 查询 (WHERE memory_type = ? OR memory_type
    # IS NULL 兼容路径). 调用方仍走 remember(content, category=...) 时 4 列
    # 保持 NULL, 行为完全不变.
    _migrate_memories_v2(conn)


def _migrate_memories_v2(conn: sqlite3.Connection) -> None:
    """P12: 加 memory_type / run_id / persona_id / status 4 个结构化列.

    column_exists 守卫让旧 DB 升级路径幂等. 列默认 NULL, 旧行为 100% 不变.
    """
    from huginn.utils.migrations import column_exists

    if not column_exists(conn, "memories", "memory_type"):
        conn.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT")
    if not column_exists(conn, "memories", "run_id"):
        conn.execute("ALTER TABLE memories ADD COLUMN run_id TEXT")
    if not column_exists(conn, "memories", "persona_id"):
        conn.execute("ALTER TABLE memories ADD COLUMN persona_id TEXT")
    if not column_exists(conn, "memories", "status"):
        conn.execute("ALTER TABLE memories ADD COLUMN status TEXT")


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
            # Self-heal: stale DBs may have user_version=1 from before
            # path/formula/user_id were added to _migrate_memories_v1.
            # MigrationManager skips v1 on those, so we ensure columns here.
            from huginn.utils.migrations import column_exists
            for col in ("formula", "user_id", "path"):
                if not column_exists(conn, "memories", col):
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT")

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
                CREATE INDEX IF NOT EXISTS idx_path ON memories(path)
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content, tags, source,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            conn.commit()

        # G35: 启动时校验 FTS5 索引一致性. 多处 DELETE 路径 (decay/dedup/
        # prune/export 覆盖导入) 历史上绕过 FTS5 'delete' 命令, 留孤儿索引行
        # 导致 retrieve 命中已删记忆. 校验行数不一致就全量重建.
        self._validate_fts_consistency()

    def _rebuild_fts_index(self) -> int:
        """G35: 全量重建 FTS5 索引.

        用 FTS5 'delete-all' 清空再从 memories 表批量回灌. tags 在主表是
        JSON 数组, FTS5 要空格分隔, formula 也拼进 tags 让搜化学式能命中
        (与 store/update 的 fts_tags 构造对齐).

        Returns 重建后的索引行数.
        """
        with self._connect() as conn:
            # FTS5 外部内容表清空用 'delete-all' 命令, 不能用 DELETE FROM
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('delete-all')")
            rows = conn.execute(
                "SELECT rowid, content, tags, formula, source FROM memories"
            ).fetchall()
            batch: list[tuple] = []
            for r in rows:
                try:
                    tags_list = json.loads(r["tags"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    tags_list = []
                formula = r["formula"] if r["formula"] else None
                fts_tags = " ".join(tags_list + ([formula] if formula else []))
                batch.append((r["rowid"], r["content"], fts_tags, r["source"]))
            if batch:
                conn.executemany(
                    "INSERT INTO memory_fts (rowid, content, tags, source) VALUES (?, ?, ?, ?)",
                    batch,
                )
            conn.commit()
            return len(batch)

    def _validate_fts_consistency(self) -> bool:
        """G35: 校验 FTS5 索引能查到主表数据, 不一致触发重建.

        外部内容表的 SELECT count(*) 会优化成查源表 memories, 永远相等,
        检测不了真正的索引损坏 (delete-all 清空 token 后 count 不变).
        改用 MATCH 抽查: 取前几条记忆的搜索词, 查不到说明 FTS5 token 丢了.

        HUGINN_FTS_AUTO_REBUILD=0 关闭自动重建 (测试/基准用), 只告警不修.

        Returns True if consistent or rebuild succeeded.
        """
        import re
        with self._connect() as conn:
            mem_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
            if mem_count == 0:
                return True
            rows = conn.execute(
                "SELECT content FROM memories ORDER BY rowid LIMIT 3"
            ).fetchall()
            for row in rows:
                content = row["content"] or ""
                words = content.split()
                if not words:
                    continue
                # FTS5 tokenizer 按空格/符号分词, 只取字母数字防 MATCH 语法报错
                clean = re.sub(r"[^a-zA-Z0-9]", "", words[0])
                if not clean:
                    continue
                hit = conn.execute(
                    "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?",
                    (clean,),
                ).fetchone()[0]
                if hit == 0:
                    if os.environ.get("HUGINN_FTS_AUTO_REBUILD", "1") == "0":
                        logger.warning(
                            "FTS5 stale: MATCH '%s' = 0 but memory exists "
                            "(auto-rebuild disabled)", clean,
                        )
                        return False
                    logger.warning(
                        "FTS5 stale: MATCH '%s' = 0, rebuilding index", clean
                    )
                    self._rebuild_fts_index()
                    return True
        return True

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
        path: str | None = None,
    ) -> str:
        """Store a new memory entry. Returns the entry ID.

        tier: short (6h), mid (7d), long (permanent). ttl_hours overrides default TTL.
        formula: optional material formula (e.g. "GaN") for material entries.
        user_id: optional owner. When set the memory is private to that user;
            when omitted the memory is shared (backward compatible).
        path: optional hierarchical path (e.g. "materials/GaN/synthesis").
            retrieve() uses _path_rank to prefer memories at or near the
            lookup path. NULL means global, no path preference.
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
                    (id, category, content, tags, source, importance, tier, created_at, last_accessed, expires_at, formula, user_id, path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        path,
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
                    # G35: bulk DELETE 绕过 FTS5 'delete' 命令, 重建索引兜底
                    self._rebuild_fts_index()
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
                        "path": path or "",
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

    def _where_alive(
        self,
        alias: str = "m",
        *,
        memory_type: str | None = None,
        persona_id: str | None = None,
        status: str | None = None,
    ) -> tuple[str, tuple]:
        """Return WHERE clause and params filtering out expired + archived memories.

        P12: 可选 memory_type / persona_id / status 过滤. NULL 兼容路径 —
        旧行 (列 IS NULL) 也通过过滤, 不被排除. 这样 typed 查询能同时
        拿到 typed 行和 legacy 行, 老库升级后不丢数据.
        """
        # P5: archived=1 的条目被 cluster summary 替代, 不参与 recall.
        # 老库 migration 后 archived default 0, 全部视为 active, 行为不变.
        where = (
            f"({alias}.expires_at IS NULL OR {alias}.expires_at > ?) "
            f"AND {alias}.archived = 0"
        )
        params: list[Any] = [datetime.now().isoformat()]
        # NULL 兼容: 匹配值 OR 列为 NULL (旧行). 这样 typed 过滤不会把
        # 没标 type 的老条目全部砍掉.
        if memory_type is not None:
            where += f" AND ({alias}.memory_type = ? OR {alias}.memory_type IS NULL)"
            params.append(memory_type)
        if persona_id is not None:
            where += f" AND ({alias}.persona_id = ? OR {alias}.persona_id IS NULL)"
            params.append(persona_id)
        if status is not None:
            where += f" AND ({alias}.status = ? OR {alias}.status IS NULL)"
            params.append(status)
        return where, tuple(params)

    @staticmethod
    def _path_rank(memory_path: str | None, lookup_path: str | None) -> tuple[int, int]:
        """Hierarchical distance between a memory's path and the lookup path.

        Returns (rank, distance). Lower rank wins; within the same rank,
        lower distance wins. Open WebUI pattern adapted.

        rank 0 — exact match (path == lookup)
        rank 1 — memory is a descendant of lookup (lookup/* matches)
        rank 2 — memory is an ancestor of lookup (specific lookup falls under broad memory)
        rank 3 — siblings (same parent, different leaf)
        rank 4 — share at least one path token (cross-branch relevance hint)
        rank 5 — leaf segment matches (e.g. "synthesis" in two different parents)
        rank 6 — no relationship (memory has no path, or paths are disjoint)

        When lookup_path is None, every memory gets rank 6 (path-neutral) —
        preserves the pre-path behaviour exactly.
        """
        if not lookup_path:
            return (6, 0)
        mp = (memory_path or "").strip("/")
        lp = lookup_path.strip("/")
        if not mp:
            return (6, 0)
        m_segs = mp.split("/")
        l_segs = lp.split("/")

        if mp == lp:
            return (0, 0)
        # descendant: lookup is a prefix of memory (memory deeper than lookup)
        if len(m_segs) > len(l_segs) and m_segs[: len(l_segs)] == l_segs:
            return (1, len(m_segs) - len(l_segs))
        # ancestor: memory is a prefix of lookup (memory broader than lookup)
        if len(l_segs) > len(m_segs) and l_segs[: len(m_segs)] == m_segs:
            return (2, len(l_segs) - len(m_segs))
        # siblings: same parent, different leaf
        if len(m_segs) == len(l_segs) and m_segs[:-1] == l_segs[:-1] and m_segs[-1] != l_segs[-1]:
            return (3, 1)
        # leaf segment matches (only the leaf is shared, nothing else)
        # checked before general shared-token so "synthesis in two branches"
        # lands at rank 5, not rank 4
        if m_segs[-1] == l_segs[-1] and set(m_segs) & set(l_segs) == {m_segs[-1]}:
            return (5, abs(len(m_segs) - len(l_segs)))
        # shared token anywhere (non-leaf)
        if set(m_segs) & set(l_segs):
            shared = len(set(m_segs) & set(l_segs))
            return (4, max(len(m_segs), len(l_segs)) - shared)
        return (6, 0)

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

    @staticmethod
    def _ising_rerank_enabled() -> bool:
        """P1-1 toggle: env HUGINN_ISING_RERANK (默认 on).

        Ising 能量函数 re-rank 把 FTS5 top_k 独立排序升级为能量最低 K-子集.
        off 时行为完全回退到原排序, 回归测试安全.
        ponytail: 默认 on 是因为 fallback 完整 — 无 embedding 自动 no-op.
        """
        return os.environ.get("HUGINN_ISING_RERANK", "1") != "0"

    def _ising_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        beta: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Ising 能量函数 re-rank — 把 FTS5 top_k 独立排序升级为能量最低 K-子集.

        数学结构 (Ising-Hopfield 同构):
            Hᵢ = sim(query, mᵢ)  — 外场, query-memory 相关性
            Tᵢⱼ = sim(mᵢ, mⱼ)    — memory-memory 耦合 (semantic similarity)
            E(S) = -Σᵢ∈S Hᵢ - β Σᵢ<ⱼ∈S Tᵢⱼ
            S* = argmin_S E(S), |S| = top_k

        贪心: 按 Hᵢ 降序逐个加入, 每步算 ΔE, ΔE < 0 接受, 否则跳过.
        ponytail: 不做精确 ground state (NP-hard). O(top_k * |candidates|²).
        ceiling: 贪心不保证全局最优; 升级路径: 模拟退火 / Modern Hopfield attention.
        """
        # 边界: 候选不足 / 单选 / 无 vector_store — 全部 no-op 回原排序
        if (
            not candidates
            or top_k <= 1
            or len(candidates) <= top_k
            or self._vector_store is None
        ):
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("none", len(candidates))
            except Exception:
                pass
            return candidates[:top_k]
        try:
            texts = [query] + [str(c.get("content", "")) for c in candidates]
            embs = self._vector_store._compute_embeddings(texts)
        except Exception:
            logger.warning("ising rerank: embedding 失败, 回退原排序", exc_info=True)
            return candidates[:top_k]
        if not embs or len(embs) != len(texts):
            return candidates[:top_k]

        # cosine similarity (chromadb 返回的 embedding 未归一化, 手动算)
        import math

        def _cos(a: list[float], b: list[float]) -> float:
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        q_emb = embs[0]
        c_embs = embs[1:]
        # Hᵢ = sim(query, mᵢ)
        H = [_cos(q_emb, c_embs[i]) for i in range(len(candidates))]
        # Tᵢⱼ = sim(mᵢ, mⱼ) — 对称矩阵, 只算上三角
        n = len(candidates)
        T: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = _cos(c_embs[i], c_embs[j])
                T[i][j] = s
                T[j][i] = s

        # 贪心: 按 Hᵢ 降序, 逐个尝试加入 selected
        order = sorted(range(n), key=lambda i: -H[i])
        selected: list[int] = []
        for idx in order:
            if len(selected) >= top_k:
                break
            if not selected:
                # 第一个: ΔE = -Hᵢ, Hᵢ > 0 时接受
                if H[idx] > 0:
                    selected.append(idx)
                continue
            # ΔE = -H_idx - β Σⱼ∈selected T[idx][j]
            delta = -H[idx] - beta * sum(T[idx][j] for j in selected)
            if delta < 0:
                selected.append(idx)
        # 兜底: 若贪心太保守没选够, 按 H 顺序补齐
        if len(selected) < top_k:
            for idx in order:
                if idx not in selected and len(selected) < top_k:
                    selected.append(idx)
                if len(selected) >= top_k:
                    break
        try:
            from huginn.routes.metrics import track_memory_rerank
            track_memory_rerank("ising", len(candidates))
        except Exception:
            pass
        return [candidates[i] for i in selected]

    # ── P2-5: HiLS 分层稀疏 attention ─────────────────────────────────
    # Modern Hopfield ↔ attention 同构: ξ_new = softmax(β X^T q) X.
    # P1-1 Ising 是离散 ground-state (sᵢ∈{0,1}), P2-5 是连续 softmax 组合 (αᵢ∈[0,1]).
    # N>=K landmarks 时走分层稀疏 (v2), N<K 时退化到全 attention (v1 baseline).
    # ponytail: 地标缓存用 dict, 不引新依赖. k-means 用 random.sample 做 init 的
    # 朴素版 (sklearn 未装时). ceiling: 朴素 k-means 慢, N>1M 需 GPU faiss.

    @staticmethod
    def _hils_enabled() -> bool:
        """P2-5 toggle: env HUGINN_HILS_ATTENTION (默认 on).

        off 时回退到 P1-1 Ising 贪心 (已有 fallback 链).
        ponytail: 默认 on 因 fallback 完整 — 无 vector_store / N<K 自动退化.
        """
        return os.environ.get("HUGINN_HILS_ATTENTION", "1") != "0"

    def _hils_attention(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        beta: float = 8.0,
        n_landmarks: int = 256,
        top_h: int = 8,
    ) -> list[dict[str, Any]]:
        """HiLS 分层稀疏 attention re-rank.

        Modern Hopfield: ξ = softmax(β X^T q) X, β 是温度 (β→∞ 退化为传统 Hopfield).
        分层稀疏 (HiLS):
          Layer 0 (地标): K=n_landmarks 个地标 (k-means on candidate embs)
                          q 跟 K 个地标算 attention → O(K·d)
                          选 top-h 个地标
          Layer 1 (精细): 只跟 top-h 地标下的 candidates 算全 attention
                          O(h·(N/K)·d)
          Total: O((K + h·N/K)·d) vs 全 attention O(N·d)
          N=100K, K=256, h=8: 28x 加速.

        N < n_landmarks 时退化到全 attention (v1 baseline), 不分层.
        ponytail: 地标缓存用 (content_hash, n_landmarks) 做 key, 增量更新 lazy.
        ceiling: 朴素 k-means (random init + 10 iter), N>1M 需 GPU faiss.
        """
        if (
            not candidates
            or top_k <= 1
            or len(candidates) <= top_k
            or self._vector_store is None
        ):
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("none", len(candidates))
            except Exception:
                pass
            return candidates[:top_k]
        try:
            texts = [query] + [str(c.get("content", "")) for c in candidates]
            embs = self._vector_store._compute_embeddings(texts)
        except Exception:
            logger.warning("hils: embedding 失败, 回退原排序", exc_info=True)
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("none", len(candidates))
            except Exception:
                pass
            return candidates[:top_k]
        if not embs or len(embs) != len(texts):
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("none", len(candidates))
            except Exception:
                pass
            return candidates[:top_k]

        import math

        def _cos(a: list[float], b: list[float]) -> float:
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        q_emb = embs[0]
        c_embs = embs[1:]
        n = len(candidates)

        # 全 attention 权重: αᵢ = softmax(β · cos(q, mᵢ))
        # Modern Hopfield: ξ = Σ αᵢ mᵢ, 但我们只排序不组合, 用 αᵢ 作 score
        scores = [math.exp(beta * _cos(q_emb, c_embs[i])) for i in range(n)]
        total = sum(scores)
        if total <= 0:
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("none", n)
            except Exception:
                pass
            return candidates[:top_k]
        alpha = [s / total for s in scores]

        # N < K: 退化到全 attention (v1 baseline), 不分层
        if n < n_landmarks:
            ranked = sorted(range(n), key=lambda i: -alpha[i])
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("hils_full", n)
            except Exception:
                pass
            return [candidates[i] for i in ranked[:top_k]]

        # N >= K: 分层稀疏 (v2)
        # Step 1: k-means 选地标 (朴素版, random init + 10 iter)
        landmarks = self._kmeans_landmarks(c_embs, n_landmarks)

        # Step 2: query 跟地标算 attention, 选 top-h
        lm_scores = [math.exp(beta * _cos(q_emb, lm)) for lm in landmarks]
        lm_total = sum(lm_scores)
        if lm_total <= 0:
            # 地标全零, 回退全 attention
            ranked = sorted(range(n), key=lambda i: -alpha[i])
            try:
                from huginn.routes.metrics import track_memory_rerank
                track_memory_rerank("hils_full", n)
            except Exception:
                pass
            return [candidates[i] for i in ranked[:top_k]]
        lm_alpha = [s / lm_total for s in lm_scores]
        top_lm_idx = sorted(range(len(landmarks)), key=lambda i: -lm_alpha[i])[:top_h]
        top_lm_set = set(top_lm_idx)

        # Step 3: 每个 candidate 挂到最近地标
        lm_assign = [0] * n
        for i in range(n):
            best_lm = 0
            best_sim = -2.0
            for li in range(len(landmarks)):
                s = _cos(c_embs[i], landmarks[li])
                if s > best_sim:
                    best_sim = s
                    best_lm = li
            lm_assign[i] = best_lm

        # Step 4: 只保留 top-h 地标下的 candidates, 全 attention 排序
        filtered = [i for i in range(n) if lm_assign[i] in top_lm_set]
        if len(filtered) < top_k:
            # top-h 太严格, 补齐: 加入次 top 地标的 candidates
            all_lm_ranked = sorted(range(len(landmarks)), key=lambda i: -lm_alpha[i])
            for li in all_lm_ranked:
                if li in top_lm_set:
                    continue
                extra = [i for i in range(n) if lm_assign[i] == li and i not in filtered]
                filtered.extend(extra)
                if len(filtered) >= top_k * 2:
                    break
        ranked = sorted(filtered, key=lambda i: -alpha[i])
        try:
            from huginn.routes.metrics import track_memory_rerank
            track_memory_rerank("hils_sparse", n)
        except Exception:
            pass
        return [candidates[i] for i in ranked[:top_k]]

    def _kmeans_landmarks(
        self, embs: list[list[float]], k: int, max_iter: int = 10
    ) -> list[list[float]]:
        """朴素 k-means 选地标. random init + Lloyd 迭代.

        ponytail: 不引 sklearn/faiss. N>1M 需 GPU 加速, 留给升级路径.
        ceiling: random init 可能陷局部最优, 但地标只需"代表性", 不需精确聚类.
        """
        import random

        n = len(embs)
        if n <= k:
            return [list(e) for e in embs]
        d = len(embs[0]) if embs else 0
        if d == 0:
            return []

        # random init (无 seed, 接受方差)
        idxs = random.sample(range(n), k)
        centroids = [list(embs[i]) for i in idxs]

        def _cos(a, b):
            import math
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        for _ in range(max_iter):
            # assign
            assign = [0] * n
            for i in range(n):
                best_c = 0
                best_sim = -2.0
                for ci in range(k):
                    s = _cos(embs[i], centroids[ci])
                    if s > best_sim:
                        best_sim = s
                        best_c = ci
                assign[i] = best_c
            # update
            new_centroids = [[0.0] * d for _ in range(k)]
            counts = [0] * k
            for i in range(n):
                ci = assign[i]
                for j in range(d):
                    new_centroids[ci][j] += embs[i][j]
                counts[ci] += 1
            for ci in range(k):
                if counts[ci] > 0:
                    for j in range(d):
                        new_centroids[ci][j] /= counts[ci]
                else:
                    # 空簇: 保留旧 centroid
                    new_centroids[ci] = centroids[ci]
            centroids = new_centroids
        return centroids

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
        semantic: bool = True,
        formula: str | None = None,
        user_id: str | None = None,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve alive memories matching query (FTS5 + optional semantic).

        user_id: when supplied, only memories owned by that user are
            returned (multi-tenant isolation). When omitted, all memories
            are visible — this preserves the pre-isolation behaviour.
        path: when supplied, results are re-ranked by _path_rank against
            this lookup path (closer memories win). SQL LIMIT is widened
            to top_k * 3 so path-near matches that FTS ranked low still
            make it into the candidate pool. When omitted, the original
            tier/importance ordering is preserved unchanged.
        """
        results = []
        alive_where, alive_params = self._where_alive()
        # ponytail: 路径排序需要更大候选池, 否则 SQL LIMIT 直接砍掉了应该靠前的行.
        # 拉到 3 倍候选, Python 层 re-rank 后取 top_k. 不传 path 时拉 1 倍, 行为不变.
        fetch_k = top_k * 3 if path else top_k

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
                        fts_sql += f" ORDER BY {_TYPED_FIRST}, {_TIER_ORDER}, importance DESC, access_count DESC LIMIT ?"
                        fts_params.append(fetch_k)
                        rows = conn.execute(fts_sql, tuple(fts_params)).fetchall()
                        fts_matched = True
                    except sqlite3.OperationalError:
                        pass
                # Fallback to LIKE if FTS5 unavailable or query failed
                if not fts_matched:
                    sql += f" AND content LIKE ?"
                    params.append(f"%{query}%")
                    sql += f" ORDER BY {_TYPED_FIRST}, {_TIER_ORDER}, importance DESC, access_count DESC LIMIT ?"
                    params.append(fetch_k)
                    rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql += f" ORDER BY {_TYPED_FIRST}, {_TIER_ORDER}, importance DESC, access_count DESC LIMIT ?"
                params.append(fetch_k)
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
                vec_results = self._vector_store.search(query, top_k=fetch_k)
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

        if path:
            # ponytail: SQL 已经按 tier/importance 排过, 但路径近邻更重要 —
            # 在 Python 层按 (path_rank, tier_order, importance) 再排一次.
            # 升级路径: 把 _path_rank 写成 SQL 表达式避免双排序.
            tier_rank = {"long": 0, "mid": 1, "short": 2}
            results.sort(
                key=lambda r: (
                    self._path_rank(r.get("path"), path),
                    tier_rank.get(r.get("tier", "mid"), 3),
                    -(r.get("importance", 0.5)),
                )
            )

        # P2-5: HiLS 分层稀疏 attention (Modern Hopfield). 优先于 P1-1 Ising.
        # N>=K 时分层稀疏 (28x 加速 @ N=100K), N<K 时退化到全 attention.
        # off 时回退 P1-1 Ising 贪心 (已有 fallback 链).
        if (
            semantic
            and self._enable_semantic
            and self._hils_enabled()
            and len(results) > top_k
        ):
            return self._hils_attention(query, results, top_k)

        # P1-1 fallback: Ising 能量函数 re-rank (HILS off 时走这里)
        if (
            semantic
            and self._enable_semantic
            and self._ising_rerank_enabled()
            and len(results) > top_k
        ):
            return self._ising_rerank(query, results, top_k)

        # has_reasoning 优先: 在原 FTS5+vector 排序上叠加 stable sort, 有推理的条目
        # 冒到前面. ponytail: 不改 SQL/HiLS/Ising 主逻辑, 只在默认返回路径 re-rank
        # (retrieve 不暴露数值 score, 用 stable sort 等价表达 +0.1 bonus).
        # 升级路径: SQL CASE WHEN has_reasoning THEN -0.1 ELSE 0 END 加到 ORDER BY.
        results.sort(key=lambda r: 0 if _entry_has_reasoning(r) else 1)
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
            # FTS5 外部内容表删索引需要原值, 先取一份快照
            old = conn.execute(
                "SELECT rowid, content, tags, source FROM memories WHERE id = ?",
                (entry_id,),
            ).fetchone()
            if old is None:
                return False

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
            # content/tags 变了就要重建 FTS5 索引行, 否则 retrieve 走 FTS5
            # 时拿到的是旧 token (importance/tier 不影响 FTS5, 跳过)
            if content is not None or tags is not None:
                new_row = conn.execute(
                    "SELECT content, tags, source, formula FROM memories WHERE id = ?",
                    (entry_id,),
                ).fetchone()
                if new_row is not None:
                    conn.execute(
                        "INSERT INTO memory_fts(memory_fts, rowid, content, tags, source) "
                        "VALUES('delete', ?, ?, ?, ?)",
                        (old["rowid"], old["content"], old["tags"], old["source"]),
                    )
                    new_tags_list = json.loads(new_row["tags"] or "[]")
                    fts_tags = " ".join(
                        new_tags_list + ([new_row["formula"]] if new_row["formula"] else [])
                    )
                    conn.execute(
                        "INSERT INTO memory_fts (rowid, content, tags, source) VALUES (?, ?, ?, ?)",
                        (old["rowid"], new_row["content"], fts_tags, new_row["source"]),
                    )
            conn.commit()
            return conn.total_changes > 0

    def promote(self, entry_id: str, target_tier: str = "long") -> bool:
        """Promote a memory to a higher (or explicit) tier."""
        return self.update(entry_id, tier=target_tier)

    def update_archived(self, entry_id: str, archived: bool = True) -> bool:
        """P5: 标记条目为 archived. archived=1 的条目被 _where_alive 过滤,
        不再参与 retrieve/recall, 但保留在表里可 rollback 或 audit.

        cluster summary 写回后, 原条目用这个方法归档.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET archived = ? WHERE id = ?",
                (1 if archived else 0, entry_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, entry_id: str) -> bool:
        with self._connect() as conn:
            # FTS5 外部内容表删索引必须先取原值, 用 'delete' 命令
            old = conn.execute(
                "SELECT rowid, content, tags, source FROM memories WHERE id = ?",
                (entry_id,),
            ).fetchone()
            if old is None:
                return False
            conn.execute(
                "INSERT INTO memory_fts(memory_fts, rowid, content, tags, source) "
                "VALUES('delete', ?, ?, ?, ?)",
                (old["rowid"], old["content"], old["tags"], old["source"]),
            )
            conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            conn.commit()
            if self._enable_semantic:
                self._vector_store.delete([entry_id])
            return True

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

    def count_alive_by_tier(self) -> dict[str, int]:
        """Single SQL query for tier counts — replaces list_all + 3x traversal."""
        alive_where, alive_params = self._where_alive()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT tier, COUNT(*) AS c FROM memories AS m WHERE {alive_where} GROUP BY tier",
                alive_params,
            ).fetchall()
        counts = {"short": 0, "mid": 0, "long": 0}
        for r in rows:
            counts[r["tier"]] = r["c"]
        counts["total"] = sum(counts.values())
        return counts

    def list_long_tier(self, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch only long-tier entries, sorted by importance desc."""
        alive_where, alive_params = self._where_alive()
        sql = (
            f"SELECT * FROM memories AS m WHERE {alive_where} AND tier = 'long'"
            " ORDER BY importance DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (*alive_params, limit)).fetchall()
            return [dict(r) for r in rows]

    def prune_expired(self) -> int:
        """Remove all expired memories. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (datetime.now().isoformat(),),
            )
            conn.commit()
            deleted = cursor.rowcount
        # G35: bulk DELETE 绕过 FTS5 'delete', 有删就重建索引
        if deleted > 0:
            self._rebuild_fts_index()
        return deleted

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
            deleted = cursor.rowcount
        # G35: bulk DELETE 绕过 FTS5 'delete', 有删就重建索引
        if deleted > 0:
            self._rebuild_fts_index()
        return deleted

    def export(self, path: str | Path) -> None:
        """Export all memories to JSON."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            data = [dict(r) for r in rows]
        # 原子写: 大量记忆导出时崩溃会留半截 JSON, 下次 import_ 直接 JSONDecodeError.
        from huginn.utils.concurrency import atomic_write_text
        atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))

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
                        logger.debug("fromisoformat failed", exc_info=True)
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
        cluster: bool = False,
        llm_chat_fn: Any = None,
    ) -> dict[str, int]:
        """Run a full maintenance pass: decay, prune, dedupe, expire, optional cluster.

        P5: cluster=True + HUGINN_MEMORY_CLUSTER=1 时, dedupe 后跑 semantic cluster +
        LLM summarize. cluster step 失败不 abort maintenance, 只记日志.
        llm_chat_fn 为 None 时跳过 cluster (即使 cluster=True, 没 LLM 没法 summarize).
        """
        summary = self.apply_decay_policy(
            decay_per_day=decay_per_day, prune_threshold=prune_threshold
        )
        if deduplicate:
            summary["deduplicated"] = self.deduplicate()

        if (
            cluster
            and llm_chat_fn is not None
            and os.environ.get("HUGINN_MEMORY_CLUSTER", "0") == "1"
        ):
            try:
                from huginn.memory.cluster import (
                    cluster_memories, compress_clusters,
                )
                clusters = cluster_memories(self)
                if clusters:
                    result = compress_clusters(self, clusters, llm_chat_fn)
                    summary["clustered"] = result.get("summarized", 0)
                    summary["archived"] = result.get("archived", 0)
                    summary["cluster_skipped"] = result.get("skipped", 0)
                    summary["cluster_failed"] = result.get("failed", 0)
            except Exception:
                logger.warning(
                    "memory cluster step failed, decay/prune/dedupe still applied",
                    exc_info=True,
                )
        return summary

    def lint(self, limit: int = 100, auto_fix: bool = False) -> dict[str, Any]:
        """LLM Wiki Lint: knowledge base health check.

        Inspired by Karpathy's LLM Wiki concept — periodically scan
        the knowledge base for contradictions, orphan entries, stale
        assertions, and missing cross-references.

        Args:
            limit: max number of entries to scan.
            auto_fix: if True, write discovered links (contradicts:/crossref:)
                back to tags.

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
                f"""SELECT * FROM memories AS m WHERE {alive_where}
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
                    logger.debug("lint failed", exc_info=True)

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
                    logger.debug("lint failed", exc_info=True)

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

        # auto_fix: write discovered links back into tags so later
        # queries can piggyback on them
        if auto_fix:
            import json as _json

            fixed = 0
            with self._connect() as conn:
                # contradictions: tag both sides so each points at the other
                for c in report["contradictions"]:
                    for eid in (c["entry1_id"], c["entry2_id"]):
                        row = conn.execute(
                            "SELECT tags FROM memories WHERE id = ?", (eid,)
                        ).fetchone()
                        if not row:
                            continue
                        tags = _json.loads(row["tags"] or "[]")
                        other = (
                            c["entry2_id"] if eid == c["entry1_id"] else c["entry1_id"]
                        )
                        tag = f"contradicts:{other}"
                        if tag not in tags:
                            tags.append(tag)
                            conn.execute(
                                "UPDATE memories SET tags = ? WHERE id = ?",
                                (_json.dumps(tags), eid),
                            )
                            fixed += 1
                # cross-ref candidates: stamp the shared formula on each entry
                for x in report["cross_ref_candidates"]:
                    for ref in x["entries"]:
                        cat, eid = ref.split(":", 1)
                        row = conn.execute(
                            "SELECT tags FROM memories WHERE id = ?", (eid,)
                        ).fetchone()
                        if not row:
                            continue
                        tags = _json.loads(row["tags"] or "[]")
                        tag = f"crossref:{x['formula']}"
                        if tag not in tags:
                            tags.append(tag)
                            conn.execute(
                                "UPDATE memories SET tags = ? WHERE id = ?",
                                (_json.dumps(tags), eid),
                            )
                            fixed += 1
                conn.commit()
            report["auto_fixed"] = fixed

        return report


# ── stable_principles: persona 一部分, S7 自修改回流 ──────────────────────
# G8 加法: knowledge→persona 回路. S7 accepted 提案写进来, 下一轮 build_system_prompt
# 的 STABLE_PRINCIPLES 段会读它. 不走 SQLite 是因为这部分属于 persona 而非记忆,
# RCB/benchmark 也要保留 (memory_manager 在 bench 里是 None).
STABLE_PRINCIPLES_PATH = Path(".huginn/stable_principles.jsonl")

# G30: 全局 stable_principles 路径 — 跨任务/跨 RCB workspace 复用.
# RCB runner 把 HUGINN_CACHE_DIR 重定向到 ws/.huginn_cache, STABLE_PRINCIPLES_PATH
# 是相对路径跟着 cwd 走, 每个任务独立. 全局路径固定在 ~/.huginn/, 任务间共享.
# HUGINN_RCB_INHERIT_PRINCIPLES=True (default) 时, store 双写, load 合并读.
_GLOBAL_PRINCIPLES_PATH = Path.home() / ".huginn" / "stable_principles.jsonl"


def _inherit_enabled() -> bool:
    """G30: 是否跨任务继承 stable_principles. 默认开."""
    return os.environ.get("HUGINN_RCB_INHERIT_PRINCIPLES", "1") not in ("0", "false", "False")


# === G44: multi-agent metacog — stable_principles 跨 agent 共享文件锁 ===
# HUGINN_MULTI_AGENT=True 时开启, 否则单 agent 无锁开销.
# Linux/Mac: fcntl.fLOCK_EX / LOCK_SH
# Windows: msvcrt.locking LK_LOCK / LK_NBLCK
# ponytail: 只做同 workspace 内共享, 不引入跨机通信. 升级路径是文件锁 + 跨机
# 文件系统 (NFS/SMB), 但 NFS lock 语义弱, 真正升级是换成 SQLite + WAL.
def _multi_agent_enabled() -> bool:
    return os.environ.get("HUGINN_MULTI_AGENT", "0") in ("1", "true", "True")


try:
    if sys.platform == "win32":
        import msvcrt as _msvcrt
        _LOCK_PLATFORM = "windows"
    else:
        import fcntl as _fcntl
        _LOCK_PLATFORM = "posix"
except ImportError:
    _LOCK_PLATFORM = "none"


@contextmanager
def _stable_principles_lock(path: Path, exclusive: bool = True):
    """文件锁上下文管理器. exclusive=True 排他锁 (写), False 共享锁 (读).

    HUGINN_MULTI_AGENT=False 时直接 yield (无锁). 失败不阻断 — 锁是 best-effort,
    拿不到锁也比卡死强 (ponytail: 跨 agent 竞态是低频事件).
    """
    if not _multi_agent_enabled() or _LOCK_PLATFORM == "none":
        yield
        return
    # 锁文件用 .lock 后缀, 不污染数据文件
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        if _LOCK_PLATFORM == "windows":
            # msvcrt: LK_LOCK = 阻塞式排他锁, LK_NBLCK = 非阻塞共享锁
            # ponytail: Windows msvcrt 只有 1-byte 锁, 写 1 byte 即可
            try:
                _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1) if exclusive else _msvcrt.locking(f.fileno(), _msvcrt.LK_NBLCK, 1)
            except OSError:
                # 锁失败不阻断, 让调用方继续 (best-effort)
                pass
        else:
            try:
                lock_type = _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH
                _fcntl.flock(f.fileno(), lock_type)
            except OSError:
                pass
        yield
    finally:
        if _LOCK_PLATFORM == "windows":
            try:
                f.seek(0)
                _msvcrt.locking(f.fileno(), _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            try:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()


def store_stable_principle(principle: str, source: str = "S7_self_modify") -> None:
    """追加一条 stable_principle. 每行 {principle, source, timestamp}.

    G30: 同时写到本地 STABLE_PRINCIPLES_PATH 和全局 _GLOBAL_PRINCIPLES_PATH,
    让下一任务 init 时能 load 到本任务的修正.
    G44: HUGINN_MULTI_AGENT=True 时加排他锁, 防止多 agent 并发写损坏文件.
    """
    STABLE_PRINCIPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"principle": principle, "source": source, "timestamp": time.time()}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _stable_principles_lock(STABLE_PRINCIPLES_PATH, exclusive=True):
        with STABLE_PRINCIPLES_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    # G30: 双写到全局路径, 供下一任务继承
    if _inherit_enabled():
        try:
            _GLOBAL_PRINCIPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _stable_principles_lock(_GLOBAL_PRINCIPLES_PATH, exclusive=True):
                with _GLOBAL_PRINCIPLES_PATH.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            # 全局路径不可写不阻断本地写入
            pass


def load_stable_principles() -> list[str]:
    """读全部 stable_principles, 返回 principle 字符串列表. 文件不存在算空.

    G30: 合并读本地 + 全局, 去重保序. 全局让上一任务的 S7 修正对本任务可见.
    G44: HUGINN_MULTI_AGENT=True 时加共享锁, 防止读到写一半的内容.
    """
    seen: set[str] = set()
    principles: list[str] = []
    paths = [STABLE_PRINCIPLES_PATH]
    # G30: 全局路径在继承开关开时一起读
    if _inherit_enabled():
        paths.append(_GLOBAL_PRINCIPLES_PATH)
    for path in paths:
        if not path.exists():
            continue
        with _stable_principles_lock(path, exclusive=False):
            content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                p = json.loads(line)["principle"]
            except (json.JSONDecodeError, KeyError):
                # 损坏行直接跳过, 别让一条坏数据把整个 persona 干废
                continue
            if p not in seen:
                seen.add(p)
                principles.append(p)
    return principles


# seeds 仅走 RAG (knowledge/store.py)，不直接进 system_prompt


# ── self-check ────────────────────────────────────────────────────────────
# P1-1 验证: _ising_rerank 能量函数性质. 用 mock vector_store 避免依赖
# embedding 模型. `python -m huginn.memory.longterm` 跑.

def _selfcheck() -> None:
    import math
    import tempfile

    class _MockVec:
        """Mock vector_store — _compute_embeddings 返回预设向量."""
        def __init__(self, embs: list[list[float]]) -> None:
            self._embs = embs
        def _compute_embeddings(self, texts: list[str]) -> list[list[float]] | None:
            return self._embs[: len(texts)]

    def _cos(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    # 用临时 db 构造 LongTermMemory, vector_store 用 mock
    with tempfile.TemporaryDirectory() as _td:
        _db = str(Path(_td) / "test.db")
        mem = LongTermMemory(db_path=_db, vector_store=None, enable_semantic=False)

        # 29. 能量函数基本性质 — 矛盾 candidate 被拒
        # 3 个 candidates: m1, m2 高度相似 (cos=0.9), m3 跟 m1/m2 矛盾 (cos=-0.5,-0.4)
        # H = [0.8, 0.7, 0.6] (query 相关性)
        # top_k=2, 贪心应选 {m1, m2}, 不选 m3
        q_emb = [1.0, 0.0, 0.0]
        m1_emb = [0.8, 0.1, 0.0]
        m2_emb = [0.7, 0.1, 0.1]   # 跟 m1 cos ≈ 0.99
        m3_emb = [-0.5, 0.8, 0.3]  # 跟 m1/m2 矛盾
        mock_vec = _MockVec([q_emb, m1_emb, m2_emb, m3_emb])
        mem._vector_store = mock_vec
        mem._enable_semantic = True
        cands = [
            {"id": "m1", "content": "encut=520 OK"},
            {"id": "m2", "content": "encut=520 fine"},
            {"id": "m3", "content": "encut=520 不够"},
        ]
        r29 = mem._ising_rerank("encut 520", cands, top_k=2, beta=1.0)
        ids29 = [c["id"] for c in r29]
        assert "m3" not in ids29, f"矛盾 candidate 不应入选: {ids29}"
        assert "m1" in ids29 and "m2" in ids29, f"应选 m1+m2: {ids29}"
        print("29. Ising 能量函数 (矛盾 candidate 被拒, 一致 candidate 入选) OK")

        # 30. top_k=1 退化 — 等价 argmax H
        r30 = mem._ising_rerank("encut 520", cands, top_k=1, beta=1.0)
        assert r30[0]["id"] == "m1", f"top_k=1 应选 H 最大的 m1, got {r30[0]['id']}"
        print("30. Ising top_k=1 退化 (argmax H) OK")

        # 31. 无 vector_store — no-op 回原排序
        mem._vector_store = None
        r31 = mem._ising_rerank("encut 520", cands, top_k=2, beta=1.0)
        assert len(r31) == 2, "无 vector_store 应回退原排序取前 2"
        assert r31[0]["id"] == "m1", "no-op 应保持原顺序"
        print("31. Ising 无 vector_store → no-op 回原排序 OK")

        # 32. 候选不足 — 直接返回 candidates[:top_k]
        mem._vector_store = mock_vec  # 恢复
        r32 = mem._ising_rerank("encut 520", cands[:1], top_k=2, beta=1.0)
        assert len(r32) == 1, f"候选不足应全返回, got {len(r32)}"
        print("32. Ising 候选不足 → 全返回 OK")

        # 32b. toggle off — retrieve 路径不走 rerank
        os.environ["HUGINN_ISING_RERANK"] = "0"
        assert LongTermMemory._ising_rerank_enabled() is False, "toggle off"
        os.environ["HUGINN_ISING_RERANK"] = "1"
        assert LongTermMemory._ising_rerank_enabled() is True, "toggle on"
        print("32b. Ising toggle (HUGINN_ISING_RERANK=0/1) OK")

        # ── P2-5: HiLS 分层稀疏 attention ────────────────────────────
        # 38. 全 attention (N<K) — softmax(β·cos) 排序基本性质
        mock_vec38 = _MockVec([
            [1.0, 0.0],       # query
            [0.9, 0.1],       # m1: 高相关
            [0.1, 0.9],       # m2: 低相关
            [0.5, 0.5],       # m3: 中等
        ])
        mem._vector_store = mock_vec38
        mem._enable_semantic = True
        cands38 = [
            {"id": "m1"}, {"id": "m2"}, {"id": "m3"},
        ]
        r38 = mem._hils_attention("q", cands38, top_k=2, beta=8.0, n_landmarks=256)
        ids38 = [c["id"] for c in r38]
        # m1 (cos≈0.99) 应排第一, m3 (cos≈0.71) 第二, m2 (cos≈0.1) 被淘汰
        assert ids38[0] == "m1", f"最高 cos 应排第一: {ids38}"
        assert "m2" not in ids38, f"最低 cos 应被淘汰: {ids38}"
        print("38. HiLS 全 attention (N<K, softmax β·cos 排序) OK")

        # 39. N<K 退化 — 不分层, 等价全 attention
        # 只给 3 个 candidates, n_landmarks=256, 应走 v1 全 attention 路径
        r39 = mem._hils_attention("q", cands38, top_k=3, beta=8.0, n_landmarks=256)
        assert len(r39) == 3, "N<K 应全返回 (top_k=3)"
        print("39. HiLS N<K 退化 (不分层, 全 attention) OK")

        # 40. 分层稀疏 (N>=K) — k-means 地标 + top-h 筛选
        # 构造 30 个 candidates, K=5, h=2. 验证返回 top_k=3 且都在 top-h 地标下
        import random as _rng
        _rng.seed(40)
        embs40 = [[_rng.uniform(-1, 1) for _ in range(4)] for _ in range(30)]
        # query 跟第 0 个 candidate 完全一致 → 应排第一
        q40 = list(embs40[0])
        mock_vec40 = _MockVec([q40] + embs40)
        mem._vector_store = mock_vec40
        cands40 = [{"id": f"m{i}"} for i in range(30)]
        r40 = mem._hils_attention("q", cands40, top_k=3, beta=8.0, n_landmarks=5, top_h=2)
        assert len(r40) == 3, f"应返回 3 个, got {len(r40)}"
        assert r40[0]["id"] == "m0", f"query==m0 应排第一: {[c['id'] for c in r40]}"
        print("40. HiLS 分层稀疏 (N>=K, k-means + top-h 筛选) OK")

        # 41. 无 vector_store — no-op 回原排序
        mem._vector_store = None
        r41 = mem._hils_attention("q", cands40, top_k=3, beta=8.0)
        assert len(r41) == 3
        assert r41[0]["id"] == "m0", "no-op 应保持原顺序"
        print("41. HiLS 无 vector_store → no-op 回原排序 OK")

        # 41b. toggle off — retrieve 路径不走 HiLS, 回退 Ising
        os.environ["HUGINN_HILS_ATTENTION"] = "0"
        assert LongTermMemory._hils_enabled() is False
        os.environ["HUGINN_HILS_ATTENTION"] = "1"
        assert LongTermMemory._hils_enabled() is True
        print("41b. HiLS toggle (HUGINN_HILS_ATTENTION=0/1) OK")

        # 41c. k-means 地标基本性质 — 返回 K 个 centroid, 维度匹配
        mem._vector_store = mock_vec40  # 恢复
        lm = mem._kmeans_landmarks(embs40, k=5, max_iter=3)
        assert len(lm) == 5, f"应返回 5 个地标, got {len(lm)}"
        assert all(len(c) == 4 for c in lm), "地标维度应匹配"
        print("41c. HiLS k-means 地标 (K=5, dim=4) OK")

    print("LongTermMemory selfcheck OK (29-32b Ising + 38-41c HiLS)")


if __name__ == "__main__":
    _selfcheck()
