"""计算溯源注册表 — 追踪文件产出关系, 构建科学计算的 DAG.

每个工具调用产生的文件自动注册:
  (tool_name, inputs, parameters) → (file_path, format, key_properties)

agent 可以查询:
  - "Si 结构的弛豫结果在哪?" → 通过溯源链找到 OUTCAR
  - "哪些计算用了 PBE 泛函?" → 按参数查询
  - "这个结构的能量是多少?" → 从 key_properties 直接取

双层存储: 内存 dict 走热路径, SQLite 走持久化+恢复+全文搜索.
Codex rollout.jsonl + SQLite state_db 启发, 但我们不需要 rollout 级
回放, 只要重启不丢 + 查询快就行.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 内存缓存上限, SQLite 不限
_MEM_CACHE_LIMIT = 200


class VersionConflict(Exception):
    """乐观并发冲突: 注册时版本号已变, 说明有其他写入插队."""


@dataclass
class ProvenanceEntry:
    """一次文件产出的溯源记录."""

    file_path: str
    produced_by: str  # tool name
    produced_at: float  # timestamp
    input_files: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    file_format: str = ""  # poscar/outcar/cif/...
    key_properties: dict[str, Any] = field(default_factory=dict)
    # 能量/带隙/晶格常数等关键值, 直接存在这里,
    # agent 压缩后仍可查询, 不需要重新解析文件.
    snapshot_step_id: str | None = None  # 关联的文件快照 step_id (VC-1)
    reverted: bool = False  # 是否已被 rollback 标记 (VC-2)
    random_seed: int | None = None  # 随机种子, 保证可复现

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "input_files": list(self.input_files),
            "parameters": self.parameters,
            "file_format": self.file_format,
            "key_properties": self.key_properties,
            "snapshot_step_id": self.snapshot_step_id,
            "reverted": self.reverted,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ProvenanceEntry:
        """从 SQLite 行重建 entry."""
        keys = set(row.keys())
        return cls(
            file_path=row["file_path"],
            produced_by=row["produced_by"],
            produced_at=row["produced_at"],
            input_files=json.loads(row["input_files"] or "[]"),
            parameters=json.loads(row["parameters"] or "{}"),
            file_format=row["file_format"] or "",
            key_properties=json.loads(row["key_properties"] or "{}"),
            snapshot_step_id=row["snapshot_step_id"] if "snapshot_step_id" in keys else None,
            reverted=bool(row["reverted"]) if "reverted" in keys else False,
            random_seed=row["random_seed"] if "random_seed" in keys and row["random_seed"] is not None else None,
        )


class _ProvenanceStore:
    """SQLite 持久层. append-only 写, 索引加速查询.

    单连接 + 线程锁 + WAL (已在 __init__ 开启). ponytail: provenance 写入
    频率低 (每个 tool call 一次), 单连接够用. 高吞吐时换连接池.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # 跟项目其他 SQLite 模块保持一致: WAL 减少写延迟 (不触发 fsync),
        # synchronous=NORMAL 兼顾安全和速度. Windows AV 不会扫 WAL 文件.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path   TEXT NOT NULL,
                    produced_by TEXT NOT NULL,
                    produced_at REAL NOT NULL,
                    input_files TEXT,
                    parameters  TEXT,
                    file_format TEXT,
                    key_properties TEXT,
                    snapshot_step_id TEXT,
                    reverted INTEGER DEFAULT 0,
                    random_seed INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_path ON entries(file_path);
                CREATE INDEX IF NOT EXISTS idx_tool ON entries(produced_by);
                CREATE INDEX IF NOT EXISTS idx_time ON entries(produced_at);
                CREATE INDEX IF NOT EXISTS idx_fmt  ON entries(file_format);
            """)
            self._migrate_schema()
            self._conn.commit()

    def _migrate_schema(self) -> None:
        """老库可能缺 snapshot_step_id / reverted 列, 补上."""
        cur = self._conn.execute("PRAGMA table_info(entries)")
        existing = {row[1] for row in cur.fetchall()}
        if "snapshot_step_id" not in existing:
            self._conn.execute("ALTER TABLE entries ADD COLUMN snapshot_step_id TEXT")
        if "reverted" not in existing:
            self._conn.execute("ALTER TABLE entries ADD COLUMN reverted INTEGER DEFAULT 0")
        if "random_seed" not in existing:
            self._conn.execute("ALTER TABLE entries ADD COLUMN random_seed INTEGER")

    def save(self, entry: ProvenanceEntry) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO entries
                   (file_path, produced_by, produced_at, input_files,
                    parameters, file_format, key_properties,
                    snapshot_step_id, reverted, random_seed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.file_path,
                    entry.produced_by,
                    entry.produced_at,
                    json.dumps(entry.input_files),
                    json.dumps(entry.parameters),
                    entry.file_format,
                    json.dumps(entry.key_properties),
                    entry.snapshot_step_id,
                    int(entry.reverted),
                    entry.random_seed,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def find_by_path(self, path: str) -> ProvenanceEntry | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entries WHERE file_path = ? ORDER BY produced_at DESC LIMIT 1",
                (path,),
            )
            row = cur.fetchone()
            return ProvenanceEntry.from_row(row) if row else None

    def find_by_tool(self, tool_name: str) -> list[ProvenanceEntry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entries WHERE produced_by = ? ORDER BY produced_at DESC",
                (tool_name,),
            )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def find_by_format(self, fmt: str) -> list[ProvenanceEntry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entries WHERE file_format = ? ORDER BY produced_at DESC",
                (fmt,),
            )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def find_by_property(self, key: str, value: Any = None) -> list[ProvenanceEntry]:
        # key_properties 是 JSON, 用 LIKE 做粗筛再 Python 过滤
        # ponytail: 不上 FTS5, key_properties 的 key 不固定, LIKE 够用
        pattern = f'%"{key}"%'
        with self._lock:
            if value is None:
                cur = self._conn.execute(
                    "SELECT * FROM entries WHERE key_properties LIKE ? ORDER BY produced_at DESC",
                    (pattern,),
                )
            else:
                pattern2 = f'%"{key}": {json.dumps(value)}%'
                pattern3 = f'%"{key}":"{json.dumps(value)}"%'
                cur = self._conn.execute(
                    """SELECT * FROM entries WHERE key_properties LIKE ?
                       OR key_properties LIKE ?
                       ORDER BY produced_at DESC""",
                    (pattern2, pattern3),
                )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def get_lineage(self, file_path: str, depth: int = 5) -> list[ProvenanceEntry]:
        """递归查溯源链: file → input_files → 它们的 input_files → ..."""
        chain: list[ProvenanceEntry] = []
        visited: set[str] = set()
        current_path = file_path
        while depth > 0 and current_path and current_path not in visited:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT * FROM entries WHERE file_path = ? ORDER BY produced_at DESC LIMIT 1",
                    (current_path,),
                )
                row = cur.fetchone()
            if not row:
                break
            entry = ProvenanceEntry.from_row(row)
            chain.append(entry)
            visited.add(current_path)
            if entry.input_files:
                current_path = entry.input_files[0]
            else:
                break
            depth -= 1
        return chain

    def search(self, query_str: str) -> list[dict[str, Any]]:
        """LIKE-based 全文搜索, 按 file_path / tool / format / properties 匹配."""
        q = f"%{query_str.lower()}%"
        results: list[tuple[int, dict]] = []
        with self._lock:
            cur = self._conn.execute(
                """SELECT * FROM entries
                   WHERE LOWER(file_path) LIKE ?
                      OR LOWER(produced_by) LIKE ?
                      OR LOWER(file_format) LIKE ?
                      OR LOWER(key_properties) LIKE ?
                      OR LOWER(parameters) LIKE ?
                   ORDER BY produced_at DESC""",
                (q, q, q, q, q),
            )
            for row in cur.fetchall():
                entry = ProvenanceEntry.from_row(row)
                score = 0
                ql = query_str.lower()
                if ql in entry.file_path.lower():
                    score += 2
                if ql in entry.produced_by.lower():
                    score += 2
                if ql in entry.file_format.lower():
                    score += 1
                for k, v in entry.key_properties.items():
                    if ql in k.lower() or ql in str(v).lower():
                        score += 3
                results.append((score, entry.to_dict()))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:20]]

    def recent(self, n: int = 10) -> list[ProvenanceEntry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entries ORDER BY produced_at DESC LIMIT ?", (n,)
            )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM entries")
            return cur.fetchone()[0]

    def cleanup_old(self, days: int = 30) -> int:
        """删除超过 N 天的记录, 返回删除条数."""
        cutoff = time.time() - days * 86400
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entries WHERE produced_at < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    def get_events_since(self, since_id: int = 0, limit: int = 100) -> list[ProvenanceEntry]:
        """Event sourcing: fetch all events after a given id."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT * FROM entries WHERE id > ? ORDER BY id ASC LIMIT ?""",
                (since_id, limit),
            )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def get_events_by_tool(
        self, tool_name: str, limit: int = 50
    ) -> list[ProvenanceEntry]:
        """Fetch events for a specific tool, oldest first (for replay)."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT * FROM entries WHERE produced_by = ? ORDER BY id ASC LIMIT ?""",
                (tool_name, limit),
            )
            return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]

    def get_event_by_id(self, event_id: int) -> ProvenanceEntry | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entries WHERE id = ?", (event_id,)
            )
            row = cur.fetchone()
            return ProvenanceEntry.from_row(row) if row else None

    def get_max_id(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT MAX(id) FROM entries")
            val = cur.fetchone()[0]
            return val if val is not None else 0

    def mark_reverted(self, event_ids: list[int], reverted: bool = True) -> int:
        """批量标记事件为已回滚/取消回滚. 返回受影响行数."""
        if not event_ids:
            return 0
        placeholders = ",".join("?" * len(event_ids))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE entries SET reverted = ? WHERE id IN ({placeholders})",
                [int(reverted)] + event_ids,
            )
            self._conn.commit()
            return cur.rowcount

    def get_unreverted_ids_since(self, target_id: int) -> list[int]:
        """返回 target_id 之后未回滚的事件 id (倒序, 供 mark_reverted 用)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM entries WHERE id > ? AND reverted = 0 "
                "ORDER BY id DESC",
                (target_id,),
            )
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ProvenanceRegistry:
    """全局溯源注册表, 进程级单例.

    双层存储: 内存 dict 走热路径 (O(1) 查找), SQLite 走持久化
    (重启恢复 + 全文搜索). register() 同时写两层; find_* 先查内存,
    miss 时回退 SQLite 并 warm cache.
    """

    _instance: ProvenanceRegistry | None = None

    def __init__(self) -> None:
        self._entries: list[ProvenanceEntry] = []
        self._by_path: dict[str, ProvenanceEntry] = {}
        self._by_tool: dict[str, list[ProvenanceEntry]] = {}

        # SQLite 持久层
        cache_dir = os.environ.get("HUGINN_CACHE_DIR", "")
        if cache_dir:
            db_path = str(Path(cache_dir) / "provenance.db")
        else:
            db_path = str(Path.home() / ".huginn" / "provenance.db")
        self._store: _ProvenanceStore | None = None
        try:
            self._store = _ProvenanceStore(db_path)
            # warm cache: 从 SQLite 拉最近的 entries 填充内存
            for entry in self._store.recent(_MEM_CACHE_LIMIT):
                self._entries.append(entry)
                self._by_path[entry.file_path] = entry
                self._by_tool.setdefault(entry.produced_by, []).append(entry)
            if self._entries:
                logger.info(
                    "ProvenanceRegistry: warmed cache with %d entries from SQLite",
                    len(self._entries),
                )
        except Exception:
            logger.warning("SQLite provenance store init failed, running in-memory only", exc_info=True)
            self._store = None

    @classmethod
    def shared(cls) -> ProvenanceRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        file_path: str,
        produced_by: str,
        input_files: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        file_format: str = "",
        key_properties: dict[str, Any] | None = None,
        snapshot_step_id: str | None = None,
        expected_version: int | None = None,
    ) -> ProvenanceEntry:
        """注册一条产出记录. 同时写内存和 SQLite.

        Args:
            snapshot_step_id: 关联的文件快照 step_id, 用于 rollback 联动.
            expected_version: 调用方读过的版本号; 不匹配则抛 VersionConflict.
        """
        # 乐观并发: 版本号变了说明有其他写入插队
        if expected_version is not None and self._store is not None:
            current = self._store.get_max_id()
            if current != expected_version:
                raise VersionConflict(
                    f"Expected version {expected_version}, got {current}"
                )

        # auto-extract seed from parameters if not explicitly provided
        # many tools pass seed in parameters dict (gp_tool, packing_tool, etc.)
        _seed = None
        if parameters:
            for k in ("seed", "random_seed", "rng_seed"):
                v = parameters.get(k)
                if v is not None and isinstance(v, (int, float)):
                    _seed = int(v)
                    break

        entry = ProvenanceEntry(
            file_path=file_path,
            produced_by=produced_by,
            produced_at=time.time(),
            input_files=input_files or [],
            parameters=parameters or {},
            file_format=file_format,
            key_properties=key_properties or {},
            snapshot_step_id=snapshot_step_id,
            random_seed=_seed,
        )

        # 写内存 (热路径)
        self._entries.append(entry)
        self._by_path[file_path] = entry
        self._by_tool.setdefault(produced_by, []).append(entry)

        # 内存缓存上限, FIFO 淘汰最旧的
        if len(self._entries) > _MEM_CACHE_LIMIT:
            old = self._entries.pop(0)
            # 不删 by_path, 让查询仍能命中 (虽然不在 list 里)
            # ponytail: 不精确清理 by_tool list, 热路径查询不受影响

        # 写 SQLite (持久化)
        if self._store is not None:
            try:
                self._store.save(entry)
            except Exception:
                logger.debug("SQLite save failed (non-fatal)", exc_info=True)

        return entry

    def find_by_path(self, path: str) -> ProvenanceEntry | None:
        # 先查内存
        entry = self._by_path.get(path)
        if entry is not None:
            return entry
        # miss → 查 SQLite 并 warm cache
        if self._store is not None:
            entry = self._store.find_by_path(path)
            if entry is not None:
                self._by_path[path] = entry
                self._by_tool.setdefault(entry.produced_by, []).append(entry)
            return entry
        return None

    def find_by_tool(self, tool_name: str) -> list[ProvenanceEntry]:
        # 内存有就走内存
        cached = self._by_tool.get(tool_name)
        if cached:
            return cached
        # 回退 SQLite
        if self._store is not None:
            results = self._store.find_by_tool(tool_name)
            if results:
                self._by_tool[tool_name] = results
            return results
        return []

    def find_by_format(self, fmt: str) -> list[ProvenanceEntry]:
        # 内存扫描
        results = [e for e in self._entries if e.file_format == fmt]
        if results:
            return results
        # 回退 SQLite
        if self._store is not None:
            return self._store.find_by_format(fmt)
        return []

    def find_by_property(self, key: str, value: Any = None) -> list[ProvenanceEntry]:
        # 内存扫描
        results = []
        for e in self._entries:
            if key in e.key_properties:
                if value is None or e.key_properties[key] == value:
                    results.append(e)
        if results:
            return results
        # 回退 SQLite
        if self._store is not None:
            return self._store.find_by_property(key, value)
        return []

    def find_by_seed(self, seed: int) -> list[ProvenanceEntry]:
        """按随机种子查溯源 — 复现性验证用."""
        # 内存扫描
        results = [e for e in self._entries if e.random_seed == seed]
        if results:
            return results
        # SQLite
        if self._store is not None:
            with self._store._lock:
                cur = self._store._conn.execute(
                    "SELECT * FROM entries WHERE random_seed = ? ORDER BY produced_at DESC",
                    (seed,),
                )
                return [ProvenanceEntry.from_row(r) for r in cur.fetchall()]
        return []

    def get_lineage(self, file_path: str, depth: int = 5) -> list[ProvenanceEntry]:
        # 内存能走就走 (快)
        chain: list[ProvenanceEntry] = []
        visited: set[str] = set()
        current = self._by_path.get(file_path)
        if current is not None:
            while current and depth > 0 and current.file_path not in visited:
                chain.append(current)
                visited.add(current.file_path)
                if current.input_files:
                    current = self._by_path.get(current.input_files[0])
                else:
                    break
                depth -= 1
            if chain:
                return chain
        # 内存没命中 → SQLite 递归查
        if self._store is not None:
            return self._store.get_lineage(file_path, depth)
        return chain

    def query(self, query_str: str) -> list[dict[str, Any]]:
        # 优先走 SQLite (有索引, LIKE 快)
        if self._store is not None:
            return self._store.search(query_str)
        # 回退内存扫描
        q = query_str.lower()
        results = []
        for e in self._entries:
            score = 0
            if q in e.file_path.lower():
                score += 2
            if q in e.produced_by.lower():
                score += 2
            if q in e.file_format.lower():
                score += 1
            for k, v in e.key_properties.items():
                if q in k.lower() or q in str(v).lower():
                    score += 3
            if score > 0:
                results.append((score, e.to_dict()))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:20]]

    def count(self) -> int:
        if self._store is not None:
            return self._store.count()
        return len(self._entries)

    def recent(self, n: int = 10) -> list[ProvenanceEntry]:
        # SQLite 有 produced_at 索引, 走 DESC 查询最快; 不可用时退回内存
        # 按时间倒序 (内存 _entries 不保证有序, 见 register 的 append 逻辑)
        if self._store is not None:
            return self._store.recent(n)
        entries = sorted(self._entries, key=lambda e: e.produced_at, reverse=True)
        return entries[:n] if n > 0 else []

    def summary(self) -> dict[str, Any]:
        total = self._store.count() if self._store else len(self._entries)
        return {
            "total_files": total,
            "by_tool": {k: len(v) for k, v in self._by_tool.items()},
            "by_format": {},
            "recent": [e.to_dict() for e in self._entries[-10:]],
        }

    def to_context_block(self) -> str:
        """生成可插入上下文的状态块, 跨压缩保留."""
        if not self._entries:
            return ""
        lines = ["### Provenance registry (active files):"]
        for e in self._entries[-10:]:
            props = ""
            if e.key_properties:
                props = " | " + ", ".join(
                    f"{k}={v}" for k, v in e.key_properties.items()
                )
            lines.append(f"  - {e.file_path} ({e.file_format or '?'}) by {e.produced_by}{props}")
        return "\n".join(lines)

    def cleanup_old(self, days: int = 30) -> int:
        """删除超过 N 天的记录. 仅操作 SQLite, 内存缓存自然过期."""
        if self._store is not None:
            deleted = self._store.cleanup_old(days)
            if deleted > 0:
                logger.info("Cleaned up %d provenance entries older than %d days", deleted, days)
            return deleted
        return 0

    # ── Event Sourcing API ──────────────────────────────────────

    def get_events(
        self,
        since_id: int = 0,
        limit: int = 100,
        tool: str | None = None,
    ) -> list[ProvenanceEntry]:
        """Query historical events — the core event sourcing API.

        Agent can use this to:
        - Resume from a checkpoint: get_events(since_id=last_seen_id)
        - Inspect a specific tool's history: get_events(tool="vasp_tool")
        - Full audit: get_events(limit=10000)

        Returns events in chronological order (oldest first).
        """
        if self._store is None:
            # Memory-only fallback
            if tool:
                return [e for e in self._entries if e.produced_by == tool][:limit]
            return self._entries[since_id:since_id + limit]
        if tool:
            return self._store.get_events_by_tool(tool, limit)
        return self._store.get_events_since(since_id, limit)

    def replay_to(self, target_id: int) -> list[ProvenanceEntry]:
        """Replay all events up to a given event id.

        Returns the full sequence of events from the beginning to
        target_id (inclusive). Agent can reconstruct state at any
        point by replaying from event 1 to target_id.
        """
        if self._store is None:
            return list(self._entries)
        return self._store.get_events_since(0, target_id + 1)

    def rollback_to(self, target_id: int) -> list[str]:
        """回滚到 target_id 之前的状态: 调 SnapshotManager.revert() 回退文件.

        找到 target_id 之后的所有事件, 对其中关联了 snapshot_step_id 的,
        按 reverse 顺序调 SnapshotManager.revert() 做真正的文件回滚.
        同时把事件标记为 reverted=True.

        返回受影响的文件路径列表.
        """
        if self._store is None:
            # 内存模式: 无 step_id 关联, 只返回路径
            return [e.file_path for e in self._entries if not e.reverted][::-1]

        events = self._store.get_events_since(target_id, 10000)
        if not events:
            return []

        reverted_paths: list[str] = []
        snap_step_ids: list[str] = []
        sid_to_path: dict[str, str] = {}  # revert 失败时按 sid 移除对应路径

        # reverse 顺序: 先回滚最新的
        for ev in reversed(events):
            if ev.reverted:
                continue
            reverted_paths.append(ev.file_path)
            if ev.snapshot_step_id:
                snap_step_ids.append(ev.snapshot_step_id)
                sid_to_path[ev.snapshot_step_id] = ev.file_path

        # 真正的文件回滚: 调 SnapshotManager
        if snap_step_ids:
            try:
                from huginn.snapshot import SnapshotManager

                mgr = SnapshotManager()
                for sid in snap_step_ids:
                    # workspace 从 snapshot 记录里取
                    snap = mgr._load(sid)
                    if snap is not None and not snap.reverted:
                        ws = Path(snap.workspace)
                        try:
                            mgr.revert(sid, ws)
                        except Exception:
                            logger.warning("revert %s failed (non-fatal)", sid, exc_info=True)
                            # 失败的 path 从 reverted_paths 移除, 让调用方区分
                            # 真回滚 vs 标记回滚但文件没动
                            failed_path = sid_to_path.get(sid)
                            if failed_path and failed_path in reverted_paths:
                                reverted_paths.remove(failed_path)
            except ImportError:
                logger.debug("SnapshotManager not available, skipping file revert")

        # 标记 provenance 事件为已回滚 (用 store 封装方法, 不直接访问 _lock/_conn)
        ev_ids = self._store.get_unreverted_ids_since(target_id)
        if ev_ids:
            self._store.mark_reverted(ev_ids, reverted=True)

        logger.info(
            "rollback_to(%d): %d paths, %d snapshot reverts",
            target_id, len(reverted_paths), len(snap_step_ids),
        )
        return reverted_paths

    def revert_to_version(self, version: int) -> list[str]:
        """统一版本时钟入口: 回滚到指定版本号.

        version 是 current_version() 返回的 event id.
        等价于 rollback_to(version), 但语义更明确: 调用方传的是
        "我读到的版本号", 而非 "回滚到这个 id 之后".
        """
        return self.rollback_to(version)

    def get_event(self, event_id: int) -> ProvenanceEntry | None:
        """Get a single event by id."""
        if self._store is None:
            if 0 <= event_id < len(self._entries):
                return self._entries[event_id]
            return None
        return self._store.get_event_by_id(event_id)

    def current_version(self) -> int:
        """当前版本号 (max event id). 乐观并发用:
        读 version → 操作 → register(expected_version=version) → 冲突则报错.
        """
        if self._store is None:
            return len(self._entries)
        return self._store.get_max_id()


def register_tool_output(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
) -> None:
    """从工具调用中自动提取文件路径和关键属性, 注册到溯源表.

    在 ToolAdapter._run_post_checks 里调用.
    """
    try:
        reg = ProvenanceRegistry.shared()

        # 从 snapshot integration 拿 step_id, 建立 provenance↔snapshot 双向链
        snap_step_id: str | None = None
        try:
            from huginn.snapshot.integration import consume_last_snapshot_step_id
            snap_step_id = consume_last_snapshot_step_id()
        except ImportError:
            pass

        # 从 tool_input 提取输入文件
        input_files: list[str] = []
        for key in ("file_path", "working_dir", "poscar_path", "structure_file"):
            val = tool_input.get(key)
            if val and isinstance(val, str):
                input_files.append(val)

        # 从 tool_output 提取产出文件和关键属性
        if not isinstance(tool_output, dict):
            return

        result = tool_output.get("result", tool_output)
        if not isinstance(result, dict):
            return

        # 提取文件路径
        output_paths: list[str] = []
        for key in ("output_file", "outcar_path", "trajectory_file", "file_path", "saved_to"):
            val = result.get(key)
            if val and isinstance(val, str):
                output_paths.append(val)

        # 提取关键属性
        key_props: dict[str, Any] = {}
        for key in (
            "energy", "total_energy", "free_energy", "E0",
            "band_gap", "lattice_constant", "converged",
            "forces_max", "stress_max", "pressure",
            "spacegroup", "volume", "density",
            "n_atoms", "formula",
        ):
            val = result.get(key)
            if val is not None:
                key_props[key] = val

        # 提取参数
        params: dict[str, Any] = {}
        for key in ("action", "encut", "ediff", "kpoints", "functional", "basis_set", "method"):
            val = tool_input.get(key)
            if val is not None:
                params[key] = val

        # 推断文件格式
        fmt = ""
        for key in ("file_format", "format"):
            val = result.get(key)
            if val:
                fmt = str(val)
                break
        if not fmt and output_paths:
            ext = Path(output_paths[0]).suffix.lstrip(".").lower()
            fmt = ext

        # 注册每个产出文件
        for path in output_paths:
            reg.register(
                file_path=path,
                produced_by=tool_name,
                input_files=input_files,
                parameters=params,
                file_format=fmt,
                key_properties=key_props,
                snapshot_step_id=snap_step_id,
            )
    except Exception:
        logger.debug("register_tool_output failed (non-fatal)", exc_info=True)
