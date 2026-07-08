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

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "input_files": list(self.input_files),
            "parameters": self.parameters,
            "file_format": self.file_format,
            "key_properties": self.key_properties,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ProvenanceEntry:
        """从 SQLite 行重建 entry."""
        return cls(
            file_path=row["file_path"],
            produced_by=row["produced_by"],
            produced_at=row["produced_at"],
            input_files=json.loads(row["input_files"] or "[]"),
            parameters=json.loads(row["parameters"] or "{}"),
            file_format=row["file_format"] or "",
            key_properties=json.loads(row["key_properties"] or "{}"),
        )


class _ProvenanceStore:
    """SQLite 持久层. append-only 写, 索引加速查询.

    单连接 + 线程锁, ponytail: 不做连接池, provenance 写入频率低
    (每个 tool call 一次), 单连接够用. 高吞吐时换 WAL + 连接池.
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
                    key_properties TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_path ON entries(file_path);
                CREATE INDEX IF NOT EXISTS idx_tool ON entries(produced_by);
                CREATE INDEX IF NOT EXISTS idx_time ON entries(produced_at);
                CREATE INDEX IF NOT EXISTS idx_fmt  ON entries(file_format);
            """)
            self._conn.commit()

    def save(self, entry: ProvenanceEntry) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO entries
                   (file_path, produced_by, produced_at, input_files,
                    parameters, file_format, key_properties)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.file_path,
                    entry.produced_by,
                    entry.produced_at,
                    json.dumps(entry.input_files),
                    json.dumps(entry.parameters),
                    entry.file_format,
                    json.dumps(entry.key_properties),
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
    ) -> ProvenanceEntry:
        """注册一条产出记录. 同时写内存和 SQLite."""
        entry = ProvenanceEntry(
            file_path=file_path,
            produced_by=produced_by,
            produced_at=time.time(),
            input_files=input_files or [],
            parameters=parameters or {},
            file_format=file_format,
            key_properties=key_properties or {},
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
            )
    except Exception:
        logger.debug("register_tool_output failed (non-fatal)", exc_info=True)
