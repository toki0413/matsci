"""结构化研究日志 (Research Log) — 猜想的演化树.

设计参考: Moonshine 数学 agent 的"研究日志与记忆"模块. 这不是普通的
audit log (谁在什么时候调了什么工具), 而是用来追踪 *思想本身的演化*:
每条记录是研究过程中产生的一个节点 —— 猜想、证明尝试、反例、开放问题、
障碍、跨领域搭桥、边界刻画、独立验证 —— 通过 parent_id 串成一棵演化树.

典型用法:

    log = get_research_log()
    c = log.add(RecordType.CONJECTURE,
                "钙钛矿带隙随容忍因子线性变化",
                "容忍因子 t 在 [0.9, 1.05] 时 Eg 与 t 近似线性 ...",
                tags=["perovskite", "band_gap", "DFT"])
    attempt = log.add(RecordType.PROOF_ATTEMPT,
                      "用 12 种钙钛矿 DFT 数据回归验证",
                      "R^2 = 0.62, 线性假设不成立",
                      parent_id=c.id, source="agent")
    log.update_status(attempt.id, "refuted")

所有数据落在 SQLite (WAL 模式), 路径优先取 $HUGINN_CACHE_DIR, 没有就退回
~/.huginn/. 线程安全靠一把实例锁兜底, 跟 AnomalyLogStore 一个思路.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# status 的合法取值, 散落在各处硬编码容易写错, 集中声明一下方便校验.
# 这里不做成 Enum 是因为 spec 要求 status: str, 调用方也经常自己拼字符串.
VALID_STATUSES = (
    "proposed",
    "in_progress",
    "verified",
    "refuted",
    "superseded",
)


class RecordType(str, Enum):
    """研究记录的类型.

    继承 str 是为了直接 json.dumps / 写库时当字符串用, 省得到处 .value.
    """

    CONJECTURE = "conjecture"          # 新猜想 / 假设
    PROOF_ATTEMPT = "proof_attempt"    # 证明 / 验证尝试
    COUNTEREXAMPLE = "counterexample"  # 反例
    OPEN_QUESTION = "open_question"    # 开放问题
    OBSTACLE = "obstacle"             # 障碍识别
    BRIDGE = "bridge"                 # 理论搭桥 (跨领域关联)
    BOUNDARY = "boundary"             # 边界刻画 (适用条件确定)
    VERIFICATION = "verification"    # 独立验证结果


@dataclass
class ResearchRecord:
    """一条研究记录, 对应演化树里的一个节点.

    parent_id 指向上一条记录, 形成 DAG (一般是一棵树, 但不强制唯一父节点
    的语义, 调用方可以自己解释). metadata 留给那些不好建固定字段的附加上下文,
    比如 DFT 计算的参数、引用的文献 DOI 之类.
    """

    id: str
    timestamp: str
    record_type: RecordType
    title: str
    content: str
    parent_id: str | None = None
    status: str = "proposed"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        """展平成可 JSON 化的 dict, 给上层 / API / 导出用."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "record_type": self.record_type.value,
            "title": self.title,
            "content": self.content,
            "parent_id": self.parent_id,
            "status": self.status,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "source": self.source,
        }


@dataclass
class ResearchLogConfig:
    """研究日志的运行时配置.

    max_records 触发清理时, 优先删 archived=1 的记录里 timestamp 最老的,
    再不够才动 superseded 的非归档记录 —— 保证正在用的猜想不被误删.
    """

    enabled: bool = True
    max_records: int = 10000
    auto_archive: bool = True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_records (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    record_type TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    parent_id   TEXT,
    status      TEXT NOT NULL,
    tags        TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    source      TEXT NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_research_type     ON research_records(record_type);
CREATE INDEX IF NOT EXISTS idx_research_status   ON research_records(status);
CREATE INDEX IF NOT EXISTS idx_research_parent   ON research_records(parent_id);
CREATE INDEX IF NOT EXISTS idx_research_archived ON research_records(archived);
"""


def _migrate_research_log_v1(conn: sqlite3.Connection) -> None:
    """v1 baseline -- the schema is created by _SCHEMA below, so this is a no-op.

    Exists to establish the user_version baseline so future schema changes
    can be tracked properly instead of relying on CREATE TABLE IF NOT EXISTS
    + scattered ALTER TABLE.
    """
    pass


class ResearchLog:
    """SQLite 后端的结构化研究日志.

    一条记录 = 演化树的一个节点. 所有写操作都走同一把锁, 读操作也加锁
    是为了避免 SQLite "database is locked" —— WAL 已经够并发了, 但 agent
    主体异步 + checkpointer 在别的线程跑的情况还是会出现, 加锁最省心.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        config: ResearchLogConfig | None = None,
    ) -> None:
        self.config = config or ResearchLogConfig()
        self._db_path = str(self._resolve_path(db_path))
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: 跨线程共用连接, 锁我们自己管
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        # Run versioned migrations before applying the base schema
        from huginn.utils.migrations import MigrationManager

        _mgr = MigrationManager(self._db_path)
        try:
            _mgr.run_migrations([(1, _migrate_research_log_v1)])
        finally:
            _mgr.close()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # WAL: 读写不互斥, 崩溃也比默认的 rollback journal 更不容易丢数据
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.commit()

    @staticmethod
    def _resolve_path(db_path: str | Path | None) -> Path:
        """决定 sqlite 文件位置: 显式传入 > runtime home.

        统一走 utils.runtime.get_runtime_home(), 不再各自硬编码.
        """
        if db_path is not None:
            return Path(db_path)
        try:
            from huginn.utils.runtime import get_runtime_home
            return get_runtime_home() / "research_log.sqlite"
        except Exception:
            base = os.environ.get("HUGINN_CACHE_DIR")
            if base:
                return Path(base) / "research_log.sqlite"
            return Path.home() / ".huginn" / "research_log.sqlite"

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(
        self,
        record_type: RecordType,
        title: str,
        content: str,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "agent",
        metadata: dict[str, Any] | None = None,
        status: str = "proposed",
    ) -> ResearchRecord:
        """追加一条研究记录, 返回建好的 ResearchRecord.

        parent_id 指向上一条记录就形成了演化关系; 不传就是一棵新树的根.
        """
        if status not in VALID_STATUSES:
            # 不抛异常太烦人, 退回 proposed 并记一笔, 让调用方在日志里看到
            logger.warning("unknown status %r, fallback to proposed", status)
            status = "proposed"

        record = ResearchRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            record_type=record_type,
            title=title,
            content=content,
            parent_id=parent_id,
            status=status,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
            source=source,
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO research_records (
                    id, timestamp, record_type, title, content,
                    parent_id, status, tags, metadata, source, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    record.id,
                    record.timestamp,
                    record.record_type.value,
                    record.title,
                    record.content,
                    record.parent_id,
                    record.status,
                    json.dumps(record.tags, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False, default=str),
                    record.source,
                ),
            )
            self._conn.commit()
            # 写完顺手做一次容量检查, 超了就清最老的归档记录
            self._maybe_cleanup()

        logger.info(
            "research record added: %s [%s] %s",
            record.id, record.record_type.value, record.title,
        )
        return record

    def update_status(self, record_id: str, new_status: str) -> bool:
        """更新某条记录的状态. refuted/superseded 在 auto_archive 开启时
        会顺带打上 archived 标记 (记录本身不删, 树还得留着看).

        返回是否真的更新到了 (找不到 / 状态没变都算 False).
        """
        if new_status not in VALID_STATUSES:
            logger.warning("unknown status %r, ignored", new_status)
            return False

        archive_flag = 1 if (
            self.config.auto_archive and new_status in ("refuted", "superseded")
        ) else None

        with self._lock:
            if archive_flag is not None:
                cur = self._conn.execute(
                    """
                    UPDATE research_records
                       SET status = ?, archived = ?
                     WHERE id = ? AND status != ?
                    """,
                    (new_status, archive_flag, record_id, new_status),
                )
            else:
                cur = self._conn.execute(
                    """
                    UPDATE research_records
                       SET status = ?
                     WHERE id = ? AND status != ?
                    """,
                    (new_status, record_id, new_status),
                )
            self._conn.commit()
            updated = cur.rowcount > 0

        if updated:
            logger.info("record %s status -> %s", record_id, new_status)
        return updated

    def _maybe_cleanup(self) -> None:
        """超过 max_records 时清掉最老的归档记录, 再不够才动 superseded.

        必须在持锁状态下调用. 故意不删 verified / in_progress —— 这些是
        正在用的结论, 删了树就断了.
        """
        if self.config.max_records <= 0:
            return
        total = self._conn.execute(
            "SELECT COUNT(*) AS c FROM research_records"
        ).fetchone()["c"]
        if total <= self.config.max_records:
            return

        overflow = total - self.config.max_records
        # 先清归档的, 再清 superseded 但没归档的, 都按时间从老到新
        self._conn.execute(
            """
            DELETE FROM research_records
             WHERE id IN (
                SELECT id FROM research_records
                 WHERE archived = 1
                 ORDER BY timestamp ASC
                 LIMIT ?
            )
            """,
            (overflow,),
        )
        # 归档不够删, 再补一刀 superseded
        remaining = total - self._conn.execute(
            "SELECT COUNT(*) AS c FROM research_records"
        ).fetchone()["c"]
        if remaining > self.config.max_records:
            extra = remaining - self.config.max_records
            self._conn.execute(
                """
                DELETE FROM research_records
                 WHERE id IN (
                    SELECT id FROM research_records
                     WHERE status = 'superseded'
                     ORDER BY timestamp ASC
                     LIMIT ?
                )
                """,
                (extra,),
            )
        logger.info("research log cleanup done, target=%d", self.config.max_records)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, record_id: str) -> ResearchRecord | None:
        """按 id 取单条, 没有返回 None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_by_type(self, record_type: RecordType) -> list[ResearchRecord]:
        """按类型列出, 时间升序 (演化顺序)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM research_records WHERE record_type = ? ORDER BY timestamp ASC",
                (record_type.value,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_status(self, status: str, limit: int | None = None) -> list[ResearchRecord]:
        """按状态列出, 时间升序. limit 截取前 N 条."""
        with self._lock:
            if limit is not None:
                rows = self._conn.execute(
                    "SELECT * FROM research_records WHERE status = ? ORDER BY timestamp ASC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM research_records WHERE status = ? ORDER BY timestamp ASC",
                    (status,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_tag(self, tag: str) -> list[ResearchRecord]:
        """按标签列出. tags 存成 JSON 数组文本, 用 LIKE 匹配引号包裹的标签,
        避免子串误命中 (比如 "DFT" 不会匹中 "DFT_validation").
        """
        pattern = f'%"{tag}"%'
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM research_records
                 WHERE tags LIKE ?
                 ORDER BY timestamp ASC
                """,
                (pattern,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_children(self, record_id: str) -> list[ResearchRecord]:
        """直接子记录, 时间升序."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM research_records WHERE parent_id = ? ORDER BY timestamp ASC",
                (record_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_tree(self, record_id: str) -> dict | None:
        """递归获取以 record_id 为根的整棵子树.

        返回结构: {"record": ResearchRecord, "children": [子树, ...]}.
        找不到根记录返回 None.
        """
        root = self.get(record_id)
        if root is None:
            return None
        return self._build_subtree(root)

    def _build_subtree(self, record: ResearchRecord) -> dict:
        """递归构造子树, get_tree 内部用."""
        children = self.get_children(record.id)
        return {
            "record": record,
            "children": [self._build_subtree(c) for c in children],
        }

    def search(self, query: str) -> list[ResearchRecord]:
        """标题 + 内容的模糊搜索, 大小写不敏感 (SQLite LIKE 对 ASCII 默认不敏感).

        顺序: 先命标题的, 再命内容的, 去重后返回.
        """
        if not query:
            return []
        pattern = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM research_records
                 WHERE title LIKE ? OR content LIKE ?
                 ORDER BY timestamp ASC
                """,
                (pattern, pattern),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # 统计 & 渲染
    # ------------------------------------------------------------------
    def get_stats(self) -> dict[str, Any]:
        """各 record_type / status 的数量统计, 外加总数和归档数."""
        with self._lock:
            type_rows = self._conn.execute(
                "SELECT record_type, COUNT(*) AS c FROM research_records GROUP BY record_type"
            ).fetchall()
            status_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS c FROM research_records GROUP BY status"
            ).fetchall()
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM research_records"
            ).fetchone()["c"]
            archived = self._conn.execute(
                "SELECT COUNT(*) AS c FROM research_records WHERE archived = 1"
            ).fetchone()["c"]
        return {
            "by_type": {r["record_type"]: r["c"] for r in type_rows},
            "by_status": {r["status"]: r["c"] for r in status_rows},
            "total": total,
            "archived": archived,
        }

    def render_markdown(
        self,
        record_type: RecordType | None = None,
        status: str | None = None,
    ) -> str:
        """按 type / status 过滤后渲染成 Markdown.

        没有 record_type 过滤时, 按 record_type 分节, 每节里以根记录
        (parent_id 为空或父不在结果集里) 为入口渲染演化树, 子节点缩进.
        指定了 record_type 时, 该类型的记录平铺列出, 但会带上父记录的
        标题作为上下文 (因为跨类型的树结构会断).
        """
        with self._lock:
            if record_type is not None and status is not None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM research_records
                     WHERE record_type = ? AND status = ?
                     ORDER BY timestamp ASC
                    """,
                    (record_type.value, status),
                ).fetchall()
            elif record_type is not None:
                rows = self._conn.execute(
                    "SELECT * FROM research_records WHERE record_type = ? ORDER BY timestamp ASC",
                    (record_type.value,),
                ).fetchall()
            elif status is not None:
                rows = self._conn.execute(
                    "SELECT * FROM research_records WHERE status = ? ORDER BY timestamp ASC",
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM research_records ORDER BY timestamp ASC"
                ).fetchall()

        records = [self._row_to_record(r) for r in rows]

        if record_type is not None:
            return self._render_flat(records, record_type)
        return self._render_trees_by_type(records)

    def _render_trees_by_type(self, records: list[ResearchRecord]) -> str:
        """按 record_type 分节渲染演化树. 根 = parent_id 为空或父不在结果集."""
        lines: list[str] = []
        if not records:
            lines.append("_(research log is empty)_")
            return "\n".join(lines)

        by_id = {r.id: r for r in records}
        child_map: dict[str | None, list[ResearchRecord]] = {}
        for r in records:
            # 父不在结果集里, 当成根处理 (这样 status 过滤导致中间节点缺失时不会丢记录)
            parent = r.parent_id if r.parent_id in by_id else None
            child_map.setdefault(parent, []).append(r)

        # 按 record_type 的声明顺序分节, 输出更稳定
        roots = child_map.get(None, [])
        roots_by_type: dict[RecordType, list[ResearchRecord]] = {}
        for r in roots:
            roots_by_type.setdefault(r.record_type, []).append(r)

        lines.append("# Research Log")
        lines.append("")
        for rt in RecordType:
            group = roots_by_type.get(rt, [])
            if not group:
                continue
            lines.append(f"## {rt.value}")
            lines.append("")
            for root in group:
                self._render_tree_node(root, child_map, lines, depth=0)
            lines.append("")
        return "\n".join(lines)

    def _render_tree_node(
        self,
        record: ResearchRecord,
        child_map: dict[str | None, list[ResearchRecord]],
        lines: list[str],
        depth: int,
    ) -> None:
        """递归渲染一个节点, 用缩进表达层级."""
        indent = "  " * depth
        bullet = "-" if depth == 0 else "*"
        tags = f" `{'` `'.join(record.tags)}`" if record.tags else ""
        lines.append(
            f"{indent}{bullet} **[{record.record_type.value}]** {record.title} "
            f"`{record.status}`{tags}"
        )
        # 内容太长就截断, 免得 Markdown 爆掉
        body = record.content.strip()
        if body:
            snippet = body if len(body) <= 200 else body[:200] + " ..."
            lines.append(f"{indent}    {snippet.replace(chr(10), ' ')}")
        meta = f"_{record.timestamp} · {record.source}_"
        lines.append(f"{indent}    {meta}")
        for child in child_map.get(record.id, []):
            self._render_tree_node(child, child_map, lines, depth + 1)

    def _render_flat(self, records: list[ResearchRecord], record_type: RecordType) -> str:
        """指定了 record_type 时, 平铺列出并附上父记录标题作为上下文."""
        lines: list[str] = [f"# Research Log — {record_type.value}", ""]
        if not records:
            lines.append("_(no records of this type)_")
            return "\n".join(lines)

        # 缓存一下父标题, 避免同一条父记录反复查库
        parent_titles: dict[str, str] = {}
        for r in records:
            if r.parent_id and r.parent_id not in parent_titles:
                parent = self.get(r.parent_id)
                parent_titles[r.parent_id] = parent.title if parent else "(missing parent)"
        for r in records:
            tags = f" `{'` `'.join(r.tags)}`" if r.tags else ""
            lines.append(f"## {r.title} `{r.status}`{tags}")
            lines.append("")
            lines.append(f"- **id**: `{r.id}`")
            lines.append(f"- **time**: {r.timestamp}")
            lines.append(f"- **source**: {r.source}")
            if r.parent_id:
                p_title = parent_titles.get(r.parent_id, "(missing parent)")
                lines.append(f"- **parent**: {p_title} (`{r.parent_id}`)")
            lines.append("")
            lines.append(r.content.strip() or "_(no content)_")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _row_to_record(self, row: sqlite3.Row) -> ResearchRecord:
        """sqlite3.Row -> ResearchRecord. JSON 字段反序列化."""
        return ResearchRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            record_type=RecordType(row["record_type"]),
            title=row["title"],
            content=row["content"],
            parent_id=row["parent_id"],
            status=row["status"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source=row["source"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ----------------------------------------------------------------------
# 模块级单例. 第一次调用时懒加载, 用默认路径和默认配置.
# 测试要隔离的话直接 new 一个 ResearchLog(db_path=...) 就行, 别碰这个单例.
# ----------------------------------------------------------------------
_research_log_singleton: ResearchLog | None = None
_singleton_lock = threading.Lock()


def get_research_log() -> ResearchLog:
    """拿模块级单例 ResearchLog. 线程安全地懒加载."""
    global _research_log_singleton
    if _research_log_singleton is None:
        with _singleton_lock:
            # 双检锁, 避免两个线程同时过了第一道 None 检查
            if _research_log_singleton is None:
                _research_log_singleton = ResearchLog()
                logger.info("research log singleton initialized at %s", _research_log_singleton._db_path)
    return _research_log_singleton


__all__ = [
    "RecordType",
    "ResearchRecord",
    "ResearchLog",
    "ResearchLogConfig",
    "get_research_log",
    "VALID_STATUSES",
]
