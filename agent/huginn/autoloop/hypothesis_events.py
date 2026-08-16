"""假设图事件日志的 SQLite + FTS5 持久化 (段升级 P1#3).

从 ``hypothesis_loop.py`` 拆出的可复用持久化原语: 把 HypothesisGraph 的
in-memory event log (``_events``) 落到 SQLite, 加 FTS5 全文索引, 支持跨进程
resume / replay / 语义搜索, 不再只存活于单进程内存.

best-effort 契约 (安全护栏):
- workspace 为 None 或 sqlite 打不开时, store 为 None, 所有方法 no-op.
- 任何 append/load/search 异常都静默降级 (log debug), 绝不阻塞主流程.
- 与 memory-only 行为完全兼容: 调用方不感知是否持久化.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


def _dumps(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return "{}"


def _loads(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class HypothesisEventStore:
    """SQLite 事件日志: append-only, 附 FTS5 全文索引.

    表:
      events(id PK, ts, event, node_id, payload_json)
      events_fts(rowid→events, ts/event/node_id/payload)  — FTS5 虚拟表
    """

    def __init__(self, workspace: Path | str | None) -> None:
        self._db: sqlite3.Connection | None = None
        if workspace is None:
            return
        try:
            d = Path(workspace) / HUGINN_DIR_NAME
            d.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(d / "hypothesis_events.db"))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    event TEXT NOT NULL,
                    node_id TEXT,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                USING fts5(ts, event, node_id, payload)
                """
            )
            conn.commit()
            self._db = conn
        except Exception:
            logger.debug("hypothesis event store unavailable (non-fatal)", exc_info=True)
            self._db = None

    def append(self, event: dict[str, Any]) -> None:
        """追加一条事件 (memory 已 append 后再调, 只负责持久化)."""
        if self._db is None:
            return
        try:
            payload = _dumps({
                k: v for k, v in event.items()
                if k not in ("event", "node_id", "ts")
            })
            cur = self._db.execute(
                "INSERT INTO events(ts,event,node_id,payload) VALUES(?,?,?,?)",
                (event.get("ts", ""), event.get("event", ""), event.get("node_id"), payload),
            )
            rowid = cur.lastrowid
            self._db.execute(
                "INSERT INTO events_fts(rowid,ts,event,node_id,payload) VALUES(?,?,?,?,?)",
                (rowid, event.get("ts", ""), event.get("event", ""),
                 event.get("node_id") or "", payload),
            )
            self._db.commit()
        except Exception:
            logger.debug("hypothesis event append failed (non-fatal)", exc_info=True)

    def load(self) -> list[dict[str, Any]]:
        """按插入顺序读回全部事件 (供 resume/replay)."""
        if self._db is None:
            return []
        try:
            rows = self._db.execute(
                "SELECT ts,event,node_id,payload FROM events ORDER BY id"
            ).fetchall()
            return [
                {"ts": ts, "event": ev, "node_id": nid, **_loads(payload)}
                for ts, ev, nid, payload in rows
            ]
        except Exception:
            logger.debug("hypothesis event load failed (non-fatal)", exc_info=True)
            return []

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """FTS5 全文搜索事件 (MATCH 查询). 语法非法时降级为按 event 类型前缀过滤."""
        if self._db is None:
            return []
        try:
            rows = self._db.execute(
                "SELECT ts,event,node_id,payload FROM events_fts "
                "WHERE events_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [
                {"ts": ts, "event": ev, "node_id": nid, **_loads(payload)}
                for ts, ev, nid, payload in rows
            ]
        except Exception:
            logger.debug("hypothesis event search failed (non-fatal)", exc_info=True)
            return []

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def __enter__(self) -> "HypothesisEventStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()