"""PRT Level 0 — 异常登记表 AnomalyLog.

主动回顾 (Proactive Retrospective) 的最底层: 把 agent 跑工具时碰到的
异常信号登记进 SQLite, 后面 Level 1+ 再来消费. 这里只管登记, 不做回顾,
也不打断主流程.

表结构和 Anomaly 字段一一对应; context_snapshot / unresolved_dimensions /
linked_hypotheses 三个字段用 JSON 文本存.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """一条异常记录.

    id 形如 "ANOM-001", 由 store 在 log() 时自动分配, 调用方传进来的会被覆盖.
    linked_hypotheses 留给 Level 2 用, 现在恒为空列表.
    """

    id: str
    ts: datetime
    category: str  # INPUT_DATA / DATA_CONFLICT / TOOL_FAILURE / COMPUTATION_RESULT
    severity: str  # HIGH / MEDIUM / LOW
    description: str
    detection_method: str  # standard_value_compare / tool_call_failed / validate_tool_error / keyword_match
    source: str  # user_input / tool_output / multi_source / environment
    context_snapshot: dict
    unresolved_dimensions: list[str]
    attribution: str | None = None
    attribution_confidence: float = 0.0
    linked_hypotheses: list[str] = field(default_factory=list)
    resolution: str | None = None
    resolved: bool = False
    resolved_by: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS anomalies (
    id                     TEXT PRIMARY KEY,
    ts                     TEXT NOT NULL,
    category               TEXT NOT NULL,
    severity               TEXT NOT NULL,
    description            TEXT NOT NULL,
    detection_method       TEXT NOT NULL,
    source                 TEXT NOT NULL,
    context_snapshot       TEXT NOT NULL,
    unresolved_dimensions  TEXT NOT NULL,
    attribution            TEXT,
    attribution_confidence REAL NOT NULL,
    linked_hypotheses      TEXT NOT NULL,
    resolution             TEXT,
    resolved               INTEGER NOT NULL,
    resolved_by            TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomalies_resolved ON anomalies(resolved);
CREATE INDEX IF NOT EXISTS idx_anomalies_category  ON anomalies(category);
"""


class AnomalyLogStore:
    """SQLite 后端的异常登记表.

    线程安全靠一把全局锁兜底 —— agent 主体是单线程异步, 但 checkpointer
    之类的组件可能在别的线程里跑, 加锁省心. SQLite 连接本身用 check_same_thread=False.
    """

    def __init__(self, db_path: str = "anomalies.db") -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: 允许跨线程用同一个连接, 锁我们自己来管
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _next_id(self) -> str:
        """取下一个自增 id, 格式 ANOM-001. 没记录就从 1 开始."""
        row = self._conn.execute(
            "SELECT id FROM anomalies ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return "ANOM-001"
        last = row["id"]
        try:
            n = int(last.split("-", 1)[1]) + 1
        except (IndexError, ValueError):
            # id 被人改坏过, 退回按行数兜底
            count = self._conn.execute("SELECT COUNT(*) AS c FROM anomalies").fetchone()["c"]
            n = count + 1
        return f"ANOM-{n:03d}"

    def log(self, anomaly: Anomaly) -> str:
        """写入一条异常, 返回分配到的 id.

        anomaly.id 会被 store 自动覆盖, 保证唯一且单调递增.
        """
        with self._lock:
            anomaly.id = self._next_id()
            anomaly.ts = anomaly.ts or datetime.now()
            self._conn.execute(
                """
                INSERT INTO anomalies (
                    id, ts, category, severity, description,
                    detection_method, source, context_snapshot,
                    unresolved_dimensions, attribution, attribution_confidence,
                    linked_hypotheses, resolution, resolved, resolved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anomaly.id,
                    anomaly.ts.isoformat(),
                    anomaly.category,
                    anomaly.severity,
                    anomaly.description,
                    anomaly.detection_method,
                    anomaly.source,
                    json.dumps(anomaly.context_snapshot, ensure_ascii=False, default=str),
                    json.dumps(anomaly.unresolved_dimensions, ensure_ascii=False),
                    anomaly.attribution,
                    float(anomaly.attribution_confidence),
                    json.dumps(anomaly.linked_hypotheses, ensure_ascii=False),
                    anomaly.resolution,
                    1 if anomaly.resolved else 0,
                    anomaly.resolved_by,
                ),
            )
            self._conn.commit()
            logger.info(
                "anomaly logged: %s [%s/%s] %s",
                anomaly.id, anomaly.category, anomaly.severity, anomaly.description,
            )
            return anomaly.id

    def list_unresolved(self, category: str | None = None) -> list[Anomaly]:
        """查未解决的异常, 可按 category 过滤."""
        with self._lock:
            if category is None:
                rows = self._conn.execute(
                    "SELECT * FROM anomalies WHERE resolved = 0 ORDER BY ts ASC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM anomalies WHERE resolved = 0 AND category = ? ORDER BY ts ASC",
                    (category,),
                ).fetchall()
        return [self._row_to_anomaly(r) for r in rows]

    def resolve(
        self, anomaly_id: str, resolution: str, resolved_by: str
    ) -> bool:
        """标记某条异常已解决. 返回是否真的更新到了(找不到/已解决都算 False)."""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE anomalies
                   SET resolved = 1, resolution = ?, resolved_by = ?
                 WHERE id = ? AND resolved = 0
                """,
                (resolution, resolved_by, anomaly_id),
            )
            self._conn.commit()
            updated = cur.rowcount > 0
        if updated:
            logger.info("anomaly %s resolved by %s: %s", anomaly_id, resolved_by, resolution)
        return updated

    def get(self, anomaly_id: str) -> Anomaly | None:
        """按 id 取单条."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM anomalies WHERE id = ?", (anomaly_id,)
            ).fetchone()
        return self._row_to_anomaly(row) if row is not None else None

    def to_dict(self, anomaly: Anomaly) -> dict:
        """序列化成可 JSON 化的 dict, 给上层 / API 用."""
        return {
            "id": anomaly.id,
            "ts": anomaly.ts.isoformat() if anomaly.ts else None,
            "category": anomaly.category,
            "severity": anomaly.severity,
            "description": anomaly.description,
            "detection_method": anomaly.detection_method,
            "source": anomaly.source,
            "context_snapshot": anomaly.context_snapshot,
            "unresolved_dimensions": anomaly.unresolved_dimensions,
            "attribution": anomaly.attribution,
            "attribution_confidence": anomaly.attribution_confidence,
            "linked_hypotheses": anomaly.linked_hypotheses,
            "resolution": anomaly.resolution,
            "resolved": anomaly.resolved,
            "resolved_by": anomaly.resolved_by,
        }

    def _row_to_anomaly(self, row: sqlite3.Row) -> Anomaly:
        return Anomaly(
            id=row["id"],
            ts=datetime.fromisoformat(row["ts"]),
            category=row["category"],
            severity=row["severity"],
            description=row["description"],
            detection_method=row["detection_method"],
            source=row["source"],
            context_snapshot=json.loads(row["context_snapshot"]),
            unresolved_dimensions=json.loads(row["unresolved_dimensions"]),
            attribution=row["attribution"],
            attribution_confidence=float(row["attribution_confidence"]),
            linked_hypotheses=json.loads(row["linked_hypotheses"]),
            resolution=row["resolution"],
            resolved=bool(row["resolved"]),
            resolved_by=row["resolved_by"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
