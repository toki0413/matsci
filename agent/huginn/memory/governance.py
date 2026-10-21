"""记忆状态事件账 — 每次保留/影响/巩固/遗忘迁移都可审计.

对齐 spec §18.3：`ingest / index / consolidate / recall / revise / forget /
retrain / promote / quarantine` 等迁移必须留下 `policy_id + policy_version +
decision + route + authority + evidence`。

本模块只追加，不修改；查询按 (tenant_id, memory_id, created_at) 索引。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

MemoryEventType = Literal[
    "ingest",
    "index",
    "consolidate",
    "recall",
    "revise",
    "forget",
    "retrain",
    "promote",
    "quarantine",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class MemoryStateEvent:
    tenant_id: str
    memory_id: str
    event_type: MemoryEventType
    policy_id: str
    policy_version: str
    decision: str
    route: str = ""
    authority: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    actor: str = "system"
    soul_scope: str = "agent"
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_now_iso)


class MemoryStateLedger:
    """SQLite 追加式记忆治理账。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "memory_state_ledger.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_state_events (
                  event_id        TEXT PRIMARY KEY,
                  tenant_id       TEXT NOT NULL,
                  memory_id       TEXT NOT NULL,
                  event_type      TEXT NOT NULL,
                  policy_id       TEXT NOT NULL,
                  policy_version  TEXT NOT NULL,
                  decision        TEXT NOT NULL,
                  route           TEXT NOT NULL DEFAULT '',
                  authority_json  TEXT NOT NULL DEFAULT '{}',
                  evidence_json   TEXT NOT NULL DEFAULT '{}',
                  actor           TEXT NOT NULL,
                  soul_scope      TEXT NOT NULL,
                  created_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mem_state
                  ON memory_state_events(tenant_id, memory_id, created_at);
                """
            )
            conn.commit()

    def record(self, event: MemoryStateEvent) -> str:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO memory_state_events ("
                "event_id, tenant_id, memory_id, event_type, policy_id, "
                "policy_version, decision, route, authority_json, evidence_json, "
                "actor, soul_scope, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.tenant_id,
                    event.memory_id,
                    event.event_type,
                    event.policy_id,
                    event.policy_version,
                    event.decision,
                    event.route,
                    json.dumps(event.authority, ensure_ascii=False),
                    json.dumps(event.evidence, ensure_ascii=False),
                    event.actor,
                    event.soul_scope,
                    event.created_at,
                ),
            )
            conn.commit()
        return event.event_id

    def record_influence(
        self,
        *,
        tenant_id: str,
        memory_id: str,
        decision: Any,
        policy_version: str = "v1",
        evidence: dict[str, Any] | None = None,
    ) -> str:
        """记录一条影响闸判定（route / influence / authority_class）."""
        return self.record(
            MemoryStateEvent(
                tenant_id=tenant_id,
                memory_id=memory_id,
                event_type="recall",
                policy_id="memory_influence",
                policy_version=policy_version,
                decision=str(getattr(decision, "route", "unknown")),
                route=str(getattr(decision, "route", "")),
                authority={
                    "influence": float(getattr(decision, "influence", 0.0)),
                    "authority_class": getattr(decision, "authority_class", ""),
                },
                evidence=evidence or {},
            )
        )

    def record_retention(
        self,
        *,
        tenant_id: str,
        memory_id: str,
        decision: Any,
        policy_version: str = "v1",
        evidence: dict[str, Any] | None = None,
    ) -> str:
        """记录一条保留闸判定（keep/archive/consolidate/prune）."""
        return self.record(
            MemoryStateEvent(
                tenant_id=tenant_id,
                memory_id=memory_id,
                event_type="forget",
                policy_id="memory_retention",
                policy_version=policy_version,
                decision=str(getattr(decision, "action", "unknown")),
                evidence=evidence or {},
            )
        )

    def list_events(
        self,
        *,
        tenant_id: str,
        memory_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM memory_state_events WHERE tenant_id = ? "
        )
        params: list[Any] = [tenant_id]
        if memory_id is not None:
            sql += "AND memory_id = ? "
            params.append(memory_id)
        sql += "ORDER BY created_at ASC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("authority_json", "evidence_json"):
                try:
                    item[key] = json.loads(item.get(key) or "{}")
                except (json.JSONDecodeError, TypeError):
                    item[key] = {}
            out.append(item)
        return out


_ledger: MemoryStateLedger | None = None


def get_memory_state_ledger() -> MemoryStateLedger:
    global _ledger
    if _ledger is None:
        _ledger = MemoryStateLedger()
    return _ledger


__all__ = [
    "MemoryEventType",
    "MemoryStateEvent",
    "MemoryStateLedger",
    "get_memory_state_ledger",
]
