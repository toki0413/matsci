"""梦境产物存储 — DreamObject 与修复提案的追加式落库.

只持久化候选产物；修复是否执行由调用方显式 `mark_repair_applied`，不在梦境周期里
自动删除或改写 direct 证据。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huginn.memory.dream import DreamObject, RepairAction

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DreamStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "dream_store.db"
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
                CREATE TABLE IF NOT EXISTS dream_objects (
                  dream_id      TEXT PRIMARY KEY,
                  persona_id    TEXT NOT NULL,
                  tenant_id     TEXT NOT NULL DEFAULT 'default',
                  deployment_id TEXT NOT NULL DEFAULT 'local',
                  status        TEXT NOT NULL,
                  payload_json  TEXT NOT NULL,
                  created_at    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dream_repairs (
                  repair_id     TEXT PRIMARY KEY,
                  dream_id      TEXT NOT NULL,
                  action        TEXT NOT NULL,
                  target_json   TEXT NOT NULL,
                  risk          TEXT NOT NULL,
                  reversible    INTEGER NOT NULL,
                  evidence_json TEXT NOT NULL DEFAULT '{}',
                  applied       INTEGER NOT NULL DEFAULT 0,
                  created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dream_persona
                  ON dream_objects(persona_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_dream_tenant
                  ON dream_objects(tenant_id, created_at);
                """
            )
            # 旧库迁移：补租户/部署列（幂等）.
            for column, ddl in (
                ("tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("deployment_id", "TEXT NOT NULL DEFAULT 'local'"),
            ):
                existing = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(dream_objects)")
                }
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE dream_objects ADD COLUMN {column} {ddl}"
                    )
            conn.commit()

    def save(
        self,
        dream: DreamObject,
        repairs: list[RepairAction] | None = None,
    ) -> str:
        payload = asdict(dream)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dream_objects "
                "(dream_id, persona_id, tenant_id, deployment_id, status, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dream.dream_id,
                    dream.persona_id,
                    dream.tenant_id,
                    dream.deployment_id,
                    dream.status,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    _now_iso(),
                ),
            )
            for index, repair in enumerate(repairs or []):
                repair_id = f"{dream.dream_id}:{index}"
                conn.execute(
                    "INSERT OR IGNORE INTO dream_repairs "
                    "(repair_id, dream_id, action, target_json, risk, reversible, "
                    "evidence_json, applied, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        repair_id,
                        dream.dream_id,
                        repair.action,
                        json.dumps(list(repair.target_ids), ensure_ascii=False),
                        repair.risk,
                        1 if repair.reversible else 0,
                        json.dumps(repair.evidence, ensure_ascii=False, default=str),
                        _now_iso(),
                    ),
                )
            conn.commit()
        return dream.dream_id

    def get(self, dream_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dream_objects WHERE dream_id = ?", (dream_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def list_dreams(
        self,
        *,
        persona_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM dream_objects "
        clauses: list[str] = []
        params: list[Any] = []
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            out.append(item)
        return out

    def list_repairs(self, dream_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dream_repairs WHERE dream_id = ? ORDER BY created_at ASC",
                (dream_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["target_json"] = json.loads(item.get("target_json") or "[]")
            item["evidence_json"] = json.loads(item.get("evidence_json") or "{}")
            out.append(item)
        return out

    def mark_repair_applied(self, repair_id: str) -> bool:
        return self.set_repair_result(repair_id, applied=True)

    def get_repair(self, repair_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dream_repairs WHERE repair_id = ?", (repair_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["target_json"] = json.loads(item.get("target_json") or "[]")
        item["evidence_json"] = json.loads(item.get("evidence_json") or "{}")
        return item

    def set_repair_result(
        self,
        repair_id: str,
        *,
        applied: bool,
        evidence_patch: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT evidence_json FROM dream_repairs WHERE repair_id = ?",
                (repair_id,),
            ).fetchone()
            if row is None:
                return False
            try:
                evidence = json.loads(row["evidence_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                evidence = {}
            if evidence_patch:
                evidence.update(evidence_patch)
            cur = conn.execute(
                "UPDATE dream_repairs SET applied = ?, evidence_json = ? "
                "WHERE repair_id = ?",
                (
                    1 if applied else 0,
                    json.dumps(evidence, ensure_ascii=False, default=str),
                    repair_id,
                ),
            )
            conn.commit()
            return cur.rowcount > 0


_store: DreamStore | None = None


def get_dream_store() -> DreamStore:
    global _store
    if _store is None:
        _store = DreamStore()
    return _store


__all__ = ["DreamStore", "get_dream_store"]

