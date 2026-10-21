"""认知账 — 用户信念与科学结论的原子记录 + 取代链（spec §7.2 / P1-a）.

设计约束：
- **保留双份**：被取代的旧记录不删除、不覆盖，只追加 ``supersedes`` 链；
- **按 valid-time 检索**：``current()`` 取每个 subject 的最新有效记录；
- **冲突必须可寻址**：``obstruction()`` 返回 ≥2 个 belief_id，纯文本冲突不算；
- 租户隔离：所有查询必须带 ``tenant_id``；
- 认识论记录只读，不写规范账（不做授权）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BeliefScope = Literal["user", "science", "self"]
SourceClass = Literal["direct", "derived", "hypothesis"]


@dataclass(frozen=True)
class BeliefRecord:
    belief_id: str
    tenant_id: str
    subject: str
    claim: str
    subject_scope: BeliefScope = "science"
    owner_id: str = ""
    source_class: SourceClass = "direct"
    epistemic_status: str = "accepted"
    evidential_authority: float = 0.9
    provenance: tuple[str, ...] = ()
    valid_from: str = ""
    valid_to: str = ""
    supersedes: str = ""


@dataclass(frozen=True)
class BeliefObstruction:
    kind: Literal["epistemic"] = "epistemic"
    subject: str = ""
    reason: str = ""
    refs: tuple[str, ...] = ()
    blocked: bool = False


class BeliefLedger:
    """SQLite 认知账；追加式 + 取代链投影。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "belief_ledger.db"
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS belief_records (
                  belief_id          TEXT PRIMARY KEY,
                  tenant_id          TEXT NOT NULL,
                  subject            TEXT NOT NULL,
                  claim              TEXT NOT NULL,
                  subject_scope      TEXT NOT NULL DEFAULT 'science',
                  owner_id           TEXT NOT NULL DEFAULT '',
                  source_class       TEXT NOT NULL DEFAULT 'direct',
                  epistemic_status   TEXT NOT NULL DEFAULT 'accepted',
                  evidential_authority REAL NOT NULL DEFAULT 0.9,
                  provenance_json    TEXT NOT NULL DEFAULT '[]',
                  valid_from         TEXT NOT NULL,
                  valid_to           TEXT NOT NULL DEFAULT '',
                  supersedes         TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_belief_subject
                  ON belief_records(tenant_id, subject, valid_from);
                CREATE TABLE IF NOT EXISTS belief_links (
                  tenant_id   TEXT NOT NULL,
                  from_id     TEXT NOT NULL,
                  to_id       TEXT NOT NULL,
                  relation    TEXT NOT NULL DEFAULT 'supersedes',
                  created_at  TEXT NOT NULL,
                  PRIMARY KEY (tenant_id, from_id, to_id, relation)
                );
                """
            )
            conn.commit()

    def append(self, record: BeliefRecord) -> str:
        """追加一条信念；若带 supersedes，同时写取代链（旧记录保留）."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO belief_records ("
                "belief_id, tenant_id, subject, claim, subject_scope, owner_id, "
                "source_class, epistemic_status, evidential_authority, "
                "provenance_json, valid_from, valid_to, supersedes"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.belief_id,
                    record.tenant_id,
                    record.subject,
                    record.claim,
                    record.subject_scope,
                    record.owner_id,
                    record.source_class,
                    record.epistemic_status,
                    float(record.evidential_authority),
                    json.dumps(list(record.provenance), ensure_ascii=False),
                    record.valid_from,
                    record.valid_to,
                    record.supersedes,
                ),
            )
            if record.supersedes:
                conn.execute(
                    "INSERT OR IGNORE INTO belief_links "
                    "(tenant_id, from_id, to_id, relation, created_at) "
                    "VALUES (?, ?, ?, 'supersedes', ?)",
                    (
                        record.tenant_id,
                        record.supersedes,
                        record.belief_id,
                        record.valid_from,
                    ),
                )
            conn.commit()
        return record.belief_id

    def supersede(
        self, old_id: str, *, claim: str, provenance: tuple[str, ...] = (),
        valid_from: str = "", **overrides: Any
    ) -> BeliefRecord:
        """构造并追加一条取代记录；旧记录与双份都保留."""
        old = self.get(old_id)
        if old is None:
            raise KeyError(f"belief not found: {old_id}")
        record = BeliefRecord(
            belief_id=overrides.pop("belief_id", uuid.uuid4().hex),
            tenant_id=old["tenant_id"],
            subject=old["subject"],
            claim=claim,
            subject_scope=old["subject_scope"],
            owner_id=old["owner_id"],
            source_class=overrides.pop("source_class", old["source_class"]),
            epistemic_status=overrides.pop("epistemic_status", "accepted"),
            evidential_authority=float(
                overrides.pop("evidential_authority", old["evidential_authority"])
            ),
            provenance=tuple(provenance) or tuple(old["provenance"]),
            valid_from=valid_from,
            supersedes=old_id,
            **overrides,
        )
        self.append(record)
        return record

    def get(self, belief_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM belief_records WHERE belief_id = ?"
        params: list[Any] = [belief_id]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(tenant_id)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return _row_to_dict(row) if row is not None else None

    def superseded_by(self, belief_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT to_id FROM belief_links "
                "WHERE from_id = ? AND relation = 'supersedes' "
                "ORDER BY created_at DESC LIMIT 1",
                (belief_id,),
            ).fetchone()
        return str(row["to_id"]) if row else None

    def current(self, tenant_id: str, subject: str) -> dict[str, Any] | None:
        """该 subject 的最新有效记录（按 valid_from 排序，取代链尾部）."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM belief_records WHERE tenant_id = ? AND subject = ? "
                "AND valid_to = '' ORDER BY valid_from ASC",
                (tenant_id, subject),
            ).fetchall()
        if not rows:
            return None
        superseded = {
            row["from_id"]
            for row in self._all_links(tenant_id)
            if row["relation"] == "supersedes"
        }
        active = [r for r in rows if r["belief_id"] not in superseded]
        latest = active[-1] if active else rows[-1]
        return _row_to_dict(latest)

    def obstruction(self, tenant_id: str, subject: str) -> BeliefObstruction | None:
        """认识论 H¹ 代理：同 subject ≥2 条有效且 claim 不同 → 可寻址冲突."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM belief_records WHERE tenant_id = ? AND subject = ? "
                "ORDER BY valid_from ASC",
                (tenant_id, subject),
            ).fetchall()
        if len(rows) < 2:
            return None
        records = [_row_to_dict(row) for row in rows]
        distinct = {r["claim"] for r in records}
        if len(distinct) < 2:
            return None
        refs = tuple(r["belief_id"] for r in records)
        return BeliefObstruction(
            subject=subject,
            reason=(
                f"subject '{subject}' has {len(distinct)} incompatible claims "
                f"across {len(refs)} addressable records"
            ),
            refs=refs,
        )

    def _all_links(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM belief_links WHERE tenant_id = ?", (tenant_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_records(self, tenant_id: str, subject: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM belief_records WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject)
        sql += " ORDER BY valid_from ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]


_ledger: BeliefLedger | None = None


def get_belief_ledger() -> BeliefLedger:
    global _ledger
    if _ledger is None:
        _ledger = BeliefLedger()
    return _ledger


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["provenance"] = tuple(json.loads(item.get("provenance_json") or "[]"))
    except (json.JSONDecodeError, TypeError):
        item["provenance"] = ()
    item.pop("provenance_json", None)
    return item


__all__ = [
    "BeliefLedger",
    "BeliefObstruction",
    "BeliefRecord",
    "BeliefScope",
    "SourceClass",
    "get_belief_ledger",
]
