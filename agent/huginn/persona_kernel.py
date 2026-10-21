"""Persona Kernel / Params / Revision（spec §10，I5）.

边界：
- ``PersonaKernel`` 不可变：安全约束与授权边界永不被 revision 触碰；
- ``PersonaParams`` 只能由 revision 改，不能直接赋值；
- ``adaptive_layer`` 降级为**只读投影**：由 params + 最近证据渲染，不再前缀累加；
- revision 晋升必须过采纳门：≥N 条跨会话独立证据 + 真实 outcome 改善 +
  OOD 不退步 + 安全/授权回归通过；失败也落盘（rejected），可回滚。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

RevisionStatus = Literal["candidate", "active", "superseded", "rejected"]

_PARAM_FIELDS = frozenset({
    "clarification_threshold",
    "evidence_strictness",
    "risk_tolerance",
    "exploration_rate",
    "verbosity",
    "skepticism",
})
_KERNEL_FIELDS = frozenset({
    "safety_constraints",
    "authorization_boundaries",
    "core_digest",
})


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PersonaKernel:
    persona_id: str
    role: str
    safety_constraints: tuple[str, ...] = ()
    authorization_boundaries: tuple[str, ...] = ()
    core_digest: str = ""

    def __post_init__(self) -> None:
        if not self.core_digest:
            raw = "|".join(
                [self.persona_id, self.role, *self.safety_constraints,
                 *self.authorization_boundaries]
            )
            object.__setattr__(
                self, "core_digest", hashlib.sha256(raw.encode()).hexdigest()[:16]
            )


@dataclass(frozen=True)
class PersonaParams:
    clarification_threshold: float = 0.5
    evidence_strictness: float = 0.7
    risk_tolerance: float = 0.3
    exploration_rate: float = 0.2
    verbosity: float = 0.5
    skepticism: float = 0.5
    tool_bias: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarification_threshold": self.clarification_threshold,
            "evidence_strictness": self.evidence_strictness,
            "risk_tolerance": self.risk_tolerance,
            "exploration_rate": self.exploration_rate,
            "verbosity": self.verbosity,
            "skepticism": self.skepticism,
            "tool_bias": dict(self.tool_bias),
        }


@dataclass(frozen=True)
class PersonaRevision:
    revision_id: str
    persona_id: str
    base_version: int
    delta: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    candidate_since: str = ""
    status: RevisionStatus = "candidate"
    eval_report: dict[str, Any] = field(default_factory=dict)


def evaluate_revision(
    revision: PersonaRevision,
    *,
    cross_session_evidence: int,
    outcome_improved: bool,
    ood_regressed: bool,
    safety_regression: bool,
    n_required: int = 3,
) -> PersonaRevision:
    """采纳门：不满足则保持 candidate；硬失败（安全/OOD/越界）→ rejected."""
    report = {
        "cross_session_evidence": int(cross_session_evidence),
        "n_required": int(n_required),
        "outcome_improved": bool(outcome_improved),
        "ood_regressed": bool(ood_regressed),
        "safety_regression": bool(safety_regression),
    }
    illegal = set(revision.delta) - _PARAM_FIELDS
    if illegal or set(revision.delta) & _KERNEL_FIELDS:
        return replace(
            revision,
            status="rejected",
            eval_report={**report, "reason": f"illegal delta fields: {sorted(illegal)}"},
        )
    if safety_regression or ood_regressed:
        return replace(
            revision,
            status="rejected",
            eval_report={**report, "reason": "hard gate failed"},
        )
    if cross_session_evidence < n_required or not outcome_improved:
        return replace(revision, status="candidate", eval_report=report)
    return replace(revision, status="active", eval_report=report)


class PersonaRevisionStore:
    """追加式 revision 账；active 是状态投影，旧 revision 保留可回滚."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "persona_revisions.db"
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
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persona_revisions (
                  revision_id    TEXT PRIMARY KEY,
                  persona_id     TEXT NOT NULL,
                  base_version   INTEGER NOT NULL,
                  delta_json     TEXT NOT NULL,
                  evidence_json  TEXT NOT NULL,
                  candidate_since TEXT NOT NULL,
                  status         TEXT NOT NULL,
                  eval_json      TEXT NOT NULL,
                  created_at     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_persona_rev
                  ON persona_revisions(persona_id, created_at);
                """
            )
            conn.commit()

    def save(self, revision: PersonaRevision) -> str:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO persona_revisions ("
                "revision_id, persona_id, base_version, delta_json, evidence_json, "
                "candidate_since, status, eval_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision.revision_id,
                    revision.persona_id,
                    int(revision.base_version),
                    json.dumps(revision.delta, ensure_ascii=False),
                    json.dumps(list(revision.evidence_ids), ensure_ascii=False),
                    revision.candidate_since or _now_iso(),
                    revision.status,
                    json.dumps(revision.eval_report, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            conn.commit()
        return revision.revision_id

    def get(self, revision_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persona_revisions WHERE revision_id = ?",
                (revision_id,),
            ).fetchone()
        return _row(row) if row else None

    def active(self, persona_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persona_revisions WHERE persona_id = ? "
                "AND status = 'active' ORDER BY created_at DESC LIMIT 1",
                (persona_id,),
            ).fetchone()
        return _row(row) if row else None

    def activate(self, revision_id: str) -> bool:
        """激活一条 revision；同 persona 旧 active 标 superseded（保留可回滚）."""
        record = self.get(revision_id)
        if record is None:
            return False
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE persona_revisions SET status = 'superseded' "
                "WHERE persona_id = ? AND status = 'active'",
                (record["persona_id"],),
            )
            conn.execute(
                "UPDATE persona_revisions SET status = 'active' WHERE revision_id = ?",
                (revision_id,),
            )
            conn.commit()
        return True

    def rollback(self, persona_id: str, revision_id: str) -> bool:
        return self.activate(revision_id)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("delta_json", "evidence_json", "eval_json"):
        try:
            item[key] = json.loads(item.get(key) or "{}")
        except (json.JSONDecodeError, TypeError):
            item[key] = {}
    return item


# ── adaptive_layer 只读投影 ────────────────────────────────────────────


class PersonaEvidenceStore:
    """PMK/行为证据的追加式投影源；只保留最近 N 条，不做前缀累加."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "persona_evidence.db"
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS persona_evidence ("
                "persona_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def append(self, persona_id: str, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO persona_evidence (persona_id, text, created_at) "
                "VALUES (?, ?, ?)",
                (persona_id, text[:240], _now_iso()),
            )
            conn.commit()

    def recent(self, persona_id: str, limit: int = 5) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM persona_evidence WHERE persona_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (persona_id, max(1, int(limit))),
            ).fetchall()
        return [str(row["text"]) for row in rows]


_evidence_store: PersonaEvidenceStore | None = None


def get_persona_evidence_store() -> PersonaEvidenceStore:
    global _evidence_store
    if _evidence_store is None:
        _evidence_store = PersonaEvidenceStore()
    return _evidence_store


def render_adaptive_layer(
    params: PersonaParams,
    evidence: list[str],
    *,
    limit: int = 5,
) -> str:
    """只读投影：params + 最近证据；同输入同输出，不累积历史前缀."""
    lines = [
        "[persona params] "
        + ", ".join(
            f"{key}={value:.2f}" for key, value in params.to_dict().items()
            if isinstance(value, (int, float))
        )
    ]
    for text in evidence[-max(1, int(limit)):]:
        lines.append(f"- {text[:200]}")
    return "\n".join(lines)


__all__ = [
    "PersonaEvidenceStore",
    "PersonaKernel",
    "PersonaParams",
    "PersonaRevision",
    "PersonaRevisionStore",
    "RevisionStatus",
    "evaluate_revision",
    "get_persona_evidence_store",
    "render_adaptive_layer",
]
