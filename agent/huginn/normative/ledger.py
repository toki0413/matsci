"""规范账（Normative Ledger）— 追加式事件 + 确定性投影.

本模块只回答一个问题：**此刻哪些动作被显式授权**。它不做任务执行、不调 LLM、
不读聊天全文；所有状态都是 ``normative_events`` 的纯投影。

设计约束（对齐 spec §7.1 / I1–I12）：
- 追加式：事件不可修改；删/撤是新的 ``revoke`` 事件。
- 投影确定性：同一事件序列 + 同一 ``now`` → 同一 ``GroundingView``。
- actor 能力隔离：agent 只能 propose/commit/withdraw，不能自授权。
- 租户隔离：查询必须带 ``tenant_id``，跨租户不可见。
- TTL 在读取时按服务端时间判定，不写过期事件。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ActType = Literal[
    "propose",
    "acknowledge",
    "accept",
    "reject",
    "authorize",
    "commit",
    "revoke",
    "withdraw",
]
Actor = Literal["user", "agent", "system"]

# actor → 允许写入的 act_type（I11：agent 不得自授权/自接受）
_ACTOR_CAPABILITIES: dict[str, frozenset[str]] = {
    "user": frozenset({
        "propose", "acknowledge", "accept", "reject",
        "authorize", "commit", "revoke", "withdraw",
    }),
    "agent": frozenset({"propose", "commit", "withdraw"}),
    "system": frozenset({"authorize", "revoke", "reject"}),
}

_ALLOWED_ACT_TYPES = frozenset(_ACTOR_CAPABILITIES["user"])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str) -> datetime:
    """解析 ISO 8601；无时区按 UTC；非法值抛 ValueError（不静默）。"""
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _resource_covered(granted: str, requested: str) -> bool:
    """资源覆盖：精确相等，或授权以 ``*`` 结尾时按前缀覆盖。"""
    if granted == requested:
        return True
    if granted.endswith("*"):
        return requested.startswith(granted[:-1])
    return False


def scope_covers(granted: dict[str, Any] | None,
                 requested: dict[str, Any] | None) -> bool:
    """结构判定授权 scope 是否覆盖请求 scope（不做语义相似度）.

    scope 形如 ``{"actions": [...], "resources": [...], "limits": {...}}``。
    - actions: 请求集合必须是授权集合子集；授权含 ``*`` 表示全部。
    - resources: 每个请求资源至少被一个授权资源覆盖。
    - limits: 请求的每个限额必须存在且不超过授权限额。
    """
    granted = granted or {}
    requested = requested or {}

    g_actions = set(granted.get("actions") or [])
    r_actions = set(requested.get("actions") or [])
    if "*" not in g_actions and not r_actions <= g_actions:
        return False

    g_resources = list(granted.get("resources") or [])
    for req in requested.get("resources") or []:
        if not any(_resource_covered(str(g), str(req)) for g in g_resources):
            return False

    g_limits = granted.get("limits") or {}
    for key, value in (requested.get("limits") or {}).items():
        if key not in g_limits:
            return False
        try:
            if float(value) > float(g_limits[key]):
                return False
        except (TypeError, ValueError):
            return False
    return True


@dataclass(frozen=True)
class NormativeAct:
    """一条规范行为事件（append-only）。"""

    tenant_id: str
    session_id: str
    actor: Actor
    act_type: ActType
    proposition_id: str
    act_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    proposition_digest: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    ttl_s: int | None = None
    supersedes: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    seq: int | None = None
    deployment_id: str = "local"
    device_id: str = "local"

    def __post_init__(self) -> None:
        if self.act_type not in _ALLOWED_ACT_TYPES:
            raise ValueError(f"未知 act_type: {self.act_type!r}")
        if self.actor not in _ACTOR_CAPABILITIES:
            raise ValueError(f"未知 actor: {self.actor!r}")
        if self.act_type not in _ACTOR_CAPABILITIES[self.actor]:
            raise PermissionError(
                f"actor={self.actor} 不得写 {self.act_type}"
            )
        if not self.proposition_digest:
            object.__setattr__(self, "proposition_digest", _digest(self.proposition_id))
        if self.act_type == "revoke" and not self.supersedes:
            raise ValueError("revoke 必须用 supersedes 指向被撤销的事件")
        if self.act_type == "authorize" and not (self.scope.get("actions") or []):
            raise ValueError("authorize 必须给出 scope.actions")


@dataclass
class GroundingView:
    """某一会话在某一时刻的规范状态投影。"""

    tenant_id: str
    session_id: str
    proposals: list[NormativeAct] = field(default_factory=list)
    accepted: list[NormativeAct] = field(default_factory=list)
    rejected: list[NormativeAct] = field(default_factory=list)
    authorizations: list[NormativeAct] = field(default_factory=list)
    commitments: list[NormativeAct] = field(default_factory=list)
    revoked: list[NormativeAct] = field(default_factory=list)
    expired: list[NormativeAct] = field(default_factory=list)
    events: list[NormativeAct] = field(default_factory=list)

    def active_authorization_for(
        self, scope: dict[str, Any] | None
    ) -> NormativeAct | None:
        """返回覆盖该 scope 的第一条 active 授权；无则 None。"""
        for act in self.authorizations:
            if scope_covers(act.scope, scope):
                return act
        return None

    def covers(self, scope: dict[str, Any] | None) -> bool:
        return self.active_authorization_for(scope) is not None


class NormativeLedger:
    """SQLite 规范账。线程安全（写锁），读走独立连接。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "normative_ledger.db"
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
                CREATE TABLE IF NOT EXISTS normative_events (
                  act_id             TEXT PRIMARY KEY,
                  tenant_id          TEXT NOT NULL,
                  deployment_id      TEXT NOT NULL,
                  device_id          TEXT NOT NULL,
                  session_id         TEXT NOT NULL,
                  seq                INTEGER NOT NULL,
                  actor              TEXT NOT NULL,
                  act_type           TEXT NOT NULL,
                  proposition_id     TEXT NOT NULL,
                  proposition_digest TEXT NOT NULL,
                  scope_json         TEXT NOT NULL DEFAULT '{}',
                  ttl_s              INTEGER,
                  supersedes         TEXT,
                  evidence_json      TEXT NOT NULL DEFAULT '{}',
                  created_at         TEXT NOT NULL,
                  UNIQUE(tenant_id, session_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_norm_prop
                  ON normative_events(tenant_id, proposition_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_norm_scope
                  ON normative_events(tenant_id, act_type, created_at);
                """
            )
            conn.commit()

    def append(self, act: NormativeAct) -> str:
        """追加一条事件并返回 act_id；同 act_id 幂等。"""
        if not act.tenant_id or not act.session_id:
            raise ValueError("tenant_id / session_id 不能为空")
        with self._lock, self._connect() as conn:
            seq = act.seq
            if seq is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM normative_events "
                    "WHERE tenant_id = ? AND session_id = ?",
                    (act.tenant_id, act.session_id),
                ).fetchone()
                seq = int(row["n"])
            try:
                conn.execute(
                    "INSERT INTO normative_events ("
                    "act_id, tenant_id, deployment_id, device_id, session_id, seq, "
                    "actor, act_type, proposition_id, proposition_digest, "
                    "scope_json, ttl_s, supersedes, evidence_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        act.act_id,
                        act.tenant_id,
                        act.deployment_id,
                        act.device_id,
                        act.session_id,
                        seq,
                        act.actor,
                        act.act_type,
                        act.proposition_id,
                        act.proposition_digest,
                        json.dumps(act.scope, ensure_ascii=False),
                        act.ttl_s,
                        act.supersedes,
                        json.dumps(act.evidence, ensure_ascii=False),
                        act.created_at,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    "SELECT act_id FROM normative_events WHERE act_id = ?",
                    (act.act_id,),
                ).fetchone()
                if existing:
                    return act.act_id
                raise
        return act.act_id

    def append_act(
        self,
        *,
        tenant_id: str,
        session_id: str,
        actor: Actor,
        act_type: ActType,
        proposition_id: str,
        scope: dict[str, Any] | None = None,
        ttl_s: int | None = None,
        supersedes: str | None = None,
        evidence: dict[str, Any] | None = None,
        created_at: str | None = None,
        seq: int | None = None,
        deployment_id: str = "local",
        device_id: str = "local",
    ) -> NormativeAct:
        """构造并追加一条事件，返回落库后的 act。"""
        act = NormativeAct(
            tenant_id=tenant_id,
            session_id=session_id,
            actor=actor,
            act_type=act_type,
            proposition_id=proposition_id,
            scope=dict(scope or {}),
            ttl_s=ttl_s,
            supersedes=supersedes,
            evidence=dict(evidence or {}),
            created_at=created_at or _now_iso(),
            seq=seq,
            deployment_id=deployment_id,
            device_id=device_id,
        )
        self.append(act)
        return act

    def _fetch_events(self, tenant_id: str, session_id: str) -> list[NormativeAct]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM normative_events "
                "WHERE tenant_id = ? AND session_id = ? "
                "ORDER BY seq ASC, created_at ASC",
                (tenant_id, session_id),
            ).fetchall()
        return [self._row_to_act(r) for r in rows]

    @staticmethod
    def _row_to_act(row: sqlite3.Row) -> NormativeAct:
        try:
            scope = json.loads(row["scope_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            scope = {}
        try:
            evidence = json.loads(row["evidence_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            evidence = {}
        return NormativeAct(
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            actor=row["actor"],
            act_type=row["act_type"],
            proposition_id=row["proposition_id"],
            act_id=row["act_id"],
            proposition_digest=row["proposition_digest"],
            scope=scope,
            ttl_s=row["ttl_s"],
            supersedes=row["supersedes"],
            evidence=evidence,
            created_at=row["created_at"],
            seq=int(row["seq"]),
            deployment_id=row["deployment_id"],
            device_id=row["device_id"],
        )

    def project(
        self,
        tenant_id: str,
        session_id: str,
        *,
        now: str | None = None,
    ) -> GroundingView:
        """确定性投影：事件序列 + now → GroundingView。"""
        events = self._fetch_events(tenant_id, session_id)
        now_dt = _parse_ts(now) if now else datetime.now(UTC)

        revoked_ids = {
            e.supersedes for e in events
            if e.act_type == "revoke" and e.supersedes
        }
        expired_ids: set[str] = set()
        for e in events:
            if e.ttl_s is not None and e.act_id not in revoked_ids:
                try:
                    deadline = _parse_ts(e.created_at) + timedelta(seconds=e.ttl_s)
                except (TypeError, ValueError):
                    continue
                if deadline <= now_dt:
                    expired_ids.add(e.act_id)

        latest_decision: dict[str, NormativeAct] = {}
        for e in events:
            if e.act_type not in ("accept", "reject"):
                continue
            if e.act_id in revoked_ids or e.act_id in expired_ids:
                continue
            prev = latest_decision.get(e.proposition_id)
            if prev is None or (int(e.seq or 0), e.created_at) > (
                int(prev.seq or 0), prev.created_at
            ):
                latest_decision[e.proposition_id] = e

        decided = set(latest_decision)
        proposals = [
            e for e in events
            if e.act_type == "propose"
            and e.proposition_id not in decided
            and e.act_id not in revoked_ids
        ]
        accepted = [
            e for e in latest_decision.values() if e.act_type == "accept"
        ]
        rejected = [
            e for e in latest_decision.values() if e.act_type == "reject"
        ]
        authorizations = [
            e for e in events
            if e.act_type == "authorize"
            and e.act_id not in revoked_ids
            and e.act_id not in expired_ids
        ]
        commitments = [
            e for e in events
            if e.act_type == "commit"
            and e.act_id not in revoked_ids
            and e.act_id not in expired_ids
        ]
        revoked = [
            e for e in events
            if e.act_type == "revoke" or e.act_id in revoked_ids
        ]
        expired = [e for e in events if e.act_id in expired_ids]

        return GroundingView(
            tenant_id=tenant_id,
            session_id=session_id,
            proposals=proposals,
            accepted=accepted,
            rejected=rejected,
            authorizations=authorizations,
            commitments=commitments,
            revoked=revoked,
            expired=expired,
            events=events,
        )

    def export_events(self, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        """导出原始事件（供回放测试/审计），按 seq 排序。"""
        return [
            {
                "act_id": e.act_id,
                "seq": e.seq,
                "actor": e.actor,
                "act_type": e.act_type,
                "proposition_id": e.proposition_id,
                "scope": e.scope,
                "ttl_s": e.ttl_s,
                "supersedes": e.supersedes,
                "created_at": e.created_at,
            }
            for e in self._fetch_events(tenant_id, session_id)
        ]


__all__ = [
    "ActType",
    "Actor",
    "GroundingView",
    "NormativeAct",
    "NormativeLedger",
    "scope_covers",
]
