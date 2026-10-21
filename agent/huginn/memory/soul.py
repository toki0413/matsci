"""Soul 层级 — 价值排序，不是授权排序（spec §18.5，I18/I22）.

四层全序：``TenantSoul (4) ≻ OrgUnitSoul (3) ≻ AgentSoul (2) ≻ UserSoul (1)``。

适用：梦境种子选择、显著性、边界厚度、价值冲突裁决。
**不适用**：事实真假（认识论）、动作授权（规范账）。高优先级只能约束低优先级，
不能替它写 ``authorize``；每次覆盖写 ``soul_resolution_event``，可审计。
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SoulScope = Literal["tenant", "org_unit", "agent", "user"]

_SOUL_RANK: dict[str, int] = {
    "tenant": 4,
    "org_unit": 3,
    "agent": 2,
    "user": 1,
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SoulContext:
    """一次决策可见的四层 Soul 视图；按租户隔离."""

    tenant_id: str
    org_unit_id: str = ""
    agent_id: str = ""
    user_id: str = ""
    values: dict[str, float] = field(default_factory=dict)

    def priority(self, scope: SoulScope) -> int:
        return _SOUL_RANK[scope]


def soul_priority(scope: SoulScope) -> int:
    return _SOUL_RANK[str(scope)]


def total_order() -> dict[str, int]:
    """全序（高 → 低）；测试用于断言无循环、无并列."""
    return dict(_SOUL_RANK)


@dataclass(frozen=True)
class SoulResolution:
    event_id: str
    dimension: str
    higher_scope: SoulScope
    lower_scope: SoulScope
    decision: str
    reason: str
    created_at: str


def resolve(
    *,
    dimension: str,
    higher: SoulScope,
    lower: SoulScope,
    decision: str,
    reason: str = "",
) -> SoulResolution:
    """裁决同一维度的价值冲突：高优先级胜出，低优先级只能增加约束.

    只写裁决事件，不写规范账（``authorize`` 仍由 NormativeLedger 独立判定）。
    """
    if soul_priority(higher) <= soul_priority(lower):
        raise ValueError(
            f"{higher} 不得压过 {lower}（Soul 优先级必须严格更高）"
        )
    return SoulResolution(
        event_id=uuid.uuid4().hex,
        dimension=str(dimension),
        higher_scope=higher,
        lower_scope=lower,
        decision=str(decision),
        reason=str(reason),
        created_at=_now_iso(),
    )


class SoulResolutionLedger:
    """``soul_resolution_events`` 追加账；低层覆盖高层是违规，必须可审计."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "soul_resolution.db"
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
                CREATE TABLE IF NOT EXISTS soul_resolution_events (
                  event_id     TEXT PRIMARY KEY,
                  tenant_id    TEXT NOT NULL,
                  dimension    TEXT NOT NULL,
                  higher_scope TEXT NOT NULL,
                  lower_scope  TEXT NOT NULL,
                  decision     TEXT NOT NULL,
                  reason       TEXT NOT NULL,
                  created_at   TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def record(self, tenant_id: str, resolution: SoulResolution) -> str:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO soul_resolution_events ("
                "event_id, tenant_id, dimension, higher_scope, lower_scope, "
                "decision, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resolution.event_id,
                    tenant_id,
                    resolution.dimension,
                    resolution.higher_scope,
                    resolution.lower_scope,
                    resolution.decision,
                    resolution.reason,
                    resolution.created_at,
                ),
            )
            conn.commit()
        return resolution.event_id

    def list_events(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM soul_resolution_events WHERE tenant_id = ? "
                "ORDER BY created_at ASC",
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def filter_dream_seeds(
    seeds: list[dict[str, Any]],
    *,
    tenant_id: str,
    user_id: str = "",
) -> list[dict[str, Any]]:
    """梦境种子池：跨租户零共享；同一租户内按 Soul 优先级稳定排序.

    seed 形如 ``{"seed_id": ..., "tenant_id": ..., "soul_priority": "user"}``。
    缺失 tenant 的种子一律丢弃（宁缺勿猜）。
    """
    allowed: list[tuple[int, dict[str, Any]]] = []
    for seed in seeds:
        if str(seed.get("tenant_id") or "") != str(tenant_id):
            continue
        scope = str(seed.get("soul_priority") or "agent")
        if scope not in _SOUL_RANK:
            scope = "agent"
        if user_id and seed.get("user_id") and str(seed["user_id"]) != str(user_id):
            continue
        allowed.append((_SOUL_RANK[scope], seed))
    allowed.sort(key=lambda pair: pair[0], reverse=True)
    return [seed for _, seed in allowed]


def dream_authority(scope: SoulScope = "agent") -> dict[str, float]:
    """Soul 只调制显著性/抽象权威，不提升证据权威（不可授权动作）."""
    return {
        "salience": min(1.0, 0.4 + 0.15 * soul_priority(scope)),
        "abstraction_authority": 0.7,
        "evidential_authority": 0.4,
        "influence_authority": 0.6,
    }


__all__ = [
    "SoulContext",
    "SoulResolution",
    "SoulResolutionLedger",
    "SoulScope",
    "dream_authority",
    "filter_dream_seeds",
    "resolve",
    "soul_priority",
    "total_order",
]
