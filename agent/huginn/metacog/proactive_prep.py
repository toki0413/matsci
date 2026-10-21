"""空闲期主动准备 — PreparedContext 只是候选，不是已授权上下文（spec §19.2）.

I26: 预取产物必须带 TTL，过期即不可用；命中仍要过影响闸。
I27: 空闲环不得产生用户可见动作；任何动作仍需规范账授权。
I34: 空闲任务按租户隔离且有预算上限；超预算即停。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

PrepStatus = Literal["candidate", "expired", "consumed"]
JobType = Literal["demand_probe", "structure_probe", "surprise_probe", "prepare_context"]
JobStatus = Literal["queued", "running", "done", "failed", "expired"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass(frozen=True)
class PreparedContext:
    prep_id: str
    tenant_id: str
    persona_id: str
    predicted_need: str
    query_hint: str
    evidence_ids: tuple[str, ...] = ()
    authority_vector: dict[str, float] = field(default_factory=dict)
    ttl_s: int = 900
    created_at: str = field(default_factory=_now_iso)
    status: PrepStatus = "candidate"


class PreparedContextStore:
    """按租户隔离的预取缓存；只存候选，不自动注入."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "prepared_contexts.db"
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
                CREATE TABLE IF NOT EXISTS prepared_contexts (
                  prep_id        TEXT PRIMARY KEY,
                  tenant_id      TEXT NOT NULL,
                  persona_id     TEXT NOT NULL,
                  predicted_need TEXT NOT NULL,
                  query_hint     TEXT NOT NULL,
                  evidence_json  TEXT NOT NULL,
                  authority_json TEXT NOT NULL,
                  ttl_s          INTEGER NOT NULL,
                  status         TEXT NOT NULL,
                  created_at     TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def save(self, prep: PreparedContext) -> str:
        import json

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO prepared_contexts ("
                "prep_id, tenant_id, persona_id, predicted_need, query_hint, "
                "evidence_json, authority_json, ttl_s, status, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prep.prep_id,
                    prep.tenant_id,
                    prep.persona_id,
                    prep.predicted_need,
                    prep.query_hint,
                    json.dumps(list(prep.evidence_ids), ensure_ascii=False),
                    json.dumps(prep.authority_vector, ensure_ascii=False),
                    int(prep.ttl_s),
                    prep.status,
                    prep.created_at,
                ),
            )
            conn.commit()
        return prep.prep_id

    def get(self, prep_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prepared_contexts WHERE prep_id = ? AND tenant_id = ?",
                (prep_id, tenant_id),
            ).fetchone()
        return _prep_row(row) if row else None

    def candidates(self, tenant_id: str, *, now: str | None = None) -> list[dict[str, Any]]:
        """只返回未过期候选；过期即标 expired，不得作为当前共同基础."""
        now_dt = _parse_ts(now) if now else datetime.now(UTC)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prepared_contexts WHERE tenant_id = ? "
                "AND status = 'candidate'",
                (tenant_id,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                item = _prep_row(row)
                created = _parse_ts(item["created_at"])
                if created + timedelta(seconds=int(item["ttl_s"])) <= now_dt:
                    conn.execute(
                        "UPDATE prepared_contexts SET status = 'expired' "
                        "WHERE prep_id = ?",
                        (item["prep_id"],),
                    )
                    item["status"] = "expired"
                    continue
                out.append(item)
            conn.commit()
        return out

    def consume(self, prep_id: str, *, tenant_id: str, now: str | None = None) -> bool:
        """消费候选；过期/跨租户一律失败（不自动注入）."""
        item = self.get(prep_id, tenant_id=tenant_id)
        if item is None or item["status"] != "candidate":
            return False
        now_dt = _parse_ts(now) if now else datetime.now(UTC)
        created = _parse_ts(item["created_at"])
        if created + timedelta(seconds=int(item["ttl_s"])) <= now_dt:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE prepared_contexts SET status = 'expired' WHERE prep_id = ?",
                    (prep_id,),
                )
                conn.commit()
            return False
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE prepared_contexts SET status = 'consumed' WHERE prep_id = ?",
                (prep_id,),
            )
            conn.commit()
        return True


def _prep_row(row: sqlite3.Row) -> dict[str, Any]:
    import json

    item = dict(row)
    for key in ("evidence_json", "authority_json"):
        try:
            item[key] = json.loads(item.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            item[key] = [] if key == "evidence_json" else {}
    return item


@dataclass(frozen=True)
class IdleJob:
    job_id: str
    tenant_id: str
    persona_id: str
    job_type: JobType
    budget_ms: int
    sleep_state: Literal["active", "sleep"] = "sleep"
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now_iso)
    finished_at: str = ""


class IdleJobQueue:
    """按租户隔离、有预算上限的空闲任务队列；超预算即 DROPPED."""

    def __init__(self, *, per_tenant_max: int = 100) -> None:
        self._lock = threading.RLock()
        self._queues: dict[str, list[IdleJob]] = {}
        self._spent_ms: dict[str, int] = {}
        self._reserved_ms: dict[str, int] = {}
        self.per_tenant_max = max(1, int(per_tenant_max))
        self.visible_actions = 0

    def submit(self, job: IdleJob, *, budget_cap_ms: int = 60_000) -> str:
        with self._lock:
            if job.sleep_state != "sleep":
                return "DROPPED"  # 活跃状态不跑巩固/预取
            queue = self._queues.setdefault(job.tenant_id, [])
            if len(queue) >= self.per_tenant_max:
                return "DROPPED"
            committed = self._spent_ms.get(job.tenant_id, 0) + self._reserved_ms.get(
                job.tenant_id, 0
            )
            if committed + int(job.budget_ms) > int(budget_cap_ms):
                return "DROPPED"
            queue.append(job)
            self._reserved_ms[job.tenant_id] = (
                self._reserved_ms.get(job.tenant_id, 0) + int(job.budget_ms)
            )
            return job.job_id

    def run_due(
        self,
        tenant_id: str,
        executor: Any,
        *,
        budget_cap_ms: int = 60_000,
    ) -> dict[str, int]:
        """执行该租户队列；executor 只做预取/预计算，返回候选（不产生可见动作）."""
        counts = {"done": 0, "failed": 0, "dropped": 0}
        with self._lock:
            queue = self._queues.get(tenant_id, [])
        while True:
            with self._lock:
                if not queue:
                    break
                job = queue.pop(0)
                self._reserved_ms[tenant_id] = max(
                    0, self._reserved_ms.get(tenant_id, 0) - int(job.budget_ms)
                )
                spent = self._spent_ms.get(tenant_id, 0)
                if spent + int(job.budget_ms) > int(budget_cap_ms):
                    counts["dropped"] += 1
                    continue
                self._spent_ms[tenant_id] = spent + int(job.budget_ms)
            try:
                executor(job)
                counts["done"] += 1
            except Exception:  # 防御: 单个 idle job 失败不影响其他租户
                counts["failed"] += 1
        return counts

    def spent_ms(self, tenant_id: str) -> int:
        with self._lock:
            return int(self._spent_ms.get(tenant_id, 0))


__all__ = [
    "IdleJob",
    "IdleJobQueue",
    "JobStatus",
    "JobType",
    "PrepStatus",
    "PreparedContext",
    "PreparedContextStore",
]
