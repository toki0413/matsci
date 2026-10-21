"""漂移事件账 — VIGIL / Smith / 世界模型 prediction_error 共用的审计表."""
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

DriftLevel = Literal["review", "intervene", "pause"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class DriftEvent:
    tenant_id: str
    session_id: str
    detector: str
    baseline_id: str
    observed: float
    threshold: float
    level: DriftLevel
    evidence: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_now_iso)


class DriftLedger:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "drift_events.db"
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
                CREATE TABLE IF NOT EXISTS drift_events (
                  event_id      TEXT PRIMARY KEY,
                  tenant_id     TEXT NOT NULL,
                  session_id    TEXT NOT NULL,
                  detector      TEXT NOT NULL,
                  baseline_id   TEXT NOT NULL,
                  observed      REAL NOT NULL,
                  threshold     REAL NOT NULL,
                  level         TEXT NOT NULL,
                  evidence_json TEXT NOT NULL DEFAULT '{}',
                  created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_drift
                  ON drift_events(tenant_id, session_id, created_at);
                """
            )
            conn.commit()

    def record(self, event: DriftEvent) -> str:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO drift_events ("
                "event_id, tenant_id, session_id, detector, baseline_id, "
                "observed, threshold, level, evidence_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.tenant_id,
                    event.session_id,
                    event.detector,
                    event.baseline_id,
                    float(event.observed),
                    float(event.threshold),
                    event.level,
                    json.dumps(event.evidence, ensure_ascii=False),
                    event.created_at,
                ),
            )
            conn.commit()
        return event.event_id

    def list_events(
        self,
        *,
        tenant_id: str,
        session_id: str | None = None,
        level: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM drift_events WHERE tenant_id = ? "
        params: list[Any] = [tenant_id]
        if session_id is not None:
            sql += "AND session_id = ? "
            params.append(session_id)
        if level is not None:
            sql += "AND level = ? "
            params.append(level)
        sql += "ORDER BY created_at ASC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["evidence_json"] = json.loads(item.get("evidence_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                item["evidence_json"] = {}
            out.append(item)
        return out


_ledger: DriftLedger | None = None


def get_drift_ledger() -> DriftLedger:
    global _ledger
    if _ledger is None:
        _ledger = DriftLedger()
    return _ledger


def record_prediction_error(
    record: Any,
    *,
    tenant_id: str,
    session_id: str,
    review_threshold: float = 0.05,
    intervene_threshold: float = 0.15,
    ledger: DriftLedger | None = None,
) -> str | None:
    """把世界模型 reconcile 误差落成漂移事件；ok/no_signal 不写."""
    from huginn.research.world_model_binding import prediction_error_signal

    signal = prediction_error_signal(
        record,
        review_threshold=review_threshold,
        intervene_threshold=intervene_threshold,
    )
    level = signal.get("level")
    if level not in ("review", "intervene"):
        return None
    observed = signal.get("observed")
    if observed is None:
        return None
    threshold = (
        intervene_threshold if level == "intervene" else review_threshold
    )
    event = DriftEvent(
        tenant_id=tenant_id,
        session_id=session_id,
        detector="prediction_error",
        baseline_id=str(getattr(record, "prediction_id", "unknown")),
        observed=float(observed),
        threshold=float(threshold),
        level=level,
        evidence={
            "actual_source": getattr(record, "actual_source", ""),
            "errors": getattr(record, "errors", {}),
            "status": getattr(record, "status", ""),
        },
    )
    return (ledger or get_drift_ledger()).record(event)


def record_prediction_error_value(
    error: float | None,
    *,
    tenant_id: str,
    session_id: str,
    prediction_id: str = "",
    review_threshold: float = 0.05,
    intervene_threshold: float = 0.15,
    ledger: DriftLedger | None = None,
) -> str | None:
    """记录裸 prediction_error（ForwardPredictor 路径）；ok/no_signal 不写."""
    if error is None:
        return None
    value = float(error)
    if value > intervene_threshold:
        level, threshold = "intervene", intervene_threshold
    elif value > review_threshold:
        level, threshold = "review", review_threshold
    else:
        return None
    event = DriftEvent(
        tenant_id=tenant_id,
        session_id=session_id,
        detector="prediction_error",
        baseline_id=prediction_id or "forward_predictor",
        observed=value,
        threshold=float(threshold),
        level=level,
        evidence={"source": "forward_predictor"},
    )
    return (ledger or get_drift_ledger()).record(event)


__all__ = [
    "DriftEvent",
    "DriftLedger",
    "DriftLevel",
    "get_drift_ledger",
    "record_prediction_error",
    "record_prediction_error_value",
]
