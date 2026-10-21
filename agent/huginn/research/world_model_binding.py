"""世界模型 binding — 预测升格链、worldview 隔离、prediction_error 漂移信号.

对齐 spec §20：
- prediction 默认 hypothesis/derived，未经 reconcile 不得作为 direct 证据；
- actual 必须来自真实执行/可重放经验并带 source，禁止合成；
- 跨 worldview/domain/contract_version 不得粘合；
- 世界模型不得授权动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from huginn.research.law_model import LawState, reconcile

ReconcileStatus = Literal["borne_out", "falsified", "insufficient"]
PromotionStatus = Literal["hypothesis", "reconciled", "replicated", "refuted", "insufficient"]
_FORBIDDEN_SOURCES = frozenset({"synthetic", "fabricated", "llm", "model", ""})


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class WorldModelPrediction:
    prediction_id: str
    world_model_id: str
    worldview: str
    domain: str
    contract_version: str
    predicted: dict[str, float]
    uncertainty: dict[str, float] = field(default_factory=dict)
    epoch: int = 0
    restored: bool = False
    created_at: str = field(default_factory=_now_iso)


@dataclass(frozen=True)
class ReconcileRecord:
    reconcile_id: str
    prediction_id: str
    actual_source: str
    actual: dict[str, float]
    tol: float
    status: ReconcileStatus
    errors: dict[str, float] = field(default_factory=dict)
    prediction_error: float | None = None
    worldview_consistent: bool = True
    created_at: str = field(default_factory=_now_iso)


@dataclass(frozen=True)
class AuthorityVector:
    abstraction_authority: float
    evidential_authority: float
    influence_authority: float


def reconcile_prediction(
    prediction: WorldModelPrediction,
    actual: dict[str, Any],
    *,
    actual_source: str,
    tol: float = 0.03,
    metrics: tuple[str, ...] | None = None,
    reconcile_id: str | None = None,
    worldview_consistent: bool = True,
) -> ReconcileRecord:
    """对账；actual_source 缺失/合成 → ValueError（不伪造 ground truth）."""
    source = str(actual_source or "").strip().lower()
    if source in _FORBIDDEN_SOURCES:
        raise ValueError(
            "actual_source 必须是真实执行/可重放经验（tool_run_id / pybullet_run_id / replay_id）"
        )
    if metrics is None:
        metrics = tuple(
            key for key in prediction.predicted if key in (actual or {})
        )
    response = reconcile(
        LawState(dict(prediction.predicted), domain=prediction.domain),
        dict(actual or {}),
        tol=tol,
        metrics=tuple(metrics),
    )
    errors = dict(response.get("errors") or {})
    prediction_error = max(errors.values()) if errors else None
    return ReconcileRecord(
        reconcile_id=reconcile_id or f"rec:{prediction.prediction_id}",
        prediction_id=prediction.prediction_id,
        actual_source=source,
        actual=dict(actual or {}),
        tol=tol,
        status=response.get("status", "insufficient"),
        errors=errors,
        prediction_error=prediction_error,
        worldview_consistent=bool(worldview_consistent),
        created_at=_now_iso(),
    )


def promotion_status(
    record: ReconcileRecord,
    *,
    independent_runs: int = 0,
) -> PromotionStatus:
    if record.status == "falsified":
        return "refuted"
    if record.status == "insufficient":
        return "insufficient"
    if independent_runs >= 1:
        return "replicated"
    return "reconciled"


def authority_for(
    record: ReconcileRecord | None,
    *,
    independent_runs: int = 0,
) -> AuthorityVector:
    """预测默认低证据权威；reconcile 成功升 0.6；独立复现升 0.9."""
    if record is None:
        return AuthorityVector(0.5, 0.2, 0.4)
    status = promotion_status(record, independent_runs=independent_runs)
    if status == "replicated":
        return AuthorityVector(0.6, 0.9, 0.8)
    if status == "reconciled":
        return AuthorityVector(0.6, 0.6, 0.6)
    if status == "refuted":
        return AuthorityVector(0.6, 0.1, 0.1)
    return AuthorityVector(0.4, 0.1, 0.2)


def worldview_compatible(a: WorldModelPrediction, b: WorldModelPrediction) -> bool:
    return (
        a.worldview == b.worldview
        and a.domain == b.domain
        and a.contract_version == b.contract_version
    )


def prediction_error_signal(
    record: ReconcileRecord,
    *,
    review_threshold: float = 0.05,
    intervene_threshold: float = 0.15,
) -> dict[str, Any]:
    """把 reconcile 误差映射成漂移信号；无误差 → no_signal."""
    error = record.prediction_error
    if error is None:
        return {"detector": "prediction_error", "level": "no_signal", "observed": None}
    if error > intervene_threshold:
        level = "intervene"
    elif error > review_threshold:
        level = "review"
    else:
        level = "ok"
    return {
        "detector": "prediction_error",
        "level": level,
        "observed": error,
        "source": record.actual_source,
        "prediction_id": record.prediction_id,
    }


def to_pmk_stance(prediction: WorldModelPrediction) -> dict[str, Any]:
    """给 PMK 认识论 sheaf 的 world_model stalk（不是事实，必须带标签）."""
    summary = ", ".join(
        f"{key}={value:.4g}" for key, value in prediction.predicted.items()
    )
    return {
        "source": "world_model",
        "text": f"[{prediction.worldview}/{prediction.domain}] {summary}",
        "worldview": prediction.worldview,
        "domain": prediction.domain,
        "contract_version": prediction.contract_version,
        "epoch": prediction.epoch,
        "restored": prediction.restored,
    }


__all__ = [
    "AuthorityVector",
    "PromotionStatus",
    "ReconcileRecord",
    "ReconcileStatus",
    "WorldModelPrediction",
    "authority_for",
    "prediction_error_signal",
    "promotion_status",
    "reconcile_prediction",
    "to_pmk_stance",
    "worldview_compatible",
]
