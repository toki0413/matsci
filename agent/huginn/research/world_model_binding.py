"""世界模型 binding — 预测升格链、worldview 隔离、prediction_error 漂移信号.

对齐 spec §20：
- prediction 默认 hypothesis/derived，未经 reconcile 不得作为 direct 证据；
- actual 必须来自真实执行/可重放经验并带 source，禁止合成；
- 跨 worldview/domain/contract_version 不得粘合；
- 世界模型不得授权动作。
"""
from __future__ import annotations

import re
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


@dataclass(frozen=True)
class WorldModelRiskSignal:
    """世界模型的风险信号（§20.5）：只约束，不授权."""

    mode: Literal["allow", "ask", "degrade", "deny"]
    reasons: tuple[str, ...] = ()
    reversible: bool = True
    constraints: tuple[str, ...] = ()


def world_model_risk(
    *,
    constraints: tuple[str, ...] = (),
    reversible: bool = True,
    uncertainty: float = 0.0,
    risk: str = "low",
) -> WorldModelRiskSignal:
    """约束违规 → deny；不可逆 → ask；高风险 + 高不确定 → ask；其余 allow."""
    if constraints:
        return WorldModelRiskSignal(
            mode="deny", reasons=tuple(str(c) for c in constraints),
            reversible=reversible, constraints=tuple(constraints),
        )
    if not reversible:
        return WorldModelRiskSignal(
            mode="ask", reasons=("irreversible action requires explicit scope/confirmation",),
            reversible=False,
        )
    if str(risk).lower() in ("high", "critical") and float(uncertainty) > 0.5:
        return WorldModelRiskSignal(
            mode="ask",
            reasons=(f"high risk with uncertainty={float(uncertainty):.2f}",),
            reversible=reversible,
        )
    return WorldModelRiskSignal(mode="allow", reversible=reversible)


@dataclass(frozen=True)
class WorldModelRevision:
    """世界模型参数/定律修订提案；只有过采纳门才 active（I47）."""

    revision_id: str
    world_model_id: str
    delta: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    status: Literal["candidate", "active", "rejected"] = "candidate"


def propose_revision(
    *,
    world_model_id: str,
    delta: dict[str, Any],
    evidence_ids: tuple[str, ...] = (),
    revision_id: str | None = None,
) -> WorldModelRevision:
    return WorldModelRevision(
        revision_id=revision_id or f"rev:{world_model_id}:{len(delta)}",
        world_model_id=str(world_model_id),
        delta=dict(delta),
        evidence_ids=tuple(evidence_ids),
    )


def apply_revision(
    revision: WorldModelRevision,
    *,
    actor: str,
    adoption_green: bool,
    independent_runs: int = 0,
) -> WorldModelRevision:
    """修订只有 ``actor=user/system/meta_improver`` + GREEN 采纳门才 active.

    监控环/梦境/agent 不能直接改模型（I47）；未过门保持 candidate，被拒则 rejected。
    """
    actor = str(actor or "unknown").lower()
    if actor in ("agent", "dream", "monitor", "unknown"):
        return WorldModelRevision(
            revision_id=revision.revision_id,
            world_model_id=revision.world_model_id,
            delta=revision.delta,
            evidence_ids=revision.evidence_ids,
            status="rejected",
        )
    if not adoption_green or independent_runs < 1:
        return revision
    return WorldModelRevision(
        revision_id=revision.revision_id,
        world_model_id=revision.world_model_id,
        delta=revision.delta,
        evidence_ids=revision.evidence_ids,
        status="active",
    )


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


_LABEL_KEYS = ("worldview", "domain", "contract")


def parse_world_labels(text: str) -> dict[str, str]:
    """从 PMK 文本里解析 ``worldview/domain/contract`` 标签（缺哪个就少哪个）."""
    raw = str(text or "")
    labels: dict[str, str] = {}
    for key in _LABEL_KEYS:
        match = re.search(
            rf"{key}(?:_version)?\s*[=:]\s*([A-Za-z0-9_.\-]+)", raw, re.IGNORECASE
        )
        if match:
            labels[key] = match.group(1)
    return labels


def cross_glue_labels(
    world_model_text: str,
    epistemic_texts: dict[str, str],
) -> list[dict[str, str]]:
    """世界模型 stalk 与 memory/kb/timeseries 的 worldview/contract 兼容性检查.

    Spec §20.4 的 restriction map 第一层：标签不兼容时**不建边**，记为
    ``worldview_mismatch`` 或 ``contract_violation``；跨 worldview 的差异不是
    认识论 contradiction，绝不能升级成 pause。

    只有**双方都带完整标签**时才判定；无标签的证据不是 glue 候选（未检查 ≠ 通过）。
    """
    wm = parse_world_labels(world_model_text)
    if not all(key in wm for key in _LABEL_KEYS):
        return []
    obstructions: list[dict[str, str]] = []
    for source, text in (epistemic_texts or {}).items():
        other = parse_world_labels(text)
        if not all(key in other for key in _LABEL_KEYS):
            continue
        if other["worldview"] != wm["worldview"]:
            obstructions.append({
                "kind": "worldview_mismatch",
                "source": str(source),
                "reason": (
                    f"{source} worldview={other['worldview']} vs world_model "
                    f"worldview={wm['worldview']}; not glued, not a contradiction"
                ),
            })
            continue
        if other["domain"] != wm["domain"] or other["contract"] != wm["contract"]:
            obstructions.append({
                "kind": "contract_violation",
                "source": str(source),
                "reason": (
                    f"{source} domain={other['domain']}/contract={other['contract']} "
                    f"vs world_model domain={wm['domain']}/contract={wm['contract']}"
                ),
            })
    return obstructions


@dataclass(frozen=True)
class WorldModelFinding:
    """Dreamer 可消费的世界模型 finding（§20.8）；只作候选，不改模型."""

    finding_id: str
    subject: str
    probe_type: str = "surprise"
    surprise: float = 0.0
    expected_impact: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    status: str = ""


def findings_from_reconciles(
    records: list[ReconcileRecord],
) -> list[WorldModelFinding]:
    """falsified / insufficient 的 reconcile → 高惊奇 finding；borne_out 跳过."""
    findings: list[WorldModelFinding] = []
    for record in records:
        if record.status == "borne_out":
            continue
        error = float(record.prediction_error or 0.0)
        surprise = (
            min(1.0, 0.5 + error * 2.0)
            if record.status == "falsified"
            else 0.5
        )
        findings.append(
            WorldModelFinding(
                finding_id=f"wm:{record.prediction_id}:{record.status}",
                subject=record.prediction_id,
                surprise=round(surprise, 4),
                expected_impact=0.8 if record.status == "falsified" else 0.5,
                evidence_ids=(
                    record.prediction_id,
                    record.actual_source,
                ),
                status=record.status,
            )
        )
    return findings


def diagnose_world_model_failure(
    record: ReconcileRecord,
    *,
    contract_compatible: bool = True,
) -> dict[str, Any]:
    """把 reconcile 失败归因到 law_mismatch / contract_violation / stale_model."""
    if not contract_compatible or not record.worldview_consistent:
        cause = "contract_violation"
    elif record.status == "falsified":
        cause = "law_mismatch"
    elif record.status == "insufficient":
        cause = "stale_model"
    else:
        cause = "none"
    return {
        "cause_class": cause,
        "confidence": 0.8 if cause != "none" else 0.5,
        "abstain": cause == "none",
        "prediction_id": record.prediction_id,
        "evidence_ids": [record.prediction_id, record.actual_source],
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
    "WorldModelFinding",
    "WorldModelPrediction",
    "WorldModelRevision",
    "WorldModelRiskSignal",
    "apply_revision",
    "authority_for",
    "cross_glue_labels",
    "diagnose_world_model_failure",
    "findings_from_reconciles",
    "propose_revision",
    "parse_world_labels",
    "prediction_error_signal",
    "promotion_status",
    "reconcile_prediction",
    "to_pmk_stance",
    "world_model_risk",
    "worldview_compatible",
]
