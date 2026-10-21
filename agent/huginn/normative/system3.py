"""System3 准入调度 — 在 admit() 之上叠加监控信任度.

规则（spec §19.12.6 / §8）：
- trust=full/partial：沿用 admit()；
- trust=blind/unknown：收紧自主边界——低风险动作也受 contested/recheck/refuted
  claim 约束；决策 reason 显式标注 trust；
- 只读动作仍允许；监控失败不阻断任务可用性。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionDecision, AdmissionRequest, admit
from huginn.normative.ledger import GroundingView

_STRICT_TRUST = frozenset({"blind", "unknown"})
_BLOCKING_CLAIMS = frozenset({"contested", "recheck", "refuted"})


@dataclass(frozen=True)
class AutonomyLimits:
    """trust → 自主边界；调用方按此收紧预算/并发/审批门槛."""

    max_parallel_jobs: int
    token_budget_scale: float
    require_approval_for_risk: str
    strict: bool


def autonomy_limits(trust: str) -> AutonomyLimits:
    trust = str(trust or "unknown").lower()
    if trust == "full":
        return AutonomyLimits(5, 1.0, "medium", False)
    if trust == "partial":
        return AutonomyLimits(4, 0.75, "medium", False)
    return AutonomyLimits(2, 0.5, "low", True)


def current_autonomy_limits() -> AutonomyLimits:
    """读取当前 VIGIL trust 对应的自主边界；读不到按 unknown 最严格处理."""
    try:
        from huginn.metacog.vigil_runtime import get_vigil_runtime

        trust = get_vigil_runtime().status().trust
    except Exception:
        trust = "unknown"
    return autonomy_limits(trust)


def parallel_cap(requested: int, *, trust: str | None = None) -> int:
    """把请求的并发数夹到当前自主边界内（最低 1）."""
    limits = autonomy_limits(trust) if trust is not None else current_autonomy_limits()
    return max(1, min(int(requested), int(limits.max_parallel_jobs)))


def scaled_token_budget(
    budget: int,
    *,
    trust: str | None = None,
    floor: int = 1000,
) -> int:
    """按当前 trust 缩放 token/字符预算；不降到 floor 以下."""
    limits = autonomy_limits(trust) if trust is not None else current_autonomy_limits()
    scaled = int(float(budget) * float(limits.token_budget_scale))
    return max(int(floor), scaled)


def admit_with_trust(
    request: AdmissionRequest,
    grounding: GroundingView,
    *,
    trust: str,
    claim_statuses: dict[str, str] | None = None,
    persona_boundaries: dict[str, Any] | None = None,
) -> AdmissionDecision:
    """在 admit() 结果上叠加 trust；不调 LLM、不做 I/O."""
    decision = admit(
        request,
        grounding,
        claim_statuses=claim_statuses,
        persona_boundaries=persona_boundaries,
    )
    trust = str(trust or "unknown").lower()
    limits = autonomy_limits(trust)
    reasons = tuple(decision.reasons) + (f"monitor trust={trust}",)

    if trust in _STRICT_TRUST and decision.allow and request.requires_authorization:
        claim_statuses = claim_statuses or {}
        blocked = tuple(
            cid for cid in request.claim_ids
            if claim_statuses.get(cid) in _BLOCKING_CLAIMS
        )
        if blocked and request.risk in (RiskLevel.NONE, RiskLevel.LOW):
            return AdmissionDecision(
                allow=False,
                mode="degrade",
                reasons=reasons + (
                    "monitor blind: low-risk action still depends on contested claim",
                ),
                cited_act_ids=decision.cited_act_ids,
                cited_claim_ids=blocked,
                autonomy=asdict(limits),
            )

    return AdmissionDecision(
        allow=decision.allow,
        mode=decision.mode,
        reasons=reasons,
        required_acts=decision.required_acts,
        cited_act_ids=decision.cited_act_ids,
        cited_claim_ids=decision.cited_claim_ids,
        autonomy=asdict(limits),
    )


__all__ = [
    "AutonomyLimits",
    "admit_with_trust",
    "autonomy_limits",
    "current_autonomy_limits",
    "parallel_cap",
    "scaled_token_budget",
]
