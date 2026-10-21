"""System3 准入调度 — 在 admit() 之上叠加监控信任度.

规则（spec §19.12.6 / §8）：
- trust=full/partial：沿用 admit()；
- trust=blind/unknown：收紧自主边界——低风险动作也受 contested/recheck/refuted
  claim 约束；决策 reason 显式标注 trust；
- 只读动作仍允许；监控失败不阻断任务可用性。
"""
from __future__ import annotations

from typing import Any

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionDecision, AdmissionRequest, admit
from huginn.normative.ledger import GroundingView

_STRICT_TRUST = frozenset({"blind", "unknown"})
_BLOCKING_CLAIMS = frozenset({"contested", "recheck", "refuted"})


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
            )

    return AdmissionDecision(
        allow=decision.allow,
        mode=decision.mode,
        reasons=reasons,
        required_acts=decision.required_acts,
        cited_act_ids=decision.cited_act_ids,
        cited_claim_ids=decision.cited_claim_ids,
    )


__all__ = ["admit_with_trust"]
