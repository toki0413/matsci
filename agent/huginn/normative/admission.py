"""准入调度器（Admission）— 纯函数、无 LLM、确定性.

输入是一份 ``GroundingView`` 投影 + 一个 ``AdmissionRequest``；输出
``AdmissionDecision``。它只回答「此刻是否允许执行」，不执行、不写账。

规则优先级（对齐 spec §8 / I1–I12）：
1. 租户/会话不匹配 → deny。
2. persona 边界命中 → deny。
3. 只读动作 → allow（不需要授权）。
4. 有覆盖 scope 的 active 授权 → 继续检查 claim。
5. 无授权 → ask，并返回最小 required_acts（机器可执行，不泛问）。
6. 依赖 claim 处于 contested/recheck/refuted 且风险 >= MEDIUM → degrade。
7. 其余 → allow。

热路径不调 LLM；同一输入必得同一输出。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from huginn.core_types import RiskLevel
from huginn.normative.ledger import GroundingView, NormativeAct

DecisionMode = Literal["allow", "ask", "deny", "degrade"]

_RISK_RANK = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}
_BLOCKING_CLAIM_STATUSES = frozenset({"contested", "recheck", "refuted"})


@dataclass(frozen=True)
class AdmissionRequest:
    tenant_id: str
    session_id: str
    action_name: str
    resources: tuple[str, ...] = ()
    risk: RiskLevel = RiskLevel.LOW
    persona_id: str = ""
    claim_ids: tuple[str, ...] = ()
    proposed_by: str = "agent"
    requires_authorization: bool = True
    requested_scope: dict[str, Any] | None = None

    def scope(self) -> dict[str, Any]:
        if self.requested_scope is not None:
            return self.requested_scope
        return {
            "actions": [self.action_name],
            "resources": list(self.resources),
            "limits": {},
        }


@dataclass(frozen=True)
class AdmissionDecision:
    allow: bool
    mode: DecisionMode
    reasons: tuple[str, ...] = ()
    required_acts: tuple[str, ...] = ()
    cited_act_ids: tuple[str, ...] = ()
    cited_claim_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow": self.allow,
            "mode": self.mode,
            "reasons": list(self.reasons),
            "required_acts": list(self.required_acts),
            "cited_act_ids": list(self.cited_act_ids),
            "cited_claim_ids": list(self.cited_claim_ids),
        }


def _deny(*reasons: str) -> AdmissionDecision:
    return AdmissionDecision(allow=False, mode="deny", reasons=tuple(reasons))


def _required_act(request: AdmissionRequest) -> str:
    scope = request.scope()
    actions = ",".join(scope.get("actions") or [request.action_name])
    resources = ",".join(scope.get("resources") or ["*"])
    return f"authorize:{actions}:{resources}"


def _persona_violation(
    request: AdmissionRequest, boundaries: dict[str, Any] | None
) -> str | None:
    if not boundaries:
        return None
    denied_actions = set(boundaries.get("denied_actions") or [])
    if request.action_name in denied_actions or "*" in denied_actions:
        return f"persona 边界禁止动作: {request.action_name}"
    for pattern in boundaries.get("denied_resources") or []:
        for resource in request.resources:
            if str(pattern).endswith("*"):
                if str(resource).startswith(str(pattern)[:-1]):
                    return f"persona 边界禁止资源: {resource}"
            elif str(resource) == str(pattern):
                return f"persona 边界禁止资源: {resource}"
    return None


def admit(
    request: AdmissionRequest,
    grounding: GroundingView,
    *,
    claim_statuses: dict[str, str] | None = None,
    persona_boundaries: dict[str, Any] | None = None,
) -> AdmissionDecision:
    """纯函数准入判定；不调 LLM、不做 I/O、不抛业务异常。"""
    if request.tenant_id != grounding.tenant_id:
        return _deny("租户不匹配：请求与规范账不属于同一 tenant")
    if request.session_id != grounding.session_id:
        return _deny("会话不匹配：请求与规范账不属于同一 session")

    violation = _persona_violation(request, persona_boundaries)
    if violation:
        return _deny(violation)

    if not request.requires_authorization:
        return AdmissionDecision(
            allow=True,
            mode="allow",
            reasons=("只读动作，无需授权",),
        )

    auth: NormativeAct | None = grounding.active_authorization_for(request.scope())
    if auth is None:
        return AdmissionDecision(
            allow=False,
            mode="ask",
            reasons=("缺少覆盖该 scope 的 active 授权",),
            required_acts=(_required_act(request),),
        )

    claim_statuses = claim_statuses or {}
    blocked_claims = tuple(
        cid for cid in request.claim_ids
        if claim_statuses.get(cid) in _BLOCKING_CLAIM_STATUSES
    )
    if blocked_claims and _RISK_RANK.get(request.risk, 1) >= _RISK_RANK[RiskLevel.MEDIUM]:
        return AdmissionDecision(
            allow=False,
            mode="degrade",
            reasons=(
                "依赖的 claim 处于 contested/recheck/refuted，禁止按确定结论执行",
            ),
            cited_act_ids=(auth.act_id,),
            cited_claim_ids=blocked_claims,
        )

    return AdmissionDecision(
        allow=True,
        mode="allow",
        reasons=(f"授权覆盖 scope: {auth.act_id}",),
        cited_act_ids=(auth.act_id,),
        cited_claim_ids=blocked_claims,
    )


__all__ = [
    "AdmissionDecision",
    "AdmissionRequest",
    "DecisionMode",
    "admit",
]
