"""会话级规范账适配器 — 把一场会话的 propose/accept/authorize 收在一处.

薄封装，不复制投影/判定逻辑：``view`` 与 ``admit`` 直接委托给
``NormativeLedger`` 与 ``admission.admit``。
"""
from __future__ import annotations

from typing import Any

from huginn.normative.admission import AdmissionDecision, AdmissionRequest, admit
from huginn.normative.ledger import GroundingView, NormativeAct, NormativeLedger


class SessionGrounding:
    """一场会话的规范行为入口。"""

    def __init__(
        self,
        ledger: NormativeLedger,
        tenant_id: str,
        session_id: str,
    ) -> None:
        if not tenant_id or not session_id:
            raise ValueError("tenant_id / session_id 不能为空")
        self.ledger = ledger
        self.tenant_id = tenant_id
        self.session_id = session_id

    def propose(
        self,
        proposition_id: str,
        *,
        scope: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> NormativeAct:
        return self.ledger.append_act(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            actor="agent",
            act_type="propose",
            proposition_id=proposition_id,
            scope=scope,
            evidence=evidence,
        )

    def accept(
        self, proposition_id: str, *, evidence: dict[str, Any] | None = None
    ) -> NormativeAct:
        return self.ledger.append_act(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            actor="user",
            act_type="accept",
            proposition_id=proposition_id,
            evidence=evidence,
        )

    def reject(
        self, proposition_id: str, *, evidence: dict[str, Any] | None = None
    ) -> NormativeAct:
        return self.ledger.append_act(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            actor="user",
            act_type="reject",
            proposition_id=proposition_id,
            evidence=evidence,
        )

    def authorize(
        self,
        proposition_id: str,
        scope: dict[str, Any],
        *,
        ttl_s: int | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> NormativeAct:
        return self.ledger.append_act(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            actor="user",
            act_type="authorize",
            proposition_id=proposition_id,
            scope=scope,
            ttl_s=ttl_s,
            evidence=evidence,
        )

    def revoke(
        self,
        act_id: str,
        proposition_id: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> NormativeAct:
        return self.ledger.append_act(
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            actor="user",
            act_type="revoke",
            proposition_id=proposition_id,
            supersedes=act_id,
            evidence=evidence,
        )

    def view(self, *, now: str | None = None) -> GroundingView:
        return self.ledger.project(self.tenant_id, self.session_id, now=now)

    def admit(
        self,
        request: AdmissionRequest,
        *,
        claim_statuses: dict[str, str] | None = None,
        persona_boundaries: dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        return admit(
            request,
            self.view(),
            claim_statuses=claim_statuses,
            persona_boundaries=persona_boundaries,
        )


__all__ = ["SessionGrounding"]
