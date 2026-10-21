"""System3 准入确定性：相同输入必得相同输出，且不调 LLM."""
from __future__ import annotations

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionRequest
from huginn.normative.ledger import NormativeLedger
from huginn.normative.system3 import admit_with_trust

TENANT = "t1"
SESSION = "s1"
SCOPE = {"actions": ["write"], "resources": ["*"], "limits": {}}


def _grounding(tmp_path):
    ledger = NormativeLedger(tmp_path / "norm.db")
    ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="authorize", proposition_id="p1", scope=SCOPE,
    )
    return ledger.project(TENANT, SESSION)


def test_deterministic_no_llm(tmp_path, monkeypatch):
    """I8: admit_with_trust 是纯函数；禁止热路径 LLM."""
    view = _grounding(tmp_path)
    request = AdmissionRequest(
        tenant_id=TENANT,
        session_id=SESSION,
        action_name="write",
        risk=RiskLevel.MEDIUM,
        claim_ids=("c1",),
    )

    # 任何 LLM 入口被调用都应让测试失败（纯函数不应触达）
    def _boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("System3 admission 不得调用 LLM")

    import huginn.llm_retry as retry_mod

    monkeypatch.setattr(retry_mod, "with_retry", _boom)

    first = admit_with_trust(
        request, view, trust="blind", claim_statuses={"c1": "contested"}
    )
    second = admit_with_trust(
        request, view, trust="blind", claim_statuses={"c1": "contested"}
    )
    assert first.to_dict() == second.to_dict()
    assert first.allow is False and first.mode == "degrade"
    assert first.cited_claim_ids == ("c1",)
