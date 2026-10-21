"""System3 trust-aware admission 测试."""
from __future__ import annotations

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionRequest
from huginn.normative.ledger import NormativeLedger
from huginn.normative.system3 import (
    admit_with_trust,
    autonomy_limits,
    parallel_cap,
    scaled_token_budget,
)

TENANT = "t1"
SESSION = "s1"
SCOPE = {"actions": ["write"], "resources": ["*"], "limits": {}}


def _grounding(tmp_path):
    ledger = NormativeLedger(tmp_path / "norm.db")
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=SCOPE,
    )
    return ledger.project(TENANT, SESSION)


def _request(**overrides) -> AdmissionRequest:
    data = {
        "tenant_id": TENANT,
        "session_id": SESSION,
        "action_name": "write",
        "risk": RiskLevel.LOW,
        "claim_ids": ("c1",),
    }
    data.update(overrides)
    return AdmissionRequest(**data)


def test_full_trust_allows_low_risk_with_contested_claim(tmp_path):
    view = _grounding(tmp_path)
    decision = admit_with_trust(
        _request(),
        view,
        trust="full",
        claim_statuses={"c1": "contested"},
    )
    assert decision.allow is True
    assert decision.mode == "allow"


def test_blind_trust_degrades_low_risk_with_contested_claim(tmp_path):
    view = _grounding(tmp_path)
    decision = admit_with_trust(
        _request(),
        view,
        trust="blind",
        claim_statuses={"c1": "contested"},
    )
    assert decision.allow is False
    assert decision.mode == "degrade"
    assert "monitor trust=blind" in decision.reasons


def test_blind_trust_still_allows_read_only(tmp_path):
    view = _grounding(tmp_path)
    decision = admit_with_trust(
        _request(requires_authorization=False),
        view,
        trust="blind",
        claim_statuses={"c1": "contested"},
    )
    assert decision.allow is True


def test_blind_trust_without_authorization_asks(tmp_path):
    ledger = NormativeLedger(tmp_path / "norm.db")
    view = ledger.project(TENANT, SESSION)
    decision = admit_with_trust(
        _request(),
        view,
        trust="blind",
    )
    assert decision.allow is False
    assert decision.mode == "ask"


def test_autonomy_limits_contract_by_trust():
    full = autonomy_limits("full")
    blind = autonomy_limits("blind")
    assert full.max_parallel_jobs == 5
    assert full.token_budget_scale == 1.0
    assert blind.max_parallel_jobs == 2
    assert blind.token_budget_scale == 0.5
    assert blind.strict is True


def test_parallel_cap_clamps_by_trust():
    assert parallel_cap(20, trust="full") == 5
    assert parallel_cap(20, trust="partial") == 4
    assert parallel_cap(20, trust="blind") == 2
    assert parallel_cap(1, trust="blind") == 1


def test_scaled_token_budget_by_trust():
    assert scaled_token_budget(12000, trust="full") == 12000
    assert scaled_token_budget(12000, trust="partial") == 9000
    assert scaled_token_budget(12000, trust="blind") == 6000
    assert scaled_token_budget(1200, trust="blind") == 1000


def test_tool_output_budget_scales_with_trust(monkeypatch):
    """工具输出预算随 trust 收紧（§19.12.6），显式预算不受影响."""
    import huginn.normative.system3 as s3
    from huginn.tools import adapter

    monkeypatch.delenv("HUGINN_MAX_TOOL_OUTPUT_TOKENS", raising=False)
    monkeypatch.setattr(
        s3, "current_autonomy_limits", lambda: s3.autonomy_limits("full")
    )
    assert adapter._default_tool_output_tokens() == 25000
    monkeypatch.setattr(
        s3, "current_autonomy_limits", lambda: s3.autonomy_limits("partial")
    )
    assert adapter._default_tool_output_tokens() == 18750
    monkeypatch.setattr(
        s3, "current_autonomy_limits", lambda: s3.autonomy_limits("blind")
    )
    assert adapter._default_tool_output_tokens() == 12500


def test_decision_carries_autonomy_limits(tmp_path):
    view = _grounding(tmp_path)
    decision = admit_with_trust(
        _request(),
        view,
        trust="blind",
        claim_statuses={"c1": "contested"},
    )
    assert decision.autonomy["max_parallel_jobs"] == 2
    assert decision.autonomy["strict"] is True
    assert decision.to_dict()["autonomy"]["token_budget_scale"] == 0.5
