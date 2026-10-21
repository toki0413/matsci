"""准入调度器不变量测试：fail-closed、最小 required_acts、claim 降级、边界拒绝."""
from __future__ import annotations

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionRequest, admit
from huginn.normative.ledger import NormativeLedger

TENANT = "tenant-a"
SESSION = "session-1"
WRITE_SCOPE = {"actions": ["write"], "resources": ["workspace/*"], "limits": {}}


def _view(tmp_path):
    ledger = NormativeLedger(tmp_path / "normative.db")
    return ledger, ledger.project(TENANT, SESSION)


def _request(**overrides) -> AdmissionRequest:
    data = {
        "tenant_id": TENANT,
        "session_id": SESSION,
        "action_name": "write",
        "resources": ("workspace/a.txt",),
        "risk": RiskLevel.MEDIUM,
    }
    data.update(overrides)
    return AdmissionRequest(**data)


def test_no_authorization_asks_with_minimal_act(tmp_path):
    """I1: 无 active 授权 → 不 allow，并给出机器可执行的 required_acts."""
    _, view = _view(tmp_path)
    decision = admit(_request(), view)
    assert decision.allow is False
    assert decision.mode == "ask"
    assert decision.required_acts == (
        "authorize:write:workspace/a.txt",
    )


def test_authorization_covers_scope_allows(tmp_path):
    ledger, _ = _view(tmp_path)
    grant = ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    decision = admit(_request(), ledger.project(TENANT, SESSION))
    assert decision.allow is True
    assert decision.mode == "allow"
    assert decision.cited_act_ids == (grant.act_id,)


def test_scope_mismatch_asks(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope={"actions": ["read"], "resources": ["workspace/*"]},
    )
    decision = admit(_request(), ledger.project(TENANT, SESSION))
    assert decision.allow is False
    assert decision.mode == "ask"


def test_revoked_authorization_does_not_allow(tmp_path):
    ledger, _ = _view(tmp_path)
    grant = ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="revoke",
        proposition_id="p1",
        supersedes=grant.act_id,
    )
    decision = admit(_request(), ledger.project(TENANT, SESSION))
    assert decision.allow is False
    assert decision.mode == "ask"


def test_contested_claim_degrades_medium_risk(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    decision = admit(
        _request(claim_ids=("claim-1",)),
        ledger.project(TENANT, SESSION),
        claim_statuses={"claim-1": "contested"},
    )
    assert decision.allow is False
    assert decision.mode == "degrade"
    assert decision.cited_claim_ids == ("claim-1",)


def test_low_risk_contested_claim_is_not_blocked(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    decision = admit(
        _request(risk=RiskLevel.LOW, claim_ids=("claim-1",)),
        ledger.project(TENANT, SESSION),
        claim_statuses={"claim-1": "contested"},
    )
    assert decision.allow is True


def test_persona_boundary_denies(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    decision = admit(
        _request(),
        ledger.project(TENANT, SESSION),
        persona_boundaries={"denied_actions": ["write"]},
    )
    assert decision.allow is False
    assert decision.mode == "deny"


def test_tenant_mismatch_denies(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    decision = admit(
        _request(tenant_id="tenant-b"),
        ledger.project(TENANT, SESSION),
    )
    assert decision.allow is False
    assert decision.mode == "deny"


def test_read_only_needs_no_authorization(tmp_path):
    _, view = _view(tmp_path)
    decision = admit(_request(requires_authorization=False), view)
    assert decision.allow is True
    assert decision.mode == "allow"
    assert not decision.required_acts


def test_admission_is_deterministic(tmp_path):
    ledger, _ = _view(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
    )
    view = ledger.project(TENANT, SESSION)
    first = admit(_request(), view)
    second = admit(_request(), view)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_shadow_evaluates_without_blocking(tmp_path):
    """shadow 模式：评估真实工具调用，返回 ask/allow，但由调用方决定是否阻断."""
    from huginn.normative.shadow import evaluate_tool_call

    ledger, _ = _view(tmp_path)
    decision = evaluate_tool_call(
        tenant_id=TENANT,
        session_id=SESSION,
        tool_name="bash_tool",
        resources=("workspace/x",),
        risk=RiskLevel.HIGH,
        ledger=ledger,
        enabled=True,
    )
    assert decision is not None
    assert decision.allow is False
    assert decision.mode == "ask"

    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="tool-1",
        scope={"actions": ["bash_tool"], "resources": ["*"], "limits": {"uses": 1}},
    )
    allowed = evaluate_tool_call(
        tenant_id=TENANT,
        session_id=SESSION,
        tool_name="bash_tool",
        risk=RiskLevel.HIGH,
        ledger=ledger,
        enabled=True,
    )
    assert allowed is not None and allowed.allow is True


def test_shadow_disabled_returns_none(tmp_path):
    from huginn.normative.shadow import evaluate_tool_call

    ledger, _ = _view(tmp_path)
    assert evaluate_tool_call(
        tenant_id=TENANT,
        session_id=SESSION,
        tool_name="bash_tool",
        ledger=ledger,
        enabled=False,
    ) is None


def test_shadow_read_only_allows_without_authorization(tmp_path):
    from huginn.normative.shadow import evaluate_tool_call

    ledger, _ = _view(tmp_path)
    decision = evaluate_tool_call(
        tenant_id=TENANT,
        session_id=SESSION,
        tool_name="file_read_tool",
        requires_authorization=False,
        ledger=ledger,
        enabled=True,
    )
    assert decision is not None and decision.allow is True
