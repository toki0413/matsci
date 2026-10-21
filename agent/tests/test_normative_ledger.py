"""规范账不变量测试：追加式、投影确定性、撤销支配、TTL、租户隔离."""
from __future__ import annotations

import pytest

from huginn.normative.ledger import NormativeLedger, scope_covers

TENANT = "tenant-a"
SESSION = "session-1"
WRITE_SCOPE = {
    "actions": ["write"],
    "resources": ["workspace/*"],
    "limits": {"files": 10},
}


def _ledger(tmp_path) -> NormativeLedger:
    return NormativeLedger(tmp_path / "normative.db")


def test_propose_accept_authorize_are_distinct(tmp_path):
    """I2: propose ≠ accept ≠ authorize."""
    ledger = _ledger(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="agent",
        act_type="propose",
        proposition_id="plan-1",
    )
    view = ledger.project(TENANT, SESSION)
    assert len(view.proposals) == 1
    assert not view.accepted
    assert not view.authorizations
    assert not view.covers(WRITE_SCOPE)

    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="accept",
        proposition_id="plan-1",
    )
    view = ledger.project(TENANT, SESSION)
    assert len(view.accepted) == 1
    assert not view.authorizations, "接受命题不蕴含授权执行"
    assert not view.covers(WRITE_SCOPE)

    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="plan-1",
        scope=WRITE_SCOPE,
    )
    view = ledger.project(TENANT, SESSION)
    assert len(view.authorizations) == 1
    assert view.covers(WRITE_SCOPE)
    assert view.covers({"actions": ["write"], "resources": ["workspace/a.txt"]})
    assert not view.covers({"actions": ["publish"], "resources": ["workspace/a.txt"]})


def test_agent_cannot_self_authorize_or_accept(tmp_path):
    """I11: agent 只能 propose/commit/withdraw."""
    ledger = _ledger(tmp_path)
    with pytest.raises(PermissionError):
        ledger.append_act(
            tenant_id=TENANT,
            session_id=SESSION,
            actor="agent",
            act_type="authorize",
            proposition_id="plan-1",
            scope=WRITE_SCOPE,
        )
    with pytest.raises(PermissionError):
        ledger.append_act(
            tenant_id=TENANT,
            session_id=SESSION,
            actor="agent",
            act_type="accept",
            proposition_id="plan-1",
        )


def test_revoke_dominates_prior_authorization(tmp_path):
    """I3: revoke 支配更早的 authorize；之后的新授权仍有效."""
    ledger = _ledger(tmp_path)
    grant = ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="plan-1",
        scope=WRITE_SCOPE,
    )
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="revoke",
        proposition_id="plan-1",
        supersedes=grant.act_id,
    )
    view = ledger.project(TENANT, SESSION)
    assert not view.covers(WRITE_SCOPE)
    assert any(a.act_id == grant.act_id for a in view.revoked)
    assert not any(a.act_id == grant.act_id for a in view.authorizations)

    # 撤销后重新显式授权 → 新 act_id 不受旧 revoke 影响
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="plan-1",
        scope=WRITE_SCOPE,
    )
    assert ledger.project(TENANT, SESSION).covers(WRITE_SCOPE)


def test_ttl_is_evaluated_at_read_time(tmp_path):
    """I12: 过期授权不得出现在 active 视图；TTL 读取时判定."""
    ledger = _ledger(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="plan-1",
        scope=WRITE_SCOPE,
        ttl_s=10,
        created_at="2026-01-01T00:00:00+00:00",
    )
    fresh = ledger.project(
        TENANT, SESSION, now="2026-01-01T00:00:05+00:00"
    )
    assert fresh.covers(WRITE_SCOPE)
    expired = ledger.project(
        TENANT, SESSION, now="2026-01-01T00:00:11+00:00"
    )
    assert not expired.covers(WRITE_SCOPE)
    assert expired.expired


def test_tenant_isolation(tmp_path):
    """I9: 跨租户零泄漏."""
    ledger = _ledger(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="plan-1",
        scope=WRITE_SCOPE,
    )
    other = ledger.project("tenant-b", SESSION)
    assert not other.events
    assert not other.covers(WRITE_SCOPE)


def test_projection_is_deterministic(tmp_path):
    """I4: 同一事件序列 + 同一 now → 同一投影."""
    ledger = _ledger(tmp_path)
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="agent",
        act_type="propose",
        proposition_id="p1",
    )
    ledger.append_act(
        tenant_id=TENANT,
        session_id=SESSION,
        actor="user",
        act_type="authorize",
        proposition_id="p1",
        scope=WRITE_SCOPE,
        ttl_s=3600,
        created_at="2026-01-01T00:00:00+00:00",
    )
    now = "2026-01-01T00:10:00+00:00"
    first = ledger.project(TENANT, SESSION, now=now)
    second = ledger.project(TENANT, SESSION, now=now)
    assert [a.act_id for a in first.events] == [a.act_id for a in second.events]
    assert [a.act_id for a in first.authorizations] == [
        a.act_id for a in second.authorizations
    ]
    assert first.covers(WRITE_SCOPE) == second.covers(WRITE_SCOPE)


def test_scope_covers_limits_and_prefix():
    assert scope_covers(
        {"actions": ["write"], "resources": ["workspace/*"],
         "limits": {"files": 10}},
        {"actions": ["write"], "resources": ["workspace/x"], "limits": {"files": 5}},
    )
    assert not scope_covers(
        {"actions": ["write"], "resources": ["workspace/*"],
         "limits": {"files": 10}},
        {"actions": ["write"], "resources": ["workspace/x"], "limits": {"files": 20}},
    )
    assert not scope_covers(
        {"actions": ["write"], "resources": ["workspace/*"]},
        {"actions": ["publish"], "resources": ["workspace/x"]},
    )


def test_session_state_records_proposal_and_outcome(tmp_path):
    """session_state 接入后：request 写 propose，clear(accepted=...) 写 accept/reject."""
    from huginn.normative.session import SessionGrounding
    from huginn.session_state import UnifiedSessionState

    ledger = _ledger(tmp_path)
    state = UnifiedSessionState(session_id=SESSION)
    state.grounding = SessionGrounding(ledger, TENANT, SESSION)

    state.request_confirmation("new_plan", "execute plan?", {"plan": "p1"})
    view = ledger.project(TENANT, SESSION)
    assert len(view.proposals) == 1
    assert not view.accepted
    assert not view.authorizations, "提议不自动变成授权"

    state.clear_confirmation(accepted=True)
    view = ledger.project(TENANT, SESSION)
    assert len(view.accepted) == 1
    assert state.pending_confirmation is None


def test_session_state_does_not_guess_outcome(tmp_path):
    """clear_confirmation() 不带 accepted 时只清工作状态，不伪造接受/拒绝."""
    from huginn.normative.session import SessionGrounding
    from huginn.session_state import UnifiedSessionState

    ledger = _ledger(tmp_path)
    state = UnifiedSessionState(session_id=SESSION)
    state.grounding = SessionGrounding(ledger, TENANT, SESSION)

    state.request_confirmation("new_plan", "execute plan?")
    state.clear_confirmation()
    view = ledger.project(TENANT, SESSION)
    assert len(view.proposals) == 1, "未表态的提议应保持 open"
    assert not view.accepted
    assert not view.rejected
