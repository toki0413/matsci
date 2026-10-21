"""Grounding awareness 不变量：act 分离、撤销/过期 mask."""
from __future__ import annotations

from huginn.normative.ledger import NormativeLedger

TENANT = "t1"
SESSION = "s1"
SCOPE = {"actions": ["write"], "resources": ["workspace/*"], "limits": {}}


def _ledger(tmp_path) -> NormativeLedger:
    return NormativeLedger(tmp_path / "norm.db")


def test_act_separation(tmp_path):
    """I2: propose 不蕴含 accept；accept 不蕴含 authorize."""
    ledger = _ledger(tmp_path)
    ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="agent",
        act_type="propose", proposition_id="p1",
    )
    view = ledger.project(TENANT, SESSION)
    assert view.proposals and not view.accepted and not view.authorizations

    ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="accept", proposition_id="p1",
    )
    view = ledger.project(TENANT, SESSION)
    assert view.accepted and not view.authorizations
    assert not view.covers(SCOPE), "accept 不是 authorize"

    ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="authorize", proposition_id="p1", scope=SCOPE,
    )
    assert ledger.project(TENANT, SESSION).covers(SCOPE)


def test_mask_revoked(tmp_path):
    """I12: 撤销/过期授权不得出现在后续 AdmissionView."""
    ledger = _ledger(tmp_path)
    active = ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="authorize", proposition_id="p1", scope=SCOPE,
    )
    expired = ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="authorize", proposition_id="p2", scope=SCOPE, ttl_s=1,
    )
    ledger.append_act(
        tenant_id=TENANT, session_id=SESSION, actor="user",
        act_type="revoke", proposition_id="p1", supersedes=active.act_id,
    )
    view = ledger.project(TENANT, SESSION, now="2100-01-01T00:00:00+00:00")

    ids = {act.act_id for act in view.authorizations}
    assert active.act_id not in ids
    assert expired.act_id not in ids
    assert view.active_authorization_for(SCOPE) is None
    assert not view.covers(SCOPE)
