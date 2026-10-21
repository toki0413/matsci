"""I18/I22: Soul 全序、覆盖留痕、跨租户梦境零泄漏."""
from __future__ import annotations

import pytest

from huginn.memory.soul import (
    SoulResolutionLedger,
    dream_authority,
    filter_dream_seeds,
    resolve,
    soul_priority,
    total_order,
)


def test_total_order_and_audit(tmp_path):
    order = total_order()
    assert order == {"tenant": 4, "org_unit": 3, "agent": 2, "user": 1}
    assert soul_priority("tenant") > soul_priority("org_unit") > \
        soul_priority("agent") > soul_priority("user")

    resolution = resolve(
        dimension="dream_seed",
        higher="tenant",
        lower="user",
        decision="tenant_value_wins",
        reason="tenant safety constraint",
    )
    ledger = SoulResolutionLedger(tmp_path / "soul.db")
    ledger.record("t1", resolution)
    events = ledger.list_events("t1")
    assert len(events) == 1
    assert events[0]["higher_scope"] == "tenant"
    assert events[0]["lower_scope"] == "user"

    # 低层不得覆盖高层：顺序反了直接拒绝
    with pytest.raises(ValueError):
        resolve(
            dimension="dream_seed",
            higher="user",
            lower="tenant",
            decision="user_wins",
        )


def test_tenant_isolation_of_dream_seeds():
    seeds = [
        {"seed_id": "a", "tenant_id": "t1", "soul_priority": "user"},
        {"seed_id": "b", "tenant_id": "t1", "soul_priority": "tenant"},
        {"seed_id": "c", "tenant_id": "t2", "soul_priority": "tenant"},
        {"seed_id": "d", "tenant_id": "t1"},
        {"seed_id": "e"},
    ]
    filtered = filter_dream_seeds(seeds, tenant_id="t1")
    assert {seed["seed_id"] for seed in filtered} == {"a", "b", "d"}
    assert filtered[0]["seed_id"] == "b", "高优先级种子先选"
    assert all(seed["tenant_id"] == "t1" for seed in filtered)


def test_soul_does_not_authorize():
    """Soul 只调制显著性/抽象权威，不提升证据权威、不授权动作."""
    authority = dream_authority("tenant")
    assert authority["evidential_authority"] <= 0.4
    assert authority["influence_authority"] <= 0.6
    assert authority["abstraction_authority"] <= 0.7
    import huginn.memory.soul as soul

    assert not hasattr(soul, "authorize")
