"""I21/I23: 梦境产物不覆盖 source；晋升需 provenance + 独立佐证 + 无矛盾."""
from __future__ import annotations

from huginn.memory.dream import (
    DreamObject,
    promote_dream_object,
)


def _obj(**overrides) -> DreamObject:
    data = {
        "dream_id": "d1",
        "persona_id": "p1",
        "sleep_window": ("a", "b"),
        "seed_event_ids": ("e1",),
        "provenance": ("e1",),
        "status": "candidate",
    }
    data.update(overrides)
    return DreamObject(**data)


def test_no_authorize_no_kernel():
    """I15: derived/hypothesis 不得授权动作，不得修改 PersonaKernel."""
    from huginn.persona_kernel import (
        PersonaKernel,
        PersonaRevision,
        evaluate_revision,
    )

    kernel = PersonaKernel(
        persona_id="p1",
        role="scientist",
        safety_constraints=("no_unsafe_write",),
        authorization_boundaries=("workspace/*",),
    )
    digest = kernel.core_digest

    # 梦境提出的"人格修订"若试图改安全/授权边界 → 直接 rejected
    dream_revision = PersonaRevision(
        revision_id="dream-rev",
        persona_id="p1",
        base_version=1,
        delta={"safety_constraints": ("allow_all",), "influence_authority": 0.99},
        evidence_ids=("dream:1", "dream:2", "dream:3"),
    )
    evaluated = evaluate_revision(
        dream_revision,
        cross_session_evidence=10,
        outcome_improved=True,
        ood_regressed=False,
        safety_regression=False,
    )
    assert evaluated.status == "rejected"
    assert kernel.core_digest == digest, "内核不可变"

    # 梦境产物即使晋升，权威上限也不足以授权动作
    obj = promote_dream_object(
        _obj(), corroboration=3, provenance_complete=True
    )
    assert obj.status == "promoted"
    assert obj.influence_authority <= 0.6
    assert not hasattr(obj, "authorize")


def test_no_overwrite():
    """I21: 晋升返回新对象，原 source/dream 对象不被覆盖."""
    original = _obj()
    promoted = promote_dream_object(
        original, corroboration=2, provenance_complete=True
    )
    assert original.status == "candidate", "不得原地覆盖"
    assert promoted.status == "promoted"
    assert promoted.dream_id == original.dream_id
    assert promoted.evidential_authority <= 0.6
    assert promoted.influence_authority <= 0.6


def test_promotion_preconditions():
    """I23: provenance 链完整 + ≥1 独立佐证；未解矛盾一律 quarantine."""
    assert promote_dream_object(_obj()).status == "candidate"
    assert promote_dream_object(
        _obj(), corroboration=3, provenance_complete=False
    ).status == "candidate"
    assert promote_dream_object(
        _obj(), corroboration=0, provenance_complete=True
    ).status == "candidate"

    promoted = promote_dream_object(
        _obj(), corroboration=1, provenance_complete=True
    )
    assert promoted.status == "promoted"

    quarantined = promote_dream_object(
        _obj(), corroboration=5, provenance_complete=True,
        has_unresolved_contradiction=True,
    )
    assert quarantined.status == "quarantined"
    assert quarantined.influence_authority == 0.0
    assert quarantined.evidential_authority <= 0.1
