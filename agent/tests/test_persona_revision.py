"""I5: PersonaKernel 不可变；active revision 需 ≥N 证据 + 采纳门，且可回滚."""
from __future__ import annotations

import dataclasses

import pytest

from huginn.persona_kernel import (
    PersonaEvidenceStore,
    PersonaKernel,
    PersonaParams,
    PersonaRevision,
    PersonaRevisionStore,
    evaluate_revision,
    render_adaptive_layer,
)


def _revision(delta=None, **overrides) -> PersonaRevision:
    data = {
        "revision_id": "rev1",
        "persona_id": "p1",
        "base_version": 1,
        "delta": delta or {"skepticism": 0.7},
        "evidence_ids": ("e1", "e2", "e3"),
    }
    data.update(overrides)
    return PersonaRevision(**data)


def test_kernel_immutable():
    kernel = PersonaKernel(
        persona_id="p1", role="scientist",
        safety_constraints=("no_unsafe_write",),
        authorization_boundaries=("workspace/*",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        kernel.role = "hacker"  # type: ignore[misc]
    assert kernel.core_digest


def test_revision_requires_evidence_and_gate():
    revision = _revision()
    pending = evaluate_revision(
        revision, cross_session_evidence=2, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    assert pending.status == "candidate"

    active = evaluate_revision(
        revision, cross_session_evidence=3, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    assert active.status == "active"
    assert active.eval_report["cross_session_evidence"] == 3

    rejected = evaluate_revision(
        revision, cross_session_evidence=5, outcome_improved=True,
        ood_regressed=True, safety_regression=False,
    )
    assert rejected.status == "rejected"

    unsafe = evaluate_revision(
        _revision(delta={"safety_constraints": ("none",)}),
        cross_session_evidence=5, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    assert unsafe.status == "rejected", "revision 不得触碰安全/授权边界"


def test_active_revision_and_rollback(tmp_path):
    store = PersonaRevisionStore(tmp_path / "rev.db")
    rev1 = evaluate_revision(
        _revision(revision_id="rev1"),
        cross_session_evidence=3, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    rev2 = evaluate_revision(
        _revision(revision_id="rev2", base_version=2, delta={"verbosity": 0.8}),
        cross_session_evidence=4, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    store.save(rev1)
    store.activate("rev1")
    assert store.active("p1")["revision_id"] == "rev1"

    store.save(rev2)
    store.activate("rev2")
    assert store.active("p1")["revision_id"] == "rev2"
    assert store.get("rev1")["status"] == "superseded", "旧 revision 必须保留"

    assert store.rollback("p1", "rev1") is True
    assert store.active("p1")["revision_id"] == "rev1"
    assert store.get("rev1")["status"] == "active"


def test_adaptive_layer_is_readonly_projection(tmp_path):
    store = PersonaEvidenceStore(tmp_path / "evidence.db")
    store.append("p1", "iter 1: memory says A")
    store.append("p1", "iter 2: kb says B")
    evidence = store.recent("p1")
    params = PersonaParams(skepticism=0.7)
    first = render_adaptive_layer(params, evidence)
    second = render_adaptive_layer(params, store.recent("p1"))
    assert first == second, "同 params + 同证据必须投影出同一文本"
    assert "iter 1" in first and "iter 2" in first
    assert "skepticism=0.70" in first
    # 再 append 一次不会把旧渲染结果当前缀累积
    store.append("p1", "iter 3: memory says C")
    third = render_adaptive_layer(params, store.recent("p1"))
    assert third.count("[persona params]") == 1
    assert "iter 1: memory says A" in third
