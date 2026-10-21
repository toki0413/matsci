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


def test_params_drive_runtime_thresholds():
    """I5/DoD 4: PersonaParams 真正驱动 plan_check 门槛，默认值保持旧行为."""
    default = PersonaParams()
    assert default.plan_accept_threshold() == 0.5
    assert default.plan_clarify_fails() == 3

    strict = PersonaParams(evidence_strictness=0.9, clarification_threshold=0.7)
    assert strict.plan_accept_threshold() > default.plan_accept_threshold()
    assert strict.plan_clarify_fails() < default.plan_clarify_fails()

    loose = PersonaParams(evidence_strictness=0.3, clarification_threshold=0.2)
    assert loose.plan_accept_threshold() < default.plan_accept_threshold()
    assert loose.plan_clarify_fails() > default.plan_clarify_fails()


def test_runtime_reads_persona_params_and_active_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huginn.persona_kernel as kernel
    from huginn.autoloop.engine_observe import EngineObserve

    store = PersonaRevisionStore(tmp_path / "rev.db")
    active = evaluate_revision(
        _revision(revision_id="rev-active", delta={"verbosity": 0.9}),
        cross_session_evidence=3, outcome_improved=True,
        ood_regressed=False, safety_regression=False,
    )
    store.save(active)
    store.activate("rev-active")
    monkeypatch.setattr(kernel, "get_persona_revision_store", lambda: store)

    persona = SimpleNamespace(
        variables={"persona_params": {"evidence_strictness": 0.8}}
    )

    class _PM:
        def get(self, name):
            return persona

    class _Engine:
        _last_persona = "p1"

        def _get_persona_manager(self):
            return _PM()

    params = EngineObserve(_Engine())._persona_params()
    assert params.evidence_strictness == 0.8
    assert params.verbosity == 0.9, "active revision 必须覆盖默认参数"


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
