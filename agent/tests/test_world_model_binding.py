"""世界模型 binding 测试：预测不直接成事实、对账升格、worldview 隔离、漂移信号."""
from __future__ import annotations

import pytest

from huginn.research.world_model_binding import (
    WorldModelPrediction,
    apply_revision,
    authority_for,
    cross_glue_labels,
    parse_world_labels,
    prediction_error_signal,
    promotion_status,
    propose_revision,
    reconcile_prediction,
    to_pmk_stance,
    worldview_compatible,
)


def _prediction(**overrides) -> WorldModelPrediction:
    data = {
        "prediction_id": "p1",
        "world_model_id": "law.exoplanet",
        "worldview": "physics_causal",
        "domain": "exoplanet",
        "contract_version": "v1",
        "predicted": {"T_eq_K": 200.0, "S_Wm2": 1000.0},
    }
    data.update(overrides)
    return WorldModelPrediction(**data)


def test_prediction_defaults_to_low_evidential_authority():
    authority = authority_for(None)
    assert authority.evidential_authority <= 0.2
    assert authority.influence_authority <= 0.4


def test_reconcile_requires_real_actual_source():
    prediction = _prediction()
    with pytest.raises(ValueError):
        reconcile_prediction(prediction, {"T_eq_K": 200.0}, actual_source="")
    with pytest.raises(ValueError):
        reconcile_prediction(
            prediction, {"T_eq_K": 200.0}, actual_source="synthetic"
        )


def test_reconcile_three_states_and_promotion():
    prediction = _prediction()
    ok = reconcile_prediction(
        prediction,
        {"T_eq_K": 200.0, "S_Wm2": 1000.0},
        actual_source="tool_run:1",
    )
    assert ok.status == "borne_out"
    assert promotion_status(ok) == "reconciled"
    assert authority_for(ok).evidential_authority == 0.6
    assert authority_for(ok, independent_runs=1).evidential_authority == 0.9

    bad = reconcile_prediction(
        prediction,
        {"T_eq_K": 300.0},
        actual_source="tool_run:1",
    )
    assert bad.status == "falsified"
    assert promotion_status(bad) == "refuted"
    assert authority_for(bad).evidential_authority == 0.1

    insufficient = reconcile_prediction(
        prediction,
        {},
        actual_source="tool_run:1",
    )
    assert insufficient.status == "insufficient"
    assert promotion_status(insufficient) == "insufficient"


def test_worldview_must_match():
    a = _prediction()
    b = _prediction(prediction_id="p2", worldview="latent_transition")
    assert not worldview_compatible(a, b)
    assert worldview_compatible(a, _prediction(prediction_id="p3"))


def test_prediction_error_signal_levels():
    prediction = _prediction()
    record = reconcile_prediction(
        prediction,
        {"T_eq_K": 200.0},
        actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    signal = prediction_error_signal(record)
    assert signal["detector"] == "prediction_error"
    assert signal["level"] == "ok"

    bad = reconcile_prediction(
        prediction,
        {"T_eq_K": 240.0},
        actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    signal_bad = prediction_error_signal(bad)
    assert signal_bad["level"] in ("review", "intervene")


def test_prediction_not_direct():
    """I41: 预测默认 hypothesis/derived，未经 reconcile 不得作为 direct 证据."""
    assert authority_for(None).evidential_authority <= 0.2
    prediction = _prediction()
    insufficient = reconcile_prediction(
        prediction, {}, actual_source="tool_run:1"
    )
    assert insufficient.status == "insufficient"
    assert authority_for(insufficient).evidential_authority <= 0.1
    reconciled = reconcile_prediction(
        prediction,
        {"T_eq_K": 200.0, "S_Wm2": 1000.0},
        actual_source="tool_run:1",
    )
    assert authority_for(reconciled).evidential_authority == 0.6


def test_required_labels():
    """I42: 预测必须带 worldview/domain/contract；缺失即 advisory/quarantine."""
    from huginn.runtime.task_lifecycle import _check_pmk_axes

    stance = to_pmk_stance(_prediction())
    assert stance["worldview"] and stance["domain"] and stance["contract_version"]
    axes = _check_pmk_axes({"world_model": "predicted T_eq_K=200"})
    assert "missing labels" in axes.advisory
    assert not axes.blocked


def test_no_authority():
    """I44: 世界模型不得授权动作；authorize 只能由规范账写."""
    import huginn.research.world_model_binding as binding

    assert not hasattr(binding, "authorize")
    assert not hasattr(binding, "append_act")
    record = reconcile_prediction(
        _prediction(),
        {"T_eq_K": 200.0, "S_Wm2": 1000.0},
        actual_source="tool_run:1",
    )
    assert not hasattr(record, "scope"), "对账记录不携带授权 scope"


def test_revision_gate():
    """I47: 世界模型修订必须过采纳门；监控/梦境不得直接改模型."""
    revision = propose_revision(
        world_model_id="law.exoplanet", delta={"albedo": 0.2},
        evidence_ids=("run:1",),
    )
    for actor in ("monitor", "dream", "agent"):
        rejected = apply_revision(
            revision, actor=actor, adoption_green=True, independent_runs=1
        )
        assert rejected.status == "rejected", actor

    pending = apply_revision(
        revision, actor="meta_improver", adoption_green=False, independent_runs=1
    )
    assert pending.status == "candidate"
    active = apply_revision(
        revision, actor="meta_improver", adoption_green=True, independent_runs=1
    )
    assert active.status == "active"


def test_availability_degrade(tmp_path):
    """I48: 世界模型不可用只降预测能力，不阻断任务."""
    from huginn.core_types import RiskLevel
    from huginn.normative.admission import AdmissionRequest, admit
    from huginn.normative.ledger import NormativeLedger

    prediction = _prediction()
    record = reconcile_prediction(prediction, {}, actual_source="tool_run:1")
    assert promotion_status(record) == "insufficient"

    ledger = NormativeLedger(tmp_path / "norm.db")
    scope = {"actions": ["write"], "resources": ["*"], "limits": {}}
    ledger.append_act(
        tenant_id="t1", session_id="s1", actor="user",
        act_type="authorize", proposition_id="p1", scope=scope,
    )
    decision = admit(
        AdmissionRequest(
            tenant_id="t1", session_id="s1", action_name="write",
            risk=RiskLevel.MEDIUM,
        ),
        ledger.project("t1", "s1"),
    )
    assert decision.allow is True, "没有世界模型也必须能执行已授权任务"


def test_world_model_findings_and_diagnosis():
    """§20.8: reconcile 失败转 Dreamer finding，并归因到 law/contract/stale."""
    from huginn.research.world_model_binding import (
        diagnose_world_model_failure,
        findings_from_reconciles,
    )

    prediction = _prediction()
    falsified = reconcile_prediction(
        prediction, {"T_eq_K": 400.0}, actual_source="tool_run:1"
    )
    insufficient = reconcile_prediction(
        prediction, {}, actual_source="tool_run:1"
    )
    borne_out = reconcile_prediction(
        prediction, {"T_eq_K": 200.0, "S_Wm2": 1000.0},
        actual_source="tool_run:1",
    )

    findings = findings_from_reconciles([falsified, insufficient, borne_out])
    assert len(findings) == 2
    assert {f.status for f in findings} == {"falsified", "insufficient"}
    assert findings[0].evidence_ids[1] == "tool_run:1"

    assert diagnose_world_model_failure(falsified)["cause_class"] == "law_mismatch"
    assert diagnose_world_model_failure(insufficient)["cause_class"] == "stale_model"
    assert diagnose_world_model_failure(
        falsified, contract_compatible=False
    )["cause_class"] == "contract_violation"
    none = diagnose_world_model_failure(borne_out)
    assert none["cause_class"] == "none" and none["abstain"] is True


def test_cross_glue_worldview_mismatch_is_not_contradiction():
    """跨 worldview 的预测/证据不粘合，记 worldview_mismatch，不升级矛盾."""
    wm = "world_model=x worldview=physics_causal domain=exoplanet contract=v1"
    obs = cross_glue_labels(
        wm,
        {"kb": "T_eq=200 [worldview=latent_transition domain=exoplanet contract=v1]"},
    )
    assert len(obs) == 1, obs
    assert obs[0]["kind"] == "worldview_mismatch"
    assert "not a contradiction" in obs[0]["reason"]


def test_cross_glue_contract_violation_same_worldview():
    wm = "world_model=x worldview=physics_causal domain=exoplanet contract=v1"
    obs = cross_glue_labels(
        wm,
        {
            "kb": "T_eq=200 [worldview=physics_causal domain=mechanics contract=v1]",
            "memory": "T_eq=210 [worldview=physics_causal domain=exoplanet contract=v2]",
        },
    )
    assert {o["kind"] for o in obs} == {"contract_violation"}
    assert {o["source"] for o in obs} == {"kb", "memory"}


def test_cross_glue_skips_unlabeled_evidence():
    """无标签 = 未检查，不是通过，也不是 mismatch."""
    wm = "world_model=x worldview=physics_causal domain=exoplanet contract=v1"
    assert cross_glue_labels(wm, {"kb": "plain unlabeled text"}) == []
    assert parse_world_labels("plain text") == {}
    assert cross_glue_labels("plain stalk", {"kb": wm}) == []


def test_pmk_stance_is_labeled_not_fact():
    stance = to_pmk_stance(_prediction())
    assert stance["source"] == "world_model"
    assert stance["worldview"] == "physics_causal"
    assert stance["domain"] == "exoplanet"
    assert "physics_causal" in stance["text"]
