"""学习回路：轨迹索引、分类器建议、action mask、OPE 与影子零未授权."""
from __future__ import annotations

from huginn.core_types import RiskLevel
from huginn.normative.learning import (
    ActClassifier,
    AdmissionPolicy,
    AdmissionQuery,
    AdmissionSample,
    MemoryOpPolicy,
    TrajectoryIndex,
    ope_report,
    shadow_evaluate,
)


def _sample(**overrides) -> AdmissionSample:
    data = {
        "query": AdmissionQuery(intent="write file"),
        "context_features": {"authorized": True, "risk": "low"},
        "decision": {"mode": "allow"},
        "outcome": {"reward": 1.0},
        "tenant_id": "t1",
        "behaviour_prob": 0.5,
    }
    data.update(overrides)
    return AdmissionSample(**data)


def test_trajectory_index_tenant_partition():
    index = TrajectoryIndex(":memory:")
    index.record(_sample())
    index.record(_sample())
    index.record(_sample(tenant_id="t2"))
    assert index.count("t1") == 2
    assert index.count("t2") == 1
    records = index.list_samples("t1")
    assert len(records) == 2
    assert records[0]["payload"]["decision"]["mode"] == "allow"


def test_act_classifier_advisory_and_calibration():
    classifier = ActClassifier()
    authorized = classifier.classify("I authorize you to write workspace/x")
    assert authorized["act_type"] == "authorize"
    assert authorized["advisory_only"] is True
    assert classifier.classify("I reject approach A")["act_type"] == "reject"
    assert classifier.classify("I propose a new plan")["act_type"] == "propose"
    assert classifier.classify("hmm maybe")["confidence"] < 0.5

    labelled = [
        ("I authorize write", "authorize"),
        ("grant permission to write", "authorize"),
        ("I reject this", "reject"),
        ("I propose this", "propose"),
    ]
    accuracy = classifier.fit(labelled)
    assert accuracy["authorize"] == 1.0
    metrics = classifier.sensitivity_specificity(labelled, positive="authorize")
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_admission_policy_hard_mask():
    policy = AdmissionPolicy()
    for _ in range(100):
        assert policy.select(authorized=False) != "auto_allow"
        assert policy.select(authorized=True, revoked=True) != "auto_allow"
        assert policy.select(authorized=True, cross_tenant=True) == "deny"
    # 高风险即使有授权也不自动放行
    assert policy.select(authorized=True, risk=RiskLevel.HIGH) == "ask"

    # 训练后仍不能越过 mask
    for _ in range(20):
        policy.update("auto_allow", 1.0)
    assert policy.select(authorized=False) != "auto_allow"


def test_memory_op_policy_mask():
    policy = MemoryOpPolicy()
    for _ in range(100):
        assert policy.select(is_direct=True) != "discard"
    policy.update("discard", 1.0)
    assert policy.select(is_direct=True) != "discard", "direct 证据不可丢弃"


def test_ope_report_and_shadow_no_unauthorized():
    samples = [
        _sample(),
        _sample(),
        _sample(decision={"mode": "ask"}, outcome={"reward": 0.2},
                behaviour_prob=1.0),
    ]

    def target(sample: AdmissionSample) -> str:
        return sample.action()

    report = ope_report(samples, target)
    assert report["n"] == 3
    # (1/0.5 + 1/0.5 + 0.2/1.0) / 3
    assert report["ips"] == 1.4, report
    # (2*1 + 2*1 + 1*0.2) / (2 + 2 + 1)
    assert report["snips"] == 0.84, report
    assert report["unauthorized_rate"] == 0.0

    unauthorized_sample = _sample(
        context_features={"authorized": False, "risk": "low"},
        decision={"mode": "allow"},
        outcome={"unauthorized": True, "reward": 1.0},
    )
    bad_report = ope_report([unauthorized_sample], target)
    assert bad_report["unauthorized_rate"] == 1.0
    assert unauthorized_sample.reward() == 0.0, "未授权执行是硬失败"

    shadow = shadow_evaluate(
        [_sample(context_features={"authorized": False, "risk": "low"})],
        AdmissionPolicy(),
    )
    assert shadow["unauthorized_executions"] == 0
    assert shadow["decisions"] == ["ask"]


def test_policy_replay_deterministic():
    def _run() -> dict:
        policy = AdmissionPolicy(seed=3)
        for arm, reward in [("ask", 0.4), ("degrade", 0.6), ("ask", 0.8)]:
            policy.update(arm, reward)
        return policy.snapshot()

    assert _run() == _run()
