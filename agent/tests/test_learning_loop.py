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
    build_ope_report,
    ope_report,
    sample_from_payload,
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


def test_shadow_trajectory_sampling(monkeypatch, tmp_path):
    """真实工具准入可采样成轨迹，并在执行后回填 outcome（默认关闭）."""
    import huginn.normative.shadow as shadow
    from huginn.normative.admission import AdmissionDecision

    index = TrajectoryIndex(tmp_path / "traj.db")
    monkeypatch.setattr(shadow, "_trajectory", index)
    monkeypatch.setenv("HUGINN_NORMATIVE_TRAJECTORY", "true")

    decision = AdmissionDecision(allow=True, mode="allow")
    sample_id = shadow.record_admission_sample(
        tenant_id="t1",
        session_id="s1",
        tool_name="vasp_tool",
        decision=decision,
        risk=RiskLevel.LOW,
    )
    assert sample_id
    assert index.count("t1") == 1
    assert shadow.annotate_admission_sample(
        sample_id, outcome={"success": True, "reward": 1.0}
    )
    payload = index.list_samples("t1")[0]["payload"]
    assert payload["context_features"]["authorized"] is True
    assert payload["outcome"]["success"] is True

    monkeypatch.setenv("HUGINN_NORMATIVE_TRAJECTORY", "false")
    assert shadow.record_admission_sample(
        tenant_id="t1",
        session_id="s1",
        tool_name="vasp_tool",
        decision=decision,
        risk=RiskLevel.LOW,
    ) == ""


def test_build_ope_report_time_split_and_shadow():
    """DoD 7: 离线 OPE 报告按时间切分 train/eval，影子模式零未授权."""
    samples = []
    for i in range(10):
        authorized = i % 2 == 0
        samples.append(
            _sample(
                context_features={"authorized": authorized, "risk": "low"},
                decision={"mode": "allow" if authorized else "ask"},
                outcome={"reward": 1.0 if authorized else 0.3},
                ts=f"2026-01-01T00:00:{i:02d}+00:00",
            )
        )
    report = build_ope_report(samples, train_ratio=0.7)
    assert report["train_n"] == 7
    assert report["eval_n"] == 3
    assert report["eval_n"] > 0
    assert report["shadow"]["unauthorized_executions"] == 0
    assert report["note"].startswith("offline OPE")


def test_cli_ope_report_roundtrip(tmp_path):
    """CLI 能从轨迹库生成 JSON 报告."""
    import json

    from huginn.cli.ope_report import main

    index = TrajectoryIndex(tmp_path / "traj.db")
    for i in range(5):
        index.record(
            _sample(
                ts=f"2026-01-01T00:00:{i:02d}+00:00",
                decision={"mode": "allow"},
                outcome={"reward": 1.0},
            )
        )
    out = tmp_path / "report.json"
    assert main([
        "--tenant", "t1", "--db", str(tmp_path / "traj.db"),
        "--out", str(out),
    ]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["tenant_id"] == "t1"
    assert report["total_samples"] == 5
    assert report["shadow"]["unauthorized_executions"] == 0


def test_sample_payload_roundtrip():
    sample = _sample()
    payload = {
        "query": {"intent": "write", "goal": "s1", "persona_params": {}},
        "context_features": sample.context_features,
        "decision": sample.decision,
        "outcome": sample.outcome,
        "behaviour_prob": 0.5,
        "ts": sample.ts,
    }
    restored = sample_from_payload(payload, tenant_id="t1")
    assert restored is not None
    assert restored.action() == "auto_allow"
    assert restored.reward() == 1.0


def test_policy_replay_deterministic():
    def _run() -> dict:
        policy = AdmissionPolicy(seed=3)
        for arm, reward in [("ask", 0.4), ("degrade", 0.6), ("ask", 0.8)]:
            policy.update(arm, reward)
        return policy.snapshot()

    assert _run() == _run()
