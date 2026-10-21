"""I31: 案例策略更新不改模型权重，且可回放审计."""
from __future__ import annotations

from huginn.research.case_policy import Case, CasePolicy


def _cases() -> list[Case]:
    return [
        Case("c1", "q1", "retrieve_bm25", 0.8),
        Case("c2", "q2", "retrieve_bm25", 0.6),
        Case("c3", "q3", "retrieve_dense", 0.9),
        Case("c4", "q4", "retrieve_dense", 0.1),
    ]


def test_replayable_no_weight_update():
    policy = CasePolicy()
    for case in _cases():
        policy.update(case)
    snapshot = policy.snapshot()
    assert snapshot.to_dict()["model_weights"] is None, "不得更新模型权重"
    assert snapshot.version == 4
    assert snapshot.means["retrieve_bm25"] == 0.7
    assert snapshot.means["retrieve_dense"] == 0.5

    decision = policy.select("q5")
    assert decision == "retrieve_bm25"
    assert policy.select("q6") == "retrieve_bm25", "确定性选择"

    replayed = CasePolicy()
    replayed.replay(policy.export_events())
    assert replayed.snapshot().means == snapshot.means
    assert replayed.snapshot().version == snapshot.version
    assert replayed.select("q7") == decision


def test_no_examples_returns_none():
    assert CasePolicy().select("q") is None
