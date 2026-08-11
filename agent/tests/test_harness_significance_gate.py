"""Unit tests for huginn/harness/significance_gate.py (H5).

测试覆盖:
1. 配对记录 + 持久化
2. 样本不足 → 不通过
3. 全正差值 → 通过 (exact test)
4. 全负差值 → 不通过
5. 混合差值 → Wilcoxon
6. 持久化 reload
7. passes_gate 快捷方法
8. clear
"""
from __future__ import annotations

import pytest

from huginn.harness.significance_gate import (
    ScorePair,
    SignificanceGate,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """每个测试用独立 cache 目录 + 新单例."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    SignificanceGate._instance = None
    yield
    SignificanceGate._instance = None


def test_insufficient_samples():
    """样本不足时不通过, reason 含 'insufficient'."""
    gate = SignificanceGate.get_instance()
    gate.record_pair("cfg_a", 0.5, 0.6, "t1")
    gate.record_pair("cfg_a", 0.5, 0.6, "t2")
    d = gate.gate_decision("cfg_a")
    assert not d.passed
    assert "insufficient" in d.reason
    assert d.n_samples == 2


def test_all_positive_diffs_pass():
    """全正差值 → 通过, p_value=0.0."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_b", 0.4, 0.7, f"t{i}")
    d = gate.gate_decision("cfg_b")
    assert d.passed
    assert d.p_value == 0.0
    assert d.median_diff > 0


def test_all_negative_diffs_fail():
    """全负差值 → 不通过, p_value=1.0."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_c", 0.7, 0.4, f"t{i}")
    d = gate.gate_decision("cfg_c")
    assert not d.passed
    assert d.p_value == 1.0


def test_wilcoxon_mixed_diffs():
    """混合差值 → Wilcoxon 检验, 7正1负应该通过.

    关键: 负差值的绝对值必须最小 (秩=1), 否则 n=8 时 W- 过大,
    p > 0.05 不显著. 之前用 (0.9, 0.5) → diff=-0.4, |diff| 最大,
    秩=8, W-=8, p=0.1016 >= 0.05, 测试失败.
    """
    gate = SignificanceGate.get_instance()
    pairs = [
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.2, 0.5),
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.5, 0.48),
    ]
    for i, (b, c) in enumerate(pairs):
        gate.record_pair("cfg_d", b, c, f"t{i}")
    d = gate.gate_decision("cfg_d")
    assert d.p_value is not None
    assert d.passed, f"7-positive-1-negative should pass: {d}"


def test_wilcoxon_noisy_diffs_fail():
    """5正3负 → 不应该通过 (噪声不够显著)."""
    gate = SignificanceGate.get_instance()
    pairs = [
        (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6),
        (0.5, 0.4), (0.5, 0.4), (0.5, 0.4),
    ]
    for i, (b, c) in enumerate(pairs):
        gate.record_pair("cfg_e", b, c, f"t{i}")
    d = gate.gate_decision("cfg_e", alpha=0.05)
    assert d.p_value is not None
    # 5正3负在 n=8 时 p ≈ 0.36, 不显著
    assert not d.passed, f"5-positive-3-negative should not pass: {d}"


def test_persistence_reload(tmp_path):
    """持久化 reload: 数据跨 session 保留."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_persist", 0.3, 0.6, f"t{i}")
    assert len(gate.get_pairs("cfg_persist")) == 5

    SignificanceGate._instance = None
    gate2 = SignificanceGate.get_instance()
    pairs = gate2.get_pairs("cfg_persist")
    assert len(pairs) == 5
    assert all(p.candidate_score == 0.6 for p in pairs)


def test_passes_gate_shortcut():
    """passes_gate 返回 bool, 不是 GateDecision."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_f", 0.3, 0.7, f"t{i}")
    assert gate.passes_gate("cfg_f") is True
    assert gate.passes_gate("nonexistent") is False


def test_clear():
    """clear 清除指定 config 的数据."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_g", 0.3, 0.7, f"t{i}")
    assert len(gate.get_pairs("cfg_g")) == 5
    gate.clear("cfg_g")
    assert len(gate.get_pairs("cfg_g")) == 0


def test_all_zero_diffs():
    """所有差值为零 → 不通过, reason 含 'zero'."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_h", 0.5, 0.5, f"t{i}")
    d = gate.gate_decision("cfg_h")
    assert not d.passed
    assert "zero" in d.reason.lower()


def test_custom_alpha_and_min_samples():
    """自定义 alpha 和 min_samples."""
    gate = SignificanceGate.get_instance()
    for i in range(3):
        gate.record_pair("cfg_i", 0.3, 0.7, f"t{i}")
    # min_samples=3 → 够了
    d = gate.gate_decision("cfg_i", min_samples=3)
    assert d.passed, f"3 samples with min=3 should pass: {d}"
    # min_samples=5 → 不够
    d2 = gate.gate_decision("cfg_i", min_samples=5)
    assert not d2.passed
    assert "insufficient" in d2.reason


def test_gate_decision_to_dict():
    """GateDecision.to_dict 序列化正常."""
    gate = SignificanceGate.get_instance()
    for i in range(5):
        gate.record_pair("cfg_j", 0.3, 0.7, f"t{i}")
    d = gate.gate_decision("cfg_j")
    d_dict = d.to_dict()
    assert d_dict["config_id"] == "cfg_j"
    assert d_dict["passed"] is True
    assert "p_value" in d_dict
    assert "reason" in d_dict


def test_score_pair_diff():
    """ScorePair.diff 属性."""
    p = ScorePair(baseline_score=0.3, candidate_score=0.7)
    assert abs(p.diff - 0.4) < 1e-9
    p2 = ScorePair(baseline_score=0.8, candidate_score=0.5)
    assert abs(p2.diff - (-0.3)) < 1e-9


def test_summary():
    """summary 返回统计信息."""
    gate = SignificanceGate.get_instance()
    for i in range(3):
        gate.record_pair("cfg_k", 0.3, 0.7, f"t{i}")
    for i in range(2):
        gate.record_pair("cfg_l", 0.3, 0.7, f"t{i}")
    s = gate.summary()
    assert s["total_configs"] == 2
    assert s["configs"]["cfg_k"] == 3
    assert s["configs"]["cfg_l"] == 2
