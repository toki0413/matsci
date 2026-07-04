"""Belief Entropy 模块的测试.

覆盖三个维度的计算 (R_loss / H_logprob / summary 估计) 以及自适应参数、
历史追踪、趋势判断和单例行为. 所有测试都关闭事实检查, 避免调 LLM.
"""

from __future__ import annotations

import math

import pytest

from huginn.utils.belief_entropy import (
    BeliefEntropy,
    BeliefEntropyConfig,
    get_belief_entropy,
)
import huginn.utils.belief_entropy as be_mod


@pytest.fixture()
def be() -> BeliefEntropy:
    """干净的 BeliefEntropy 实例, 关掉事实检查省调用."""
    return BeliefEntropy(BeliefEntropyConfig(fact_check_enabled=False))


# ── R_loss: 压缩比损失 ───────────────────────────────────────


def test_ratio_loss(be: BeliefEntropy) -> None:
    # 5000 -> 500, 压缩到 1/10, 损失 0.9
    assert be._compute_ratio_loss(5000, 500) == pytest.approx(0.9)


def test_ratio_loss_zero(be: BeliefEntropy) -> None:
    # 原始 0 token, 没法算比例, 直接返回 0
    assert be._compute_ratio_loss(0, 0) == 0.0
    assert be._compute_ratio_loss(0, 100) == 0.0


# ── H_logprob: token 级 Shannon 熵 ──────────────────────────


def test_logprob_entropy_uniform(be: BeliefEntropy) -> None:
    # top-k=5, 每个 token 概率相等 -> 最大不确定 -> 归一化熵 ~1.0
    logprobs = [[{"logprob": 0.0}] for _ in range(5)]
    logprobs = [[{"logprob": 0.0}, {"logprob": 0.0}, {"logprob": 0.0},
                 {"logprob": 0.0}, {"logprob": 0.0}]]
    entropy = be._compute_logprob_entropy(logprobs)
    assert entropy == pytest.approx(1.0, abs=0.01)


def test_logprob_entropy_certain(be: BeliefEntropy) -> None:
    # 一个 token logprob=0 (概率 1.0), 其余接近 0 -> 熵 ~0
    logprobs = [[
        {"logprob": 0.0},
        {"logprob": -100.0},
        {"logprob": -100.0},
        {"logprob": -100.0},
        {"logprob": -100.0},
    ]]
    entropy = be._compute_logprob_entropy(logprobs)
    assert entropy == pytest.approx(0.0, abs=0.05)


# ── summary 粗估 ────────────────────────────────────────────


def test_estimate_entropy_from_summary_dense(be: BeliefEntropy) -> None:
    # 信息密度高: 数字、术语都有, 模型对状态很清楚 -> 低熵
    summary = "DFT calculation with ENCUT=520 eV, k-points 4x4x4, energy=-1.23 eV"
    entropy = be._estimate_entropy_from_summary(summary)
    assert entropy < 0.5


def test_estimate_entropy_from_summary_sparse(be: BeliefEntropy) -> None:
    # 泛泛而谈, 没有具体信息 -> 高熵
    summary = "The user discussed various topics related to materials science and computation"
    entropy = be._estimate_entropy_from_summary(summary)
    assert entropy > 0.5


# ── 自适应参数 ─────────────────────────────────────────────


def test_adaptive_params_low(be: BeliefEntropy) -> None:
    # 低熵: 模型很清楚, 可以更激进地压缩
    keep_last_n, _ = be._adaptive_params(0.2)
    assert keep_last_n == -1


def test_adaptive_params_high(be: BeliefEntropy) -> None:
    # 高熵: 模型很迷糊, 保守一点, 多留几条消息
    keep_last_n, _ = be._adaptive_params(0.8)
    assert keep_last_n == 2


def test_adaptive_params_mid(be: BeliefEntropy) -> None:
    # 中间区间: 不动
    keep_last_n, _ = be._adaptive_params(0.5)
    assert keep_last_n is None


# ── measure 整体流程 ───────────────────────────────────────


def test_measure_combined(be: BeliefEntropy) -> None:
    # 完整调用, h_belief 应该落在 [0, 1]
    result = be.measure(
        summary="DFT calculation with ENCUT=520 eV",
        original_tokens=5000,
        compressed_tokens=500,
    )
    assert 0.0 <= result.h_belief <= 1.0
    # r_loss 应该算出来了
    assert result.r_loss == pytest.approx(0.9)
    # 没开事实检查, c_fact 默认 1.0
    assert result.c_fact == 1.0


def test_history_tracking(be: BeliefEntropy) -> None:
    # 多次调用, history 应该全部记下来
    be.measure(summary="DFT calculation", original_tokens=100, compressed_tokens=50)
    be.measure(summary="VASP relaxation", original_tokens=200, compressed_tokens=100)
    be.measure(summary="band gap analysis", original_tokens=300, compressed_tokens=150)
    history = be.get_history()
    assert len(history) == 3
    assert all(0.0 <= h <= 1.0 for h in history)


def test_trend(be: BeliefEntropy) -> None:
    # 模拟多次调用, 熵值递增 (模型越来越迷糊)
    for h in [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]:
        be._history.append(h)
    trend = be.get_trend()
    # 近期均值 > 早期均值, trend 为正
    assert trend > 0


# ── 单例 ───────────────────────────────────────────────────


def test_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    # 重置单例, 确保拿到的是全新实例
    monkeypatch.setattr(be_mod, "_singleton", None)
    a = get_belief_entropy()
    b = get_belief_entropy()
    assert a is b
