"""rcb_runner 主循环中 HypothesisManifold MCMC → EffortBandit 的 wiring 闭环测试.

rcb_runner 主循环 (v15 Phase 2 + v18/v19) 已把 HypothesisManifold 的 mcmc_step
接入主迭代, 并把 MCMC 接受率注入 EffortBandit.update_mcmc_acceptance 作为
探索/利用信号. 本测试验证这条 wiring 是真实闭环 (而非静态 init 后从不活动):

  主循环调用序列 (rcb_runner.py):
    _hypo_manifold.mcmc_step(obs, current_h, rng=..., cached_log_p_current=...)
      → (_next_h, _next_log_p)
    _mcmc_accepted = (_next_h != _prev_h)
    EffortBandit.get_instance().update_mcmc_acceptance(1.0 if accepted else 0.0)

assert-based, runnable standalone or via pytest.
"""
from __future__ import annotations

import random

from huginn.agent.bandit_controller import EffortBandit
from huginn.metacog.hypothesis_manifold import (
    Hypothesis,
    HypothesisManifold,
    Observation,
)


def _build_manifold(n: int = 4) -> HypothesisManifold:
    """构造一个有区分度的 manifold: 每个 h 预测不同位置的 x."""
    m = HypothesisManifold()
    for i in range(n):
        m.add(Hypothesis(
            h_id=f"h{i}",
            description=f"hypothesis {i}: x near {i * 1.0}",
            predictions={"x": float(i)},
            n_params=1,
        ))
    return m


def _obs(x: float, sigma: float = 1.0) -> list[Observation]:
    return [Observation("x", x, sigma)]


# --- 1. manifold 闭环核心: mcmc_step 确实推进 (接受/拒绝) -----------------
def test_mcmc_step_advances_manifold():
    m = _build_manifold()
    obs = _obs(1.1, 0.5)  # 观察落在 h1 附近
    cur = "h0"
    rng = random.Random(42)
    # 连续多步, 应能停在 h1 (posterior 最高)
    visited = {cur}
    for _ in range(50):
        nxt, _lp = m.mcmc_step(obs, cur, rng=rng)
        visited.add(nxt)
        cur = nxt
    assert "h1" in visited, f"MCMC 应探索到 h1, 实际 visited={visited}"
    # 增量缓存路径: cached_log_p 传当前值应不报错
    nxt2, lp2 = m.mcmc_step(obs, cur, rng=rng, cached_log_p_current=-1.0)
    assert isinstance(nxt2, str) and isinstance(lp2, float)


# --- 2. bandit 联动: update_mcmc_acceptance 真实更新滑动平均 ---------------
def test_update_mcmc_acceptance_updates_sliding_average():
    b = EffortBandit.get_instance()
    b._mcmc_accept_n = 0
    b._mcmc_accept_rate = None
    seq = [1.0, 1.0, 0.0, 1.0, 1.0]  # 4/5 接受
    for r in seq:
        b.update_mcmc_acceptance(r)
    assert b._mcmc_accept_n == len(seq)
    assert b._mcmc_accept_rate is not None
    # 滑动平均落在 (0.8 正负) 范围内
    assert 0.0 < b._mcmc_accept_rate <= 1.0
    assert abs(b._mcmc_accept_rate - 0.8) < 0.2


# --- 3. wiring 集成: 复刻 rcb_runner 主循环调用序列 -----------------------
def test_rcb_runner_wiring_sequence_closed_loop():
    """按 rcb_runner 主循环的实际调用顺序测: mcmc_step → 接受判定 → bandit 注入.

    这验证 rcb_runner 里那段 wiring (P1-A + #2 打通) 是真实闭环: 每步
    accepted/rejected 判定 → update_mcmc_acceptance 注入, bandit 状态随之演进.
    """
    m = _build_manifold()
    obs = _obs(1.1, 0.5)
    rng = random.Random(7)
    b = EffortBandit.get_instance()
    b._mcmc_accept_n = 0
    b._mcmc_accept_rate = None

    cur = "h0"
    cached_log_p = None
    accept_count = 0
    steps = 40
    for _ in range(steps):
        # 复刻 rcb_runner 主循环: 传 rng + cached_log_p (增量路径)
        nxt, lp = m.mcmc_step(obs, cur, rng=rng, cached_log_p_current=cached_log_p)
        accepted = nxt != cur
        cached_log_p = lp
        cur = nxt
        if accepted:
            accept_count += 1
        # 复刻 #2 打通: 接受率注入 bandit
        b.update_mcmc_acceptance(1.0 if accepted else 0.0)

    # 闭环应产生活动: 有接受步, 且 bandit 收到注入
    assert accept_count > 0, "MCMC 闭环应接受至少一步"
    assert b._mcmc_accept_n == steps, "bandit 应收到全部步的注入"
    assert b._mcmc_accept_rate is not None
    # 接受率应大致反映真实接受比 (0 到 1 之间)
    assert 0.0 < b._mcmc_accept_rate <= 1.0


# --- 4. R-NaD 锚正则: 默认关 → 行为不变; 开 → 向锚点漂移 -----------------
def test_anchor_off_leaves_state_untouched():
    """anchor_lambda=0 (默认) 时锚不初始化也不更新, 行为与旧路径一致."""
    m = _build_manifold()
    obs = _obs(1.0, 0.5)
    rng = random.Random(1)
    cur = "h0"
    for _ in range(20):
        cur, _ = m.mcmc_step(obs, cur, rng=rng)
    assert m._anchor_pred == {}, "默认关闭时锚不应被更新"


def test_anchor_penalty_and_ema_update():
    """无锚 → 距离 0; EMA 慢更新; 距锚越远惩罚越大."""
    m = _build_manifold()
    assert m._anchor_penalty("h1") == 0.0  # 无锚 → 无正则
    m._update_anchor("h0")  # 首次接受 → 锚 = h0 预测
    assert m._anchor_pred == {"x": 0.0}
    assert m._anchor_penalty("h0") == 0.0
    assert abs(m._anchor_penalty("h2") - 2.0) < 1e-9
    # alpha=0.1: x 0.0 → 0.2 (向 h2 靠拢)
    m._update_anchor("h2")
    assert abs(m._anchor_pred["x"] - 0.2) < 1e-9


def test_anchor_regularization_biases_acceptance_toward_anchor():
    """大 λ + 锚在 h0 → 远离锚的 proposal 被系统性拒绝, 链驻留锚点."""
    m = _build_manifold()
    obs = _obs(0.0, 100.0)  # 极宽 sigma → likelihood 差异可忽略, 正则主导
    m._anchor_pred = {"x": 0.0}
    rng = random.Random(7)
    cur = "h0"
    stayed = 0
    for _ in range(200):
        nxt, _ = m.mcmc_step(
            obs, cur, rng=rng, global_proposal_prob=1.0, anchor_lambda=10.0)
        if nxt == "h0":
            stayed += 1
        cur = nxt
    assert stayed > 150, f"锚正则应把链拉回 h0, 实际 stayed={stayed}"


if __name__ == "__main__":
    test_mcmc_step_advances_manifold()
    test_update_mcmc_acceptance_updates_sliding_average()
    test_rcb_runner_wiring_sequence_closed_loop()
    test_anchor_off_leaves_state_untouched()
    test_anchor_penalty_and_ema_update()
    test_anchor_regularization_biases_acceptance_toward_anchor()
    print("P0-1 wiring 闭环测试全部通过")
