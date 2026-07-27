"""score history 滑动窗口 + LLM temperature 动态调节.

数据源: v19 bandit reward_slow (Δdarwin_score_mean).
  reward_slow 是 darwin_score 均值在 slow timescale 上的变化量,
  正值=改进, 负值=退化, 振荡=探索不稳定.

控制论思想 (最小机制, ponytail: 不引入完整控制论框架):
  - 信号: window 内 dispersion → 监测稳定性
  - 执行器: temperature 下调 0.2 → 抑制探索, 集中利用
  - 终止条件: 连续 N 轮单调下降 → 系统退化, 应终止

不做 (YAGNI):
  - Lyapunov 函数 — 需要系统模型, 这里只有标量信号
  - PID 控制器 — 需要设定点 + 误差积分, 过度参数化
  - 状态空间模型 — 单变量观测无需状态估计
  - 完整 bandit 策略 — 已由 v19 实现, 这里只做监测器

天花板: 纯标量反馈, 多维信号需扩展为向量统计. 升级: 协方差矩阵.
"""
from __future__ import annotations

import statistics
from collections import deque


class ScoreHistory:
    def __init__(self, window_size: int = 5, variance_threshold: float = 0.1,
                 monotonic_decrease_limit: int = 3, base_temperature: float = 0.7):
        # deque maxlen 满后自动从左端丢, 不用手动 pop
        self._window: deque[float] = deque(maxlen=window_size)
        self._variance_threshold = variance_threshold
        self._monotonic_decrease_limit = monotonic_decrease_limit
        self._base_temperature = base_temperature
        # 连续严格下降轮数; 持平或上升清零
        self._monotonic_streak = 0
        # 累计 push 次数, 不会随 window 滚动丢失
        self._n_samples = 0

    def push(self, score: float) -> None:
        """添加一个新 score 到窗口."""
        prev = self._window[-1] if self._window else None
        if prev is not None:
            if score < prev:
                self._monotonic_streak += 1
            else:
                # 持平或上升都打断下降 streak
                self._monotonic_streak = 0
        self._window.append(score)
        self._n_samples += 1

    def get_temperature(self) -> float:
        """返回当前应使用的 LLM temperature.

        variance > threshold → base - 0.2
        否则 → base
        """
        if len(self._window) < 2:
            # 单点无 dispersion, 不外推
            return self._base_temperature
        # ponytail: spec 字段叫 variance_threshold, 实际比较用 population std-dev.
        # 阈值 0.1 在 std-dev (与数据同尺度) 上直观, 在 squared variance 上不直观.
        # 升级路径: 改用 pvariance 时把阈值下调到 0.01 量级.
        if statistics.pstdev(self._window) > self._variance_threshold:
            return self._base_temperature - 0.2
        return self._base_temperature

    def should_terminate(self) -> bool:
        """返回是否应终止.

        最近 N 轮单调下降 → True
        否则 → False
        """
        return self._monotonic_streak >= self._monotonic_decrease_limit

    def stats(self) -> dict:
        """返回统计信息: mean / variance / min / max / n_samples / monotonic_streak."""
        w = list(self._window)
        if not w:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n_samples": 0,
                "monotonic_streak": 0,
            }
        # ponytail: 用 pvariance (总体方差), 样本小不引入 Bessel 修正.
        # 与 get_temperature() 的 pstdev 同源: sqrt(pvariance) == pstdev.
        return {
            "mean": statistics.mean(w),
            "variance": statistics.pvariance(w),
            "min": min(w),
            "max": max(w),
            "n_samples": self._n_samples,
            "monotonic_streak": self._monotonic_streak,
        }


if __name__ == "__main__":
    h = ScoreHistory()
    # 模拟振荡
    for s in [0.5, 0.3, 0.6, 0.2, 0.5]:
        h.push(s)
    assert h.get_temperature() < 0.7, "oscillation should lower temperature"
    # 模拟单调下降
    h2 = ScoreHistory()
    for s in [0.5, 0.4, 0.3, 0.2]:
        h2.push(s)
    assert h2.should_terminate(), "monotonic decrease should terminate"
    print("score_history self-check OK")
