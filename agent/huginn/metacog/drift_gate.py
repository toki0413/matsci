"""Smith 式漂移检测 — 倾向性回归 / 求真下降 / 因果图编辑距离.

纯函数：返回 review / intervene / pause 三级信号，不执行动作。
阈值只是默认起点，生产应由策略注册表按 tenant/persona/task family 覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DriftLevel = Literal["review", "intervene", "pause"]
DriftDetector = Literal["propensity", "truth_seeking", "causal_edit"]


@dataclass(frozen=True)
class DriftThresholds:
    propensity_drop: float = 0.15
    truth_drop: float = 0.20
    causal_edit: float = 0.30


@dataclass(frozen=True)
class DriftSignal:
    detector: DriftDetector
    observed: float
    baseline: float
    delta: float
    level: DriftLevel
    reason: str


def _drop_ratio(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    return max(0.0, (baseline - current) / baseline)


def evaluate_drift(
    *,
    propensity: tuple[float, float] | None = None,
    truth_seeking: tuple[float, float] | None = None,
    causal_edit_distance: float | None = None,
    thresholds: DriftThresholds | None = None,
) -> list[DriftSignal]:
    """返回按严重度排序的漂移信号；无输入 → 空列表."""
    t = thresholds or DriftThresholds()
    signals: list[DriftSignal] = []

    if propensity is not None:
        baseline, current = propensity
        delta = _drop_ratio(baseline, current)
        if delta > t.propensity_drop:
            signals.append(DriftSignal(
                detector="propensity",
                observed=current,
                baseline=baseline,
                delta=round(delta, 4),
                level="review",
                reason=f"倾向性回归下降 {delta:.0%} > {t.propensity_drop:.0%}",
            ))

    if truth_seeking is not None:
        baseline, current = truth_seeking
        delta = _drop_ratio(baseline, current)
        if delta > t.truth_drop:
            signals.append(DriftSignal(
                detector="truth_seeking",
                observed=current,
                baseline=baseline,
                delta=round(delta, 4),
                level="intervene",
                reason=f"求真率下降 {delta:.0%} > {t.truth_drop:.0%}",
            ))

    if causal_edit_distance is not None and causal_edit_distance > t.causal_edit:
        signals.append(DriftSignal(
            detector="causal_edit",
            observed=causal_edit_distance,
            baseline=0.0,
            delta=round(causal_edit_distance, 4),
            level="pause",
            reason=(
                f"因果图编辑距离 {causal_edit_distance:.2f} "
                f"> {t.causal_edit:.2f}"
            ),
        ))

    rank = {"review": 0, "intervene": 1, "pause": 2}
    signals.sort(key=lambda s: rank[s.level], reverse=True)
    return signals


__all__ = [
    "DriftDetector",
    "DriftLevel",
    "DriftSignal",
    "DriftThresholds",
    "evaluate_drift",
]
