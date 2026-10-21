"""贝叶斯惊奇 — 问「什么改变了信念」，不是问「这有用吗」（spec §19.6，I32）.

- 完整版：``surprise = KL(P(belief|evidence) || P(belief))``；
- 廉价代理：预测误差 / 新矛盾 / 检索失败 / 权威断层；
- 惊奇**不是权威**：高惊奇不得绕过 Hygiene Gate，也不提升 evidential authority；
- 代理与完整版必须可区分（版本化），不能把代理当定理。
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def _normalize(probs: Iterable[float], *, eps: float = 1e-9) -> list[float]:
    values = [max(eps, float(p)) for p in probs]
    total = sum(values)
    if total <= 0:
        return []
    return [value / total for value in values]


def kl_divergence(prior: Iterable[float], posterior: Iterable[float]) -> float:
    """KL(posterior || prior)；长度不足/全零返回 0（无信号，不臆造）."""
    p = _normalize(prior)
    q = _normalize(posterior)
    if not p or not q or len(p) != len(q):
        return 0.0
    return sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))


def belief_surprise(prior: Iterable[float], posterior: Iterable[float]) -> dict[str, object]:
    """返回带估计器版本的惊奇；``method=kl`` 是完整版."""
    value = max(0.0, kl_divergence(prior, posterior))
    normalized = value / (1.0 + value)  # 压到 [0,1)，供排序
    return {
        "surprise": round(normalized, 6),
        "raw_kl": round(value, 6),
        "method": "kl",
        "version": "belief_surprise/v1",
    }


def surrogate_surprise(
    *,
    prediction_error: float = 0.0,
    new_contradiction: bool = False,
    retrieval_failure: bool = False,
    authority_gap: float = 0.0,
) -> dict[str, object]:
    """廉价代理；``method=surrogate`` 明确区别于定理版，不能当校准依据."""
    score = (
        max(0.0, min(1.0, float(prediction_error)))
        + (0.5 if new_contradiction else 0.0)
        + (0.4 if retrieval_failure else 0.0)
        + max(0.0, min(1.0, float(authority_gap))) * 0.3
    )
    return {
        "surprise": round(min(1.0, score / 2.2), 6),
        "method": "surrogate",
        "version": "belief_surprise/v1",
        "components": {
            "prediction_error": float(prediction_error),
            "new_contradiction": bool(new_contradiction),
            "retrieval_failure": bool(retrieval_failure),
            "authority_gap": float(authority_gap),
        },
    }


def surprise_rank(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    """按惊奇排序；只决定诊断优先级，不改变权威/授权."""
    return sorted(
        findings,
        key=lambda item: float(item.get("surprise", 0.0) or 0.0),
        reverse=True,
    )


__all__ = [
    "belief_surprise",
    "kl_divergence",
    "surprise_rank",
    "surrogate_surprise",
]
