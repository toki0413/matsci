"""P-C · 证据驱动提前终止门 (evidence-based early termination) —— 缺陷一(BSP 结算让"等"变得必要)的另一半解药.

P-A 让每层结算增量可见, P-B 让边在证据到达后可修订; 但"到底要等多少层"仍是盲的
—— 默认等完整个计划(全 BSP), 即使占优方向早已稳定. 本模块把"稳定"定义成**可证伪
的判据**, 在层间结算时判定是否值得继续烧预算:

  给定已结算层的逐实验成绩, 若**分数进入高原** —— 连续两个已观测层的 top-1 实验
  得分的相对变化 <= margin —— 说明继续探索不再产出更好的占优方向, 剩余层可以
  提前终止(预算回收). 相对变化大(不管上升还是下降) = 探索未饱和 → 继续.

诚实红线(不可逾越):
  1. 稳定度判定只基于 cache 里**真实执行过**的证据 —— 无分数层不计入, 绝不猜测;
  2. 防早停门: 已观测(有分数)层数 < min_layers 时一律判不稳定 —— 至少 N 层才允许
     查稳定, 防止"第一层成绩就当占优"的早停;
  3. 提前终止 = 预算决策: 被终止的实验**不执行、不产生证据**(绝不伪造);
  4. 判定/终止全程记录(观测层数 + top 实验 + 相对变化 + 终止层), 可证伪复核.

注意: 执行编排是 orchestration/executor 的职责 —— 本模块只提供确定性判定,
不发起终止. 与 :func:`huginn.research.replan_gate` 同族(P-B 逐实验修订边, P-C
全局终止剩余层), 同 share 全仓统一的 ``_first_scalar`` 取数语义.
"""
from __future__ import annotations

from typing import Any


def _first_scalar(node: Any) -> float | None:
    """从任意结构取第一个可量标量(与 program/decision_gate/replan_gate 同语义)."""
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for k in ("T_eq_K", "T", "score", "value", "y", "actual", "S_Wm2"):
            if k in node:
                v = _first_scalar(node[k])
                if v is not None:
                    return v
        for v in node.values():
            s = _first_scalar(v)
            if s is not None:
                return s
    elif isinstance(node, (list, tuple)):
        for v in node:
            s = _first_scalar(v)
            if s is not None:
                return s
    return None


def _top1(pairs: list[tuple[str, float]]) -> tuple[str, float] | None:
    """层内 top-1 (score 大者胜; 平分时取名字典序小者 —— 确定性 tiebreak)."""
    if not pairs:
        return None
    best = max(p[1] for p in pairs)
    cands = [p for p in pairs if p[1] == best]
    return min(cands, key=lambda p: p[0])


def stability_check(
    settled_layers: list[list[tuple[str, float]]],
    *,
    min_layers: int = 2,
    margin: float = 0.02,
) -> dict[str, Any]:
    """层间稳定度判定: 分数是否进入高原 (纯函数, 无副作用).

    settled_layers[k] = 第 k 层**已执行**实验的成绩 [(name, score)] (无分则空).

    判据(全部可证伪):
      - 有分数的观测层 >= min_layers 才允许判 stable(防早停门);
      - 最后两个有分层的 top-1 得分相对变化 <= margin(占优成绩趋平) → stable;
      - 相对变化 > margin(top 上升或下降)、层数不足、最后两层缺分 → 一律 not_stable.

    返回:
      {
        "stable": bool,
        "verdict": "stable_top_plateau" | "top_not_saturated" | "insufficient_layers" | "missing_scores",
        "layers_observed": 有分观测层数,
        "top": 最近层 top-1 实验名, "prev_top": 上一层 top-1 实验名,
        "score": 最近层 top-1 分数, "prev_score": 上一层 top-1 分数,
        "relative_change": 相对变化(0..),
      }
    """
    scored = [[p for p in lay if p[1] is not None and _is_finite(p[1])]
              for lay in settled_layers]
    tops = [_top1(lay) for lay in scored]
    observed = sum(1 for t in tops if t is not None)

    # 防早停: 结算的总层数(len(scored), 空层也算) < min_layers → 不允许查稳定.
    if len(scored) < min_layers:
        return {"stable": False, "verdict": "insufficient_layers",
                "layers_observed": observed, "top": None, "prev_top": None,
                "score": None, "prev_score": None, "relative_change": None}

    if len(tops) < 2:
        return {"stable": False, "verdict": "missing_scores",
                "layers_observed": observed, "top": None, "prev_top": None,
                "score": None, "prev_score": None, "relative_change": None}

    prev = tops[-2]
    cur = tops[-1]
    if prev is None or cur is None:
        return {"stable": False, "verdict": "missing_scores",
                "layers_observed": observed, "top": cur[0] if cur else None,
                "prev_top": prev[0] if prev else None,
                "score": cur[1] if cur else None,
                "prev_score": prev[1] if prev else None,
                "relative_change": None}

    rel = abs(cur[1] - prev[1]) / (abs(prev[1]) if abs(prev[1]) > 1e-12 else 1e-12)
    stable = rel <= margin
    return {
        "stable": stable,
        "verdict": "stable_top_plateau" if stable else "top_not_saturated",
        "layers_observed": observed,
        "top": cur[0], "prev_top": prev[0],
        "score": cur[1], "prev_score": prev[1],
        "relative_change": round(rel, 6),
    }


def _is_finite(v: float) -> bool:
    """分数必须有限(排除 inf/nan 伪造面): None 已在外层过滤."""
    return v == v and v not in (float("inf"), float("-inf"))