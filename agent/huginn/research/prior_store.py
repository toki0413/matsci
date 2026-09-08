"""A2 · 跨 run 稳定度先验 (cross-run settlement prior) —— 把一次 run 的分数高原
沉淀为可传递、可注入的预算先验.

背景: 同类目标域反复研究时, "这个领域通常在第几层进入分数高原"是**可复用**的调度
经验. 本模块把一次 run 的稳定度观察提取成纯 dict(extract_prior), 再由下一次 run
以保守方式注入(resolve_early_stop_args) —— 但它只是**预算参数先验**, 绝不参与、
更不替代任何真实实验.

A4(goal 归一化匹配): 先验携带 goal_slug(与 program._slug_goal 同语义), 注入时
若当前 goal 的 slug 与先验不一致 → 视为异域先验, 拒绝套用(行为=无先验) ——
防止把 A 领域的"高原层数"错套到 B 领域.

诚实红线(不可逾越):
  1. extract_prior 只读 out.consolidated(聚合头唯一出口), 不新增 out.* 字段;
  2. 先验只能把早停参数推向**更保守**方向(min_layers 单调不减), 永不因"上次稳定"
     而激进提前终止 —— 领域可能漂移, 先验不为历史背书;
  3. 异域先验(goal_slug 不匹配)一律拒绝套用 —— 先验只对**同域**生效;
  4. applicable=False / 无 slug 的字典可安全注入(等价无先验/未验证), 不抛错.
"""
from __future__ import annotations

import re
from typing import Any


def goal_slug(goal: str, limit: int = 40) -> str:
    """goal 归一化 slug (与 program._slug_goal 同语义, 保证跨模块匹配口径一致).

    小写 + 非字母/数字/下划线 → "_", 去首尾 "_", 截断上限 —— "a4 match" → "a4_match".
    """
    s = re.sub(r"[^0-9a-z_]+", "_", (goal or "").lower()).strip("_")
    return s[:limit] or "final"


def extract_prior(out: Any) -> dict[str, Any]:
    """从 ResearchOutcome 提取一次 run 的稳定度先验 (纯函数, 无副作用).

    只消费 out.consolidated["early_stop"] —— 未启用早停/无观测 → applicable=False.
    返回说明:
      - applicable: 是否有可复用的高原观察(early_stopped 才为 True);
      - plateau: {layer_index, top, score, relative_change} 首次判稳的那一层;
      - goal: 目标原文; goal_slug: 归一化 slug(A4 跨 run 匹配凭据, 与 program 同口径).
    """
    con = getattr(out, "consolidated", None) or {}
    es = con.get("early_stop") or {}
    st = es.get("stability") or {}
    stopped = es.get("verdict") == "early_stopped"
    layer_index = es.get("stopped_after_layer")
    _goal = ((getattr(out, "plan_summary", None) or {}).get("goal")
             or (getattr(out, "harness", None) or {}).get("goal") or "")
    _base = {
        "source": "layered_settlement",
        "goal": _goal,
        "goal_slug": goal_slug(_goal),
    }
    if not stopped or layer_index is None or not st:
        _base.update({"applicable": False, "reason": "no_early_stopped"})
        return _base
    _base.update({
        "applicable": True,
        "plateau": {
            "layer_index": int(layer_index),
            "top": st.get("top"),
            "score": st.get("score"),
            "relative_change": st.get("relative_change"),
        },
    })
    return _base


def resolve_early_stop_args(
    prior: dict | None,
    *,
    default_min_layers: int = 2,
    default_margin: float = 0.02,
    current_goal_slug: str | None = None,
) -> dict[str, Any]:
    """把先验映射为早停参数(纯函数, 确定性). A4: 异域先验拒绝套用.

    规则(全部可证伪):
      - prior 为空/不可用 → 默认参数, note="no_prior", goal_matched=None;
      - prior 带 goal_slug 且 != current_goal_slug → **异域先验**: 拒绝套用,
        返回默认参数, note="goal_mismatch", goal_matched=False(绝不张冠李戴);
      - prior 可用且(同域 或 无 slug 旧格式) → min_layers = plateau.layer_index + 1,
        钳制在 [default_min_layers, 4] —— 上次 N 层才稳定, 这次至少等 N 层
        才允许查稳定(**更保守**, 防领域漂移误停); margin 原样传默认;
        goal_matched: True(有 slug 且匹配) / None(无 slug, 未验证, 向后兼容).
    """
    no_prior = not prior or not prior.get("applicable")
    prior_slug = (prior or {}).get("goal_slug")
    mismatch = (prior_slug is not None and current_goal_slug is not None
                and str(prior_slug) != str(current_goal_slug))
    if no_prior:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "no_prior", "goal_matched": None}
    if mismatch:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "rejected", "note": "goal_mismatch", "goal_matched": False}
    plateau = prior.get("plateau") or {}
    idx = plateau.get("layer_index")
    if idx is None:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "prior_without_plateau",
                "goal_matched": (None if prior_slug is None else True)}
    min_layers = max(default_min_layers, min(int(idx) + 1, 4))
    return {"min_layers": min_layers, "margin": default_margin,
            "source": "cross_run_prior", "note": f"plateau_layer={idx}",
            "goal_matched": (None if prior_slug is None else True)}