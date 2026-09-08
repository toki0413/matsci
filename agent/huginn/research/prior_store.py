"""A2 · 跨 run 稳定度先验 (cross-run settlement prior) —— 把一次 run 的分数高原
沉淀为可传递、可注入的预算先验.

背景: 同类目标域反复研究时, "这个领域通常在第几层进入分数高原"是**可复用**的调度
经验. 本模块把一次 run 的稳定度观察提取成纯 dict(extract_prior), 再由下一次 run
以保守方式注入(resolve_early_stop_args, 见 Task 3) —— 但它只是**预算参数先验**,
绝不参与、更不替代任何真实实验.

诚实红线(不可逾越):
  1. extract_prior 只读 out.consolidated(聚合头唯一出口), 不新增 out.* 字段;
  2. 先验只能把早停参数推向**更保守**方向(min_layers 单调不减), 永不因"上次稳定"
     而激进提前终止 —— 领域可能漂移, 先验不为历史背书;
  3. applicable=False 的字典可安全注入(等价无先验), 不抛错、不改行为.
"""
from __future__ import annotations

from typing import Any


def extract_prior(out: Any) -> dict[str, Any]:
    """从 ResearchOutcome 提取一次 run 的稳定度先验 (纯函数, 无副作用).

    只消费 out.consolidated["early_stop"] —— 未启用早停/无观测 → applicable=False.
    返回说明:
      - applicable: 是否有可复用的高原观察(early_stopped 才为 True);
      - plateau: {layer_index, top, score, relative_change} 首次判稳的那一层;
      - goal: 目标原文(跨 run 匹配建议用归一化 slug, 由调用方决定).
    """
    con = getattr(out, "consolidated", None) or {}
    es = con.get("early_stop") or {}
    st = es.get("stability") or {}
    stopped = es.get("verdict") == "early_stopped"
    layer_index = es.get("stopped_after_layer")
    if not stopped or layer_index is None or not st:
        return {
            "source": "layered_settlement",
            "applicable": False,
            "reason": "no_early_stopped",
            "goal": ((getattr(out, "plan_summary", None) or {}).get("goal")
                 or (getattr(out, "harness", None) or {}).get("goal") or ""),
        }
    return {
        "source": "layered_settlement",
        "applicable": True,
        "goal": ((getattr(out, "plan_summary", None) or {}).get("goal")
                 or (getattr(out, "harness", None) or {}).get("goal") or ""),
        "plateau": {
            "layer_index": int(layer_index),
            "top": st.get("top"),
            "score": st.get("score"),
            "relative_change": st.get("relative_change"),
        },
    }


def resolve_early_stop_args(
    prior: dict | None,
    *,
    default_min_layers: int = 2,
    default_margin: float = 0.02,
) -> dict[str, Any]:
    """把先验映射为早停参数(纯函数, 确定性).

    规则(全部可证伪):
      - prior 为空/不可用 → 返回默认参数, note="no_prior";
      - prior 可用(early_stopped) → min_layers = plateau.layer_index + 1,
        且钳制在 [default_min_layers, 4] —— 上次 N 层才稳定, 这次至少等 N 层
        才允许查稳定(**更保守**, 防领域漂移误停); margin 原样传默认(不因先验放宽).
    """
    if not prior or not prior.get("applicable"):
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "no_prior"}
    plateau = prior.get("plateau") or {}
    idx = plateau.get("layer_index")
    if idx is None:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "prior_without_plateau"}
    min_layers = max(default_min_layers, min(int(idx) + 1, 4))
    return {"min_layers": min_layers, "margin": default_margin,
            "source": "cross_run_prior", "note": f"plateau_layer={idx}"}