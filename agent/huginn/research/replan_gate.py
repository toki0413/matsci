"""P-B · 层间重规划门 (inter-layer replan gate) —— 缺陷二("DAG 边=先验信念的一次性编译")的解药.

背景: 计划期把依赖边编成 DAG 后, 执行期就照单执行 —— 即使前序实验的真实证据已经
表明某个方向冗余/被证伪, 后序实验照样排队执行 (预算空转). 这正是 BSP 结算的另一面:
计划在"等"期间被冻结, 初始计划被当作**一次性的先验信念编译物**, 而非可修订的工作假说.

本模块的命题: **边可以在证据到达后被修订**. 每完成一层 (前序层全部结算), 用已观测
证据对后序实验做**单实验层间门控** (纯函数、确定性、可证伪):

  - ``redundant_direction`` : 实验假说与前序层已执行实验的假说 Jaccard 重叠 >= 阈值
    → 该方向已被前序层覆盖采样, 继续执行是重复采样 (预算空转).
  - ``falsified_direction`` : 前序层某实验带 predicted 且对账偏差 > 容差 (方向被真实
    执行证伪) 且假说与之重叠 → 死磕一个已被证伪的方向.

诚实红线 (不可逾越):
  1. 判定只基于 cache 里**真实执行过**的前序层证据 —— 绝不基于未执行实验的猜测;
  2. 前序层未全部结算 (实验被乱序调度) → 一律放行, 不误伤 (保守);
  3. 被跳过的实验**永不产生证据** (不进 cache/stream_view/trace) ——
     重规划是预算决策, 不是"没做装作做了";
  4. 每条跳过决策记录 {name, layer, reasons, decided_after}, 可反向审计 (可证伪);
  5. 本门只做"跳过 (预算再分配)", 不伪造也不修改任何已执行结果.

注意: 触发执行的顺序编排是 orchestration/executor 的职责 —— 本模块只提供确定性
判定, 不发起执行. 判定与 :func:`huginn.research.decision_gate.distill_tool_output`
同关键词切法, 与 world-model reconcile 同对账容差, 保证全仓治理口径一致.
"""
from __future__ import annotations

import re
from typing import Any


def _kw(text: str) -> set[str]:
    """关键词集合 (与 decision_gate.distill_tool_output 同切法): 只留 >1 字符的词干."""
    return {w.lower() for w in re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", text or "")
            if len(w) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    """假说间重叠度: 0..1. 空集对任何集合 = 0 (无数值 = 无重叠证据, 不硬判)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _first_scalar(node: Any) -> float | None:
    """从任意结构取第一个可量标量 (与 program/decision_gate/law_model 同语义, 不伪造)."""
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


def decide_replan_skip(
    name: str,
    layer_index: int,
    prior_experiments: list[str],
    cache: dict[str, dict],
    hypotheses: dict[str, str],
    *,
    already_decided: set[str] | None = None,
    similarity: float = 0.55,
    tol: float = 0.03,
) -> dict[str, Any] | None:
    """对单个后序实验做层间重规划判定 (纯函数). None = 放行; dict = 跳过记录.

    name:            后序实验名 (待判定是否仍值得执行).
    layer_index:     该实验所在层 (0 起).
    prior_experiments: 严格在它之前的所有层的实验名 (先验边 -> 证据面).
    cache:           已真实执行实验结果 name -> {summary, objectives, predicted, ...}.
    hypotheses:      name -> 假说文本 (判定重叠用; 缺省视为 "" → 无重叠证据 → 放行).
    already_decided: 已执行 ∪ 已被跳过 的名字 (用于"前序层是否结算"的判定);
                     缺省会退化为只看 cache (executor 场景须传入, 含 replan 跳过项).
    similarity:      Jaccard 重叠阈值 (>= 视为冗余方向).
    tol:             预测-真实相对误差容差 (超过 = 方向被证伪; 同 world-model reconcile).

    返回跳过记录格式: {"name", "layer", "reasons": [{rule, with, 证据量}], "decided_after"}.
    """
    if layer_index <= 0 or not prior_experiments:
        return None                                    # 首层 / 无前序: 无条件执行

    # 保守红线 #2: 前序层未全部结算 → 放行, 不误伤乱序调度.
    # 结算 = 已执行(在 cache) 或 已被决策(在被重规划跳过名单里) 二者之一.
    decided = set(cache) | (already_decided or set())
    for pn in prior_experiments:
        if pn not in decided:
            return None

    k_cur = _kw(hypotheses.get(name, ""))
    reasons: list[dict[str, Any]] = []
    for pn in prior_experiments:
        res = cache.get(pn) or {}
        if not res:
            continue
        ph = hypotheses.get(pn, "")
        ov = _jaccard(k_cur, _kw(ph))
        if ov >= similarity:
            reasons.append({"rule": "redundant_direction", "with": pn,
                            "overlap": round(ov, 3)})
        pred = _first_scalar(res.get("predicted"))
        actual = _first_scalar(res.get("summary"))
        err = (abs(pred - actual) / (abs(actual) or 1.0)) if pred is not None and actual is not None else None
        if err is not None and err > tol and ov >= similarity * 0.7:
            reasons.append({"rule": "falsified_direction", "with": pn,
                            "rel_err": round(err, 3)})
    if not reasons:
        return None
    return {
        "name": name,
        "layer": layer_index,
        "reasons": reasons,
        "decided_after": [n for n in prior_experiments if n in cache],
    }


def replan_gate(
    layers: list[list[str]],
    cache: dict[str, dict],
    hypotheses: dict[str, str],
    *,
    similarity: float = 0.55,
    tol: float = 0.03,
) -> dict[str, Any]:
    """整计划批判定 (按层结算): 逐层推进, 产出全部重规划决策 (纯函数, 无副作用).

    layers:       拓扑层划分 [[层0名字...], [层1名字...], ...].
    cache:        已执行实验结果.
    hypotheses:   name -> 假说文本.

    返回 {layers, checked, skipped:[记录], proceeded:[名字], verdict}:
      - checked   被评估的后序实验数 (未执行且被门控查看过);
      - skipped   判定跳过 (预算再分配) 的记录列表 —— 每条约 {name, layer, reasons};
      - proceeded 判定放行的实验;
      - verdict   all_proceed / replanned.
    """
    skipped: list[dict[str, Any]] = []
    considered: list[str] = []
    decided = set(cache)
    for i, layer in enumerate(layers):
        prior = [n for j in range(i) for n in layers[j]]
        for name in layer:
            if name in cache:
                continue                              # 已执行, 不在重规划范围
            d = decide_replan_skip(name, i, prior, cache, hypotheses,
                                   already_decided=decided,
                                   similarity=similarity, tol=tol)
            considered.append(name)
            if d is not None:
                skipped.append(d)
            decided.add(name)                         # 跳过/放行都算已结算, 供后层判定
    skipped_names = {s["name"] for s in skipped}
    return {
        "layers": len(layers),
        "checked": len(considered),
        "skipped": skipped,
        "proceeded": [n for n in considered if n not in skipped_names],
        "verdict": "replanned" if skipped else "all_proceed",
    }