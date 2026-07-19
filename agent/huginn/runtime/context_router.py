"""Context Router (P3) — 信息路径多样性驱动的动态 context 选择.

参考: "Diversity of information pathways drives sparsity in real-world
networks" (Nature Physics 2023, s41567-023-02330-x).

论文核心: 真实网络在 wiring cost 固定下, 最大化 response diversity D
(密度矩阵 von Neumann 熵) → 自然得到稀疏拓扑.

翻译到 agent context 层面:
- 节点 = context 段 (memory / kg / kb / meta_trace / emotion / plan /
  cognitive / tool_hint / evolution / continuity / subgoal / goal)
- 连接 = LLM 对该段的 attend (是否塞进 prompt)
- wiring cost = prompt token 数 (长=慢+贵)
- response diversity D = "不同 task 用不同 context 子集" 的程度

huginn 现状: build_input_messages 硬编码全塞 10 段, 粗暴 truncation 兜底.
P3: 根据当前 phase + task 语义, 给每段打 0-1 权重, 只塞高分段.

D_proxy = -sum(w_i * ln w_i) 作为监控指标:
- 高 D_proxy = 多段都有贡献 (信息路径多)
- 低 D_proxy = 只有 1-2 段有贡献 (信息路径集中, 可能漏信息)

接入: ContextBuilder.build_input_messages 在拼 ctx_parts 之前调
route_context_segments, 只保留 weight > threshold 的段.

升级路径:
- LLM 版: 小模型给 8 段各打 0-1 分 (覆盖未知模式, 但有 LLM 成本)
- 真正的 density matrix: 用 embedding cosine 算 U_{2tau}, 严格但贵
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# === Context 段定义 ===

# ContextBuilder.build_input_messages 里的所有 context 段 (顺序敏感)
# 顺序对应 ctx_parts.append 的顺序, 改顺序要同步改 build_input_messages.
CONTEXT_SEGMENTS: tuple[str, ...] = (
    "meta_trace",     # 历史 trajectory trace
    "emotion",        # 情感状态 (v9 legacy, 多数 task 不需要)
    "plan",           # 当前 plan
    "cognitive",      # 认知状态
    "tool_hint",      # 工具偏好提示
    "evolution",      # evolution rules
    "continuity",     # 会话连续性
    "subgoal",        # 子目标
    "goal",           # 主目标
    "memory",         # 长期记忆 recall
    "kg",             # knowledge graph
    "kb",             # knowledge base (RAG)
)

# 默认权重阈值: > 该值才塞进 prompt
DEFAULT_THRESHOLD: float = 0.3


# === Phase → 段权重规则表 ===
#
# ponytail: 规则版, 不调 LLM. 覆盖已知 phase 模式:
# - perceive: 需要感知环境, memory+kb 重, emotion/evolution 轻
# - hypothesize: 需要历史假设 + 领域知识, meta_trace+kb 重
# - plan: 需要目标 + 认知状态, goal+cognitive+subgoal 重
# - execute: 需要计划 + 工具, plan+tool_hint 重, memory/kg 轻
# - validate: 需要目标 + 历史, goal+meta_trace 重, emotion 轻
# - learn: 需要历史 + evolution, meta_trace+evolution 重
# - report: 需要目标 + 历史, goal+meta_trace 重, emotion 轻
#
# 未知 phase → 全 1.0 (fallback 到现状)
#
# 升级路径: LLM 版给 8 段打分, 覆盖未知模式.

_PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "perceive": {
        "meta_trace": 0.8, "emotion": 0.2, "plan": 0.3, "cognitive": 0.5,
        "tool_hint": 0.3, "evolution": 0.3, "continuity": 0.6,
        "subgoal": 0.5, "goal": 0.7, "memory": 0.9, "kg": 0.6, "kb": 0.9,
    },
    "hypothesize": {
        "meta_trace": 0.9, "emotion": 0.2, "plan": 0.4, "cognitive": 0.6,
        "tool_hint": 0.3, "evolution": 0.4, "continuity": 0.5,
        "subgoal": 0.6, "goal": 0.7, "memory": 0.7, "kg": 0.7, "kb": 0.9,
    },
    "plan": {
        "meta_trace": 0.5, "emotion": 0.2, "plan": 0.8, "cognitive": 0.8,
        "tool_hint": 0.7, "evolution": 0.4, "continuity": 0.5,
        "subgoal": 0.9, "goal": 0.9, "memory": 0.5, "kg": 0.4, "kb": 0.6,
    },
    "execute": {
        "meta_trace": 0.4, "emotion": 0.3, "plan": 0.9, "cognitive": 0.6,
        "tool_hint": 0.9, "evolution": 0.3, "continuity": 0.4,
        "subgoal": 0.8, "goal": 0.7, "memory": 0.4, "kg": 0.3, "kb": 0.5,
    },
    "validate": {
        "meta_trace": 0.8, "emotion": 0.2, "plan": 0.5, "cognitive": 0.6,
        "tool_hint": 0.4, "evolution": 0.4, "continuity": 0.4,
        "subgoal": 0.7, "goal": 0.9, "memory": 0.6, "kg": 0.6, "kb": 0.7,
    },
    "learn": {
        "meta_trace": 0.9, "emotion": 0.3, "plan": 0.5, "cognitive": 0.5,
        "tool_hint": 0.3, "evolution": 0.9, "continuity": 0.5,
        "subgoal": 0.6, "goal": 0.7, "memory": 0.8, "kg": 0.5, "kb": 0.6,
    },
    "report": {
        "meta_trace": 0.9, "emotion": 0.2, "plan": 0.4, "cognitive": 0.4,
        "tool_hint": 0.2, "evolution": 0.4, "continuity": 0.5,
        "subgoal": 0.7, "goal": 0.9, "memory": 0.6, "kg": 0.5, "kb": 0.6,
    },
}


# === Task keyword → 段权重增强 ===
#
# 在 phase 权重基础上, 根据 task 关键词增强某些段.
# ponytail: 关键词命中 +0.3 (不覆盖 phase 权重, 只加成).

_TASK_KEYWORD_BOOSTS: dict[str, dict[str, float]] = {
    # 检索/查询类任务: memory + kb + kg 都重要
    "search": {"memory": 0.3, "kb": 0.3, "kg": 0.3},
    "query": {"memory": 0.3, "kb": 0.3, "kg": 0.3},
    "find": {"memory": 0.3, "kb": 0.3, "kg": 0.3},
    "recall": {"memory": 0.4, "meta_trace": 0.3},
    "检索": {"memory": 0.3, "kb": 0.3, "kg": 0.3},
    "查询": {"memory": 0.3, "kb": 0.3, "kg": 0.3},

    # 计算/仿真类任务: plan + tool_hint 重要
    "compute": {"plan": 0.3, "tool_hint": 0.3},
    "calculate": {"plan": 0.3, "tool_hint": 0.3},
    "simulate": {"plan": 0.3, "tool_hint": 0.3},
    "run": {"plan": 0.3, "tool_hint": 0.3},
    "计算": {"plan": 0.3, "tool_hint": 0.3},
    "仿真": {"plan": 0.3, "tool_hint": 0.3},

    # 分析/推理类任务: cognitive + meta_trace 重要
    "analyze": {"cognitive": 0.3, "meta_trace": 0.3},
    "reason": {"cognitive": 0.3, "meta_trace": 0.3},
    "why": {"cognitive": 0.3, "meta_trace": 0.3},
    "分析": {"cognitive": 0.3, "meta_trace": 0.3},
    "推理": {"cognitive": 0.3, "meta_trace": 0.3},

    # 规划/设计类任务: goal + subgoal + cognitive 重要
    "plan": {"goal": 0.3, "subgoal": 0.3, "cognitive": 0.3},
    "design": {"goal": 0.3, "subgoal": 0.3, "cognitive": 0.3},
    "规划": {"goal": 0.3, "subgoal": 0.3, "cognitive": 0.3},
    "设计": {"goal": 0.3, "subgoal": 0.3, "cognitive": 0.3},
}


# === 数据结构 ===


@dataclass
class RoutingDecision:
    """单次 context routing 的决策结果.

    weights: 每段的 0-1 权重 (phase 基础 + task 关键词增强)
    selected: 经过 threshold 过滤后, 实际塞进 prompt 的段名列表
    d_proxy: 信息路径多样性代理度量 (-sum w ln w, 只算 selected)
    """
    weights: dict[str, float]
    selected: list[str]
    d_proxy: float

    def to_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "selected": list(self.selected),
            "d_proxy": round(self.d_proxy, 3),
            "n_segments": len(self.selected),
            "n_total": len(self.weights),
        }


# === 核心 routing 函数 ===


def route_context_segments(
    *,
    phase: str = "",
    task_message: str = "",
    threshold: float = DEFAULT_THRESHOLD,
) -> RoutingDecision:
    """根据 phase + task 语义决定塞哪些 context 段.

    Args:
        phase: autoloop 7-phase 之一 (perceive/hypothesize/plan/execute/
               validate/learn/report). 未知 phase → 全 1.0 (fallback)
        task_message: 当前 task 的用户消息/任务描述, 用于关键词增强
        threshold: 权重阈值, > 该值才塞 (默认 0.3)

    Returns:
        RoutingDecision. selected 为空时 caller 应 fallback 到全塞.
    """
    # 1. phase 基础权重
    phase_lower = (phase or "").lower().strip()
    if phase_lower in _PHASE_WEIGHTS:
        weights = dict(_PHASE_WEIGHTS[phase_lower])
    else:
        # 未知 phase → 全 1.0 (fallback 到现状, 不改变行为)
        weights = {seg: 1.0 for seg in CONTEXT_SEGMENTS}

    # 2. task 关键词增强
    msg_lower = (task_message or "").lower()
    for kw, boosts in _TASK_KEYWORD_BOOSTS.items():
        if kw in msg_lower:
            for seg, boost in boosts.items():
                if seg in weights:
                    weights[seg] = min(1.0, weights[seg] + boost)

    # 3. threshold 过滤
    selected = [seg for seg in CONTEXT_SEGMENTS
                if weights.get(seg, 0.0) > threshold]

    # 4. D_proxy = -sum(w_i ln w_i) 只算 selected (避免未选段拉低)
    d_proxy = _compute_d_proxy([weights[s] for s in selected])

    # 5. 兜底: 全空时 fallback 到全塞 (不改变现状)
    if not selected:
        selected = list(CONTEXT_SEGMENTS)
        d_proxy = _compute_d_proxy([1.0] * len(selected))

    return RoutingDecision(
        weights=weights,
        selected=selected,
        d_proxy=d_proxy,
    )


def _compute_d_proxy(weights: list[float]) -> float:
    """计算信息路径多样性代理 D_proxy = -sum(w ln w).

    论文 D = -Tr(U ln U) 是 von Neumann 熵. 这里用段权重的 Shannon 熵近似:
    - 先归一化 (sum = 1)
    - 再算 -sum(p ln p)
    - 高 = 多段有贡献 (信息路径多)
    - 低 = 集中在少数段

    ponytail: 归一化 + Shannon 熵, 非严格 von Neumann 熵.
    升级路径: 用段间 embedding cosine 算 density matrix U_{2tau} 再算 von Neumann 熵.
    """
    if not weights:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 0.0
    probs = [w / total for w in weights]
    d = 0.0
    for p in probs:
        if p > 0:
            d -= p * math.log(p)
    return d


def should_skip_segment(
    segment_name: str,
    decision: RoutingDecision,
) -> bool:
    """判断某段是否该跳过 (ContextBuilder.build_input_messages 用).

    Returns:
        True = 跳过该段 (不塞进 prompt)
        False = 保留该段
    """
    # 不在 selected 里 → 跳过
    return segment_name not in decision.selected


# === 监控 (Engine 可选调) ===


def log_routing_decision(decision: RoutingDecision, phase: str, task: str) -> None:
    """把 routing 决策 log 出来, 供事后分析.

    ponytail: 不强求 Engine 调, 只在 debug 时用. 不引入新依赖.
    """
    d = decision.to_dict()
    logger.debug(
        "ContextRouter phase=%s task=%.40s: selected %d/%d segments, D_proxy=%.3f",
        phase, task, d["n_segments"], d["n_total"], d["d_proxy"],
    )


# === 自检 ===

if __name__ == "__main__":
    # 1. 基础 routing — perceive phase
    dec = route_context_segments(phase="perceive", task_message="search band gap")
    assert "memory" in dec.selected, "perceive+search 应选 memory"
    assert "kb" in dec.selected, "perceive+search 应选 kb"
    assert "emotion" not in dec.selected, "perceive emotion 权重 0.2+0 (无 boost), 不应选"
    assert dec.d_proxy > 0, "D_proxy 应 > 0"
    assert len(dec.selected) < len(CONTEXT_SEGMENTS), "perceive 应稀疏化, 不是全塞"

    # 2. execute phase — plan + tool_hint 重
    dec = route_context_segments(phase="execute", task_message="run vasp")
    assert "plan" in dec.selected
    assert "tool_hint" in dec.selected
    assert dec.weights["tool_hint"] >= 0.9  # 0.9 base + 0 boost (run 不 boost tool_hint? 看表)

    # 3. 未知 phase → fallback 全 1.0
    dec = route_context_segments(phase="unknown_phase", task_message="")
    assert len(dec.selected) == len(CONTEXT_SEGMENTS), "未知 phase fallback 全塞"
    assert all(w == 1.0 for w in dec.weights.values())

    # 4. task 关键词增强 — "search" 增强 memory/kb/kg
    dec_perceive_no_search = route_context_segments(phase="perceive", task_message="hello")
    dec_perceive_search = route_context_segments(phase="perceive", task_message="search band gap")
    assert dec_perceive_search.weights["memory"] > dec_perceive_no_search.weights["memory"]
    assert dec_perceive_search.weights["kb"] > dec_perceive_no_search.weights["kb"]

    # 5. 权重不超 1.0 (钳位)
    dec = route_context_segments(phase="perceive", task_message="search query find recall 检索")
    assert dec.weights["memory"] <= 1.0, f"权重应钳位到 1.0, got {dec.weights['memory']}"

    # 6. threshold 过滤 — 高 threshold 时 selected 少
    dec_low = route_context_segments(phase="perceive", task_message="", threshold=0.1)
    dec_high = route_context_segments(phase="perceive", task_message="", threshold=0.8)
    assert len(dec_low.selected) >= len(dec_high.selected), \
        "低 threshold 应选更多段"

    # 7. D_proxy 单调性 — 全选 vs 稀疏
    dec_full = route_context_segments(phase="unknown", task_message="", threshold=0.0)
    dec_sparse = route_context_segments(phase="execute", task_message="run", threshold=0.5)
    # 全选 D_proxy 高 (12 段均匀), 稀疏 D_proxy 低 (少数段集中)
    # 注意: D_proxy 是 -sum p ln p, 段数多且均匀时更高
    assert dec_full.d_proxy > dec_sparse.d_proxy, \
        f"全选 D_proxy ({dec_full.d_proxy:.3f}) 应 > 稀疏 ({dec_sparse.d_proxy:.3f})"

    # 8. should_skip_segment
    dec = route_context_segments(phase="perceive", task_message="search", threshold=0.5)
    assert should_skip_segment("emotion", dec) is True
    assert should_skip_segment("memory", dec) is False

    # 9. 空 task + 未知 phase → fallback 全塞
    dec = route_context_segments(phase="", task_message="")
    assert len(dec.selected) == len(CONTEXT_SEGMENTS)

    # 10. 全空 selected → fallback 兜底
    # 设超高 threshold 让所有段都不达标
    dec = route_context_segments(phase="perceive", task_message="", threshold=2.0)
    assert len(dec.selected) == len(CONTEXT_SEGMENTS), \
        "全空 selected 应 fallback 到全塞 (不改变现状)"

    # 11. _compute_d_proxy 边界
    assert _compute_d_proxy([]) == 0.0
    assert _compute_d_proxy([0.0]) == 0.0
    assert _compute_d_proxy([1.0]) == 0.0  # 单段 100% → D=0 (无多样性)
    # 两段均匀 → D = ln(2) ≈ 0.693
    d2 = _compute_d_proxy([0.5, 0.5])
    assert abs(d2 - math.log(2)) < 1e-6, f"两段均匀 D 应=ln2, got {d2}"
    # 4 段均匀 → D = ln(4) ≈ 1.386
    d4 = _compute_d_proxy([0.25, 0.25, 0.25, 0.25])
    assert abs(d4 - math.log(4)) < 1e-6

    # 12. RoutingDecision.to_dict
    dec = route_context_segments(phase="plan", task_message="design X", threshold=0.5)
    d = dec.to_dict()
    assert set(d.keys()) == {"weights", "selected", "d_proxy", "n_segments", "n_total"}
    assert d["n_total"] == len(CONTEXT_SEGMENTS)
    assert d["n_segments"] == len(dec.selected)
    assert d["n_segments"] <= d["n_total"]

    # 13. log_routing_decision 不崩
    log_routing_decision(dec, "plan", "design X")

    # 14. 所有 phase 都有完整权重表
    for phase_name in ("perceive", "hypothesize", "plan", "execute",
                        "validate", "learn", "report"):
        weights = _PHASE_WEIGHTS[phase_name]
        assert set(weights.keys()) == set(CONTEXT_SEGMENTS), \
            f"phase {phase_name} 权重表应覆盖所有段"

    # 15. 所有段权重在 [0, 1]
    for phase_name, weights in _PHASE_WEIGHTS.items():
        for seg, w in weights.items():
            assert 0.0 <= w <= 1.0, \
                f"phase {phase_name} seg {seg} 权重 {w} 不在 [0,1]"

    print(f"context_router selfcheck All passed "
          f"(D_proxy range: 0..{math.log(len(CONTEXT_SEGMENTS)):.3f})")
