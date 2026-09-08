"""收敛聚合头 (Consolidation Head) —— 缺陷六("多头摊派、没有收敛点")的架构级解药.

问题背景: 每新增一个认知视角 (C-Space / structural_gate / law_model_used /
plan_revision / grounding_audit / harness ...) 就往 :class:`ResearchOutcome`
直写一个新字段, 报告页眉 / Self-Harness / 账本各自再 ad hoc 扫一遍 —— 注意力被
"每个模块各说各话再拼总报告"派发进越来越多的头, 且没有一条"该把注意力放在哪"的
仲裁策略 (多头注意力的架构级表达, 继承了稀释与塌缩的双重缺陷).

本模块提供一个**唯一收敛点** :func:`consolidate`: 所有治理/审计头必须先收敛成
:class:`HeadResult` 契约再注册进来, 聚合头按仲裁策略表产出单一收敛视图
:class:`Consolidated`. 对表注意力:

  - 单头投影  : 多头上加一层线性投影 → 这里 = 所有 head 收敛成一个 bounded 视图;
  - 残差连接  : 聚合不丢原头证据 → 每个 head 保留 ref(可证伪定位) 与原始产物;
  - 多头塌缩  : 显式度量头间分歧 (diversity), 拒绝静默平均相互矛盾的 head;
  - 可反驳性  : consolidate 是纯函数, 给定同批 head 与策略表输出确定, 重跑即复现.

诚实边界 (失效模式, 可检验):
  1. 若头数被无限扩充, 聚合头会退化成"又一张表" → 头数需有上限/新头须先证明增益;
  2. 若 gate 否决权配置过紧, 聚合头从仲裁者变成又一个锚 → 用 gate_blocked 比率监控.

P1 阶段定位: 纯适配器 —— 不改任何既有消费路径(报告/self-harness/账本仍读旧字段),
只新增 out.consolidated 这一份收敛视图, 供后续 P2/P3 切换消费者与执法.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# evidence 三态 (与 harness 同语义)
EVIDENCE_OBSERVED = "observed"
EVIDENCE_UNOBSERVED = "unobserved"
EVIDENCE_MISSING = "missing"

# 分值: passed=1 / unobserved=0.5 / missing=0 / observed-but-failed=0
_SCORE = {
    "passed": 1.0,
    "unobserved": 0.5,
    "missing": 0.0,
    "failed": 0.0,
}


@dataclass
class HeadResult:
    """一个治理/审计头的统一出口 (所有模块的唯一契约).

    id:      稳定标识, 如 "gate.structural" / "audit.score_usage" / "wm.actually_used"
    name:    人类可读名
    evidence:/outcome: 与 Self-Harness 同语义的三态 + 四态判据
    detail:  细节(供 reading 逐条还原)
    ref:     指向 trace-id / out.* 的来源 (可证伪定位)
    weight:  聚合加权 (默认 1.0)
    gate:    True → 该头对"报告是否以 pass 定稿"有否决权
    """

    id: str
    name: str
    evidence: str
    outcome: str
    detail: str = ""
    ref: str = ""
    weight: float = 1.0
    gate: bool = False

    def score(self) -> float:
        return _SCORE.get(self.outcome, 0.0)


@dataclass
class Consolidated:
    """多头 → 单一收敛视图 (聚合后唯一定稿依据)."""

    verdict: str                                     # 聚合判定: pass / gate_blocked / conflict
    grounding: str                                   # 透传 grounding 门禁判定 (= out.verdict, P1 一致性锚)
    gates_failed: list[str] = field(default_factory=list)   # 否决项 id 清单
    conflicts: list[list[str]] = field(default_factory=list)  # 显式冲突对(不静默合并)
    diversity: float = 0.0                           # 头间分歧度 (防御多头塌缩)
    heads: list[str] = field(default_factory=list)   # 本次真实注册的头 (有界清单)
    score: float = 0.0                               # 加权总分 (0..1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "grounding": self.grounding,
            "gates_failed": self.gates_failed,
            "conflicts": self.conflicts,
            "diversity": round(self.diversity, 3),
            "heads": self.heads,
            "score": round(self.score, 3),
        }


def consolidate(
    heads: list[HeadResult],
    *,
    grounding_verdict: str = "needs_grounding",
) -> Consolidated:
    """多头 → 单一收敛视图 (纯函数, 无副作用).

    仲裁策略 (P1):
      - **gate 否决**: 任一 ``gate=True`` 且 outcome=``failed`` 的头 → 聚合判定
        ``gate_blocked``, 并把否决项列入 ``gates_failed`` —— 报告不得以 pass 定稿.
      - **冲突显式化**: 若既有 gate 头 failed、又有 gate 头 passed → 记入
        ``conflicts``, 拒绝静默平均相互矛盾的头.
      - **多头塌缩防御**: ``diversity`` = 去重 outcome 数 / 注册头数 —— 全部头
        结果一致时接近 0(单文化雷达), 高度分化时接近 1.
      - **grounding 透传**: ``grounding`` 原样携带 grounding 门禁判定, 保证
        P1 阶段 ``consolidated.grounding == out.verdict`` 恒成立 (向后兼容锚).
    """
    heads = list(heads)
    gates_failed = [h.id for h in heads if h.gate and h.outcome == "failed"]
    gate_passed = [h.id for h in heads if h.gate and h.outcome == "passed"]
    conflicts: list[list[str]] = []
    if gates_failed and gate_passed:
        conflicts.append([",".join(gate_passed), ",".join(gates_failed)])

    n = len(heads)
    diversity = (len({h.outcome for h in heads}) / n) if n else 0.0

    if gates_failed:
        verdict = "gate_blocked"
    elif conflicts:
        verdict = "conflict"
    else:
        verdict = "pass"

    w_sum = sum(h.weight for h in heads) or 1.0
    score = sum(h.score() * h.weight for h in heads) / w_sum

    return Consolidated(
        verdict=verdict,
        grounding=grounding_verdict,
        gates_failed=gates_failed,
        conflicts=conflicts,
        diversity=diversity,
        heads=[h.id for h in heads],
        score=score,
    )