"""分支剪枝策略 + 决策点注册 (branch_policy) — 让"剪枝"在科研场景下可逆、可参与.

科研任务的价值不可知: "低潜力"可能是真金, 一刀砍掉代价不可逆。因此本模块不在
agent 内部闷头剪, 而是把**剪枝决策在关键关口召回给用户** (见
docs/cost-participation-contract.md §3)。本模块提供三件事:

1. **BranchScore / UCB 打分**: 用 `mean + 不确定性红利` 而非纯均值判分支。
   低均值 + 高不确定性 → 不敢砍 (可能是真金); 只有"探测够多、方差小、确实
   引不出东西"才建议降级。不确定性来自探测次数 / 结果方差 / 与已知解的分歧度。

2. **BranchState**: 分支生命周期 `active → hibernating → abandoned`, 以及可逆的
   `revive` (复活)。休眠 (hibernating) 保留一条廉价 lifeline 继续低频探测, 而非硬砍。

3. **DecisionPointRegistry**: 决策点注册/查询/裁决/过期。agent 在关口发起决策点,
   用户裁决 (approved/edited/denied), 逾时走保守默认 (休眠/软停, 不硬砍)。

这是后端无关的轻量层: 持久化/事件发射由调用方 (或上层) 注入, 本模块只负责
"打分 → 状态迁移 → 决策点登记" 的可观测语义。

用法::

    from huginn.branch_policy import (
        BranchScore, default_dev_stage, BranchState, Branch, DecisionPointRegistry,
    )

    score = BranchScore(mean_value=0.3, n_probes=6, variance=0.2)
    rec = score.recommendation()          # "hibernate" — 低均值高不确定性
    branch = Branch(branch_id="b_7", score=score)
    reg = DecisionPointRegistry()
    dp = reg.open("hibernate", session="s1", branch=branch,
                  narrative={"phase": "converge", "cost_usd": 12.4},
                  agent_judgment=score.to_judgment("建议休眠保留 lifeline"))
    reg.resolve(dp.id, decision="approved", option="hibernate_lifeline")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "default_dev_stage",
    "BranchScore",
    "BranchState",
    "Branch",
    "DecisionPoint",
    "DecisionPointRegistry",
    "get_decision_point_registry",
    "reset_decision_point_registry",
]

# 分支研发阶段: 探索期不确定性红利更高 (不敢砍), 收敛期收紧.
# 数值 = 不确定性红利权重 β (越大越保守, 越不砍)。
DevStage = Literal["explore", "converge", "verify"]


def default_dev_stage() -> dict[str, float]:
    """默认研发阶段 → 不确定性红利权重 β. 探索最保守, 验证最能砍."""
    return {
        "explore": 1.5,
        "converge": 1.0,
        "verify": 0.5,
    }


@dataclass(frozen=True)
class BranchScore:
    """一个假设分支的 UCB 打分: `ucb = α·mean + β(stage)·uncertainty`."""

    mean_value: float = 0.0      # 已实现价值 (归一化 0..1)
    n_probes: int = 0            # 探测次数 — 越少不确定性越高
    variance: float = 0.0        # 结果方差 — 越大越不确定
    divergence: float = 0.0      # 与已知解的分歧度 0..1 — 越大越可能藏着新东西
    alpha: float = 1.0           # 已实现价值权重
    stage: str = "explore"

    # ── 不确定性估计 ─────────────────────────────────────────────
    def uncertainty(self) -> float:
        """不确定性系数 ∈ [0,1]: 探测少 / 方差大 / 分歧大 → 高.

        组合: 先取 (1 - 探测饱和) 与 (方差+分歧)/2 的较大者, 再叠加分歧加成。
        探测饱和: n>=10 视为试透 (0), n=0 视为完全未知 (1)。
        """
        unsat = 1.0 - min(1.0, self.n_probes / 10.0)
        spread = min(1.0, (self.variance + self.divergence) / 2.0)
        return min(1.0, max(unsat, spread) + 0.3 * self.divergence)

    def beta(self, ratios: dict[str, float] | None = None) -> float:
        """当前阶段的不确定性红利权重 β."""
        ratios = ratios or default_dev_stage()
        return ratios.get(self.stage, 1.0)

    def ucb(self, ratios: dict[str, float] | None = None) -> float:
        """UCB 打分 = α·mean + β·uncertainty. 高 = 更值得继续投."""
        return self.alpha * self.mean_value + self.beta(ratios) * self.uncertainty()

    def recommendation(self, ratios: dict[str, float] | None = None) -> str:
        """给 agent 建议 (非定论, 裁决权在用户):
           explore  → 继续投
           hibernate → 低均值但高不确定性 (可能真金) → 休眠保留 lifeline
           abandon  → 高确定性且低价值 → 建议砍 (仍可召回用户)
        """
        u = self.uncertainty()
        if u >= 0.6:
            # 还没试探透 — 不确定是"低潜力"还是"真金", 不能砍
            return "hibernate"
        if self.ucb(ratios) < 0.3:
            return "abandon"
        return "explore"

    def to_judgment(self, reason: str = "") -> dict[str, Any]:
        """转成决策点里的 agent_judgment payload (契约 §3.2)."""
        return {
            "mean_value": round(self.mean_value, 3),
            "uncertainty": round(self.uncertainty(), 3),
            "ucb": round(self.ucb(), 3),
            "recommendation": self.recommendation(),
            "reason": reason,
        }


class BranchState:
    """分支生命周期状态 — 枚举常量 (用字符串以免引入额外依赖)."""

    ACTIVE = "active"
    HIBERNATING = "hibernating"
    ABANDONED = "abandoned"


@dataclass
class Branch:
    """一个科研假设分支. 休眠时保留 lifeline + 复活条件 (可逆)."""

    branch_id: str
    score: BranchScore = field(default_factory=BranchScore)
    state: str = BranchState.ACTIVE
    # 休眠时保留: 廉价 lifeline 描述 (低精度/低档位/低频探测).
    lifeline: str = ""
    # 复活条件: 满足任一 → 回来看它 (可写, 供上层求值).
    revive_conditions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def hibernate(self, *, lifeline: str, revive_conditions: list[str] | None = None) -> None:
        """软停止: 保留 lifeline 继续低频探测, 不硬砍. 可逆 via revive."""
        self.state = BranchState.HIBERNATING
        self.lifeline = lifeline
        if revive_conditions:
            self.revive_conditions = list(revive_conditions)

    def revive(self, *, reason: str = "") -> Branch:
        """复活一条休眠分支 (带回 active). 可逆性的关键."""
        self.state = BranchState.ACTIVE
        if reason:
            self.revive_conditions.append(f"revived: {reason}")
        return self

    def abandon(self) -> None:
        """真正放弃 (高确定性 + 低价值). 不可逆前应经用户裁决."""
        self.state = BranchState.ABANDONED

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "state": self.state,
            "lifeline": self.lifeline,
            "revive_conditions": list(self.revive_conditions),
            "score": self.score.to_judgment(),
        }


@dataclass
class DecisionPoint:
    """一次决策点 (契约 §3.2)."""

    kind: str                      # prune | hibernate | degrade | pause | resume
    session_id: str = "default"
    branch_id: str = ""
    status: str = "pending"        # pending | approved | edited | denied | expired
    narrative: dict[str, Any] = field(default_factory=dict)
    agent_judgment: dict[str, Any] = field(default_factory=dict)
    options: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: f"dp_{uuid.uuid4().hex[:12]}")
    created_at: float = field(default_factory=time.time)

    def resolve(self, *, decision: str, option: str = "") -> None:
        """用户裁决: approved / edited / denied."""
        self.status = decision
        self.response = {"option": option, "at": time.time()}

    def expire(self) -> None:
        """逾时未响应 → 走保守默认 (见契约 §3.1)."""
        self.status = "expired"
        self.response = {"option": "_default", "at": time.time()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "session_id": self.session_id,
            "branch_id": self.branch_id,
            "narrative": self.narrative,
            "agent_judgment": self.agent_judgment,
            "options": self.options,
            "response": self.response,
        }


class DecisionPointRegistry:
    """决策点注册器: 登记/查询/裁决/过期. sev 可观测."""

    def __init__(self) -> None:
        self._points: dict[str, DecisionPoint] = {}

    def open(
        self,
        kind: str,
        *,
        session_id: str = "default",
        branch: Branch | None = None,
        narrative: dict[str, Any] | None = None,
        agent_judgment: dict[str, Any] | None = None,
        options: list[dict[str, Any]] | None = None,
    ) -> DecisionPoint:
        """发起一个决策点 (agent 主动召回用户)."""
        if options is None:
            options = [
                {"id": "hibernate_lifeline", "label": "休眠(保留 lifeline)", "risk": "low"},
                {"id": "continue_invest", "label": "继续投", "risk": "medium"},
                {"id": "abandon", "label": "放弃", "risk": "low"},
            ]
        dp = DecisionPoint(
            kind=kind,
            session_id=session_id,
            branch_id=branch.branch_id if branch else "",
            narrative=narrative or {},
            agent_judgment=agent_judgment or {},
            options=options,
        )
        self._points[dp.id] = dp
        return dp

    def get(self, dp_id: str) -> DecisionPoint | None:
        return self._points.get(dp_id)

    def resolve(self, dp_id: str, *, decision: str, option: str = "") -> DecisionPoint | None:
        """用户裁决; 返回更新后的决策点 (不存在返回 None)."""
        dp = self._points.get(dp_id)
        if dp is None:
            return None
        dp.resolve(decision=decision, option=option)
        return dp

    def expire(self, dp_id: str) -> DecisionPoint | None:
        dp = self._points.get(dp_id)
        if dp is None:
            return None
        dp.expire()
        return dp

    def pending(self, session_id: str | None = None) -> list[DecisionPoint]:
        """未决决策点."""
        return [
            p for p in self._points.values()
            if p.status == "pending" and (session_id is None or p.session_id == session_id)
        ]

    def list(self, session_id: str | None = None) -> list[DecisionPoint]:
        if session_id is None:
            return list(self._points.values())
        return [p for p in self._points.values() if p.session_id == session_id]

    def clear(self) -> None:
        self._points.clear()


# 进程级单例.
_registry: DecisionPointRegistry | None = None


def get_decision_point_registry() -> DecisionPointRegistry:
    """全局单例."""
    global _registry
    if _registry is None:
        _registry = DecisionPointRegistry()
    return _registry


def reset_decision_point_registry() -> None:
    """测试辅助: 重建单例."""
    global _registry
    _registry = None


if __name__ == "__main__":
    # 自检
    # 1. 低均值高不确定性 → hibernate (不敢砍, 可能真金)
    s_unknown = BranchScore(mean_value=0.3, n_probes=2, variance=0.2, divergence=0.6)
    assert s_unknown.recommendation() == "hibernate", s_unknown.recommendation()
    # 2. 高确定性低价值 → abandon
    s_dead = BranchScore(mean_value=0.05, n_probes=12, variance=0.02, divergence=0.05)
    assert s_dead.recommendation() == "abandon", s_dead.recommendation()
    # 3. 高价值 → explore
    s_good = BranchScore(mean_value=0.8, n_probes=8, variance=0.1, divergence=0.2)
    assert s_good.recommendation() == "explore", s_good.recommendation()
    # 4. 分支休眠/复活可逆
    b = Branch(branch_id="b_7", score=s_unknown)
    b.hibernate(lifeline="vasp 粗精度", revive_conditions=["主支收敛后回看"])
    assert b.state == BranchState.HIBERNATING and b.lifeline
    b.revive(reason="用户要求续投")
    assert b.state == BranchState.ACTIVE and b.revive_conditions
    # 5. 决策点登记/裁决/过期
    reg = DecisionPointRegistry()
    dp = reg.open("hibernate", session_id="s1", branch=b, narrative={"phase": "converge"},
                  agent_judgment=s_unknown.to_judgment("建议休眠保留 lifeline"))
    assert len(reg.pending("s1")) == 1
    reg.resolve(dp.id, decision="approved", option="hibernate_lifeline")
    assert dp.status == "approved" and dp.response["option"] == "hibernate_lifeline"
    assert len(reg.pending("s1")) == 0
    dp2 = reg.open("prune", session_id="s1")
    reg.expire(dp2.id)
    assert dp2.status == "expired" and dp2.response["option"] == "_default"
    print("branch_policy self-check passed")
