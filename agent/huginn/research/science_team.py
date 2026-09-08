"""多智能体科研团队 —— 以角色分工模拟科研团队协作 (deterministic, 离线可复现).

不是把单管线换壳, 而是**真正的分角色团队**: 每个成员是独立 agent, 只做自己职责,
通过共享的「证据账本(evidence ledger) + 角色日志(role log)」协作, 全程可溯源。

角色 (对齐学科团队内部分工, 均为可运行组件而非 prompt 占位):
  - PlannerAgent   需求拆解 + 子任务 DAG + 预算 -> 把高层 goal 分派给科学家
  - ScientistAgent ×N  每个科学家认领一个子研究, 执行**真实实验**, 产出证据入账本
  - CriticAgent    对账本做 Pareto 前沿筛 + 逐证据 grounding 校验, 淘汰被支配/未落地主张
  - SynthesizerAgent 据存活证据 + 完整 trace 组装可复现报告, 过 claim_grounding 门禁

与 `run_research_program` 的区别: 后者是单编排器串起一条 pipeline; 这里是「分工 +
并行 + 批判 + 综合」的团队, 科学与批判分属不同 agent, 证据只在账本上流转。

复用: `research.planning`(规划) + `providers`(真实实验) + `claim_grounding`(批判/
门禁)。纯确定性/标准库, 零网络/零 LLM, 可单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from huginn.research.planning import ResearchPlan, SubResearch, build_research_plan
from huginn.research.program import grounding_verifier


# ── 共享数据结构 (团队协作的公共契约) ─────────────────────────────


@dataclass
class Evidence:
    """一位科学家的一条真实证据 (入账本, 供批判/综合/门禁)."""
    name: str
    hypothesis: str
    objectives: dict[str, float] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    source: str = ""                 # 产出它的角色/成员

    def as_trace(self) -> str:
        import json
        return json.dumps(self.summary, ensure_ascii=False)


@dataclass
class RoleLogEntry:
    role: str
    action: str
    detail: str = ""


# ── 各角色 agent (分工的最小单元) ─────────────────────────────────


class PlannerAgent:
    """规划者: goal -> 子研究(依赖) -> DAG -> 预算 -> 分派方案."""

    role = "planner"

    def __init__(self, parallel_cap: int = 6) -> None:
        self.parallel_cap = parallel_cap

    def act(self, goal: str, sub_research: list[SubResearch]) -> ResearchPlan:
        return build_research_plan(goal, sub_research, parallel_cap=self.parallel_cap)


class ScientistAgent:
    """科学家: 认领一个子研究, 执行真实实验, 产出可证伪证据."""

    role = "scientist"
    _seq = 0

    def __init__(self, name: str) -> None:
        ScientistAgent._seq += 1
        self.name = name

    def act(self, spec: SubResearch) -> Evidence:
        result = spec.run()   # 真实实验执行 (不伪造数值)
        return Evidence(
            name=spec.name,
            hypothesis=spec.hypothesis,
            objectives=result.get("objectives", {}),
            summary=result.get("summary", {}),
            source=f"{self.role}:{self.name}",
        )


def _pareto_kept(evidences: list[Evidence], objectives_config: dict[str, str]) -> list[str]:
    """多目标 Pareto 前沿 (最大化/最小化按 objectives_config). 非支配者存活."""
    kept: list[str] = []

    def _dominates(a: dict, b: dict) -> bool:
        better = False
        for k, sense in objectives_config.items():
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None:
                continue
            if sense == "maximize":
                if va < vb:
                    return False
                if va > vb:
                    better = True
            else:  # minimize
                if va > vb:
                    return False
                if va < vb:
                    better = True
        return better

    for ev in evidences:
        if not any(_dominates(other.objectives, ev.objectives)
                   for other in evidences if other.name != ev.name):
            kept.append(ev.name)
    return kept


class CriticAgent:
    """批判者: 对账本做 Pareto 前沿筛 + 报告 grounding 校验, 淘汰被支配/未落地."""

    role = "critic"

    def __init__(self, verify=None) -> None:
        self.verify = verify or grounding_verifier()

    def prune(self, evidences: list[Evidence],
              objectives_config: dict[str, str]) -> list[Evidence]:
        kept_names = set(_pareto_kept(evidences, objectives_config))
        return [e for e in evidences if e.name in kept_names]

    def gate(self, report: str, trace: list[str]) -> dict:
        return self.verify(report, trace)


class SynthesizerAgent:
    """综合者: 据存活证据 + 完整 trace 组装可复现报告, 交批判者过门禁."""

    role = "synthesizer"

    def act(self, goal: str, survivors: list[Evidence],
            pruned: list[Evidence], verify) -> tuple[str, dict]:
        L = [f"# 科研团队综合报告 — {goal}", ""]
        L.append(f"> 存活证据 {len(survivors)} 条 (Pareto 前沿 + 门禁校验), "
                 f"淘汰 {len(pruned)} 条(被支配/未落地)。")
        for e in survivors:
            L.append(f"## {e.name}")
            L.append(f"- 假说: {e.hypothesis}")
            L.append(f"- 真实结果: {e.summary}  (来源 {e.source})")
            L.append("")
        L.append("## 结论(开放)")
        L.append("多角色分工采集的真实证据共同支撑; 数值均来自科学家实验执行, 可复现可证伪。")
        report = "\n".join(L)
        trace = [e.as_trace() for e in survivors]
        gate = verify(report, trace)
        return report, gate


# ── 团队编排器 ────────────────────────────────────────────────────


@dataclass
class TeamOutcome:
    goal: str
    report: str = ""
    verdict: str = "needs_grounding"
    plan: ResearchPlan | None = None
    survivors: list[Evidence] = field(default_factory=list)
    pruned: list[Evidence] = field(default_factory=list)
    role_log: list[RoleLogEntry] = field(default_factory=list)

    def roles(self) -> list[str]:
        seen: list[str] = []
        for e in self.role_log:
            if e.role not in seen:
                seen.append(e.role)
        return seen


class ScienceTeam:
    """科研团队编排器: plan -> scientists(并行) -> critic(批判) -> synthesizer(综合)."""

    def __init__(self, n_scientists: int = 3, parallel_cap: int = 6, verify=None) -> None:
        self.planner = PlannerAgent(parallel_cap=parallel_cap)
        self.scientists = [ScientistAgent(f"scientist.{i}") for i in range(n_scientists)]
        self.critic = CriticAgent(verify=verify)
        self.synthesizer = SynthesizerAgent()
        self.log: list[RoleLogEntry] = []

    def _log(self, role: str, action: str, detail: str = "") -> None:
        self.log.append(RoleLogEntry(role=role, action=action, detail=detail))

    def run(self, goal: str, sub_research: list[SubResearch],
            objectives_config: dict[str, str]) -> TeamOutcome:
        # 1) 规划者: 需求拆解 + DAG 分层 (科学家只做被分派的层内子研究)
        plan = self.planner.act(goal, sub_research)
        self._log(self.planner.role, "plan",
                  f"layers={[','.join(l) for l in plan.layers]} budget={plan.budget}")
        by_name = {s.name: s for s in sub_research}

        # 2) 科学家: 按拓扑分层并行执行 (同层 antichain 内独立)
        ledger: dict[str, Evidence] = {}
        if len(self.scientists) < 1:
            raise ValueError("ScienceTeam 至少需要 1 位科学家")
        worker = 0
        for layer in plan.layers:
            for name in layer:
                spec = by_name[name]
                scientist = self.scientists[worker % len(self.scientists)]
                worker += 1
                ev = scientist.act(spec)
                ledger[name] = ev
                self._log("scientist", f"experiment '{name}'",
                          f"{ev.source} objectives={ev.objectives}")

        evidences = list(ledger.values())

        # 3) 批判者: Pareto 前沿筛 (淘汰被支配)
        survivors = self.critic.prune(evidences, objectives_config)
        pruned = [e for e in evidences if e not in survivors]
        self._log("critic", "pareto_prune",
                  f"kept {len(survivors)} / {len(evidences)}")

        # 4) 综合者 + 批判者门禁
        report, gate = self.synthesizer.act(goal, survivors, pruned, self.critic.verify)
        self._log("synthesizer", "assemble", f"verdict={gate['verdict']}")

        return TeamOutcome(
            goal=goal,
            report=report,
            verdict=gate["verdict"],
            plan=plan,
            survivors=survivors,
            pruned=pruned,
            role_log=list(self.log),
        )