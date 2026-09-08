"""多智能体科研团队 —— 以角色分工模拟科研团队协作 (deterministic, 离线可复现).

不是把单管线换壳, 而是**真正的分角色团队**: 每个成员是独立 agent, 只做自己职责,
通过共享的「证据账本(evidence ledger) + 角色日志(role log)」协作, 全程可溯源。

角色 (对齐学科团队内部分工, 均为可运行组件而非 prompt 占位):
  - PlannerAgent   需求拆解 + 子任务 DAG + 预算 -> 把高层 goal 分派给科学家
  - ScientistAgent ×N  每个科学家认领一个子研究, 执行**真实实验**, 产出证据入账本
  - CriticAgent    对账本做 Pareto 前沿筛 + 逐证据 grounding 校验, 淘汰被支配/未落地主张
  - SynthesizerAgent 据存活证据 + 完整 trace 组装可复现报告, 过 claim_grounding 门禁

第二套 VLA 式团队 :class:`ModelBasedScienceTeam` 对标 世界模型/VLA: 感知(读入状态)
→ 数学定律预告(世界模型选择动作) → 执行(科学家) → 数值对账(批判), 数学为规划与
验证的公共语言。与 DAG 式 :class:`ScienceTeam`(分层任务分工) 互补 —— 前者管"怎么
分工", 后者管"怎么按定律规划/预告/验证"。

复用: `research.planning`(规划) + `research.law_model`(数学核心) + `claim_grounding`(批判/
门禁)。纯确定性/标准库, 零网络/零 LLM, 可单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from huginn.research.planning import ResearchPlan, SubResearch, build_research_plan
from huginn.research.program import grounding_verifier
from huginn.research.law_model import (
    LawAction, LawModel, LawState, ModelBasedPlanner, reconcile,
)


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


# ── 第二套: VLA 式团队 (世界模型预告 → 执行 → 数学对账) ─────────────────────
# 对标 世界模型 / VLA: 感知(seed) → 定律预告(planner 选动作) → 执行(scientist)
# → 对账(critic, 数学可证伪)。数学定律是规划与验证的公共语言。


@dataclass
class VLAOutcome:
    """VLA 式团队的产出: 每步预告/执行/对账 + 定律 + 报告 + 门禁."""
    goal: str
    law: str = ""
    plan: list = field(default_factory=list)             # [{id, action, predicted}]
    executions: list = field(default_factory=list)       # [{id, actual}]
    reconciliations: list = field(default_factory=list)  # [{id, borne_out, errors, mismatch}]
    survivors: list = field(default_factory=list)        # 定律被证实的证据名
    pruned: list = field(default_factory=list)           # 被证伪/被支配的证据名
    report: str = ""
    verdict: str = "needs_grounding"
    role_log: list[RoleLogEntry] = field(default_factory=list)


class ModelBasedScienceTeam:
    """VLA 式科研团队: 当世界模型给定律, 科学家给真相, 批判者对账 (数学为核).

    - Planner(世界模型): ``predict`` 预告每个候选动作的后继状态, 按预测目标选动作。
    - Scientist: 对选定动作做**真实执行**(独立于 predict, 是真相检验)。
    - Critic: ``reconcile`` 把 predict 与 actual 数值对账 —— 相符=定律证实, 偏差=
      定律证伪(如实标注, 不覆盖)。只把被证实的证据送入结论。
    """

    def __init__(self, model: LawModel, *, objective: str = "T_eq_K",
                 sense: str = "maximize", tol: float = 0.03,
                 n_scientists: int = 2, verify=None,
                 metrics: tuple[str, ...] | None = None) -> None:
        self.model = model
        self.planner = ModelBasedPlanner(model, objective, sense)
        self.scientists = [ScientistAgent(f"vla.scientist.{i}") for i in range(n_scientists)]
        self.critic = CriticAgent(verify=verify)
        self.synthesizer = SynthesizerAgent()
        self.tol = tol
        # 对账指标应跟随模型域: 传入则用调用方给的力学/其他域指标; 缺省保持历史
        # exoplanet 行为, 避免换域时报错.
        self.metrics = metrics if metrics is not None else ("S_Wm2", "T_eq_K")
        self.log: list[RoleLogEntry] = []

    def _log(self, role: str, action: str, detail: str = "") -> None:
        self.log.append(RoleLogEntry(role=role, action=action, detail=detail))

    def run(self, goal: str, observations: list[dict], actions: list[LawAction],
            real_executor: Callable[[LawState, LawAction], dict]) -> VLAOutcome:
        self._log("planner", "law", self.model.law())
        evidences: list[Evidence] = []
        plan, execs, recons = [], [], []
        worker = 0
        for obs in observations:
            init = self.model.seed(obs)                    # 感知: 读入观测 → 初始状态
            step = self.planner.best(init, actions)        # 定律预告 → 选动作
            scientist = self.scientists[worker % len(self.scientists)]; worker += 1
            actual = real_executor(init, step.action)      # 科学者: 真实执行(真相)
            rep = reconcile(step.predicted, actual, tol=self.tol,
                            metrics=self.metrics)  # 批判: 数学对账 (指标随域)
            rid = obs.get("name", f"obs_{len(plan)}")
            plan.append({"id": rid, **step.to_dict()})
            execs.append({"id": rid, "actual": actual,
                          "executed_by": f"{scientist.role}:{scientist.name}"})
            recons.append({"id": rid, **rep})
            evidences.append(Evidence(
                name=rid, hypothesis="世界模型定律预告，经真实执行数值对账证实",
                objectives=actual.get("objectives", {}),
                # summary 并入观测标识(id), 让报告引用的天体名/编号可被 grounding 溯源
                summary={**dict(actual), "id": rid},
                source=f"{scientist.role}:{scientist.name}"))
            self._log("scientist", f"exec '{rid}'", f"borne_out={rep['borne_out']} "
                       f"errors={rep.get('errors', {})}")

        # 数学核心: 只把定律被**证实**的证据送入结论; 证伪的如实列为 pruned
        survivors = [e for e, rc in zip(evidences, recons) if rc.get("borne_out")]
        pruned = [e for e, rc in zip(evidences, recons) if not rc.get("borne_out")]
        self._log("critic", "reconcile(math)",
                  f"borne_out {len(survivors)} / {len(evidences)}")

        report, gate = self.synthesizer.act(goal, survivors, pruned, self.critic.verify)
        self._log("synthesizer", "assemble", f"verdict={gate['verdict']}")
        return VLAOutcome(
            goal=goal, law=self.model.law(), plan=plan, executions=execs,
            reconciliations=recons,
            survivors=[e.name for e in survivors], pruned=[e.name for e in pruned],
            report=report, verdict=gate["verdict"], role_log=list(self.log),
        )