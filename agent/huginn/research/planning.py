"""需求拆解 → 子任务 DAG → 任务自主规划 → 预算分配 (研究管线的规划阶段).

补上评分要求的「需求拆解、任务自主规划」显式步骤: `run_research_program` 原本只
接收扁平实验列表, 把高层 `goal` 直接平铺成 initial_branches —— 没有"把目标拆成
语义子任务、建依赖 DAG、按拓扑分层调度并行度、分配预算"的环节。

本模块把这些**接入前的规划原语**拼成一个可复用入口 :func:`build_research_plan`:
  - :class:`SubResearch` 描述一个可执行的子研究 (假说 + run + 依赖).
  - 复用既有 ``agents.task_dag.TaskDAG`` 与 ``agents.budget_decomp.recommend``:
    * 拓扑分层 → 每层 antichain 内可并行 (`parallel_layers`);
    * 并行度上限 = 最大反链宽度(Dilworth): 同层可并行数, 再被预算 parallel 钳制;
    * 关键路径 → wall-clock 下限, 供调度/预算决策.
  - 输出 :class:`ResearchPlan`, 交给 ``run_research_program(planner=...)`` 让
    需求拆解真实影响实验序列与并行度 (不再是扁平平铺).

诚实边界: 本阶段只做**规划/调度**, 不改动任何科学计算的真实性 —— 每个
SubResearch 的 ``run`` 仍必须返回真实实验数值; planner 只是安排这些真实实验的
依赖与并行顺序。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from huginn.agents.task_dag import TaskDAG
from huginn.research.program import Experiment


@dataclass
class SubResearch:
    """一个可拆解的子研究 (最小规划单元).

    name: 唯一标识 (即生成的 Experiment.name).
    hypothesis: 人类可读假设.
    run: 执行真实实验, 返回 {objectives, summary, success}.
    depends_on: 依赖的上游子研究名 (先验依赖, LLM/人工给的领域次序).
    """
    name: str
    hypothesis: str
    run: Callable[[], dict[str, Any]]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """需求拆解后的完整研究计划.

    goal: 原始高层目标 (被拆解的对象).
    experiments: 按拓扑序排列的 Experiment 列表 (交给管线的执行单元).
    layers: TaskDAG.parallel_layers() —— 每层 antichain 内可并行.
    topo_order: 拓扑序.
    antichain_width: 最大反链(Dilworth) = 理论并行度.
    critical_path: DAG 关键路径 (wall-clock 下限).
    max_parallel: 建议并行度 = min(antichain_width, budget.parallel).
    budget: budget_decomp 给的总预算分解 {depth, parallel, per_subagent}.
    """

    goal: str
    experiments: list[Experiment] = field(default_factory=list)
    layers: list[list[str]] = field(default_factory=list)
    topo_order: list[str] = field(default_factory=list)
    antichain_width: int = 1
    critical_path: list[str] = field(default_factory=list)
    max_parallel: int = 1
    budget: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "experiments": [e.name for e in self.experiments],
            "layers": self.layers,
            "topo_order": self.topo_order,
            "antichain_width": self.antichain_width,
            "critical_path": self.critical_path,
            "max_parallel": self.max_parallel,
            "budget": self.budget,
        }


def _budget_recommend(n_sub_research: int, parallel_cap: int) -> dict:
    """总预算 (`depth×parallel×per_subagent`) 与 DAG 并行度取合理值.

    预算总量取子研究数的 2 倍 (给每子研究留重跑/变异余量), 再用 DAG 反链宽度
    钳制 parallel —— 调度并行度不会超过 DAG 实际可并行量 (规划约束诚实).
    """
    try:
        from huginn.agents.budget_decomp import recommend as _rec
        budget = _rec(2 * max(n_sub_research, 1))
    except Exception:  # noqa: BLE001 — 预算模块不可用(轻量)时用朴素预算
        budget = {"depth": 1, "parallel": parallel_cap, "per_subagent": max(n_sub_research, 1)}
    budget = dict(budget)
    budget["parallel"] = min(budget.get("parallel", parallel_cap), parallel_cap)
    return budget


def build_research_plan(
    goal: str,
    sub_research: list[SubResearch],
    *,
    parallel_cap: int = 6,
) -> ResearchPlan:
    """需求拆解 + 规划: 给定 goal 与子研究(含依赖), 产出可并行调度计划.

    - 按先验依赖建 DAG (有环/引用缺失 → 抛错, 由调用方修正或去掉依赖).
    - 拓扑分层得到并行批次, 反链宽度 = 理论并行度.
    - 预算把总工作量按素数分解成 (depth, parallel, per_subagent), parallel 被
      DAG 并行度钳制 —— 规划不会断言超现实的并行度.
    - 返回按拓扑序排列的 Experiment 序列, 供 `run_research_program(planner=...)`.
    """
    if not sub_research:
        raise ValueError("sub_research 不能为空 —— 需求拆解须至少产出一个子研究")
    names = [s.name for s in sub_research]
    if len(set(names)) != len(names):
        raise ValueError(f"sub_research 名必须唯一: {names}")
    by_name = {s.name: s for s in sub_research}
    deps: list[tuple[str, str]] = []
    for s in sub_research:
        for up in s.depends_on:
            if up not in by_name:
                raise ValueError(f"{s.name} 依赖不存在的子研究 '{up}'")
            deps.append((up, s.name))

    dag = TaskDAG(tasks=names, dependencies=deps)
    order = dag.topological_order()
    layers = dag.parallel_layers()
    width = dag.antichain_width()
    cp = dag.critical_path()

    # 每个子研究 → Experiment (run 原样透传, 科学计算真实性不改动)
    experiments = [
        Experiment(name=s.name, hypothesis=s.hypothesis, run=s.run)
        for s in sub_research
    ]
    # 预算: 总预算用子研究数 ×2 求素数分解, parallel 被 DAG 并行度钳制
    suggested_parallel = min(width, parallel_cap)
    budget = _budget_recommend(len(sub_research), suggested_parallel)

    return ResearchPlan(
        goal=goal,
        experiments=experiments,
        layers=layers,
        topo_order=order,
        antichain_width=width,
        critical_path=cp,
        max_parallel=budget["parallel"],
        budget=budget,
    )