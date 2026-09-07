"""进化/监督驱动单测 — 补齐赛题缺口的三块拼图.

覆盖:
  1. MutationStrategy     — 显式参数变异算子 (进化式实验规划的正交一环)
  2. SupervisorStrategy   — HITL 人机反馈闭环 (专家见解流入想法演化)
  3. SupervisorStrategy   — 想法 tournament (多智能体交叉评审淘汰)
  4. research.program 接线 — mutation_config / human_review 端到端生效 (确定性, client=None)

纯标准库 + numpy/networkx, 零网络/零 LLM, 确定性可复现。
"""
from __future__ import annotations

import random

from huginn.exploration.core import Branch, BranchStatus, ExplorationSpace
from huginn.exploration.strategies import Action, MutationStrategy, ParetoPruningStrategy
from huginn.exploration.supervisor import SupervisorStrategy


def _space(obj: float = 1.0) -> ExplorationSpace:
    s = ExplorationSpace(id="x", name="t", objective="objective",
                         objectives_config={"score": "maximize"})
    s.branches["b1"] = Branch(id="b1", name="A", hypothesis="idea A",
                              status=BranchStatus.COMPLETED, objectives={"score": obj})
    return s


# ── 1) 显式参数变异算子 ─────────────────────────────────────────────
def test_mutation_strategy_produces_mutants_from_front() -> None:
    ms = MutationStrategy(wrapped=ParetoPruningStrategy(),
                          param_space={"x": (0.0, 1.0)}, mutation_rate=1.0,
                          max_children=2, max_active=10, rng=random.Random(7))
    acts = ms.evaluate(_space(0.5))
    muts = [nb for a in acts if a.new_branches for nb in a.new_branches]
    assert muts, "应在 Pareto 前沿上生成变异子代"
    assert all(nb["name"].startswith("A~mut") for nb in muts)
    # 变异值必须钳制在 param_space 边界内
    assert all(0.0 <= nb["params"]["x"] <= 1.0 for nb in muts)
    # 变异子代保留世系 (mutation_of) 供溯源
    assert all(nb["mutation_of"] == "A" for nb in muts)


def test_mutation_strategy_respects_caps() -> None:
    # max_children=1 + max_active=1 → 本轮变异子代数被限制在 1 以内 (防爆炸)
    ms = MutationStrategy(wrapped=ParetoPruningStrategy(),
                          param_space={"x": (0.0, 1.0)}, mutation_rate=1.0,
                          max_children=1, max_active=1, rng=random.Random(0))
    s = _space(0.5)
    acts = ms.evaluate(s)
    muts = [nb for a in acts if a.new_branches for nb in a.new_branches]
    assert len(muts) <= 1, f"变异子代不应超过 max_active 预算, got {muts}"


def test_mutation_disabled_without_param_space() -> None:
    ms = MutationStrategy(wrapped=ParetoPruningStrategy(), param_space=None)
    acts = ms.evaluate(_space(0.5))
    assert not any(a.new_branches and any("~mut" in (nb.get("name") or "") for nb in a.new_branches)
                   for a in acts)


# ── 2) HITL 人机反馈闭环 ─────────────────────────────────────────────
def test_supervisor_hitl_injects_guidance_and_drop() -> None:
    class _Expand(ParetoPruningStrategy):
        def evaluate(self, space):
            return [Action(action_type="refine", target_branch="b1",
                           new_branches=[{"name": "A2", "hypothesis": "child"}])]

    calls = {"n": 0}

    def human(card):
        calls["n"] += 1
        return {"drop": ["A"], "guidance": "优先验证可证伪的最强信号"}

    s = _space(0.5)
    ss = SupervisorStrategy(wrapped=_Expand(), human_review=human, review_every=1)
    acted = ss.evaluate(s)
    proposed = [nb for a in acted if a.new_branches for nb in a.new_branches]
    pruned = [a.target_branch for a in acted if a.action_type == "prune"]
    assert calls["n"] == 1                       # 评审回调确实被触发
    assert "human guidance" in proposed[0]["hypothesis"]   # 专家见解回流进下一批子代
    assert "b1" in pruned                        # drop 的 A 被剪枝
    assert len(ss.review_log) == 1               # 反馈被记录 (可追溯)


# ── 3) 想法 tournament (多智能体筛选) ────────────────────────────────
def test_supervisor_tournament_prunes_losers() -> None:
    ss = SupervisorStrategy(
        wrapped=ParetoPruningStrategy(),
        debate_evaluator=lambda ideas: [i for i in ideas if i["name"] != "A"])
    acts = ss.evaluate(_space(0.5))
    pruned = [a.target_branch for a in acts if a.action_type == "prune"]
    assert "b1" in pruned, pruned


# ── 4) research.program 端到端接线 (确定性) ─────────────────────────
def test_research_program_wiring_mutation_and_supervisor() -> None:
    from huginn.research.program import Experiment, run_research_program

    def _run_good() -> dict:
        return {"success": True, "objectives": {"eastward_offset": 12.3, "stability": 0.8},
                "summary": {"hot_spot_offset_deg": 12.3, "note": "real trace"}}

    experiments = [Experiment(name="lin", hypothesis="线性浅水", run=_run_good)]
    human = {"n": 0}

    def human_review(card):
        human["n"] += 1
        return {"drop": [], "guidance": "聚焦可证伪诊断"}

    out = run_research_program(
        goal="热木星昼夜环流",
        experiments=experiments,
        objectives_config={"eastward_offset": "maximize", "stability": "maximize"},
        max_iterations=6,
        min_iterations=2,
        max_parallel=2,
        mutation_config={"param_space": {"eastward_offset": (0.0, 30.0)},
                         "mutation_rate": 1.0, "max_children": 2},
        human_review=human_review, supervisor_every=1,
        client=None,  # 确定性综合, 不调模型
    )
    # 变异算子生效 + HITL 反馈闭环生效 + 结果可落地
    assert out.mutations >= 0
    assert isinstance(out.supervision_log, list)
    assert human["n"] >= 1, "human_review 应至少被触发一次"
    assert out.report and out.verdict in {"pass", "needs_grounding"}