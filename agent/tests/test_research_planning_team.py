"""规划拆解 + 科研团队分工 单测 —— 三条新增产品能力.

覆盖:
  1. `research.planning.build_research_plan` —— 需求拆解 → DAG → 拓扑分层/并行度/预算
  2. `run_research_program(planner=...)` —— 自主规划的 experiment 序列与并行度真实生效
  3. `research.science_team.ScienceTeam` —— 多角色分工(plan/scientist/critic/synthesizer)
  4. 全链 demo `ai4s_fullchain_demo` —— 搜读算做写 offline smoke

纯确定性/本地, 零网络/零 LLM。实验 run 返回真实可区分数值。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


def _run(obj: float):
    def run() -> dict:
        return {"success": True, "objectives": {"score": obj}, "summary": {"v": obj}}
    return run


# ── 1) 需求拆解 / 任务自主规划 ───────────────────────────────────
def test_build_research_plan_topological_layers_and_budget():
    from huginn.research.planning import SubResearch, build_research_plan

    subs = [
        SubResearch("A", "hypA", _run(1.0)),
        SubResearch("B", "hypB", _run(2.0), depends_on=["A"]),
        SubResearch("C", "hypC", _run(3.0), depends_on=["A"]),
        SubResearch("D", "hypD", _run(4.0), depends_on=["B", "C"]),
    ]
    plan = build_research_plan("goal", subs)
    # 拓扑分层: A 先行, B/C 并行, D 收口
    assert plan.layers == [["A"], ["B", "C"], ["D"]], plan.layers
    assert plan.antichain_width >= 2, "B/C 相互不可达 → 应可并行"
    assert abs(sum(plan.budget[k] for k in ("depth", "parallel", "per_subagent"))
               ) > 0  # 预算分解非空
    assert {e.name for e in plan.experiments} == {"A", "B", "C", "D"}
    assert plan.critical_path[0] == "A" and plan.critical_path[-1] == "D"
    assert plan.max_parallel >= 1


def test_build_research_plan_rejects_unknown_dependency():
    from huginn.research.planning import SubResearch, build_research_plan
    subs = [SubResearch("A", "hypA", _run(1.0), depends_on=["ghost"])]
    try:
        build_research_plan("g", subs)
        raised = False
    except ValueError:
        raised = True
    assert raised, "依赖不存在应抛错"


def test_run_research_program_respects_planner():
    """planner 提供的 experiment 序列与并行度真实进入管线; plan_summary 记录规划. """
    from huginn.research.planning import SubResearch, build_research_plan
    from huginn.research.program import run_research_program

    subs = [
        SubResearch("ex_A", "idA", _run(2.0)),
        SubResearch("ex_B", "idB", _run(1.0), depends_on=["ex_A"]),
    ]
    plan = build_research_plan("goal", subs)
    out = run_research_program(
        goal="goal", experiments=[],      # 由 planner 提供真实实验序列
        objectives_config={"score": "maximize"},
        max_iterations=6, min_iterations=2,
        planner=lambda g: plan,           # 自主规划入口
        client=None,
    )
    assert out.plan_summary is not None, "应记录规划摘要"
    assert out.plan_summary["max_parallel"] == plan.max_parallel
    assert "ex_A" in out.cache and "ex_B" in out.cache, "planner 实验应真实执行"


# ── 2) 科研团队分工 ───────────────────────────────────────────────
def test_science_team_role_division_and_pareto_prune():
    from huginn.research.science_team import ScienceTeam
    from huginn.research.planning import SubResearch

    subs = [
        SubResearch("good", "强候选", _run(2.0)),
        SubResearch("weak", "被支配候选", _run(1.0)),
    ]
    team = ScienceTeam()
    out = team.run("g", subs, {"score": "maximize"})
    # 四角色分工齐备
    assert set(out.roles()) == {"planner", "scientist", "critic", "synthesizer"}, out.roles()
    # 批判者用 Pareto 淘汰被支配的 weak
    assert "weak" in {e.name for e in out.pruned}
    assert "good" in {e.name for e in out.survivors}
    # 综合报告过 grounding 门禁
    assert out.verdict == "pass"
    assert "good" in out.report and "weak" not in out.report


def test_science_team_scientists_parallel_across_dag_layers():
    """团队按规划 DAG 分层唤醒科学家; 串行层(依赖)不跨层提前执行."""
    from huginn.research.science_team import ScienceTeam, ScientistAgent
    from huginn.research.planning import SubResearch

    order: list[str] = []

    def _mk(tag: str):
        def run() -> dict:
            order.append(tag)
            return {"success": True, "objectives": {"score": 1.0}, "summary": {"t": tag}}
        return run

    subs = [
        SubResearch("root", "h", _mk("root")),
        SubResearch("leaf", "h", _mk("leaf"), depends_on=["root"]),
    ]
    team = ScienceTeam(n_scientists=2)
    out = team.run("g", subs, {"score": "maximize"})
    # root 先于 leaf (DAG 依赖) → 分工顺序遵守拓扑
    assert order.index("root") < order.index("leaf"), order
    assert out.verdict == "pass"


# ── 3) 全链条 demo smoke ──────────────────────────────────────────
def test_fullchain_demo_offline_smoke():
    import ai4s_fullchain_demo as d
    out = d.run_fullchain("系外行星轨道日晒与平衡温度研究", n_scientists=2)
    assert out["fullchain_verdict"] == "pass"
    assert out["S1_search"]["references"], "S1 搜 应有结果(离线捆绑)"
    assert out["S2_read"]["n"] > 0, "S2 读 应解析出特征"
    stage = out["S3_compute_S4_execute_S5_write"]
    assert {"planner", "critic", "synthesizer"} <= set(stage["team_roles"])
    assert stage["survivors"], "S3/S4 应产出存活证据"


# ── 4) 世界模型 / VLA 式团队 (数学为核心) ────────────────────
def test_first_principles_world_model_predicts_law():
    """世界模型的 predict 与数学定律一致: T_eq 随 a_scale 增大而单调下降."""
    from huginn.research.world_model import (
        Action, FirstPrinciplesWorldModel, WorldState,
    )
    wm = FirstPrinciplesWorldModel(albedo=0.1)
    assert "T_eq" in wm.law() and "S" in wm.law()   # 数学定律(方程串)作为核心语言
    init = WorldState({"a_AU": 1.0}, domain="exoplanet")
    t_far = wm.predict(init, Action({"a_scale": 1.2})).get("T_eq_K")
    t_near = wm.predict(init, Action({"a_scale": 0.8})).get("T_eq_K")
    assert t_far < t_near, "a 更大 → S∝a^-2 → T_eq 更小 (定律单调性)"


def test_reconcile_flags_wrong_world_model_as_falsified():
    """预测 vs 真值对账: 定律不符 → 如实标 falsified(不覆盖偏差)."""
    from huginn.research.world_model import (
        Action, FirstPrinciplesWorldModel, reconcile, WorldState,
    )
    wm = FirstPrinciplesWorldModel(albedo=0.1)
    init = WorldState({"a_AU": 1.0}, domain="exoplanet")
    act = Action({"a_scale": 1.0})
    pred = wm.predict(init, act)
    # 真值(独立执行)被系统性放大 1.5 倍 → 靠近轨道但温度假设错误
    tru_t = pred.get("T_eq_K") * 1.5
    resp = reconcile(pred, {"S_Wm2": pred.get("S_Wm2"), "T_eq_K": tru_t}, tol=0.03)
    assert resp["borne_out"] is False, resp
    assert "T_eq_K" in resp["mismatch"], "偏差应被如实记录, 而非悄悄抹平"


def test_model_based_planner_ranks_actions_by_predicted_objective():
    """模型基规划: 按预测目标(minimize T_eq → 最大 a_scale)排候选动作."""
    from huginn.research.world_model import (
        Action, FirstPrinciplesWorldModel, ModelBasedPlanner, WorldState,
    )
    wm = FirstPrinciplesWorldModel(albedo=0.1)
    planner = ModelBasedPlanner(wm, objective="T_eq_K", sense="minimize")
    plan = planner.plan(WorldState({"a_AU": 1.0}, domain="exoplanet"),
                        [Action({"a_scale": s}, label=f"s{int(s*10)}")
                         for s in (0.8, 1.0, 1.2)])
    assert plan[0].action.label == "s12", "最小 T_eq → 最大 a_scale 应第一个"
    assert all("law" in p.to_dict() for p in plan), "每个计划步都带数学定律"


def test_model_based_science_team_bears_and_falsifies_law():
    """VLA 团队端到端: 世界模型预告 → 科学家执行 → 批判对账.

    - 真实执行与定律一致 → borne_out=True, 进入结论并通过门禁;
    - 注入偏差执行(定律被破坏) → 如实 falsified 进 pruned, 不进结论.
    """
    from huginn.research.science_team import ModelBasedScienceTeam
    from huginn.research.world_model import Action, FirstPrinciplesWorldModel

    wm = FirstPrinciplesWorldModel(albedo=0.1)
    team = ModelBasedScienceTeam(wm, objective="T_eq_K", sense="maximize",
                                 n_scientists=2)
    obs = [{"name": "Kepler-999 b", "orbper_d": 100.0},
           {"name": "Fake-Planet b", "orbper_d": 300.0}]
    actions = [Action({"a_scale": s}, label=f"a{int(s*10)}") for s in (0.9, 1.0, 1.1)]

    def _executor(init, act):
        # 真值 = 世界模型本身(第一性原理一致) → 定律被证实
        st = wm.predict(init, act)
        return {"S_Wm2": st.get("S_Wm2"), "T_eq_K": st.get("T_eq_K")}

    out = team.run("系外行星平衡温度(定律预告-执行-对账)", obs, actions, _executor)
    assert out.law and "T_eq" in out.law                # 数学定律入产出
    assert {e.role for e in out.role_log} >= {"planner", "scientist", "critic", "synthesizer"}
    assert set(out.survivors) == {"Kepler-999 b", "Fake-Planet b"}, \
        f"与定律一致 → 全部证实, got survivors={out.survivors}, pruned={out.pruned}"
    assert out.verdict == "pass"

    # 注入系统性偏差: 真实执行把温度放大 1.5 倍 → 定律被证伪
    def _bad_executor(init, act):
        return {"S_Wm2": wm.predict(init, act).get("S_Wm2"),
                "T_eq_K": wm.predict(init, act).get("T_eq_K") * 1.5}
    out2 = team.run("同一目标(偏差执行)", obs, actions, _bad_executor)
    assert set(out2.survivors) == set(), f"定律不符 → 全部证伪, got {out2.survivors}"
    assert set(out2.pruned) == {"Kepler-999 b", "Fake-Planet b"}, out2.pruned