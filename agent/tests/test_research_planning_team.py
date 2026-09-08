"""规划拆解 + 科研团队分工 单测 —— 三条新增产品能力.

覆盖:
  1. `research.planning.build_research_plan` —— 需求拆解 → DAG → 拓扑分层/并行度/预算
  2. `run_research_program(planner=...)` —— 自主规划的 experiment 序列与并行度真实生效
  3. `research.science_team.ScienceTeam` —— 多角色分工(plan/scientist/critic/synthesizer)
  4. 全链 demo `ai4s_fullchain_demo` —— 搜读算做写 offline smoke
  5. `research.law_model` 域无关 —— 系外行星(热力学) + 力学谐振子双域, 含 reconcile
     可证伪对账与 VLA 团队跨域复用(对账指标随域, 不写死)
  6. `research.interaction_explain` 交互可解释性 —— order-1/2 博弈交互分解 + 等效交互
     度量 + 代理/定律**结构对齐**治理闸门(张拳石理论落地, shortcut 探测)
  7. 世界模型能力贯通 —— 多元论世界观(worldview 卡片) + LawModel 装箱建 Capability +
     具身闸门(不可证伪拒收) + MCP 码头可见可调, 交互审计作用在世界模型域
  8. 结构闸门自动挂进深研管线 —— `run_research_program(structural_audit=)` 让代理结论
     进报告/决策前自动过交互等效审计, 结果并入 trace + 报告(out.structural_*)
  9. 世界模型多元论清册 —— `world_model_inventory` 把并存的三套实现按世界观显著区分
     (不合并只标注: 物理因果 / 隐态转移 / 物理逆生成器)

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
    from huginn.research.law_model import (
        FirstPrinciplesLawModel, LawAction, LawState,
    )
    wm = FirstPrinciplesLawModel(albedo=0.1)
    assert "T_eq" in wm.law() and "S" in wm.law()   # 数学定律(方程串)作为核心语言
    init = LawState({"a_AU": 1.0}, domain="exoplanet")
    t_far = wm.predict(init, LawAction({"a_scale": 1.2})).get("T_eq_K")
    t_near = wm.predict(init, LawAction({"a_scale": 0.8})).get("T_eq_K")
    assert t_far < t_near, "a 更大 → S∝a^-2 → T_eq 更小 (定律单调性)"


def test_reconcile_flags_wrong_world_model_as_falsified():
    """预测 vs 真值对账: 定律不符 → 如实标 falsified(不覆盖偏差)."""
    from huginn.research.law_model import (
        FirstPrinciplesLawModel, LawAction, LawState, reconcile,
    )
    wm = FirstPrinciplesLawModel(albedo=0.1)
    init = LawState({"a_AU": 1.0}, domain="exoplanet")
    act = LawAction({"a_scale": 1.0})
    pred = wm.predict(init, act)
    # 真值(独立执行)被系统性放大 1.5 倍 → 靠近轨道但温度假设错误
    tru_t = pred.get("T_eq_K") * 1.5
    resp = reconcile(pred, {"S_Wm2": pred.get("S_Wm2"), "T_eq_K": tru_t}, tol=0.03)
    assert resp["borne_out"] is False, resp
    assert "T_eq_K" in resp["mismatch"], "偏差应被如实记录, 而非悄悄抹平"


def test_model_based_planner_ranks_actions_by_predicted_objective():
    """模型基规划: 按预测目标(minimize T_eq → 最大 a_scale)排候选动作."""
    from huginn.research.law_model import (
        FirstPrinciplesLawModel, LawAction, LawState, ModelBasedPlanner,
    )
    wm = FirstPrinciplesLawModel(albedo=0.1)
    planner = ModelBasedPlanner(wm, objective="T_eq_K", sense="minimize")
    plan = planner.plan(LawState({"a_AU": 1.0}, domain="exoplanet"),
                        [LawAction({"a_scale": s}, label=f"s{int(s*10)}")
                         for s in (0.8, 1.0, 1.2)])
    assert plan[0].action.label == "s12", "最小 T_eq → 最大 a_scale 应第一个"
    assert all("law" in p.to_dict() for p in plan), "每个计划步都带数学定律"


def test_model_based_science_team_bears_and_falsifies_law():
    """VLA 团队端到端: 世界模型预告 → 科学家执行 → 批判对账.

    - 真实执行与定律一致 → borne_out=True, 进入结论并通过门禁;
    - 注入偏差执行(定律被破坏) → 如实 falsified 进 pruned, 不进结论.
    """
    from huginn.research.science_team import ModelBasedScienceTeam
    from huginn.research.law_model import FirstPrinciplesLawModel, LawAction

    wm = FirstPrinciplesLawModel(albedo=0.1)
    team = ModelBasedScienceTeam(wm, objective="T_eq_K", sense="maximize",
                                 n_scientists=2)
    obs = [{"name": "Kepler-999 b", "orbper_d": 100.0},
           {"name": "Fake-Planet b", "orbper_d": 300.0}]
    actions = [LawAction({"a_scale": s}, label=f"a{int(s*10)}") for s in (0.9, 1.0, 1.1)]

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


# ── 5) 第二类第一性原理域: 力学谐振子 (LawModel 域无关 + VLA 团队指标随域) ──
def test_mechanics_law_model_predicts_omega_monotonicity():
    """力学域定律: ω=√(k/m)——刚度↑→ω↑, 质量↑→ω↓; 且 T=2π/ω 自洽."""
    from huginn.research.law_model import MechanicsLawModel, LawAction, LawState
    wm = MechanicsLawModel()
    assert wm.domain == "mechanics"
    assert "T = 2π/ω" in wm.law() and "√(k/m)" in wm.law()
    init = LawState({"mass_kg": 2.0, "stiffness_Nm": 8.0}, domain="mechanics")
    base = wm.predict(init, LawAction({"k_scale": 1.0, "m_scale": 1.0}))
    stiffer = wm.predict(init, LawAction({"k_scale": 2.0, "m_scale": 1.0}))
    heavier = wm.predict(init, LawAction({"k_scale": 1.0, "m_scale": 2.0}))
    assert stiffer.get("omega_rad_s") > base.get("omega_rad_s"), "刚度↑ → ω↑"
    assert heavier.get("omega_rad_s") < base.get("omega_rad_s"), "质量↑ → ω↓"
    import math
    assert abs(base.get("T_s") * base.get("omega_rad_s") - 2 * math.pi) < 1e-3, \
        "T = 2π/ω 应数值自洽"


def test_mechanics_reconcile_falsifies_perturbed_period():
    """力学域对账(显式传 metrics): 周期被放大 → 定律如实 falsified."""
    from huginn.research.law_model import (
        MechanicsLawModel, LawAction, LawState, reconcile,
    )
    wm = MechanicsLawModel()
    init = LawState({"mass_kg": 1.0, "stiffness_Nm": 4.0}, domain="mechanics")
    pred = wm.predict(init, LawAction({"k_scale": 1.0, "m_scale": 1.0}))
    tru = {"omega_rad_s": pred.get("omega_rad_s"), "T_s": pred.get("T_s") * 1.5}
    resp = reconcile(pred, tru, tol=0.03, metrics=("omega_rad_s", "T_s"))
    assert resp["borne_out"] is False, resp
    assert "T_s" in resp["mismatch"], "周期偏差应被如实记录"


def test_model_based_team_reconcile_metrics_follow_domain():
    """VLA 团队跨域复用: 力学域对账指标随域 (不再写死 exoplanet S/T_eq)."""
    from huginn.research.science_team import ModelBasedScienceTeam
    from huginn.research.law_model import MechanicsLawModel, LawAction

    wm = MechanicsLawModel()
    obs = [{"name": "Osc-A", "mass_kg": 1.0, "stiffness_Nm": 4.0},
           {"name": "Osc-B", "mass_kg": 4.0, "stiffness_Nm": 8.0}]
    actions = [LawAction({"k_scale": 1.0, "m_scale": s}, label=f"m{int(s*10)}")
               for s in (0.8, 1.0, 1.2)]
    team = ModelBasedScienceTeam(wm, objective="T_s", sense="maximize",
                                 metrics=("omega_rad_s", "T_s"), n_scientists=2)

    def _ok(init, act):
        st = wm.predict(init, act)
        return {"omega_rad_s": st.get("omega_rad_s"), "T_s": st.get("T_s")}

    out = team.run("谐振子固有节律(定律预告-执行-对账)", obs, actions, _ok)
    assert "√(k/m)" in out.law, "力学定律入产出"
    assert out.verdict == "pass"
    assert set(out.survivors) == {"Osc-A", "Osc-B"}, f"定律一致 → 全部证实, got {out.survivors}"
    assert {e.role for e in out.role_log} >= {"planner", "scientist", "critic", "synthesizer"}

    # 注入周期偏差 → 定律证伪 → 全部 pruned (跨域对账同样守住可证伪性)
    def _bad(init, act):
        st = wm.predict(init, act)
        return {"omega_rad_s": st.get("omega_rad_s"), "T_s": st.get("T_s") * 1.5}

    out2 = team.run("同一目标(偏差执行)", obs, actions, _bad)
    assert set(out2.survivors) == set(), f"定律不符 → 全部证伪, got {out2.survivors}"
    assert set(out2.pruned) == {"Osc-A", "Osc-B"}, out2.pruned


# ── 6) 交互可解释性 (张拳石交互理论落地): 等效交互度量 + 代理/定律结构对齐 ──
def _add_features() -> tuple[list[str], dict, dict]:
    features = ["x1", "x2", "x3"]
    instance = {"x1": 0.8, "x2": 1.2, "x3": 0.5}
    baseline = {"x1": 0.0, "x2": 0.0, "x3": 0.0}
    return features, instance, baseline


def _f_additive(inp):
    return 1.0 * inp["x1"] + 2.0 * inp["x2"] + 0.5 * inp["x3"]


def test_interaction_order2_detects_joint_product():
    """order-2 Shapley 交互指数: 纯加性→0; 含 x1·x2 乘积项→非零(shortcut 探测)."""
    from huginn.research.interaction_explain import (
        PAIR, interaction_primitives, pair_interactions,
    )
    features, instance, baseline = _add_features()
    inter_f = pair_interactions(_f_additive, features, instance, baseline)
    assert abs(inter_f[("x1", "x2")]) < 1e-3, "纯加性 → 无联合交互"

    def _g_shortcut(inp):
        return _f_additive(inp) + 5.0 * inp["x1"] * inp["x2"]

    inter_g = pair_interactions(_g_shortcut, features, instance, baseline)
    # 值函数 v(S) 里 Δ 只保留联合非加性: 5*x1*x2 该交互应 ≈ 5*0.8*1.2 = 4.8
    assert abs(inter_g[("x1", "x2")] - 4.8) < 1e-2, inter_g[("x1", "x2")]
    prim_g = interaction_primitives(_g_shortcut, features, instance, baseline)
    assert (PAIR, "x1", "x2") in prim_g and abs(prim_g[(PAIR, "x1", "x2")] - 4.8) < 1e-2


def test_equivalent_interaction_flags_spurious_interaction():
    """等效交互: 代理多出一条乘积交互 → 与纯加性定律 **不等效** 且定位出该交互."""
    from huginn.research.interaction_explain import equivalent_interaction

    def _g_shortcut(inp):
        return _f_additive(inp) + 5.0 * inp["x1"] * inp["x2"]

    features, instance, baseline = _add_features()
    from huginn.research.interaction_explain import interaction_primitives
    pi_f = interaction_primitives(_f_additive, features, instance, baseline)
    pi_g = interaction_primitives(_g_shortcut, features, instance, baseline)
    # 与自身等效 / 与纯加性异构等效
    assert equivalent_interaction(pi_f, pi_f)["equivalent"] is True
    assert equivalent_interaction(pi_g, pi_g)["equivalent"] is True
    # 代理带上乘积交互 → 两结构不等效, b_only 精确指出这条 shortcut 交互
    eq = equivalent_interaction(pi_f, pi_g)
    assert eq["equivalent"] is False
    assert ("2", "x1", "x2") in eq["b_only"], eq["b_only"]
    assert 0.0 <= eq["agreement"] < 1.0


def test_interaction_trace_is_json_and_grounding_ready():
    """交互基元可序列化成 grounding 证据 (可并入研究 trace)."""
    from huginn.research.interaction_explain import interaction_primitives, interaction_trace
    features, instance, baseline = _add_features()
    prim = interaction_primitives(_f_additive, features, instance, baseline)
    s = interaction_trace(prim)
    import json as _json
    obj = _json.loads(s)
    assert obj["type"] == "interaction_primitives"
    assert all({"order", "features", "interaction"} <= set(r) for r in obj["rows"])


def test_surrogate_law_alignment_is_structural_gate():
    """治理门禁: 数值可吻合的代理, 只要交互结构(shortcut)与定律不符即报 not-aligned."""
    from huginn.research.interaction_explain import surrogate_law_alignment

    def _law(inp):
        return 3.0 * inp["x1"] + 1.0 * inp["x2"]

    def _surrogate_ok(inp):
        return 3.0 * inp["x1"] + 1.0 * inp["x2"]   # 数值+结构都与定律一致

    def _surrogate_shortcut(inp):
        # 数值上刻意逼近但引入一条定律没有的 x1·x2 虚假交互
        return 3.0 * inp["x1"] + 1.0 * inp["x2"] + 0.4 * inp["x1"] * inp["x2"]

    features, instance, baseline = _add_features()
    ok = surrogate_law_alignment(surrogate=_surrogate_ok, law=_law,
                                 features=features, instance=instance)
    assert ok["aligned"] is True, ok
    bad = surrogate_law_alignment(surrogate=_surrogate_shortcut, law=_law,
                                  features=features, instance=instance)
    assert bad["aligned"] is False, "结构失调 → 治理闸门拦住进决策"
    # surrogate 是 a, 定律是 b → 代理多出来的虚假交互记在 surrogate_only
    assert ("2", "x1", "x2") in bad["surrogate_only"], bad["surrogate_only"]
    assert bad["law_only"] == [], "纯 2-feature 定律无联合交互 → 不应被误报缺失"


# ── 7) 世界模型能力 + 多元论世界观 (张拳石交互审计与 world-model 能力贯通) ──
def test_worldview_pluralism_card():
    """多元论落地: 每个世界模型能力声明 worldview + 治理卡片(可证伪/真相参照)."""
    from huginn.research.law_model import (
        FirstPrinciplesLawModel, MechanicsLawModel, Worldview, world_model_card,
    )
    w_fp = FirstPrinciplesLawModel()
    w_me = MechanicsLawModel()
    # 二者都是"物理-行动-因果"极 (LawModel 默认), 但域不同 → 卡片能区分
    for m in (w_fp, w_me):
        assert getattr(m, "worldview") is Worldview.PHYSICS_CAUSAL
        card = world_model_card(m)
        assert card["falsifiable"] is True, "定律模型须可证伪(reconcile 真相参照)"
        assert card["truth_reference"]  # 具身参照: 真实执行
        assert card["worldview"] == "physics_causal"
    assert world_model_card(w_fp)["domain"] == "exoplanet"
    assert world_model_card(w_me)["domain"] == "mechanics"


def test_world_model_capability_rejects_unfalsifiable():
    """具身闸门: 不可证伪对象(predict-only 鸭子) 一律拒绝装箱为世界模型能力."""
    from huginn.capabilities.world_model import WorldModelCapability
    from huginn.research.law_model import FirstPrinciplesLawModel
    # 合法: 第一性原理定律模型可装箱
    cap = WorldModelCapability(FirstPrinciplesLawModel())
    assert cap.name == "world_model.exoplanet"
    assert cap.read_only is True

    class _PredictOnly:  # noqa: D106
        def predict(self, *a, **k):
            return {"x": 1}

    try:
        WorldModelCapability(_PredictOnly())  # type: ignore[arg-type]
        raised = False
    except TypeError:
        raised = True
    assert raised, "无对账能力(不可证伪)的对象不能装箱为世界模型能力"


def test_world_model_capability_ops_law_predict_reconcile():
    """能力 op 面: law(数学定律) / predict(预告=假说) / reconcile(对账可证伪)."""
    import asyncio
    from huginn.capabilities.world_model import WorldModelCapability
    from huginn.research.law_model import FirstPrinciplesLawModel

    wm = FirstPrinciplesLawModel(albedo=0.1)
    cap = WorldModelCapability(wm)

    r_law = asyncio.run(cap.run({"op": "law"}))
    assert r_law.success and "T_eq" in r_law.data["law"]

    r_pred = asyncio.run(cap.run({"op": "predict",
                                  "state": {"a_AU": 1.0}, "action": {"a_scale": 1.0}}))
    assert r_pred.success
    assert r_pred.data["falsifiable"] is True, "预告必须标注为可证伪假说"

    # 对账: 数值一致 → 证实; 注入偏差 → 如实 falsified
    pred = r_pred.data["predicted"]
    r_ok = asyncio.run(cap.run({"op": "reconcile", "predicted": pred, "actual": pred}))
    assert r_ok.success and r_ok.data["borne_out"] is True
    bad_pred = {**pred, "state": {**pred["state"], "T_eq_K": pred["state"]["T_eq_K"] * 1.5}}
    r_bad = asyncio.run(cap.run({"op": "reconcile", "predicted": pred, "actual": bad_pred}))
    assert r_bad.success and r_bad.data["borne_out"] is False, "偏差须如实标 falsified"


def test_world_model_capability_mcp_surface():
    """世界模型能力进注册表 → MCP 码头可见、可调(默认 exoplanet 世界模型)."""
    import asyncio
    from huginn.capabilities.mcp_export import CapabilityMCPBackend
    from huginn.capabilities.registry import CapabilityRegistry
    from huginn.capabilities.world_model import register_world_model_capabilities
    from huginn.research.law_model import FirstPrinciplesLawModel

    class _Local(CapabilityRegistry):  # noqa: D101
        pass

    name = register_world_model_capabilities(FirstPrinciplesLawModel(), registry=_Local)
    assert name == "world_model.exoplanet"
    manifest = _Local.manifest()
    m = next(m for m in manifest if m["name"] == name)
    assert m["read_only"] is True and m["category"] == "world_model"
    assert m["input_schema"] and "op" in m["input_schema"]["properties"]

    backend = CapabilityMCPBackend(allow_write=False, registry=_Local)
    assert backend.contains(name)
    # mcp SDK 可能未装: Manifest/contains/call 均不依赖 mcp 包, 只有 as_mcp_tools 才需.
    assert any(t["function"]["name"] == name for t in backend.openai_functions())
    out = asyncio.run(backend.call(name, {"op": "law"}))
    assert out["success"] and "T_eq" in out["data"]["law"]


def test_interaction_alignment_applies_on_world_model_domain():
    """贯通: 结构门禁(张拳石)亦可作用在世界模型域 —— 代理带虚假交互即拦."""
    from huginn.research.interaction_explain import surrogate_law_alignment

    # 世界模型域标量"定律"(在 albedo/a_AU 两特征上**加性**, 无联合交互), 作为真值结构
    def _law(inp):
        return 250.0 * (1.0 - inp["albedo"]) + 30.0 * inp["a_AU"]

    def _surrogate_shortcut(inp):
        # 数值上逼近, 但引入定律没有的 albedo·a_AU 联合交互 (shortcut)
        return _law(inp) + 6.0 * inp["albedo"] * inp["a_AU"]

    features = ["a_AU", "albedo"]
    instance = {"a_AU": 1.0, "albedo": 0.2}
    baseline = {"a_AU": 5.0, "albedo": 0.3}
    bad = surrogate_law_alignment(surrogate=_surrogate_shortcut, law=_law,
                                  features=features, instance=instance,
                                  baseline=baseline)
    assert bad["aligned"] is False, "世界模型域的代理结构失调 → 闸门拦住进决策"
    assert ("2", "a_AU", "albedo") in bad["surrogate_only"], bad["surrogate_only"]


# ── 8) 结构闸门自动挂进深研管线 (代理结论进报告前过交互等效审计) ──────────
def _pipeline_with_structural_gate(gate):
    from huginn.research.program import run_research_program
    return run_research_program(
        goal="结构性结论", experiments=[_exp("e1", 1.0)],
        objectives_config={"score": "maximize"},
        max_iterations=6, min_iterations=1, client=None,
        structural_audit=gate,
    )


def _exp(name, obj):
    from huginn.research.program import Experiment
    return Experiment(name, f"hyp_{name}", _run(obj))


def test_structural_gate_pass_and_fail_lands_in_outcome_and_report():
    """结构闸门: 通过 → structural_aligned True 且报告如实; 未通过 → 报告醒目标注 shortcut."""
    gate_ok = lambda surv, goal: {"pass": True, "reason": "ok",  # noqa: E731
                                  "surrogate_only": [], "law_only": []}
    gate_bad = lambda surv, goal: {"pass": False, "reason": "shortcut",  # noqa: E731
                                   "surrogate_only": [("2", "a", "b")], "law_only": []}
    out_ok = _pipeline_with_structural_gate(gate_ok)
    assert out_ok.structural_aligned is True
    assert out_ok.structural_gate["pass"] is True
    assert "结构对齐闸门" in out_ok.report
    assert "未通过" not in out_ok.report

    out_bad = _pipeline_with_structural_gate(gate_bad)
    assert out_bad.structural_aligned is False
    assert out_bad.structural_gate["surrogate_only"] == [("2", "a", "b")]
    assert "结构对齐闸门(未通过)" in out_bad.report
    assert "shortcut" in out_bad.report, "未通过的报告必须醒目标注结构风险"
    # 数值 grounding 仍独立判(结构风险是额外治理信号, 不过度绑架数值门禁)
    assert out_bad.verdict == "pass"


def test_structural_gate_artifact_joins_trace_and_json():
    """结构闸门的可证伪工件并入 trace(可与 grounding 对账), 异常如实降级."""
    gate_bad = lambda surv, goal: {"pass": False, "reason": "albedo·a 交互",  # noqa: E731
                                   "surrogate_only": [("2", "a_AU", "albedo")], "law_only": []}
    out = _pipeline_with_structural_gate(gate_bad)
    assert out.structural_gate["surrogate_only"], "结构闸门应记录代理独有交互"
    assert out.structural_gate["pass"] is False

    # 结构闸门抛异常 → 如实降级为未通过, 不阻断主流程
    def _boom(surv, goal):  # noqa: ARG001
        raise RuntimeError("audit backend down")
    out2 = _pipeline_with_structural_gate(_boom)
    assert out2.structural_aligned is False
    assert "structural_audit failed" in out2.structural_gate["reason"]
    assert out2.verdict == "pass", "审计异常不应拖垮整个深研管线"


def test_build_alignment_outcome_gates_surrogate_conclusion_in_pipeline():
    """端到端: 用 interaction_explain.build_alignment_outcome 作 structural_audit,
    代理带 shortcut → 管线自动判未通过并在报告标注; 定律一致 → 通过."""
    from huginn.research.interaction_explain import build_alignment_outcome

    def _law(inp):
        return 250.0 * (1.0 - inp["albedo"]) + 30.0 * inp["a_AU"]

    def _surrogate_shortcut(inp):
        return _law(inp) + 6.0 * inp["albedo"] * inp["a_AU"]

    features = ["a_AU", "albedo"]
    instance = {"a_AU": 1.0, "albedo": 0.2}
    baseline = {"a_AU": 5.0, "albedo": 0.3}

    gate_shortcut = build_alignment_outcome(
        surrogate=_surrogate_shortcut, law=_law, features=features,
        instance=instance, baseline=baseline)
    out_bad = _pipeline_with_structural_gate(gate_shortcut)
    assert out_bad.structural_aligned is False
    assert ("2", "a_AU", "albedo") in out_bad.structural_gate["surrogate_only"]
    assert "未通过" in out_bad.report

    gate_ok = build_alignment_outcome(
        surrogate=_law, law=_law, features=features,
        instance=instance, baseline=baseline)
    out_ok = _pipeline_with_structural_gate(gate_ok)
    assert out_ok.structural_aligned is True
    assert "未通过" not in out_ok.report


# ── 9) 世界模型多元论清册 (不合并, 只标注 —— 治理三套并存实现) ──────────
def test_world_model_inventory_distinguishes_plural_worldviews():
    """多元论治理: 仓库并存的世界模型实现按其世界观显式区别, 而非熔成一团."""
    from huginn.research.law_model import Worldview, world_model_inventory
    inv = world_model_inventory()
    by_id = {e["id"]: e for e in inv}
    assert len(inv) >= 3
    # 隐态转移极(可学前向 s') vs 物理因果极: 世界观确实被区别开
    assert by_id["security.world_state"]["worldview"] == Worldview.LATENT_TRANSITION.value
    assert by_id["research.law_model"]["worldview"] == Worldview.PHYSICS_CAUSAL.value
    # 三者消费者各异(科研管线 / 沙箱主循环 / 可逆撤销控制环) —— 是分工不是冲突重复
    consumers = {e["consumer"] for e in inv}
    assert len(consumers) >= 3, consumers
    # 每项都声明可证伪性(物理/隐态极做前向都必须留对账/回测参照)
    assert all(e["falsifiable"] for e in inv), inv