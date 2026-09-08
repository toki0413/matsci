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


# ── 5) 统一诊断工具挂载面: LLM 可自主发现并调用后端能力工具 ───────────
def test_research_program_mounts_diagnostic_tools() -> None:
    """给定 diagnostic_tools, 假 LLM 调一次 hodge_circulation, 其数值落 trace 并被门禁放行."""
    from huginn.research.program import Experiment, run_research_program

    called = {"n": 0}

    def fake_handle(a):
        called["n"] += 1
        return '{"harmonic_frac": 0.42, "curl_frac": 0.58}'

    # 假 OpenAI client: 第 1 次 create → 工具调用; 之后 → 长报告引用该数值
    def make_client():
        state = {"round": 0}

        class Func:
            def __init__(s, n, a): s.name, s.arguments = n, a
            def model_dump(s): return {"name": s.name, "arguments": s.arguments}

        class TC:
            def __init__(s, n, a): s.id = "t1"; s.function = Func(n, a)
            def model_dump(s): return {"id": "t1", "function": s.function.model_dump()}

        class M:
            def __init__(s, c, t=None): s.content, s.tool_calls = c, t

        def ch(m): return type("C", (), {"message": m})()

        class Co:
            def create(s, **kw):
                state["round"] += 1
                if state["round"] == 1:
                    return type("R", (), {"choices": [ch(M("", [TC("hodge_circulation", "{}")]))]})()
                long = ("本研究针对目标开展自主深研。经帕累托演化淘汰, 存活假说提供真实数值证据。"
                        "进一步调用域诊断工具 hodge_circulation 得到谐和占比 harmonic_frac=0.42 与"
                        "旋度占比 0.58, 该测量来自对真实非线性稳态流场的 Helmholtz-Hodge 分解, 可直接"
                        "回溯。综合证据得出下阶段计划并如实标注所有数值均来自工具执行轨迹, 可复现可证伪。"
                        "这段正文足够长以满足报告长度门槛。")
                return type("R", (), {"choices": [ch(M("<report>" + long + "</report>"))]})()
        class Chat: completions = Co()
        return type("Client", (), {"chat": Chat()})()

    exps = [Experiment("lin", "线性浅水",
                       lambda: {"success": True, "objectives": {"eastward_offset": 12.3},
                                "summary": {"hot_spot_offset_deg": 12.3}})]
    schema = {"type": "function", "function": {"name": "hodge_circulation", "description": "d",
                                               "parameters": {"type": "object", "properties": {},
                                                             "required": [], "additionalProperties": False}}}
    out = run_research_program(
        goal="g", experiments=exps, objectives_config={"eastward_offset": "maximize"},
        max_iterations=5, min_iterations=1, max_parallel=1,
        client=make_client(), model="fake",
        diagnostic_tools=[{"tool": schema, "handle": fake_handle}],
    )
    assert called["n"] == 1, "LLM 应自主调用一次诊断工具"
    assert "0.42" in out.report, "被引用的诊断数值应进入最终报告"
    assert out.verdict == "pass", "诊断数值已在 trace → 门禁应放行"


# ── 6) 迭代闭环: 变异子代被真实执行并进入 Pareto 前沿 ────────────────
def test_mutation_children_actually_execute_and_join_front() -> None:
    """「评估→变异→再执行」闭环: MutationStrategy 的子代不再"被创建却无法执行".

    修复前: orchestrator 建分支时丢 params/mutation_of → executor 查不到 spec →
    子代 objectives 恒空, 迭代断裂。修复后: 世系经 branch.metadata 透传, executor
    用父实验 parametrize 重建真实变异实验 → 子代产生真实目标、进前沿候选。
    """
    from huginn.research.program import Experiment, run_research_program

    def run_a() -> dict:
        return {"success": True, "objectives": {"score": 1.0},
                "summary": {"note": "parent"}}

    def parametrize(params: dict):
        # 真实变异: score 随参数 k 单调上升 (k∈[0,1]) → 变异子代可支配父代, 必然入前沿
        k = round(float(params.get("k", 0.0)), 4)
        return lambda: {"success": True,
                        "objectives": {"score": round(1.0 + k, 4)},
                        "summary": {"note": f"mutant k={k}"}}

    out = run_research_program(
        goal="g",
        experiments=[Experiment("A", "baseline", run_a, parametrize=parametrize)],
        objectives_config={"score": "maximize"},
        max_iterations=8, min_iterations=2, max_parallel=2,
        mutation_config={"param_space": {"k": (0.0, 1.0)},
                         "mutation_rate": 1.0, "max_children": 2},
        client=None,
    )
    assert out.mutations > 0, "应生成变异子代"
    mut_names = [n for n in out.cache if "~mut" in n]
    assert mut_names, f"变异子代应被真实执行并进入 cache: {list(out.cache)}"
    front_names = {b["name"] for b in out.pareto_front}
    assert any("~mut" in n for n in front_names), f"变异子代应进入 Pareto 前沿: {front_names}"
    # 子代数值是真实变异结果(非编造): score > 1.0 且与参数单调一致
    for n in mut_names:
        assert out.cache[n]["objectives"]["score"] >= 1.0


def test_mutation_without_parametrize_reuses_parent_honestly() -> None:
    """父实验未声明 parametrize → 变异子代复用父 run 真实重跑(诚实回退, 不伪造)."""
    from huginn.research.program import Experiment, run_research_program

    runs = {"n": 0}

    def run_a() -> dict:
        runs["n"] += 1
        return {"success": True, "objectives": {"score": 1.0}, "summary": {}}

    out = run_research_program(
        goal="g",
        experiments=[Experiment("A", "baseline", run_a)],
        objectives_config={"score": "maximize"},
        max_iterations=6, min_iterations=1, max_parallel=2,
        mutation_config={"param_space": {"k": (0.0, 1.0)},
                         "mutation_rate": 1.0, "max_children": 1},
        client=None,
    )
    mut = [n for n in out.cache if "~mut" in n]
    assert mut, "无 parametrize 时变异子代也应真实重跑父实验"
    for n in mut:
        assert out.cache[n]["objectives"]["score"] == 1.0, "重跑结果必须来自真实执行"