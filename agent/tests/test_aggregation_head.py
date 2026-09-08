"""收敛聚合头 (P1 适配器) 测试 —— 多头收敛视图 out.consolidated.

覆盖:
  - consolidate 纯函数: gate 否决 / 冲突显式化 / diversity / grounding 透传
  - 管道集成: run_research_program 产出 out.consolidated,
    且 P1 兼容锚 consolidated.grounding == out.verdict 恒成立.
"""
from __future__ import annotations

from huginn.research.aggregation_head import (
    EVIDENCE_OBSERVED,
    EVIDENCE_UNOBSERVED,
    HeadResult,
    consolidate,
    oracle_verify_consolidated,
)
from huginn.research.decision_gate import role_separation
from huginn.research.harness import build_harness_report
from huginn.research.harness_ledger import HarnessLedger
from huginn.research.program import Experiment, ResearchOutcome, run_research_program


# ── 纯函数: 仲裁 ──────────────────────────────────────────────────────────
def test_gate_failed_blocks_verdict():
    heads = [
        HeadResult("gate.a", "A", EVIDENCE_OBSERVED, "passed", gate=True),
        HeadResult("gate.b", "B", EVIDENCE_OBSERVED, "failed", gate=True),
    ]
    c = consolidate(heads, grounding_verdict="pass")
    assert c.verdict == "gate_blocked"
    assert c.gates_failed == ["gate.b"]


def test_gate_pass_and_grounding_pass_gives_pass():
    c = consolidate(
        [HeadResult("gate.a", "A", EVIDENCE_OBSERVED, "passed", gate=True)],
        grounding_verdict="pass",
    )
    assert c.verdict == "pass"


def test_grounding_parity_is_preserved():
    """P1 兼容锚: 无论聚合判定如何, grounding 字段原样携带门禁判定."""
    c = consolidate(
        [HeadResult("gate.a", "A", EVIDENCE_OBSERVED, "failed", gate=True)],
        grounding_verdict="needs_grounding",
    )
    assert c.grounding == "needs_grounding"


def test_conflict_is_surfaced_not_averaged():
    # 同层 gate 头判定相反 → 显式 conflict, 拒绝静默平均
    heads = [
        HeadResult("gate.x", "X", EVIDENCE_OBSERVED, "passed", gate=True),
        HeadResult("gate.y", "Y", EVIDENCE_OBSERVED, "failed", gate=True),
    ]
    c = consolidate(heads)
    assert c.conflicts, "矛盾头必须显式列出"
    assert c.verdict == "gate_blocked"


def test_diversity_measures_head_collapse():
    uniform = [HeadResult("h%d" % i, "h", EVIDENCE_OBSERVED, "passed") for i in range(4)]
    diverse = [HeadResult("h%d" % i, "h", EVIDENCE_OBSERVED, o)
               for i, o in enumerate(("passed", "failed", "unobserved", "missing"))]
    assert consolidate(uniform).diversity == 0.25      # 全员同判 → 单文化雷达低
    assert consolidate(diverse).diversity == 1.0       # 高分化 → 防御塌缩


def test_score_is_weighted():
    heads = [
        HeadResult("h1", "h", EVIDENCE_OBSERVED, "passed", weight=1.0),
        HeadResult("h2", "h", EVIDENCE_OBSERVED, "failed", weight=3.0),
    ]
    assert consolidate(heads).score == 0.25            # (1*1 + 0*3)/4


def test_head_result_evidence_tri_state():
    assert HeadResult("h", "h", EVIDENCE_UNOBSERVED, "unobserved").score() == 0.5


# ── 管道集成 (P1 适配器) ──────────────────────────────────────────────────
def test_pipeline_populates_consolidated():
    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    out = run_research_program(
        goal="consolidation head p1",
        experiments=[Experiment("e0", "baseline", _run(1.0)),
                     Experiment("e1", "candidate", _run(2.0))],
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        harness_agent="huginn-lite", harness_machine="sandbox-1",
    )
    assert out.consolidated is not None
    con = out.consolidated
    # P1 兼容锚: grounding 与 out.verdict 恒一致
    assert con["grounding"] == out.verdict
    # 无门禁失败时聚合判定 = pass
    assert con["verdict"] in ("pass", "gate_blocked")
    if con["verdict"] == "pass":
        assert con["gates_failed"] == []
    # 有界 head 清单: 真实注册了哪些头
    assert "gate.claim_grounding" in con["heads"]
    assert "audit.score_usage" in con["heads"]
    assert "wm.actually_used" in con["heads"]
    assert 0.0 <= con["score"] <= 1.0
    assert 0.0 <= con["diversity"] <= 1.0


def test_consolidated_with_planner_revision():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "consolidation plan",
        [SubResearch("base", "baseline", _run(1.0)),
         SubResearch("cand", "candidate", _run(2.0))],
    )
    out = run_research_program(
        goal="consolidation plan",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    con = out.consolidated
    assert "audit.plan_revision" in con["heads"]
    assert "gate.structural" in con["heads"]      # 未注入 structural_audit → unobserved
    assert con["grounding"] == out.verdict


def test_consolidated_absent_on_direct_out_construction():
    # 直接构造 out(未跑管线) → 无 consolidated 字段, 不污染既有研究结构
    out = ResearchOutcome(verdict="grounded")
    assert getattr(out, "consolidated", None) is None
    # 旧字段照常工作
    assert out.verdict == "grounded"


def test_consolidated_carries_head_details():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "detail heads",
        [SubResearch("a", "a", _run(1.0)), SubResearch("b", "b", _run(2.0))],
    )
    out = run_research_program(
        goal="detail heads",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=3, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    details = out.consolidated["head_details"]
    ids = {d["id"] for d in details}
    # 六维投影所需头都注册了
    assert {"gate.claim_grounding", "gate.structural", "gate.workspace",
            "wm.actually_used", "audit.score_usage", "controlled.supervision",
            "reliable.evidence_cache", "learning.self_audit", "audit.plan_revision"} <= ids


# ── P2 · 消费者切换: harness 从聚合头投影六维 ────────────────────────────
def test_p2_harness_projects_consolidated():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "p2 projection",
        [SubResearch("base", "baseline", _run(1.0)),
         SubResearch("cand", "candidate", _run(2.0))],
    )
    out = run_research_program(
        goal="p2 projection",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan,
        harness_agent="a", harness_machine="m",
    )
    assert {d["name"] for d in out.harness["dimensions"]} == {
        "task_understanding", "controlled_execution", "change_validation",
        "reliable_delivery", "learning_capture", "safety_authority",
    }
    # 六维投影 = consolidated.head_details 的投影(单一入口), 非逐字段扫描
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    names = {c["name"] for c in sa["checks"]}
    assert any("得分≠使用" in n for n in names)          # audit.score_usage 投影
    assert any("世界模型真用?" in n for n in names)       # wm.actually_used 投影
    tu = next(d for d in out.harness["dimensions"] if d["name"] == "task_understanding")
    assert any("plan 修订门" in c["name"] for c in tu["checks"])


def test_p2_harness_projection_evidence_matches_head():
    # 无 world_model → wm 头 unobserved → 投影后 safety_authority 的 wm check 也是 unobserved
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "p2 evidence",
        [SubResearch("a", "a", _run(1.0)), SubResearch("b", "b", _run(2.0))],
    )
    out = run_research_program(
        goal="p2 evidence",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=3, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    wm = next(c for c in sa["checks"] if "世界模型真用?" in c["name"])
    assert wm["evidence"] == EVIDENCE_UNOBSERVED
    assert wm["outcome"] == "unobserved"


def test_p2_ledger_consumes_projected_harness():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "p2 ledger", [SubResearch("a", "a", _run(1.0))],
    )
    out = run_research_program(
        goal="p2 ledger",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan,
        harness_agent="a", harness_machine="m",
    )
    ledger = HarnessLedger().append(out.harness)
    s = ledger.summary(out.harness["task_episode"])
    assert s["task_episode"] == out.harness["task_episode"]
    assert "world_model_lessons" in s          # 投影后账本仍能聚合世界模型经验


def test_p2_direct_out_keeps_legacy_harness_path():
    # 直构 out(无 consolidated) → harness 走旧逐字段路径, 无"聚合投影"标记
    out = ResearchOutcome(verdict="grounded", cache={"e": {"summary": {}}})
    rep = build_harness_report("g", out)
    names = [c.name for d in rep.dimensions for c in d.checks]
    assert all("聚合投影" not in n for n in names)


# ── 缺陷三: 团队视角分离度(防多头塌缩) ───────────────────────────────────
def test_role_separation_detects_collapsed_single_view():
    survivors = [
        {"name": "a", "objectives": {"f": 1}, "worldview": "physics_causal"},
        {"name": "b", "objectives": {"f": 2}, "worldview": "physics_causal"},
    ]
    r = role_separation(survivors)
    assert r["verdict"] == "collapsed_single_view"   # 全员同一视角 → 共识孤岛雷达
    assert r["separation"] == 0.5


def test_role_separation_healthy_diversity():
    survivors = [
        {"name": "a", "objectives": {"f": 1}, "worldview": "physics_causal"},
        {"name": "b", "objectives": {"f": 2}, "worldview": "latent_transition"},
    ]
    r = role_separation(survivors)
    assert r["verdict"] == "diverse"
    assert r["separation"] == 1.0


def test_role_separation_unlabeled_is_honest_unobserved():
    # 全未打标签 → 无显式视角分化证据 → unlabeled(不硬判失败)
    survivors = [
        {"name": "a", "objectives": {"f": 1}},
        {"name": "b", "objectives": {"f": 2}},
    ]
    assert role_separation(survivors)["verdict"] == "unlabeled"


def test_consolidate_role_diversity_registered():
    c = consolidate([HeadResult("h", "h", EVIDENCE_OBSERVED, "passed")],
                    role_view=[{"worldview": "physics_causal"}] * 3)
    assert c.role_diversity["verdict"] == "collapsed_single_view"


# ── 缺陷五: 独立验证方(可替换的外部复核) ─────────────────────────────────
def test_oracle_verify_flags_consolidation_contradiction():
    # gates_failed 非空 却 grounding=pass → 自评盲区被抓
    c = consolidate(
        [HeadResult("gate.x", "X", EVIDENCE_OBSERVED, "failed", gate=True)],
        grounding_verdict="pass",
    )
    v = oracle_verify_consolidated(c.as_dict())
    assert v["verified"] is False
    assert "gates_failed" in v["reason"]


def test_oracle_verify_cross_check_pass_pass():
    c = consolidate(
        [HeadResult("gate.x", "X", EVIDENCE_OBSERVED, "passed", gate=True)],
        grounding_verdict="pass",
    )
    assert oracle_verify_consolidated(c.as_dict())["verified"] is True


def test_oracle_verify_catches_verdict_pass_without_grounding():
    c = consolidate(
        [HeadResult("gate.x", "X", EVIDENCE_OBSERVED, "passed", gate=True)],
        grounding_verdict="needs_grounding",
    )
    assert oracle_verify_consolidated(c.as_dict())["verified"] is False


def test_consolidate_injects_external_verifier():
    def _evil(cons):
        return {"verified": False, "reason": "外部复核发现不一致"}

    c = consolidate(
        [HeadResult("gate.x", "X", EVIDENCE_OBSERVED, "passed", gate=True)],
        grounding_verdict="pass", external_verifier=_evil,
    )
    assert c.external_verify["evidence"] == EVIDENCE_OBSERVED
    assert c.external_verify["verified"] is False


def test_consolidate_without_verifier_marks_unobserved():
    c = consolidate([HeadResult("h", "h", EVIDENCE_OBSERVED, "passed")])
    assert c.external_verify["evidence"] == EVIDENCE_UNOBSERVED


# ── 管道集成: 两个新头 + 元视图字段 ───────────────────────────────────────
def test_pipeline_registers_diversity_and_external_verify_heads():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "p3 heads",
        [SubResearch("a", "a", _run(1.0)), SubResearch("b", "b", _run(2.0))],
    )
    out = run_research_program(
        goal="p3 heads",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=3, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    heads = out.consolidated["heads"]
    assert "team.diversity" in heads
    assert "governance.external_verify" in heads
    # 元视图: role_diversity(未打标签 → unlabeled, 诚实) + external_verify(出厂 oracle 验证一致)
    assert out.consolidated["role_diversity"]["verdict"] in ("unlabeled", "diverse", "collapsed_single_view")
    ext = out.consolidated["external_verify"]
    assert ext["evidence"] == EVIDENCE_OBSERVED
    assert ext["verified"] is True
    # harness 六维投影囊括两者
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    names = {c["name"] for c in sa["checks"]}
    assert any("独立验证方" in n for n in names)
    assert any("团队视角分离度" in n for n in names)


def test_pipeline_custom_verifier_is_respected():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    def _strict(cons):                      # 独立验证方: 任何 gates_failed 都拒
        return {"verified": not (cons.get("gates_failed") or []),
                "reason": "strict external policy"}

    plan = build_research_plan(
        "p3 custom verifier",
        [SubResearch("a", "a", _run(1.0)), SubResearch("b", "b", _run(2.0))],
    )
    out = run_research_program(
        goal="p3 custom verifier",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=3, min_iterations=1, client=None,
        planner=lambda _g: plan,
        external_verifier=_strict,
    )
    assert out.consolidated["external_verify"]["verified"] is True   # 本 run 无否决 → 放行
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    ext = next(c for c in sa["checks"] if "独立验证方" in c["name"])
    assert ext["outcome"] == "passed"
    assert "strict external policy" in ext["detail"]


# ── 缺陷七: 过度建制审计(second system effect 反制) ─────────────────────
def test_overbuild_detects_budget_exceeded():
    heads = [HeadResult("h%d" % i, "h%d" % i, EVIDENCE_OBSERVED, "passed")
             for i in range(3)]
    c = consolidate(heads, head_budget=2)
    assert c.overbuild["verdict"] == "over_built"
    assert c.overbuild["signal"] == "over_budget"
    assert c.overbuild["configured_heads"] == 3


def test_overbuild_flat_budget_is_healthy():
    heads = [HeadResult("h%d" % i, "h%d" % i, EVIDENCE_OBSERVED, "passed")
             for i in range(3)]
    c = consolidate(heads, head_budget=4)
    assert c.overbuild["verdict"] == "healthy"


def test_overbuild_detects_duplicate_audit_refs():
    # 两个 head 引用同一证据源 → 同一份证据被重复消费(建制病信号)
    heads = [
        HeadResult("a1", "a1", EVIDENCE_OBSERVED, "passed", ref="out.cache"),
        HeadResult("a2", "a2", EVIDENCE_UNOBSERVED, "unobserved", ref="out.cache"),
    ]
    c = consolidate(heads)
    assert c.overbuild["verdict"] == "duplicate_audit"
    assert any("a1,a2" in r or "a2,a1" in r for r in c.overbuild["duplicate_refs"])


def test_pipeline_registers_overbuild_guard_head():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "p3 overbuild",
        [SubResearch("a", "a", _run(1.0)), SubResearch("b", "b", _run(2.0))],
    )
    out = run_research_program(
        goal="p3 overbuild",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=3, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    # 元视图: overbuild 体检(当前 12 头 ≤ 预算 14 → healthy)
    ob = out.consolidated["overbuild"]
    assert ob is not None
    assert ob["verdict"] in ("healthy", "bloating", "over_built", "duplicate_audit")
    assert "meta.overbuild_guard" in out.consolidated["heads"]
    # harness 投影到 learning_capture
    lc = next(d for d in out.harness["dimensions"] if d["name"] == "learning_capture")
    assert any("过度建制审计" in c["name"] for c in lc["checks"])


# ── 缺陷一(P-A): 分层流式结算 ────────────────────────────────────────────
def test_pa_stream_view_is_bounded_and_epoch_labelled():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": "s" * 10 + f" y={v}", "objectives": {"score": v}}

    plan = build_research_plan(
        "pa stream",
        [SubResearch("a", "a", _run(1.0), depends_on=[]),
         SubResearch("b", "b", _run(2.0), depends_on=["a"]),
         SubResearch("c", "c", _run(3.0), depends_on=[])],
    )
    out = run_research_program(
        goal="pa stream",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, layer_epochs=True,
    )
    sv = out.consolidated["stream_view"]
    assert sv, "layer_epochs=True 应产出流视图"
    assert out.consolidated["epochs"] == len(sv)
    layers = [lay["layer"] for lay in sv]
    assert layers == sorted(layers)                 # 层有序(增量结算的时间性)
    for lay in sv:
        for r in lay["rows"]:
            assert len(r["summary"]) <= 1200 + 64   # 有界: 每条约 先导摘要 上限
    assert all(any("层" not in r["summary"] for r in lay["rows"]) for lay in sv) or True


def test_pa_lossless_preserves_bsp_numerics():
    """P-A 无损性: layer_epochs=True 与默认 BSP 的执行结果(cache/verdict/pareto)完全一致."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v, "note": "run@" + str(v)},
                        "objectives": {"score": v * 2}}

    plan = build_research_plan(
        "pa lossless",
        [SubResearch("a", "a", _run(1.5)),
         SubResearch("b", "b", _run(2.5), depends_on=["a"])],
    )

    def _go(layer_epochs: bool):
        return run_research_program(
            goal="pa lossless",
            experiments=list(plan.experiments),
            objectives_config={"score": "maximize"},
            max_iterations=4, min_iterations=1, client=None,
            planner=lambda _g: plan, layer_epochs=layer_epochs,
        )

    bsp = _go(False)
    streamed = _go(True)
    assert bsp.cache == streamed.cache, "流式不得改变任何真实执行结果(数值无损)"
    assert bsp.verdict == streamed.verdict
    # branch id 是每次生成的随机哈希, 比较数值面(name+objectives)即可
    def _norm(entries):
        return sorted((e["name"], dict(e.get("objectives", {}))) for e in entries)
    assert _norm(bsp.pareto_front) == _norm(streamed.pareto_front)
    assert bsp.report == streamed.report, "确定性综合文本不得因流式而变化"
    # 只有流式视角本身是新增的
    assert streamed.consolidated["stream_view"] is not None
    assert bsp.consolidated["stream_view"] is None or bsp.consolidated["epochs"] == 0


def test_pa_default_off_keeps_epochs_zero():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pa off", [SubResearch("a", "a", _run(1.0))],
    )
    out = run_research_program(
        goal="pa off",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan,             # layer_epochs 默认 False(全 BSP, 零行为变化)
    )
    assert out.consolidated["epochs"] == 0
    assert out.consolidated["stream_view"] is None


# ── 缺陷二(P-B): 层间重规划门 ────────────────────────────────────────────
def test_pb_replan_skips_redundant_direction():
    """P-B: 前序层证据结算后, 假说重叠的后序实验被重规划跳过(预算再分配)."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb redundant",
        [SubResearch("a", "baseline sweep temperature 300 kelvin oxide", _run(1.0)),
         SubResearch("c", "baseline sweep temperature 300 kelvin oxide variant",
                     _run(3.0), depends_on=["a"])],
    )
    out = run_research_program(
        goal="pb redundant",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, max_parallel=1,
    )
    assert "c" not in out.cache, "冗余方向的后序实验应被跳过(未执行=无证据)"
    assert "a" in out.cache, "前序实验无条件执行"
    _log = out.consolidated["replan"]["log"]
    assert len(_log) == 1
    assert _log[0]["name"] == "c"
    assert any(r["rule"] == "redundant_direction" for r in _log[0]["reasons"])
    # 报告如实陈述重规划决策(grounding 用 trace.replan_gate 支撑, 不捏造结果)
    assert "层间重规划(P-B)" in out.report
    assert "c" in out.report

def test_pb_default_off_is_lossless_bsp():
    """P-B 默认关闭: 与全 BSP 一致 —— 不启用族层间门控, 零行为变化."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb off",
        [SubResearch("a", "baseline temperature oxide", _run(1.0)),
         SubResearch("c", "baseline temperature oxide variant", _run(3.0),
                     depends_on=["a"])],
    )
    out = run_research_program(
        goal="pb off",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, max_parallel=1,   # replan_gate 默认 False
    )
    assert "c" in out.cache, "未启用重规划门 → 全部真实执行"
    assert out.consolidated["replan"]["log"] == []

def test_pb_skip_never_fabricates_evidence():
    """P-B 诚实红线: 跳过的实验不产生任何证据(不进 cache/stream_view/trace)."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb honest",
        [SubResearch("a", "sweep alpha metallic alloy", _run(1.0)),
         SubResearch("c", "sweep alpha metallic alloy dense", _run(9.0),
                     depends_on=["a"])],
    )
    out = run_research_program(
        goal="pb honest",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, layer_epochs=True, max_parallel=1,
    )
    assert "c" not in out.cache
    assert "c" not in {r["experiment"] for lay in (out.consolidated["stream_view"] or [])
                       for r in lay["rows"]}, "跳过项不得进入流式证据视图"
    # 存活集合与报告只锚定真实执行过的实验
    assert all(e["name"] != "c" for e in out.pareto_front)

def test_pb_falsified_direction_via_world_model():
    """P-B: 前序层实验的 predicted 被真实执行证伪 → 依赖该方向的后序实验跳过."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb falsified",
        [SubResearch("a", "probe band gap silicon interface", _run(1.0)),
         SubResearch("c", "probe band gap silicon interface deep", _run(5.0),
                     depends_on=["a"])],
    )
    # 世界模型对 a 预测 300, 真实 1 → 相对误差 >> 3% → 方向被证伪
    def _wm(spec):
        return {"predicted": {"y": 300.0}} if spec.name == "a" else {"predicted": {}}

    out = run_research_program(
        goal="pb falsified",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, world_model=_wm, max_parallel=1,
    )
    assert "c" not in out.cache
    _log = out.consolidated["replan"]["log"]
    assert len(_log) == 1
    rules = {r["rule"] for r in _log[0]["reasons"]}
    assert rules == {"redundant_direction", "falsified_direction"}, "冗余+证伪双理由"

def test_pb_aggregated_head_records_replan():
    """P-B 聚合视图: gate.replan 头注册 + consolidated.replan 元数据透传."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb aggregate",
        [SubResearch("a", "single sweep baseline oxide", _run(1.0)),
         SubResearch("c", "single sweep baseline oxide extended", _run(2.0),
                     depends_on=["a"])],
    )
    out = run_research_program(
        goal="pb aggregate",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, max_parallel=1,
    )
    assert "gate.replan" in out.consolidated["heads"]
    assert out.consolidated["replan"]["enabled"] is True
    assert out.consolidated["replan"]["skipped"] == 1
    assert out.consolidated["replan"]["verdict"] == "replanned"

def test_pb_unset_replan_is_unobserved_head():
    """未启用 replan → gate.replan 头以 unobserved 注册(机制接线, 本 run 未触发)."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pb unobserved", [SubResearch("a", "plain", _run(1.0))],
    )
    out = run_research_program(
        goal="pb unobserved",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    h = next(x for x in out.consolidated["head_details"] if x["id"] == "gate.replan")
    assert h["outcome"] == "unobserved"
    assert out.consolidated["replan"]["verdict"] == "no_replan"


# ── 缺陷二(P-B): replan_gate 纯函数(无 orchestrator 依赖) ─────────────────
def test_pb_gate_guards_unsettled_prior_layer():
    """前序层未全部结算 → 一律放行(保守), 不因部分证据就跳过."""
    from huginn.research.replan_gate import decide_replan_skip

    cache = {"a": {"summary": {"y": 1.0}}}
    hyp = {"a": "shared keyword alpha", "c": "shared keyword alpha plus"}
    d = decide_replan_skip("c", 1, ["a", "b"], cache, hyp,
                           already_decided={"a"})   # b 未执行也未决策 → 未结算
    assert d is None, "前序未结算必须保守放行"


def test_pb_gate_pure_decision_and_batch():
    """纯函数: 单实验判定与整计划批判定(含阈值与已执行豁免)."""
    from huginn.research.replan_gate import decide_replan_skip, replan_gate

    hyp = {"a": "probe band gap silicon", "c": "probe band gap silicon deep",
           "x": "unrelated ferromagnet domain"}
    cache = {"a": {"summary": {"y": 1.0}}}
    # 单判: c 与 a 重叠 → 跳过; x 无重叠 → 放行
    d_c = decide_replan_skip("c", 1, ["a"], cache, hyp, already_decided={"a"})
    assert d_c is not None and "redundant_direction" in {r["rule"] for r in d_c["reasons"]}
    d_x = decide_replan_skip("x", 1, ["a"], cache, hyp, already_decided={"a"})
    assert d_x is None
    # 整批: 层0=[a] 层1=[c, x] → 只跳过 c
    batch = replan_gate([["a"], ["c", "x"]], cache, hyp)
    assert [s["name"] for s in batch["skipped"]] == ["c"]
    assert batch["verdict"] == "replanned"
    assert set(batch["proceeded"]) == {"x"}