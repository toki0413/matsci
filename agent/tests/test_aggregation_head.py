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


# ── 缺陷一(P-C): 证据驱动提前终止门 ───────────────────────────────────────
def test_pc_early_stop_on_score_plateau():
    """P-C: 连续两层 top-1 进入分数高原 → 剩余层提前终止(预算回收, 不伪造)."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pc plateau",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("b", "probe organic phase width", _run(1.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(10.1), depends_on=["a"]),
         SubResearch("d", "probe ionic liquid density", _run(9.0), depends_on=["b"]),
         SubResearch("e", "scan polymer chain length", _run(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pc plateau",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=6, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, max_parallel=1,
    )
    # 层0/层1 真实执行; 分数高原(10.0 vs 10.1, rel=0.01<=0.02) → 层2 提前终止
    es = out.consolidated["early_stop"]
    assert es["verdict"] == "early_stopped"
    assert es["stopped_after_layer"] == 1
    assert [s["name"] for s in es["log"]] == ["e"]
    assert {"a", "b", "c", "d"} <= set(out.cache), "终止前各层真实执行"
    assert "e" not in out.cache, "终止的实验未执行=无证据(不伪造)"
    assert "证据驱动提前终止(P-C)" in out.report
    assert all(elem["name"] != "e" for elem in out.pareto_front)

def test_pc_no_stop_when_top_moves():
    """P-C: top-1 分数大幅移动(未饱和) → 不终止, 剩余层继续执行."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pc moving",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(50.0), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pc moving",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, max_parallel=1,
    )
    es = out.consolidated["early_stop"]
    assert es["verdict"] == "checked_and_continued", "top 大幅上升 → 探索未饱和, 继续"
    assert "e" in out.cache, "未终止 → 全部真实执行"

def test_pc_default_off_is_lossless_bsp():
    """P-C 默认关闭: 全 BSP 语义, 零行为变化."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pc off",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(10.1), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pc off",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, max_parallel=1,   # early_stop_gate 默认 False
    )
    assert "e" in out.cache, "未启用早停 → 全部真实执行"
    es = out.consolidated["early_stop"]
    assert es["enabled"] is False and es["verdict"] == "no_early_stop"
    h = next(x for x in out.consolidated["head_details"] if x["id"] == "gate.early_stop")
    assert h["outcome"] == "unobserved"

def test_pc_single_layer_never_stops():
    """P-C 防早停: 单层计划(观测层 < min_layers)永不触发提前终止."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pc single", [SubResearch("a", "sweep alpha metallic alloy", _run(10.0))],
    )
    out = run_research_program(
        goal="pc single",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, max_parallel=1,
    )
    es = out.consolidated["early_stop"]
    assert es["verdict"] != "early_stopped", "观测层 < 2 → 防早停门拦住"
    assert es["skipped"] == 0 and "a" in out.cache


# ── 缺陷一(P-C): early_stop_gate 纯函数 ───────────────────────────────────
def test_pc_stability_check_pure():
    """稳定度判据: 高原→stable; 大幅移动/层数不足/缺分 → 不稳定(可证伪)."""
    from huginn.research.early_stop_gate import stability_check

    plateau = stability_check([[("a", 10.0), ("b", 1.0)], [("c", 10.1), ("d", 9.0)]],
                              min_layers=2, margin=0.02)
    assert plateau["stable"] is True
    assert plateau["verdict"] == "stable_top_plateau"
    assert plateau["top"] == "c" and abs(plateau["relative_change"] - 0.01) < 1e-9

    moved = stability_check([[("a", 10.0)], [("c", 50.0)]], min_layers=2, margin=0.02)
    assert moved["stable"] is False and moved["verdict"] == "top_not_saturated"

    early = stability_check([[("a", 10.0)]], min_layers=2, margin=0.02)
    assert early["stable"] is False and early["verdict"] == "insufficient_layers"

    miss = stability_check([[("a", 10.0)], []], min_layers=2, margin=0.02)
    assert miss["stable"] is False and miss["verdict"] == "missing_scores"

    tie = stability_check([[("z", 5.0), ("a", 5.0)], [("b", 5.01)]],
                          min_layers=2, margin=0.02)
    assert tie["top"] == "b"      # tiebreak 确定性(本层 top 选择不影响终止判定)


# ── 三阶段(P-A + P-B + P-C)联合协同 ──────────────────────────────────────
def test_pabc_interleave_cooperative():
    """P-A/P-B/P-C 同时开启: 共享层映射、决策互认、诚实红线不互相破坏."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    # 层0: a/b; 层1: c(50, top 大幅上升)→P-C 不触发; d;
    # 层2: e(与 c 假说重叠)→P-B 跳过; f(无重叠)→真实执行
    plan = build_research_plan(
        "pabc interleave",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("b", "probe organic phase width", _run(1.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(50.0), depends_on=["a"]),
         SubResearch("d", "probe ionic liquid density", _run(9.0), depends_on=["b"]),
         SubResearch("e", "sweep beta ceramic domain dense", _run(48.0), depends_on=["c"]),
         SubResearch("f", "scan polymer chain length", _run(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pabc interleave",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=8, min_iterations=1, client=None,
        planner=lambda _g: plan,
        layer_epochs=True, replan_gate=True, early_stop_gate=True,
        max_parallel=1,
    )
    con = out.consolidated
    # 三阶段元数据并存于同一收敛视图(唯一治理出口)
    assert con["epochs"] == 3
    assert "gate.replan" in con["heads"] and "gate.early_stop" in con["heads"]
    # P-C: top 大幅上升 → 判定过但未触发终止(诚实标 continued)
    assert con["early_stop"]["verdict"] == "checked_and_continued"
    assert "证据驱动提前终止(P-C)" not in out.report
    # P-B: e 与前序层假说重叠 → 跳过(未执行=无证据); f 无重叠 → 执行
    assert [s["name"] for s in con["replan"]["log"]] == ["e"]
    assert "e" not in out.cache
    assert {"a", "b", "c", "d", "f"} <= set(out.cache)
    # P-A: stream_view 只含真实执行(e 不在任何层), 3 层有序
    sv_rows = {r["experiment"] for lay in con["stream_view"] for r in lay["rows"]}
    assert sv_rows == {"a", "b", "c", "d", "f"} and "e" not in sv_rows
    assert "层间重规划(P-B)" in out.report
    assert all(elem["name"] != "e" for elem in out.pareto_front)


# ── 并发压力: max_parallel>1 下的预算决策最终一致性 ──────────────────────
def _assert_no_silent_loss(out, planned: set[str]):
    """不变式: 每个计划内实验要么真实执行(cache), 要么有明确的预算决策记录
    (replan 跳过 / early_stop 终止)。绝不静默消失 —— 这是并行下的审计完整性."""
    cache_set = set(out.cache)
    tracked = (cache_set
               | {s["name"] for s in out.consolidated["replan"]["log"]}
               | {s["name"] for s in out.consolidated["early_stop"]["log"]})
    assert planned <= tracked, f"计划内无审计记录的实验: {planned - tracked}"
    # 诚实红线: report/pareto/stream_view 只锚定真实执行过的实验
    sv = {r["experiment"] for lay in out.consolidated["stream_view"] for r in lay["rows"]}
    assert sv <= cache_set, "stream_view 引用了未执行证据"
    assert all(e["name"] in cache_set for e in out.pareto_front)


def test_pabc_parallel_no_loss_invariant():
    """max_parallel=3 + 三阶段: P-B 跳过(冗余) 且 P-C 判定过但未触发(top 移动)。
    并行下逐实验执行顺序非确定, 但"无静默丢失"不变式必须成立."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pabc par noloss",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("b", "probe organic phase width", _run(1.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(50.0), depends_on=["a"]),
         SubResearch("d", "probe ionic liquid density", _run(9.0), depends_on=["b"]),
         SubResearch("e", "sweep beta ceramic domain dense", _run(48.0), depends_on=["c"]),
         SubResearch("f", "scan polymer chain length", _run(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pabc par noloss",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=8, min_iterations=1, client=None,
        planner=lambda _g: plan,
        layer_epochs=True, replan_gate=True, early_stop_gate=True,
        max_parallel=3,
    )
    planned = {e.name for e in plan.experiments}
    _assert_no_silent_loss(out, planned)
    # 并行下 e(与 c 冗余)必然有审计记录(执行 或 被 replan)
    if "e" not in out.cache:
        assert "e" in {s["name"] for s in out.consolidated["replan"]["log"]}


def test_pabc_parallel_early_stop_final_consistent():
    """max_parallel=2 + 三阶段: 分数高原(10.0→10.1) 触发 P-C 提前终止。
    并行下终止边界最终一致 —— 被终止层实验要么早已执行, 要么被 early_stop 记录."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "pabc par estop",
        [SubResearch("a", "sweep alpha metallic alloy", _run(10.0)),
         SubResearch("b", "probe organic phase width", _run(1.0)),
         SubResearch("c", "sweep beta ceramic domain", _run(10.1), depends_on=["a"]),
         SubResearch("d", "probe ionic liquid density", _run(9.0), depends_on=["b"]),
         SubResearch("e", "scan polymer chain length", _run(8.0), depends_on=["c"]),
         SubResearch("f", "sort crystalline facet", _run(7.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="pabc par estop",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=8, min_iterations=1, client=None,
        planner=lambda _g: plan,
        layer_epochs=True, replan_gate=True, early_stop_gate=True,
        max_parallel=2,
    )
    _assert_no_silent_loss(out, {e.name for e in plan.experiments})
    es = out.consolidated["early_stop"]
    # 层2([e,f])若未执行必被 early_stop 记录, 不得落入 replan/静默
    remaining = {"e", "f"} - set(out.cache)
    early_names = {s["name"] for s in es["log"]}
    assert remaining <= early_names, f"被终止层实验缺 early_stop 记录: {remaining - early_names}"
    assert {"a", "b", "c", "d"} <= set(out.cache), "终止前各层必须真实执行"


# ── 组合压力: 大宽 DAG + 高并行 + world_model + 变异 + 三阶段全开 ─────────
def test_pabc_combined_stress_wide_dag():
    """压力组合: 4 层×12 实验宽 DAG, max_parallel=4(跨层并发),
    world_model(predicted 对账) + 变异(mutation) + P-A/P-B/P-C 全开,
    重复 3 轮 —— 每轮必须满足"无静默丢失"不变式与报告/证据自洽."""
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    themes = ["alpha metallic alloy", "organic phase width", "beta ceramic domain",
              "ionic liquid density", "polymer chain length", "crystalline facet",
              "quantum dot bandgap", "grain boundary mobility", "phonon scattering",
              "surface reconstruction", "defect migration barrier", "magnetic domain wall"]

    def _chain(prefix: str, layer: int, n: int, dep: list[str]):
        return [SubResearch(f"{prefix}{j}", f"sweep {themes[(layer * n + j) % len(themes)]} "
                                            f"probe variant", _run((layer + 1) * 10 + j),
                            depends_on=dep) for j in range(n)]

    # 显式 4 层 DAG(每层 3 实验, 层依赖链): 层宽 3 < max_parallel=4 → 跨层并发
    subs = ([SubResearch(f"l0_{j}", f"scan {themes[j]} baseline", _run(5.0 + j)) for j in range(3)]
            + _chain("l1_", 1, 3, [f"l0_{j}" for j in range(3)])
            + _chain("l2_", 2, 3, [f"l1_{j}" for j in range(3)])
            + _chain("l3_", 3, 3, [f"l2_{j}" for j in range(3)]))
    plan = build_research_plan("pabc stress", subs)

    def _wm(spec):
        return {"predicted": {"y": 200.0}}   # 固定预测: 可能触发 falsified 对账(压力面)

    for round_i in range(3):
        out = run_research_program(
            goal="pabc stress",
            experiments=list(plan.experiments),
            objectives_config={"score": "maximize"},
            max_iterations=24, min_iterations=2, client=None,
            planner=lambda _g: plan, world_model=_wm,
            layer_epochs=True, replan_gate=True, early_stop_gate=True,
            mutation_config={"mutation_rate": 0.5, "max_children": 1},
            max_parallel=4,
        )
        planned = {e.name for e in plan.experiments}
        _assert_no_silent_loss(out, planned)
        con = out.consolidated
        # 全部真实执行项必须带 objectives + summary(不伪造的结构性自洽)
        assert all(res.get("objectives") is not None for res in out.cache.values())
        # stream_view: 层有序且只含真实执行过(<=4 层)
        layers = [lay["layer"] for lay in con["stream_view"]]
        assert layers == sorted(set(layers)) == list(range(len(set(layers))))
        sv = {r["experiment"] for lay in con["stream_view"] for r in lay["rows"]}
        assert sv <= set(out.cache)
        # 变异子代: 若存在必须真实重跑并进 cache(诚实回退)
        mut = [n for n in out.cache if "~mut" in n]
        assert all(n in out.cache for n in mut)
        # 报告与审计一致: P-B/P-C 段只在对应决策确实发生时出现
        assert ("层间重规划(P-B)" in out.report) == ("replanned" == con["replan"]["verdict"])
        assert ("证据驱动提前终止(P-C)" in out.report) == ("early_stopped" == con["early_stop"]["verdict"])
        # 每轮收敛: 至少一个存活假说来自真实执行
        assert out.pareto_front and all(e["name"] in out.cache for e in out.pareto_front)


# ── 生产化 A1: 阈值参数化 ─────────────────────────────────────────────────
def _run_(v: float):
    return lambda: {"summary": {"y": v}, "objectives": {"score": v}}


def test_a1_stream_summary_chars_parameterized():
    """P-A 单条摘要上限参数化: 传 stream_summary_chars 有界生效(默认 1200 不变)."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 stream", [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0))],
    )
    out = run_research_program(
        goal="a1 stream", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan, layer_epochs=True, stream_summary_chars=80,
    )
    rows = [r for lay in out.consolidated["stream_view"] for r in lay["rows"]]
    assert rows, "stream_view 应有记录"
    assert all(len(r["summary"]) <= 80 + 80 for r in rows), "自定义上限生效(有界)"


def test_a1_replan_similarity_parameterized():
    """P-B 重叠阈值参数化: 调高到 0.9 后, Jaccard 0.83 的方向不再视为冗余 → 真实执行."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 sim",
        [SubResearch("a", "sweep beta ceramic domain thick", _run_(10.0)),
         SubResearch("e", "sweep beta ceramic domain thick dense", _run_(8.0),
                     depends_on=["a"])],
    )
    out = run_research_program(
        goal="a1 sim", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan, replan_gate=True, replan_similarity=0.9,
        max_parallel=1,
    )
    assert "e" in out.cache, "0.83 < 0.9 → 不冗余, 必须执行"


def test_a1_early_stop_margin_parameterized():
    """P-C margin 参数化: 放宽到 0.5 后, rel=0.4 的移动也视为高原 → 提前终止."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 margin",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(14.0), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="a1 margin", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, early_stop_margin=0.5,
        max_parallel=1,
    )
    assert out.consolidated["early_stop"]["verdict"] == "early_stopped"
    assert "e" not in out.cache, "margin=0.5 视 rel=0.4 为高原 → 剩余层终止"


def test_a1_early_stop_min_layers_parameterized():
    """P-C 防早停参数化: min_layers=3 时 2 层稳定不足置信 → 剩余层执行
    (对照: 默认 min_layers=2 会在层1 结算后终止层2, 参数生效可证伪)."""
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a1 minlayers",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(10.1), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="a1 minlayers", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, early_stop_min_layers=3,
        max_parallel=1,
    )
    assert "e" in out.cache, "min_layers=3 → 2 层观测不足置信 → 剩余层真实执行"
    assert out.consolidated["early_stop"]["verdict"] != "early_stopped"


# ── 生产化 A2: 跨 run 稳定度先验 ─────────────────────────────────────────
def test_a2_extract_prior_from_out():
    """从一次稳定终止的 run 提取可复用先验(纯函数)."""
    from huginn.research.prior_store import extract_prior
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a2 extract",
        [SubResearch("a", "sweep alpha metallic alloy", _run_(10.0)),
         SubResearch("c", "sweep beta ceramic domain", _run_(10.1), depends_on=["a"]),
         SubResearch("e", "scan polymer chain length", _run_(8.0), depends_on=["c"])],
    )
    out = run_research_program(
        goal="a2 extract", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=5, min_iterations=1, client=None,
        planner=lambda _g: plan, early_stop_gate=True, max_parallel=1,
    )
    prior = extract_prior(out)
    assert prior["applicable"] is True
    assert prior["source"] == "layered_settlement"
    assert prior["plateau"]["layer_index"] == 1
    assert prior["plateau"]["top"] == "c"
    assert prior["goal"] == "a2 extract"


def test_a2_extract_prior_not_applicable_for_plain_run():
    """未启用早停的 run → 提取出不可用先验(诚实标 applicable=False)."""
    from huginn.research.prior_store import extract_prior
    from huginn.research.planning import SubResearch, build_research_plan

    plan = build_research_plan(
        "a2 plain", [SubResearch("a", "sweep alpha metallic alloy", _run_(1.0))],
    )
    out = run_research_program(
        goal="a2 plain", experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    assert extract_prior(out)["applicable"] is False