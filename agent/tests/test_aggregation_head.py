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