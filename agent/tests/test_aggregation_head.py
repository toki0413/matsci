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
)
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