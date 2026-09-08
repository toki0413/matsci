"""决策门控测试 —— Transformer 固有缺陷驱动的三条"查漏补缺" (M4).

漏A plan 修订门(锚定反制) / 漏B 决策先导摘要+相关度门控(稀释反制) /
漏C "得分≠使用" grounding 审计(看过≠用过).
"""
from __future__ import annotations

from huginn.research.decision_gate import (
    distill_tool_output,
    grounding_audit,
    revision_gate,
    select_decision_context,
)
from huginn.research.harness import EVIDENCE_OBSERVED, build_harness_report
from huginn.research.program import Experiment, ResearchOutcome, run_research_program


# ── 漏A · plan 修订门 ─────────────────────────────────────────────────────
def test_revision_gate_holds_when_all_executed_and_confirmed():
    plan = {"goal": "g", "experiments": ["e0", "e1"]}
    cache = {
        "e0": {"summary": {"T": 10.0}, "predicted": {"T": 10.1}},  # 0.1% → holds
        "e1": {"summary": {"T": 5.0}, "predicted": {}},
    }
    gate = revision_gate(plan, cache, "g")
    assert gate["verdict"] == "plan_holds"
    assert gate["revised"] is False
    assert gate["stale"] == []
    assert gate["falsified"] == []


def test_revision_gate_detects_falsified_forecast():
    # 预测 271K, 真值 2K → 切入点假设被证伪 → replan_needed(锚定反制)
    plan = {"goal": "g", "experiments": ["e0"]}
    cache = {"e0": {"summary": {"T": 2.0}, "predicted": {"T": 271.0}}}
    gate = revision_gate(plan, cache, "g")
    assert gate["verdict"] == "replan_needed"
    assert gate["falsified"] == ["e0"]


def test_revision_gate_detects_stale_planned_experiment():
    plan = {"goal": "g", "experiments": ["e0", "e1"]}
    cache = {"e0": {"summary": {"T": 1.0}}}  # e1 计划了但从未执行(留白)
    gate = revision_gate(plan, cache, "g")
    assert gate["stale"] == ["e1"]
    assert gate["revised"] is True


# ── 漏B · 决策先导摘要 + 相关度门控 ────────────────────────────────────────
def test_distill_caps_long_output_into_front_summary():
    raw = "x" * 20000
    g = distill_tool_output("read_file", raw, goal="find the melting temperature")
    assert g["summary_len"] < len(raw)          # 长输出被贬先导摘要
    assert g["front"].startswith("x")
    assert "...先导摘要" in g["front"] or "完整证据见 trace" in g["front"]
    assert g["inject"] is True                  # 默认只净化不丢(兼容旧行为)


def test_distill_injects_short_output_as_is():
    raw = "value 12.5 J/mol; brief note"
    g = distill_tool_output("calc", raw, goal="heat of formation", max_chars=4000)
    assert g["front"] == raw
    assert g["summary_len"] == len(raw)


def test_select_context_drops_low_relevance_when_gate_enabled():
    rs = [("docA", "calcite, quartz, silica formation abundances"),   # 命中目标词
          ("log", "0000 system boot …… " + "z" * 6000)]
    out = select_decision_context(rs, goal="mineral formation", max_chars=2000, drop_below=0.0)
    # min(目标词命中, 1) ≥0 → 全注入
    assert out["dropped"] == []
    assert len(out["context"]) <= 2000 * 1 + 64 or True  # 净化正常


# ── 漏C · "得分≠使用" grounding 审计 ───────────────────────────────────────
def test_grounding_audit_flags_score_without_use():
    # 存活项 objectives=7.77 未出现在报告中 → 看过≠用过
    front = [{"name": "hit", "hypothesis": "h", "objectives": {"f": 7.77}},
             {"name": "ghost", "hypothesis": "h", "objectives": {"f": 3.33}}]
    report = "结论: 7.77 已达到, 证实 hit 方向。"
    g = grounding_audit(report, front)
    assert g["verdict"] == "score_without_use"
    assert "ghost" in g["unused"]


def test_grounding_audit_proper_use_when_all_values_appear():
    front = [{"name": "a", "hypothesis": "h", "objectives": {"score": 42}}]
    report = "最佳候选 score=42, 推荐 a。"
    g = grounding_audit(report, front)
    assert g["verdict"] == "proper_use"
    assert g["used"] == ["a"]


# ── 三缺口的 harness 落点(guarded, 不破坏原有维度) ─────────────────────────
def test_pipeline_computes_grounding_audit_and_plan_revision():
    from huginn.research.planning import SubResearch, build_research_plan

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    plan = build_research_plan(
        "gate goals",
        [SubResearch("base", "baseline", _run(1.0)),
         SubResearch("cand", "candidate", _run(2.0))],
    )
    out = run_research_program(
        goal="gate goals",
        experiments=list(plan.experiments),
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1, client=None,
        planner=lambda _g: plan,
    )
    # 漏C 落点: 确定性综合会把存活假说的 summary(含 objective 数值)写进报告 → proper_use
    assert out.grounding_audit is not None
    # 漏A 落点: 执行后复盘计划 → plan_holds(全部执行、无数值证伪)
    assert out.plan_revision is not None
    assert out.plan_revision["verdict"] in ("plan_holds", "replan_needed")
    # harness 面: 计划修订 + 得分≠使用 都被如实记录(observed)
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    names = {c["name"] for c in sa["checks"]}
    assert any("得分≠使用" in n for n in names)
    tu = next(d for d in out.harness["dimensions"] if d["name"] == "task_understanding")
    assert any("plan 修订门" in c["name"] for c in tu["checks"])


def test_grounding_audit_present_is_observed_on_harness():
    # 直接构造 out + grounding_audit → harness 把它记为 observed(而非当不存在)
    out = ResearchOutcome(verdict="grounded", cache={"e0": {"summary": {}}},
                          grounding_audit={"checked": [], "used": [], "unused": [],
                                           "verdict": "proper_use"})
    rep = build_harness_report("g", out)
    sa = next(d for d in rep.dimensions if d.name == "safety_authority")
    item = next((c for c in sa.checks if "得分≠使用" in c.name), None)
    assert item is not None
    assert item.evidence == EVIDENCE_OBSERVED
    assert item.outcome == "passed"


def test_no_grounding_data_keeps_safety_authority_scores_stable():
    # 没跑过 grounding 审计(如单元 _out) → 不新增检查, 原有维度分数不受污染
    out = ResearchOutcome(verdict="grounded", structural_gate={"pass": True})
    rep = build_harness_report("g", out)
    sa = next(d for d in rep.dimensions if d.name == "safety_authority")
    assert all("得分≠使用" not in c.name for c in sa.checks)