"""Tests for Self-Harness report (Better Harness 五维对齐, M1)."""
from __future__ import annotations

from huginn.research.harness import (
    EVIDENCE_OBSERVED,
    EVIDENCE_UNOBSERVED,
    EVIDENCE_MISSING,
    HarnessReport,
    build_harness_report,
    make_task_episode_id,
)
from huginn.research.program import ResearchOutcome


def _out(**kw) -> ResearchOutcome:
    o = ResearchOutcome()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_make_task_episode_id_is_stable_and_scoped():
    a = make_task_episode_id("search optimal doping for battery cathode")
    b = make_task_episode_id("search optimal doping for battery cathode")
    assert a == b                          # 同一任务同日内稳定(可聚合)
    assert a.startswith("ep-")
    assert len(a) == 23
    # salt 把 id 限定到单实例维度 (agent/machine 维度)
    assert make_task_episode_id("g", salt="a:m") != make_task_episode_id("g", salt="b:n")


def test_full_observed_run_gets_high_scores():
    out = _out(
        plan_summary={"sub_research": ["a", "b"], "max_parallel": 2},
        supervision_log=[{"round": 1, "action": "keep"}],
        verdict="grounded",
        structural_gate={"pass": True},
        cache={"exp_1": {"summary": {"m": 1}}},
        workspace_verified=True,
        law_model_used={"planner_rollout": ["a", "b"]},  # 世界模型 predict 真实参与决策
    )
    rep = build_harness_report("goal-x", out, agent="a", machine="m")
    assert rep.task_episode.startswith("ep-")
    assert set(d.name for d in rep.dimensions) == {
        "task_understanding", "controlled_execution", "change_validation",
        "reliable_delivery", "learning_capture", "safety_authority",
    }
    # 全部 observed passed
    for dim in rep.dimensions:
        assert dim.evidence == EVIDENCE_OBSERVED
        assert dim.score == 1.0
    assert rep.overall == 1.0


def test_safety_authority_reports_world_model_not_actually_used():
    # 误区二诚实审计: law_model 存在于代码 ≠ 部署时在"真规划".
    # 本 run 无 law_model_used/planner_rollout 证据 → world_model 项标 unobserved 不虚报.
    out = _out(
        verdict="grounded",
        structural_gate={"pass": True},   # 外部安全否决器真实触发
    )
    rep = build_harness_report("goal-x", out)
    sa = next(d for d in rep.dimensions if d.name == "safety_authority")
    wm = next(c for c in sa.checks if "世界模型真用?" in c.name)
    assert wm.evidence == EVIDENCE_UNOBSERVED
    assert wm.outcome == "unobserved"
    assert sa.evidence == EVIDENCE_OBSERVED  # 否决器触发 → 该维仍有 observed 证据态
    assert sa.score < 1.0


def test_missing_planner_reports_unobserved_not_fullcredit():
    # 红线: 配置存在 ≠ 能力可用 — 没跑 planner 就不能给满分
    out = _out(plan_summary=None)
    rep = build_harness_report("goal-x", out)
    tu = next(d for d in rep.dimensions if d.name == "task_understanding")
    assert tu.evidence == EVIDENCE_UNOBSERVED
    assert tu.score == 0.5
    assert rep.overall < 1.0


def test_observed_but_failed_gate_scores_zero():
    out = _out(
        verdict="needs_grounding",
        ungrounded=["unsubstantiated: x"],
        structural_gate={"pass": False},
    )
    rep = build_harness_report("goal-x", out)
    cv = next(d for d in rep.dimensions if d.name == "change_validation")
    # verdict 项 observed but failed → 一项 0; structural_gate 一项 failed → 0
    outcomes = {c.outcome for c in cv.checks}
    assert "failed" in outcomes
    assert cv.score < 1.0


def test_json_output_carries_task_episode():
    out = _out(verdict="grounded", cache={"e": {"summary": {}}})
    rep = build_harness_report("goal-x", out, agent="a", machine="m")
    payload = rep.to_dict()
    assert payload["task_episode"].startswith("ep-")
    assert payload["agent"] == "a"
    assert payload["machine"] == "m"
    assert "dimensions" in payload
    # JSON 可序列化
    rep.to_json()


def test_learning_capture_missing_when_no_mechanism():
    # 模拟自省模块不可用 → learning_capture=missing(计入缺口)
    out = _out(cache={"e": {"summary": {}}})
    rep = HarnessReport(goal="g").assess(out)
    lc = next(d for d in rep.dimensions if d.name == "learning_capture")
    assert lc.evidence in (EVIDENCE_OBSERVED, EVIDENCE_UNOBSERVED, EVIDENCE_MISSING)
    assert 0.0 <= lc.score <= 1.0


# ── M2: run_research_program 返回 out.harness ─────────────────────────────
def test_pipeline_populates_out_harness():
    """真实跑一条确定性深研管线, out.harness 被填充为五维报告 dict (一键出报告)."""
    from huginn.research.program import Experiment, run_research_program

    def _run(v: float):
        return lambda: {"summary": {"y": v}, "objectives": {"score": v}}

    out = run_research_program(
        goal="harness m2 integration",
        experiments=[Experiment("e0", "baseline", _run(1.0)),
                     Experiment("e1", "candidate", _run(2.0))],
        objectives_config={"score": "maximize"},
        max_iterations=4, min_iterations=1,
        client=None,                       # 确定性综合, 不依赖 LLM
        harness_agent="huginn-lite", harness_machine="sandbox-1",
    )
    assert out.harness is not None, "管线应填充 out.harness"
    assert out.harness["task_episode"].startswith("ep-")
    assert out.harness["agent"] == "huginn-lite"
    assert out.harness["machine"] == "sandbox-1"
    assert {d["name"] for d in out.harness["dimensions"]} == {
        "task_understanding", "controlled_execution", "change_validation",
        "reliable_delivery", "learning_capture", "safety_authority",
    }
    # 主环路未接入 law_model → 世界模型项诚实标 unobserved(非出身虚报真深思 D)
    sa = next(d for d in out.harness["dimensions"] if d["name"] == "safety_authority")
    wm = next(c for c in sa["checks"] if "世界模型真用?" in c["name"])
    assert wm["outcome"] == "unobserved"
    # 同一 goal 同日内 → 稳定 episode id, 可跨 run 聚合
    out2 = run_research_program(
        goal="harness m2 integration",
        experiments=[Experiment("e0", "baseline", _run(1.0))],
        objectives_config={"score": "maximize"},
        max_iterations=2, min_iterations=1, client=None,
        harness_agent="huginn-lite", harness_machine="sandbox-1",
    )
    assert out2.harness["task_episode"] == out.harness["task_episode"]