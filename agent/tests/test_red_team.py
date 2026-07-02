"""Tests for red-team adversarial review (W3 R3).

Locks behaviour:
- RedTeamFinding / RedTeamReport dataclass semantics
- RedTeamReviewer rule-based critique: hypothesis + validation transitions
- ReviewerFn interface: __call__ returns (approved, reason)
- PhaseGateHook integration: reviewer_fn blocks on high-severity findings
- Engine integration: red-team wired into engine doesn't break happy path
- Non-enabled transitions pass through without critique
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateConfig,
    PhaseGateHook,
    get_shared_phase_gate_state,
)
from huginn.autoloop.red_team import (
    RedTeamFinding,
    RedTeamReport,
    RedTeamReviewer,
)


# ── dataclass tests ──────────────────────────────────────────────────────────


class TestRedTeamFinding:
    def test_to_dict_roundtrip(self):
        f = RedTeamFinding(
            category="confounder",
            description="missing control",
            severity="medium",
            mitigation="add control group",
        )
        d = f.to_dict()
        assert d["category"] == "confounder"
        assert d["severity"] == "medium"
        assert d["mitigation"] == "add control group"


class TestRedTeamReport:
    def test_has_blocking_with_high_severity(self):
        report = RedTeamReport(
            transition=("validate", "learn"),
            findings=[
                RedTeamFinding("methodology_gap", "fail", "low"),
                RedTeamFinding("methodology_gap", "critical", "high"),
            ],
        )
        assert report.has_blocking is True
        assert report.n_findings == 2

    def test_has_blocking_without_high(self):
        report = RedTeamReport(
            transition=("hypothesize", "plan"),
            findings=[RedTeamFinding("confounder", "x", "medium")],
        )
        assert report.has_blocking is False

    def test_empty_report_not_blocking(self):
        report = RedTeamReport(transition=("plan", "execute"))
        assert report.has_blocking is False
        assert report.n_findings == 0

    def test_to_dict(self):
        report = RedTeamReport(
            transition=("validate", "learn"),
            findings=[RedTeamFinding("confounder", "x", "high")],
            summary="test summary",
        )
        d = report.to_dict()
        assert d["transition"] == ["validate", "learn"]
        assert d["has_blocking"] is True
        assert d["n_findings"] == 1
        assert d["summary"] == "test summary"


# ── reviewer: non-enabled transitions ────────────────────────────────────────


class TestReviewerTransitions:
    def test_non_enabled_transition_passes(self):
        reviewer = RedTeamReviewer()
        approved, reason = reviewer("plan", "execute", {"mode": "workflow"})
        assert approved is True
        assert reason == ""

    def test_hypothesize_to_plan_enabled(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("hypothesize", "plan", {"hypothesis": ""})
        assert report.transition == ("hypothesize", "plan")
        assert report.n_findings > 0  # empty hypothesis → findings

    def test_validate_to_learn_enabled(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": False})
        assert report.transition == ("validate", "learn")
        assert report.has_blocking is True

    def test_custom_enabled_transitions(self):
        reviewer = RedTeamReviewer(enabled_transitions={("plan", "execute")})
        # default transitions now disabled
        approved, _ = reviewer("validate", "learn", {"tests_passed": False})
        assert approved is True
        # custom transition enabled
        report = reviewer.review("plan", "execute", {"mode": "workflow"})
        assert report.transition == ("plan", "execute")


# ── reviewer: hypothesis critique ────────────────────────────────────────────


class TestHypothesisReview:
    def test_empty_hypothesis_high_severity(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("hypothesize", "plan", {"hypothesis": ""})
        assert report.has_blocking is True
        assert any(f.severity == "high" for f in report.findings)

    def test_missing_hypothesis_key(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("hypothesize", "plan", {"mode": "test"})
        assert report.has_blocking is True

    def test_well_formed_hypothesis_not_blocking(self):
        reviewer = RedTeamReviewer()
        # 有 if-then, 有前提, 有控制, 不太长 → 不阻断
        hyp = "如果温度升高, 则反应速率增加. 前提: 压力固定. 控制变量: 催化剂浓度"
        report = reviewer.review("hypothesize", "plan", {"hypothesis": hyp})
        assert not report.has_blocking

    def test_non_falsifiable_hypothesis_medium(self):
        reviewer = RedTeamReviewer()
        # 没有 if-then 结构, 短假设 (不触发 hidden_assumption)
        hyp = "材料性能很好"
        report = reviewer.review("hypothesize", "plan", {"hypothesis": hyp})
        assert not report.has_blocking  # medium 不阻断
        cats = [f.category for f in report.findings]
        assert "methodology_gap" in cats

    def test_long_hypothesis_without_assumptions(self):
        reviewer = RedTeamReviewer()
        # 长, 有 if-then, 但没提前提/条件
        hyp = "如果掺杂浓度增加, 那么带隙会减小, 因为额外的载流子会改变能带结构, 这是一个非常长的假设描述"
        report = reviewer.review("hypothesize", "plan", {"hypothesis": hyp})
        cats = [f.category for f in report.findings]
        assert "hidden_assumption" in cats
        assert not report.has_blocking  # medium, not high

    def test_no_control_variables_flagged(self):
        reviewer = RedTeamReviewer()
        hyp = "如果温度升高则反应加快, 前提: 压力恒定"
        report = reviewer.review("hypothesize", "plan", {"hypothesis": hyp})
        cats = [f.category for f in report.findings]
        assert "confounder" in cats


# ── reviewer: validation critique ────────────────────────────────────────────


class TestValidationReview:
    def test_tests_failed_high_severity(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": False})
        assert report.has_blocking is True
        assert any(f.category == "methodology_gap" and f.severity == "high" for f in report.findings)

    def test_tests_passed_no_blocking(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": True})
        assert not report.has_blocking

    def test_single_method_medium(self):
        reviewer = RedTeamReviewer()
        # tests_passed=True, mode 有但没提 cross/baseline
        report = reviewer.review("validate", "learn", {
            "tests_passed": True,
            "mode": "workflow",
        })
        assert not report.has_blocking
        cats = [f.category for f in report.findings]
        assert "alternative_explanation" in cats

    def test_cross_validation_no_finding(self):
        reviewer = RedTeamReviewer()
        report = reviewer.review("validate", "learn", {
            "tests_passed": True,
            "mode": "workflow",
            "note": "交叉验证通过",
        })
        cats = [f.category for f in report.findings]
        # 有"交叉"在 evidence 里, 不触发 alternative_explanation
        assert "alternative_explanation" not in cats


# ── ReviewerFn interface ─────────────────────────────────────────────────────


class TestReviewerFnInterface:
    def test_call_returns_tuple(self):
        reviewer = RedTeamReviewer()
        result = reviewer("validate", "learn", {"tests_passed": False})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_call_blocks_on_high_severity(self):
        reviewer = RedTeamReviewer()
        approved, reason = reviewer("validate", "learn", {"tests_passed": False})
        assert approved is False
        assert "Red-team" in reason

    def test_call_passes_on_no_blocking(self):
        reviewer = RedTeamReviewer()
        approved, reason = reviewer("validate", "learn", {"tests_passed": True})
        assert approved is True
        assert reason == ""

    def test_last_report_stored(self):
        reviewer = RedTeamReviewer()
        reviewer("validate", "learn", {"tests_passed": False})
        assert reviewer._last_report is not None
        assert reviewer._last_report.transition == ("validate", "learn")

    def test_mock_model_skipped(self):
        """MagicMock model should not trigger LLM path."""
        mock_model = MagicMock()
        reviewer = RedTeamReviewer(model=mock_model)
        assert reviewer._is_real_model() is False
        # review should work without calling model
        report = reviewer.review("validate", "learn", {"tests_passed": True})
        assert report.n_findings >= 0  # rule-based still works


# ── PhaseGateHook integration ────────────────────────────────────────────────


class TestPhaseGateIntegration:
    def test_hook_with_red_team_blocks_on_failure(self):
        reviewer = RedTeamReviewer()
        hook = PhaseGateHook(reviewer_fn=reviewer)
        # tests_passed=False → red-team blocks
        gate = hook.evaluate("validate", "learn", {"tests_passed": False})
        assert gate.is_blocked
        assert gate.status == "rejected"
        assert "Red-team" in gate.feedback

    def test_hook_with_red_team_passes_on_success(self):
        reviewer = RedTeamReviewer()
        hook = PhaseGateHook(reviewer_fn=reviewer)
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_hook_without_reviewer_passes(self):
        """Default hook (no reviewer) doesn't block on tests_passed=True."""
        hook = PhaseGateHook()
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_hook_missing_evidence_blocks_before_reviewer(self):
        """If required evidence is missing, hook blocks WITHOUT calling reviewer."""
        reviewer = RedTeamReviewer()
        hook = PhaseGateHook(reviewer_fn=reviewer)
        gate = hook.evaluate("validate", "learn", {})
        assert gate.is_blocked
        assert gate.status == "blocked"  # not "rejected"
        assert reviewer._last_report is None  # reviewer not called


# ── engine integration ───────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from huginn.autoloop.engine import AutoloopEngine

    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    eng = AutoloopEngine(workspace=tmp_path)

    class _DummyTracker:
        def start_task(self, *a, **kw): ...
        def update(self, *a, **kw): ...
        def complete(self, *a, **kw): ...
        def fail(self, *a, **kw): ...

    eng.progress_tracker = _DummyTracker()
    return eng


class TestEngineIntegration:
    def test_engine_has_red_team_reviewer(self, engine):
        """Engine should wire RedTeamReviewer into phase_gate_hook."""
        from huginn.autoloop.red_team import RedTeamReviewer

        assert engine.phase_gate_hook._reviewer_fn is not None
        assert isinstance(engine.phase_gate_hook._reviewer_fn, RedTeamReviewer)

    def test_engine_reviewer_uses_mock_model_safely(self, engine):
        """Engine's reviewer should detect MagicMock and skip LLM path."""
        reviewer = engine.phase_gate_hook._reviewer_fn
        assert reviewer._is_real_model() is False

    def test_engine_gate_passes_on_valid_evidence(self, engine):
        """validate→learn with tests_passed=True should pass (no high findings)."""
        get_shared_phase_gate_state().reset()
        ok = engine._check_gate("validate", "learn", {"tests_passed": True})
        assert ok is True

    def test_engine_gate_blocks_on_missing_evidence(self, engine):
        """validate→learn without tests_passed should block (hard check)."""
        get_shared_phase_gate_state().reset()
        ok = engine._check_gate("validate", "learn", {})
        assert ok is False

    def test_engine_gate_rejects_on_test_failure(self, engine):
        """validate→learn with tests_passed=False: hard check passes (key exists),
        but red-team rejects (high severity finding)."""
        get_shared_phase_gate_state().reset()
        ok = engine._check_gate("validate", "learn", {"tests_passed": False})
        assert ok is False
