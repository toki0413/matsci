"""Tests for blue-team defensive review (红/蓝评审对偶, 与 test_red_team 对称).

Locks behaviour:
- BlueTeamFinding / BlueTeamReport dataclass semantics
- BlueTeamReviewer rule-based defence checks: provenance / redundancy / guard integrity
- ReviewerFn interface: __call__ returns (approved, reason)
- PhaseGateHook integration: blue_reviewer_fn blocks on high-severity findings
- 红蓝独立: 红队通过但蓝队拒绝时仍 rejected (对偶不互抵)
- Engine integration: blue-team wired into engine alongside red-team
- Non-enabled transitions pass through without critique
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.blue_team import (
    BlueTeamFinding,
    BlueTeamReport,
    BlueTeamReviewer,
)
from huginn.autoloop.phase_gate import (
    PhaseGateHook,
    get_shared_phase_gate_state,
)

# ── dataclass tests ──────────────────────────────────────────────────────────


class TestBlueTeamFinding:
    def test_to_dict_roundtrip(self):
        f = BlueTeamFinding(
            category="provenance_gap",
            description="external content uncorroborated",
            severity="high",
            mitigation="add independent verification",
        )
        d = f.to_dict()
        assert d["category"] == "provenance_gap"
        assert d["severity"] == "high"
        assert d["mitigation"] == "add independent verification"


class TestBlueTeamReport:
    def test_has_blocking_with_high_severity(self):
        report = BlueTeamReport(
            transition=("validate", "learn"),
            findings=[
                BlueTeamFinding("no_redundancy", "x", "medium"),
                BlueTeamFinding("provenance_gap", "y", "high"),
            ],
        )
        assert report.has_blocking is True
        assert report.n_findings == 2

    def test_has_blocking_without_high(self):
        report = BlueTeamReport(
            transition=("validate", "learn"),
            findings=[BlueTeamFinding("no_redundancy", "x", "medium")],
        )
        assert report.has_blocking is False

    def test_empty_report_not_blocking(self):
        report = BlueTeamReport(transition=("plan", "execute"))
        assert report.has_blocking is False
        assert report.n_findings == 0

    def test_to_dict(self):
        report = BlueTeamReport(
            transition=("validate", "learn"),
            findings=[BlueTeamFinding("provenance_gap", "x", "high")],
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
        reviewer = BlueTeamReviewer()
        approved, reason = reviewer("plan", "execute", {"mode": "workflow"})
        assert approved is True
        assert reason == ""

    def test_hypothesize_to_plan_enabled(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review("hypothesize", "plan", {"hypothesis": "x"})
        assert report.transition == ("hypothesize", "plan")

    def test_validate_to_learn_enabled(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": True})
        assert report.transition == ("validate", "learn")

    def test_custom_enabled_transitions(self):
        reviewer = BlueTeamReviewer(enabled_transitions={("plan", "execute")})
        approved, _ = reviewer("validate", "learn", {"tests_passed": True})
        assert approved is True
        report = reviewer.review("plan", "execute", {"mode": "workflow"})
        assert report.transition == ("plan", "execute")


# ── defence check: provenance ────────────────────────────────────────────────


class TestProvenanceCheck:
    def test_external_content_without_corroboration_blocks(self):
        """外部内容被当既定事实推进 → 来源信任控制失效 (高, 阻断)."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "doc": {"source_class": "external_content"}},
        )
        assert report.has_blocking is True
        assert any(
            f.category == "provenance_gap" and f.severity == "high"
            for f in report.findings
        )

    def test_external_content_with_user_input_corroboration_not_blocking(self):
        """有 user_input 佐证 → 控制未失效 (降为 low, 不阻断)."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {
                "tests_passed": True,
                "doc": {"source_class": "external_content"},
                "user": {"source_class": "user_input"},
            },
        )
        assert report.has_blocking is False
        cats = [f.category for f in report.findings]
        assert "provenance_gap" in cats

    def test_external_content_with_tool_output_corroboration_not_blocking(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {
                "tests_passed": True,
                "doc": {"source_class": "external_content"},
                "calc": {"source_class": "tool_output"},
            },
        )
        assert report.has_blocking is False

    def test_no_external_source_no_provenance_finding(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "user": {"source_class": "user_input"}},
        )
        assert not any(f.category == "provenance_gap" for f in report.findings)

    def test_nested_external_content_detected(self):
        """深嵌套里的 external_content 也要被抓到 (递归扫描)."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {
                "tests_passed": True,
                "chain": [{"inner": {"source_class": "external_content"}}],
            },
        )
        assert report.has_blocking is True


# ── defence check: redundancy ────────────────────────────────────────────────


class TestRedundancyCheck:
    def test_tests_passed_without_independent_control_medium(self):
        """声称阳性但无独立校验 → 无冗余 (中, 不阻断)."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": True})
        assert report.has_blocking is False
        assert any(
            f.category == "no_redundancy" and f.severity == "medium"
            for f in report.findings
        )

    def test_independent_marker_suppresses_finding(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "note": "交叉验证通过"},
        )
        assert not any(f.category == "no_redundancy" for f in report.findings)

    def test_baseline_marker_suppresses_finding(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "note": "baseline reproduced"},
        )
        assert not any(f.category == "no_redundancy" for f in report.findings)

    def test_tests_not_passed_no_redundancy_finding(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review("validate", "learn", {"tests_passed": False})
        assert not any(f.category == "no_redundancy" for f in report.findings)


# ── defence check: guard integrity ───────────────────────────────────────────


class TestGuardIntegrityCheck:
    def test_low_confidence_visual_guard_medium(self):
        """守卫自报低置信但流程仍推进 → 守卫失效被忽略 (中)."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "_visual_self_check": {"confidence": 0.1}},
        )
        assert report.has_blocking is False
        assert any(f.category == "guard_integrity" for f in report.findings)

    def test_high_confidence_visual_guard_clean(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {"tests_passed": True, "_visual_self_check": {"confidence": 0.9}},
        )
        assert not any(f.category == "guard_integrity" for f in report.findings)

    def test_physics_audit_warnings_medium(self):
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {
                "tests_passed": True,
                "physics_audit": {"findings": [{"severity": "warning"}]},
            },
        )
        assert any(f.category == "guard_integrity" for f in report.findings)

    def test_physics_audit_errors_not_guard_integrity(self):
        """error 由 phase_gate 物理 oracle 处理, 蓝队只盯被忽略的 warning."""
        reviewer = BlueTeamReviewer()
        report = reviewer.review(
            "validate", "learn",
            {
                "tests_passed": True,
                "physics_audit": {"findings": [{"severity": "error"}]},
            },
        )
        assert not any(f.category == "guard_integrity" for f in report.findings)


# ── ReviewerFn interface ─────────────────────────────────────────────────────


class TestReviewerFnInterface:
    def test_call_returns_tuple(self):
        reviewer = BlueTeamReviewer()
        result = reviewer("validate", "learn", {"tests_passed": True})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_call_blocks_on_provenance_gap(self):
        reviewer = BlueTeamReviewer()
        approved, reason = reviewer(
            "validate", "learn",
            {"tests_passed": True, "doc": {"source_class": "external_content"}},
        )
        assert approved is False
        assert "Blue-team" in reason

    def test_call_passes_on_clean_evidence(self):
        reviewer = BlueTeamReviewer()
        approved, reason = reviewer("validate", "learn", {"tests_passed": True})
        assert approved is True
        assert reason == ""

    def test_last_report_stored(self):
        reviewer = BlueTeamReviewer()
        reviewer("validate", "learn", {"tests_passed": True})
        assert reviewer._last_report is not None
        assert reviewer._last_report.transition == ("validate", "learn")


# ── PhaseGateHook integration ────────────────────────────────────────────────


class TestPhaseGateIntegration:
    def test_hook_with_blue_team_blocks_on_provenance_gap(self):
        reviewer = BlueTeamReviewer()
        hook = PhaseGateHook(blue_reviewer_fn=reviewer)
        gate = hook.evaluate(
            "validate", "learn",
            {"tests_passed": True, "doc": {"source_class": "external_content"}},
        )
        assert gate.is_blocked
        assert gate.status == "rejected"
        assert gate.reviewer == "blue_reviewer"
        assert "Blue-team" in gate.feedback

    def test_hook_with_blue_team_passes_on_clean_evidence(self):
        reviewer = BlueTeamReviewer()
        hook = PhaseGateHook(blue_reviewer_fn=reviewer)
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_hook_without_blue_reviewer_passes(self):
        hook = PhaseGateHook()
        gate = hook.evaluate("validate", "learn", {"tests_passed": True})
        assert gate.status == "approved"

    def test_red_and_blue_independent_both_must_pass(self):
        """红蓝对偶不互抵: 红队通过 (tests_passed=True) 但蓝队拒绝 → rejected."""
        from huginn.autoloop.red_team import RedTeamReviewer

        red = RedTeamReviewer()
        blue = BlueTeamReviewer()
        hook = PhaseGateHook(reviewer_fn=red, blue_reviewer_fn=blue)
        gate = hook.evaluate(
            "validate", "learn",
            {"tests_passed": True, "doc": {"source_class": "external_content"}},
        )
        assert gate.status == "rejected"
        assert gate.reviewer == "blue_reviewer"


# ── engine integration ───────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from huginn.autoloop.engine import AutoloopEngine

    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    eng = AutoloopEngine(workspace=tmp_path)

    class _DummyTracker:
        def start_task(self, *a, **kw): ...
        def update(self, *a, **kw): ...
        def complete(self, *a, **kw): ...
        def fail(self, *a, **kw): ...

    eng.progress_tracker = _DummyTracker()
    return eng


class TestEngineIntegration:
    def test_engine_has_blue_team_reviewer(self, engine):
        """Engine should wire BlueTeamReviewer into phase_gate_hook."""
        assert engine.phase_gate_hook._blue_reviewer_fn is not None
        assert isinstance(engine.phase_gate_hook._blue_reviewer_fn, BlueTeamReviewer)

    def test_engine_gate_passes_on_clean_evidence(self, engine):
        get_shared_phase_gate_state().reset()
        ok = engine._check_gate("validate", "learn", {"tests_passed": True})
        assert ok is True

    def test_engine_gate_rejects_on_provenance_gap(self, engine):
        """外部内容无佐证 → 蓝队拒绝, 即使红队与硬性证据都过."""
        get_shared_phase_gate_state().reset()
        ok = engine._check_gate(
            "validate", "learn",
            {"tests_passed": True, "doc": {"source_class": "external_content"}},
        )
        assert ok is False
