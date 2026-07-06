"""Tests for plan-aware context injection and cross-session memory continuity.

Covers the new ContextBuilder methods (build_plan_text / build_session_continuity)
and the MemoryManager cross-session helpers (load_last_session_context /
store_plan_progress / load_active_plan).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


# ── ContextBuilder: plan + session continuity ────────────────────────


class TestBuildPlanText:
    """Verify build_plan_text injects the active plan when present."""

    def test_build_plan_text_no_state(self):
        """No session_state at all should yield an empty string."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        assert cb.build_plan_text(None) == ""

    def test_build_plan_text_no_plan(self):
        """session_state without an active_plan_id should yield empty string."""
        from huginn.context_builder import ContextBuilder

        state = SimpleNamespace(
            active_plan_id=None,
            active_plan_objective="",
            active_plan_step_index=0,
            l1_coordinates="",
            cognitive_mode=SimpleNamespace(value="focused"),
        )
        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        assert cb.build_plan_text(state) == ""

    def test_build_plan_text_with_plan(self):
        """An active plan should surface objective, step number, and mode."""
        from huginn.context_builder import ContextBuilder

        state = SimpleNamespace(
            active_plan_id="plan-42",
            active_plan_objective="GaN band structure calculation",
            active_plan_step_index=1,  # zero-based -> displayed as step 2
            l1_coordinates="3,2,1",
            cognitive_mode=SimpleNamespace(value="executing"),
        )
        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        result = cb.build_plan_text(state)

        assert "GaN band structure calculation" in result
        assert "Step: 2" in result  # 0-based index + 1
        assert "executing" in result
        assert "3,2,1" in result
        assert "### Current Plan" in result


class TestBuildSessionContinuity:
    """Verify build_session_continuity surfaces prior-session context."""

    def test_build_session_continuity_no_state(self):
        """No session_state should yield an empty string."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        assert cb.build_session_continuity(None) == ""

    def test_build_session_continuity_with_summary(self):
        """A last_session_summary should appear in the output."""
        from huginn.context_builder import ContextBuilder

        state = SimpleNamespace(
            last_session_summary="Ran VASP relaxation on Si 8-atom cell.",
            user_goals_history=[],
        )
        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        result = cb.build_session_continuity(state)

        assert "Previous Session" in result
        assert "VASP relaxation" in result

    def test_build_session_continuity_with_goals(self):
        """user_goals_history should be listed (capped at the last 5)."""
        from huginn.context_builder import ContextBuilder

        state = SimpleNamespace(
            last_session_summary="",
            user_goals_history=[
                "oldest goal that should drop off",  # index 0 — beyond the -5 window
                "optimize lattice params",
                "converge k-mesh",
                "run phonon spectrum",
                "compute dielectric tensor",
                "find band gap of GaAs",  # newest — kept
            ],
        )
        cb = ContextBuilder(memory_manager=MagicMock(), workspace="/tmp")
        result = cb.build_session_continuity(state)

        assert "Recent Goals" in result
        # The newest 5 goals are injected
        assert "find band gap of GaAs" in result
        assert "optimize lattice params" in result
        assert "compute dielectric tensor" in result
        # The oldest (index 0) is dropped by the [-5:] slice
        assert "oldest goal that should drop off" not in result


# ── MemoryManager: cross-session continuity + plan progress ──────────


class TestLoadLastSessionContext:
    """Verify load_last_session_context retrieves the prior session summary."""

    def test_load_last_session_context_empty(self):
        """No prior entries should return an all-empty dict."""
        from huginn.memory.manager import MemoryManager

        longterm = MagicMock()
        longterm.retrieve.return_value = []
        mgr = MemoryManager(longterm=longterm)

        result = mgr.load_last_session_context()
        assert result["summary"] == ""
        assert result["session_id"] == ""
        assert result["l1_coordinates"] == ""

    def test_load_last_session_context_with_data(self):
        """A retrieved conversation entry should populate the summary."""
        from huginn.memory.manager import MemoryManager

        longterm = MagicMock()
        longterm.retrieve.return_value = [
            {
                "content": "Explored Fe magnetic ordering with DFT.",
                "source": "session:abc-123",
            }
        ]
        mgr = MemoryManager(longterm=longterm)

        result = mgr.load_last_session_context()
        assert "Fe magnetic ordering" in result["summary"]
        assert result["session_id"] == "session:abc-123"


class TestStorePlanProgress:
    """Verify store_plan_progress writes to long-term memory as category=plan."""

    def test_store_plan_progress(self):
        from huginn.memory.manager import MemoryManager

        longterm = MagicMock()
        longterm.store.return_value = "entry-001"
        mgr = MemoryManager(longterm=longterm)

        entry_id = mgr.store_plan_progress(
            plan_id="plan-7",
            objective="Relax GaN wurtzite supercell",
            step_index=2,
            status="in_progress",
            l1_coordinates="2,1,0",
        )

        assert entry_id == "entry-001"
        longterm.store.assert_called_once()
        kwargs = longterm.store.call_args.kwargs
        assert kwargs["category"] == "plan"
        assert "plan_progress" in kwargs["tags"]
        assert "plan-7" in kwargs["tags"]
        assert kwargs["source"] == "plan:plan-7"
        # The serialized content carries the objective + step + status + position
        content = kwargs["content"]
        assert "Relax GaN wurtzite supercell" in content
        assert "2" in content
        assert "in_progress" in content
        assert "2,1,0" in content


class TestLoadActivePlan:
    """Verify load_active_plan parses the stored plan-progress string back out."""

    def test_load_active_plan(self):
        from huginn.memory.manager import MemoryManager

        longterm = MagicMock()
        longterm.retrieve.return_value = [
            {
                "content": "Plan: converge ENCUT for Si | Step: 3 | Status: in_progress | Position: 1,0,0",
                "source": "plan:plan-9",
            }
        ]
        mgr = MemoryManager(longterm=longterm)

        plan = mgr.load_active_plan()
        assert plan is not None
        assert plan["objective"] == "converge ENCUT for Si"
        assert plan["step_index"] == 3
        assert plan["status"] == "in_progress"
        assert "converge ENCUT" in plan["content"]

    def test_load_active_plan_none_when_empty(self):
        """No stored plan should return None."""
        from huginn.memory.manager import MemoryManager

        longterm = MagicMock()
        longterm.retrieve.return_value = []
        mgr = MemoryManager(longterm=longterm)

        assert mgr.load_active_plan() is None
