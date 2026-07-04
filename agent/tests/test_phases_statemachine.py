"""Tests for huginn.phases — the research phase state machine.

Covers the ResearchPhase enum, PHASE_TRANSITIONS adjacency map,
PHASE_PROMPTS content, and the PhaseManager class (transitions,
history, serialization, tool filters, reset, and edge cases).

Run with:
    pytest tests/test_phases_statemachine.py --override-ini="addopts="
"""

from __future__ import annotations

import pytest

from huginn.phases import (
    PHASE_PROMPTS,
    PHASE_TRANSITIONS,
    PHASE_TOOLS,
    PhaseManager,
    ResearchPhase,
)


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def mgr() -> PhaseManager:
    """Fresh manager, starts in the default OPEN phase."""
    return PhaseManager()


@pytest.fixture()
def lit_mgr() -> PhaseManager:
    """Manager seeded at LITERATURE for transition-graph tests."""
    return PhaseManager(initial=ResearchPhase.LITERATURE)


# ── 1. default phase ─────────────────────────────────────────


def test_default_phase_is_open(mgr: PhaseManager) -> None:
    # no args -> should land in OPEN (backwards-compat free-form mode)
    assert mgr.phase is ResearchPhase.OPEN


def test_explicit_initial_phase() -> None:
    # passing an explicit phase should honour it
    pm = PhaseManager(initial=ResearchPhase.EXECUTION)
    assert pm.phase is ResearchPhase.EXECUTION


# ── 2. can_transition: allowed vs blocked ────────────────────


def test_can_transition_literature_to_hypothesis(lit_mgr: PhaseManager) -> None:
    # forward edge in the happy-path workflow
    assert lit_mgr.can_transition(ResearchPhase.HYPOTHESIS) is True


def test_can_transition_literature_to_execution_blocked(lit_mgr: PhaseManager) -> None:
    # can't skip straight to execution — must go through planning first
    assert lit_mgr.can_transition(ResearchPhase.EXECUTION) is False


def test_can_transition_to_open_always_allowed() -> None:
    # every phase can bail out to OPEN
    for src in ResearchPhase:
        pm = PhaseManager(initial=src)
        assert pm.can_transition(ResearchPhase.OPEN) is True, (
            f"OPEN should be reachable from {src}"
        )


# ── 3. transition() return value ─────────────────────────────


def test_transition_allowed_returns_true(lit_mgr: PhaseManager) -> None:
    ok = lit_mgr.transition(ResearchPhase.HYPOTHESIS)
    assert ok is True
    assert lit_mgr.phase is ResearchPhase.HYPOTHESIS


def test_transition_disallowed_returns_false(lit_mgr: PhaseManager) -> None:
    # LITERATURE -> EXECUTION is not a direct edge
    ok = lit_mgr.transition(ResearchPhase.EXECUTION)
    assert ok is False
    # phase should stay put on a rejected transition
    assert lit_mgr.phase is ResearchPhase.LITERATURE


# ── 4. history tracking ──────────────────────────────────────


def test_history_starts_with_initial_phase() -> None:
    pm = PhaseManager(initial=ResearchPhase.PLANNING)
    assert pm.history == [ResearchPhase.PLANNING]


def test_history_records_each_transition() -> None:
    pm = PhaseManager(initial=ResearchPhase.LITERATURE)
    pm.transition(ResearchPhase.HYPOTHESIS)
    pm.transition(ResearchPhase.PLANNING)
    pm.transition(ResearchPhase.EXECUTION)
    assert pm.history == [
        ResearchPhase.LITERATURE,
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.PLANNING,
        ResearchPhase.EXECUTION,
    ]


def test_history_is_a_copy(mgr: PhaseManager) -> None:
    # mutating the returned list must not corrupt internal state
    snapshot = mgr.history
    snapshot.append(ResearchPhase.REPORTING)
    assert ResearchPhase.REPORTING not in mgr.history


def test_history_unchanged_on_failed_transition(lit_mgr: PhaseManager) -> None:
    before = lit_mgr.history
    lit_mgr.transition(ResearchPhase.REPORTING)  # not allowed from LITERATURE
    assert lit_mgr.history == before


# ── 5. prompt_prefix: non-empty for real phases ──────────────


@pytest.mark.parametrize(
    "phase",
    [
        ResearchPhase.LITERATURE,
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.PLANNING,
        ResearchPhase.EXECUTION,
        ResearchPhase.VALIDATION,
        ResearchPhase.REPORTING,
    ],
)
def test_prompt_prefix_non_empty_for_real_phases(phase: ResearchPhase) -> None:
    pm = PhaseManager(initial=phase)
    assert pm.prompt_prefix() != ""
    assert isinstance(pm.prompt_prefix(), str)


def test_prompt_prefix_empty_for_open(mgr: PhaseManager) -> None:
    # OPEN is free-form — no prefix prepended to the system prompt
    assert mgr.prompt_prefix() == ""


def test_phase_prompts_dict_has_entry_for_every_phase() -> None:
    # sanity: no phase should be missing from the prompts table
    for phase in ResearchPhase:
        assert phase in PHASE_PROMPTS, f"missing prompt for {phase}"


# ── 6. prompt_prefix: phase-specific keywords ────────────────


@pytest.mark.parametrize(
    "phase, keyword",
    [
        (ResearchPhase.LITERATURE, "Literature"),
        (ResearchPhase.HYPOTHESIS, "Hypothesis"),
        (ResearchPhase.PLANNING, "Planning"),
        (ResearchPhase.EXECUTION, "Execution"),
        (ResearchPhase.VALIDATION, "Validation"),
        (ResearchPhase.REPORTING, "Reporting"),
    ],
)
def test_prompt_prefix_contains_phase_keyword(
    phase: ResearchPhase, keyword: str
) -> None:
    pm = PhaseManager(initial=phase)
    assert keyword in pm.prompt_prefix(), (
        f"prompt for {phase} should mention '{keyword}'"
    )


# ── 7. tool_filter: None for OPEN, set otherwise ─────────────


def test_tool_filter_none_for_open(mgr: PhaseManager) -> None:
    # OPEN exposes every registered tool — represented as None
    assert mgr.tool_filter() is None


def test_tool_filter_none_when_phase_tools_unpopulated(monkeypatch) -> None:
    # before register_all_tools() runs, PHASE_TOOLS is empty so every
    # phase gracefully degrades to "all tools available"
    # Clear PHASE_TOOLS to simulate unpopulated state
    monkeypatch.setattr("huginn.phases.PHASE_TOOLS", {})
    pm = PhaseManager(initial=ResearchPhase.EXECUTION)
    assert pm.tool_filter() is None


def test_tool_filter_returns_set_when_populated(monkeypatch) -> None:
    # simulate post-registration state: non-OPEN phases map to a tool set
    fake_tools = {"file_read_tool", "bash_tool", "vasp_tool"}
    populated = {phase: set(fake_tools) for phase in ResearchPhase}
    populated[ResearchPhase.OPEN] = None
    monkeypatch.setattr("huginn.phases.PHASE_TOOLS", populated)

    pm = PhaseManager(initial=ResearchPhase.EXECUTION)
    result = pm.tool_filter()
    assert isinstance(result, set)
    assert "vasp_tool" in result


def test_tool_filter_open_always_none_even_when_populated(monkeypatch) -> None:
    # even after registration OPEN stays None
    populated = {phase: {"x"} for phase in ResearchPhase}
    populated[ResearchPhase.OPEN] = None
    monkeypatch.setattr("huginn.phases.PHASE_TOOLS", populated)

    pm = PhaseManager(initial=ResearchPhase.OPEN)
    assert pm.tool_filter() is None


# ── 8. reset ─────────────────────────────────────────────────


def test_reset_clears_history(mgr: PhaseManager) -> None:
    # build up some history first
    mgr.transition(ResearchPhase.LITERATURE)
    mgr.transition(ResearchPhase.HYPOTHESIS)
    assert len(mgr.history) > 1

    mgr.reset()
    assert mgr.phase is ResearchPhase.OPEN
    assert mgr.history == [ResearchPhase.OPEN]


def test_reset_to_specific_phase() -> None:
    pm = PhaseManager(initial=ResearchPhase.OPEN)
    pm.transition(ResearchPhase.LITERATURE)
    pm.transition(ResearchPhase.HYPOTHESIS)

    pm.reset(phase=ResearchPhase.PLANNING)
    assert pm.phase is ResearchPhase.PLANNING
    # history should only contain the new starting phase
    assert pm.history == [ResearchPhase.PLANNING]


def test_reset_default_is_open() -> None:
    pm = PhaseManager(initial=ResearchPhase.REPORTING)
    pm.reset()
    assert pm.phase is ResearchPhase.OPEN


# ── 9. serialization roundtrip ────────────────────────────────


def test_to_dict_structure(mgr: PhaseManager) -> None:
    data = mgr.to_dict()
    assert data["phase"] == "open"
    assert data["history"] == ["open"]


def test_from_dict_preserves_phase_and_history() -> None:
    original = PhaseManager(initial=ResearchPhase.LITERATURE)
    original.transition(ResearchPhase.HYPOTHESIS)
    original.transition(ResearchPhase.PLANNING)

    data = original.to_dict()
    restored = PhaseManager.from_dict(data)

    assert restored.phase is ResearchPhase.PLANNING
    assert restored.history == [
        ResearchPhase.LITERATURE,
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.PLANNING,
    ]


def test_roundtrip_stays_in_sync_after_more_transitions() -> None:
    # serialize, restore, keep going — state should be consistent
    pm = PhaseManager(initial=ResearchPhase.LITERATURE)
    pm.transition(ResearchPhase.HYPOTHESIS)
    restored = PhaseManager.from_dict(pm.to_dict())

    assert restored.transition(ResearchPhase.PLANNING) is True
    assert restored.phase is ResearchPhase.PLANNING


def test_from_dict_with_missing_history_defaults_to_phase() -> None:
    # only phase given, no history list — should still construct
    restored = PhaseManager.from_dict({"phase": "execution"})
    assert restored.phase is ResearchPhase.EXECUTION
    assert restored.history == [ResearchPhase.EXECUTION]


def test_from_dict_empty_data_defaults_to_open() -> None:
    # completely empty dict -> OPEN (the safe default)
    restored = PhaseManager.from_dict({})
    assert restored.phase is ResearchPhase.OPEN


# ── 10. full happy-path research cycle ───────────────────────


def test_full_research_cycle() -> None:
    # LITERATURE -> HYPOTHESIS -> PLANNING -> EXECUTION -> VALIDATION -> REPORTING
    pm = PhaseManager(initial=ResearchPhase.LITERATURE)
    steps = [
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.PLANNING,
        ResearchPhase.EXECUTION,
        ResearchPhase.VALIDATION,
        ResearchPhase.REPORTING,
    ]
    for target in steps:
        assert pm.transition(target) is True, (
            f"transition to {target} should succeed"
        )

    assert pm.phase is ResearchPhase.REPORTING
    assert pm.history == [
        ResearchPhase.LITERATURE,
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.PLANNING,
        ResearchPhase.EXECUTION,
        ResearchPhase.VALIDATION,
        ResearchPhase.REPORTING,
    ]


def test_reporting_can_loop_back_to_literature() -> None:
    # after reporting, a new cycle starts back at literature
    pm = PhaseManager(initial=ResearchPhase.REPORTING)
    assert pm.transition(ResearchPhase.LITERATURE) is True
    assert pm.phase is ResearchPhase.LITERATURE


# ── 11. backtracking edges ────────────────────────────────────


def test_backtrack_validation_to_planning() -> None:
    # results unreliable -> replan
    pm = PhaseManager(initial=ResearchPhase.VALIDATION)
    assert pm.transition(ResearchPhase.PLANNING) is True
    assert pm.phase is ResearchPhase.PLANNING


def test_backtrack_execution_to_planning() -> None:
    # something broke during the run -> back to planning
    pm = PhaseManager(initial=ResearchPhase.EXECUTION)
    assert pm.transition(ResearchPhase.PLANNING) is True


def test_backtrack_hypothesis_to_literature() -> None:
    # need more background -> return to lit review
    pm = PhaseManager(initial=ResearchPhase.HYPOTHESIS)
    assert pm.transition(ResearchPhase.LITERATURE) is True


def test_backtrack_validation_to_execution() -> None:
    # rerun with tweaked params
    pm = PhaseManager(initial=ResearchPhase.VALIDATION)
    assert pm.transition(ResearchPhase.EXECUTION) is True


def test_backtrack_planning_to_hypothesis() -> None:
    # refine the hypothesis before re-planning
    pm = PhaseManager(initial=ResearchPhase.PLANNING)
    assert pm.transition(ResearchPhase.HYPOTHESIS) is True


def test_backtrack_reporting_to_validation() -> None:
    # re-analyze before finalizing the report
    pm = PhaseManager(initial=ResearchPhase.REPORTING)
    assert pm.transition(ResearchPhase.VALIDATION) is True


# ── 12. invalid transitions don't raise ───────────────────────


def test_invalid_transition_no_exception(lit_mgr: PhaseManager) -> None:
    # should just return False, never throw
    result = lit_mgr.transition(ResearchPhase.REPORTING)
    assert result is False


def test_invalid_transition_leaves_state_unchanged() -> None:
    pm = PhaseManager(initial=ResearchPhase.PLANNING)
    pm.transition(ResearchPhase.LITERATURE)  # not allowed
    assert pm.phase is ResearchPhase.PLANNING
    assert pm.history == [ResearchPhase.PLANNING]


@pytest.mark.parametrize(
    "src, dst",
    [
        (ResearchPhase.LITERATURE, ResearchPhase.PLANNING),
        (ResearchPhase.LITERATURE, ResearchPhase.EXECUTION),
        (ResearchPhase.LITERATURE, ResearchPhase.VALIDATION),
        (ResearchPhase.LITERATURE, ResearchPhase.REPORTING),
        (ResearchPhase.HYPOTHESIS, ResearchPhase.EXECUTION),
        (ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION),
        (ResearchPhase.HYPOTHESIS, ResearchPhase.REPORTING),
        (ResearchPhase.PLANNING, ResearchPhase.LITERATURE),
        (ResearchPhase.PLANNING, ResearchPhase.VALIDATION),
        (ResearchPhase.PLANNING, ResearchPhase.REPORTING),
        (ResearchPhase.EXECUTION, ResearchPhase.LITERATURE),
        (ResearchPhase.EXECUTION, ResearchPhase.HYPOTHESIS),
        (ResearchPhase.EXECUTION, ResearchPhase.REPORTING),
        (ResearchPhase.VALIDATION, ResearchPhase.LITERATURE),
        (ResearchPhase.VALIDATION, ResearchPhase.HYPOTHESIS),
        (ResearchPhase.REPORTING, ResearchPhase.HYPOTHESIS),
        (ResearchPhase.REPORTING, ResearchPhase.PLANNING),
        (ResearchPhase.REPORTING, ResearchPhase.EXECUTION),
    ],
)
def test_disallowed_edges_return_false(src: ResearchPhase, dst: ResearchPhase) -> None:
    pm = PhaseManager(initial=src)
    assert pm.transition(dst) is False
    assert pm.phase is src


# ── 13. OPEN can reach any phase ─────────────────────────────


@pytest.mark.parametrize("target", list(ResearchPhase))
def test_open_can_transition_anywhere(target: ResearchPhase) -> None:
    pm = PhaseManager(initial=ResearchPhase.OPEN)
    assert pm.can_transition(target) is True, (
        f"OPEN should reach {target}"
    )


@pytest.mark.parametrize("target", list(ResearchPhase))
def test_open_transition_succeeds(target: ResearchPhase) -> None:
    pm = PhaseManager(initial=ResearchPhase.OPEN)
    if target is ResearchPhase.OPEN:
        # same-phase — covered separately, but still valid
        assert pm.transition(target) is True
    else:
        assert pm.transition(target) is True
        assert pm.phase is target


# ── 14. same-phase transition is a no-op ─────────────────────


@pytest.mark.parametrize("phase", list(ResearchPhase))
def test_same_phase_transition_returns_true(phase: ResearchPhase) -> None:
    pm = PhaseManager(initial=phase)
    assert pm.can_transition(phase) is True
    assert pm.transition(phase) is True


@pytest.mark.parametrize("phase", list(ResearchPhase))
def test_same_phase_transition_does_not_grow_history(phase: ResearchPhase) -> None:
    pm = PhaseManager(initial=phase)
    before = len(pm.history)
    pm.transition(phase)  # no-op
    assert len(pm.history) == before


def test_same_phase_transition_keeps_phase(mgr: PhaseManager) -> None:
    assert mgr.transition(ResearchPhase.OPEN) is True
    assert mgr.phase is ResearchPhase.OPEN
    # history should still be just [OPEN]
    assert mgr.history == [ResearchPhase.OPEN]


# ── PHASE_TRANSITIONS table sanity ────────────────────────────


def test_phase_transitions_covers_every_phase() -> None:
    # every phase must have an entry in the adjacency map
    for phase in ResearchPhase:
        assert phase in PHASE_TRANSITIONS, f"missing transitions for {phase}"


def test_open_transitions_include_all_phases() -> None:
    allowed = PHASE_TRANSITIONS[ResearchPhase.OPEN]
    assert allowed == set(ResearchPhase)


def test_phase_transitions_are_asymmetric() -> None:
    # just spot-check that the graph isn't fully connected (i.e. real
    # constraints exist): LITERATURE can't directly reach EXECUTION
    assert ResearchPhase.EXECUTION not in PHASE_TRANSITIONS[ResearchPhase.LITERATURE]
    # but the reverse isn't true either: EXECUTION can't go to LITERATURE
    assert ResearchPhase.LITERATURE not in PHASE_TRANSITIONS[ResearchPhase.EXECUTION]


def test_phase_transitions_open_in_each_row() -> None:
    # every phase should be able to escape to OPEN
    for src, targets in PHASE_TRANSITIONS.items():
        if src is ResearchPhase.OPEN:
            continue
        assert ResearchPhase.OPEN in targets, f"{src} can't reach OPEN"


# ── extra: PhaseManager independence ──────────────────────────


def test_multiple_managers_are_independent() -> None:
    a = PhaseManager(initial=ResearchPhase.LITERATURE)
    b = PhaseManager(initial=ResearchPhase.OPEN)

    a.transition(ResearchPhase.HYPOTHESIS)
    # b should be untouched
    assert b.phase is ResearchPhase.OPEN
    assert b.history == [ResearchPhase.OPEN]


def test_phase_property_is_read_only() -> None:
    pm = PhaseManager(initial=ResearchPhase.OPEN)
    with pytest.raises(AttributeError):
        pm.phase = ResearchPhase.LITERATURE  # type: ignore[misc]


def test_history_property_is_read_only() -> None:
    pm = PhaseManager(initial=ResearchPhase.OPEN)
    with pytest.raises(AttributeError):
        pm.history = [ResearchPhase.LITERATURE]  # type: ignore[misc]


# ── extra: enum behaviour ─────────────────────────────────────


def test_research_phase_is_str_enum() -> None:
    # values should be lowercase strings (used in serialization)
    assert ResearchPhase.LITERATURE.value == "literature"
    assert ResearchPhase.OPEN.value == "open"


def test_phase_lookup_by_string_value() -> None:
    # from_dict relies on this — round-trip through string values
    assert ResearchPhase("planning") is ResearchPhase.PLANNING
    assert ResearchPhase("reporting") is ResearchPhase.REPORTING
