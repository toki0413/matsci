"""Online Compact 机制测试: 候选驱动 + 经济/证据密封门控."""

from __future__ import annotations

from huginn.agent.streaming import _extract_plan_steps_for_compact
from huginn.autoloop.online_compact import (
    _fake_step,
    compact_preserved_summary_delta,
    gate_compaction,
    select_compact_candidates,
)


def test_candidates_only_verified_and_sealed() -> None:
    steps = [
        _fake_step("s1", status="completed", has_evidence=True),
        _fake_step("s2", status="completed", sealed=False),
        _fake_step("s3", status="in_progress"),
        _fake_step("s4", verified=True, sealed=True),
    ]
    ids = {c.step_id for c in select_compact_candidates(steps)}
    assert ids == {"s1", "s4"}


def test_gate_requires_economy_and_preserves_unverified() -> None:
    steps = [
        _fake_step("s1", status="completed", has_evidence=True),
        _fake_step("s2", status="in_progress"),
    ]
    ok = gate_compaction(steps, window_pct=70, cache_write_read_ratio=20)
    assert ok["should_compact"] is True
    assert ok["compact_candidates"] == ["s1"]
    assert "s2" in ok["preserve_evidence"]

    no_econ = gate_compaction(steps, window_pct=30, cache_write_read_ratio=1)
    assert no_econ["should_compact"] is False

    no_candidate = gate_compaction(
        [_fake_step("s9", status="in_progress")], window_pct=95, cache_write_read_ratio=99
    )
    assert no_candidate["should_compact"] is False


def test_delta_omits_active_steps() -> None:
    delta = compact_preserved_summary_delta(
        [_fake_step("s1", status="completed", has_evidence=True), _fake_step("s2", status="running")]
    )
    assert "s1" in delta and "s2" not in delta
    assert compact_preserved_summary_delta([_fake_step("s2", status="running")]) == ""


def test_extract_plan_steps_duck_typing() -> None:
    step = _fake_step("k1", status="completed")
    assert _extract_plan_steps_for_compact({"steps": [step]}) == [step]
    assert _extract_plan_steps_for_compact({"plan": {"steps": [step]}}) == [step]
    assert _extract_plan_steps_for_compact({"plan_steps": [step]}) == [step]
    assert _extract_plan_steps_for_compact({}) == []
    assert _extract_plan_steps_for_compact(None) == []
