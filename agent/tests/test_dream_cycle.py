"""梦境周期测试：睡眠门控、三角色分工、写入侧晋升门、可逆软衰减."""
from __future__ import annotations

from huginn.memory.dream import (
    SleepClock,
    apply_soft_decay,
    diagnose,
    plan_repairs,
    probe_memory_graph,
    promote_dream_object,
    restore_soft_decay,
    run_dream_cycle,
    synthesize_dream_object,
    tick,
)


def test_sleep_clock_in_window_and_overnight():
    clock = SleepClock(
        persona_id="p1",
        timezone="UTC",
        weekly_windows=(("mon", "22:00", "06:00"),),
    )
    assert tick(clock, "2026-10-19T23:00:00+00:00").state == "sleep"
    assert tick(clock, "2026-10-20T05:00:00+00:00").state == "sleep"
    assert tick(clock, "2026-10-20T07:00:00+00:00").state == "active"


def test_dream_cycle_skips_during_active():
    clock = SleepClock(
        persona_id="p1",
        timezone="UTC",
        weekly_windows=(("mon", "22:00", "06:00"),),
    )
    result = run_dream_cycle(
        clock=clock,
        dream_id="d1",
        memories=[{"memory_id": "m1", "degree": 0}],
        now="2026-10-20T12:00:00+00:00",
    )
    assert result.status == "skipped_active"
    assert not result.findings
    assert result.dream_object is None


def test_dreamer_probes_structure_and_demand():
    findings = probe_memory_graph(
        [
            {"memory_id": "iso", "degree": 0},
            {"memory_id": "old", "stale": True},
            {"memory_id": "dup", "duplicate_of": "iso"},
            {"memory_id": "conf", "contradiction_pressure": 0.95},
        ],
        probes=[{"probe_id": "q1", "failed": True, "surprise": 0.8}],
    )
    subjects = {f.subject for f in findings}
    assert {"iso", "old", "dup", "conf", "q1"} <= subjects
    assert any(f.probe_type == "demand" for f in findings)


def test_analyst_maps_causes_and_can_abstain():
    finding = probe_memory_graph(
        [{"memory_id": "dup", "duplicate_of": "other"}]
    )[0]
    dx = diagnose(finding)
    assert dx.cause_class == "duplicate"
    assert dx.repair_class == "merge_duplicate"
    assert not dx.abstain

    # 反事实不通过 → confidence 低于地板 → 弃权
    dx2 = diagnose(finding, counterfactual=lambda f, r: False)
    assert dx2.abstain
    assert dx2.confidence < 0.5


def test_consolidator_never_deletes_and_respects_risk():
    findings = probe_memory_graph(
        [
            {"memory_id": "dup", "duplicate_of": "x"},
            {"memory_id": "conf", "contradiction_pressure": 0.95},
        ]
    )
    diagnoses = [diagnose(f) for f in findings]
    repairs = plan_repairs(diagnoses, max_risk="medium")
    assert repairs, "至少 duplicate 修复应产出"
    assert all(r.action != "delete" for r in repairs)
    assert all(r.reversible for r in repairs)
    # contradiction risk=high，默认 medium 上限下不执行
    assert all(r.risk != "high" for r in repairs)


def test_dream_object_authority_and_promotion_gate():
    findings = probe_memory_graph([{"memory_id": "m1", "degree": 0}])
    dx = [diagnose(f) for f in findings]
    repairs = plan_repairs(dx)
    obj = synthesize_dream_object(
        dream_id="d1",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=findings,
        repairs=repairs,
    )
    assert obj.status == "candidate"
    assert obj.evidential_authority <= 0.4
    assert obj.influence_authority <= 0.6

    # 未满足晋升条件 → 保持 candidate
    assert promote_dream_object(obj).status == "candidate"
    # 独立佐证 + provenance 完整 → promoted, evidential 0.6
    promoted = promote_dream_object(
        obj, corroboration=1, provenance_complete=True
    )
    assert promoted.status == "promoted"
    assert promoted.evidential_authority == 0.6
    # 未解矛盾 → quarantine, influence 0
    quarantined = promote_dream_object(
        obj, corroboration=3, provenance_complete=True,
        has_unresolved_contradiction=True,
    )
    assert quarantined.status == "quarantined"
    assert quarantined.influence_authority == 0.0


def test_soft_decay_is_reversible():
    decayed, handle = apply_soft_decay(0.8, 0.5)
    assert decayed == 0.4
    assert restore_soft_decay(decayed, handle) == 0.8
