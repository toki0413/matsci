"""I19: 巩固只在 sleep 状态执行；状态迁移可审计."""
from __future__ import annotations

from huginn.memory.dream import SleepClock, run_dream_cycle, tick


def _clock() -> SleepClock:
    return SleepClock(
        persona_id="p1",
        timezone="UTC",
        weekly_windows=(("mon", "22:00", "06:00"),),
    )


def test_no_consolidation_in_active():
    clock = _clock()
    active = run_dream_cycle(
        clock=clock,
        dream_id="d1",
        memories=[{"memory_id": "m1", "degree": 0}],
        now="2026-10-20T12:00:00+00:00",
    )
    assert active.status == "skipped_active"
    assert not active.findings and not active.repairs
    assert active.dream_object is None

    sleeping = run_dream_cycle(
        clock=clock,
        dream_id="d2",
        memories=[{"memory_id": "m1", "degree": 0}],
        now="2026-10-19T23:00:00+00:00",  # Monday 23:00 在 mon 22:00–06:00 内
    )
    assert sleeping.status == "completed"
    assert sleeping.findings

    # 状态迁移可审计：last_transition_at 只在跨越窗口边界时写入
    before = tick(clock, "2026-10-19T12:00:00+00:00")
    after = tick(before, "2026-10-19T23:00:00+00:00")
    assert after.state == "sleep"
    assert after.last_transition_at
