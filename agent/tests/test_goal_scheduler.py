"""GoalScheduler check_completion 累积验证测试."""
from __future__ import annotations

from huginn.autoloop.goal_scheduler import Goal, GoalScheduler


def _make_goal(criteria: list[str]) -> Goal:
    return Goal(
        id="test-1",
        objective="test objective",
        success_criteria=criteria,
    )


def test_check_completion_all_hit():
    """所有 criteria 都在 validation 里 -> True."""
    goal = _make_goal(["band gap", "1.12"])
    v = {"result": "band gap is 1.12 eV"}
    assert GoalScheduler.check_completion(goal, v)


def test_check_completion_partial_miss():
    """部分 criteria 没命中 -> False."""
    goal = _make_goal(["band gap", "mechanism"])
    v = {"result": "band gap is 1.12 eV"}  # no "mechanism"
    assert not GoalScheduler.check_completion(goal, v)


def test_check_completion_accumulated():
    """累积模式: 第一轮缺 mechanism, 第二轮补上 -> True."""
    goal = _make_goal(["band gap", "mechanism"])
    goal.metadata = {}

    v1 = {"result": "band gap is 1.12 eV"}
    # 第一轮: 只有 band gap, 没有 mechanism
    assert not GoalScheduler.check_completion(goal, v1)

    v2 = {"result": "the mechanism is phonon scattering"}
    # 第二轮: mechanism 命中, 加上第一轮的 band gap
    assert GoalScheduler.check_completion(goal, v2)

    # 验证历史被累积了
    assert len(goal.metadata["_validation_history"]) == 2


def test_check_completion_no_criteria():
    """没有 criteria -> False."""
    goal = _make_goal([])
    assert not GoalScheduler.check_completion(goal, {"result": "ok"})


def test_check_completion_none_validation():
    """validation 为 None -> False."""
    goal = _make_goal(["something"])
    assert not GoalScheduler.check_completion(goal, None)


def test_check_completion_across_iterations():
    """跨迭代累积: 3 轮分别命中不同 criteria, 最终 True."""
    goal = _make_goal(["energy", "force", "stress"])
    goal.metadata = {}

    assert not GoalScheduler.check_completion(goal, {"r": "energy = -24"})
    assert not GoalScheduler.check_completion(goal, {"r": "force = 0.01"})
    # 第三轮 stress 命中, 加上前两轮的 energy + force
    assert GoalScheduler.check_completion(goal, {"r": "stress = 0"})
