"""物理世界接入核心结构 (Cordis 时空可组合的物理实例化) 测试.

验证:
- OP_ACTION 逆类型: track_world_action 登记可序列化逆, revert_all 时经
  物理执行器执行 inverse 动作 (时间可组合).
- journal/崩溃重放: OP_ACTION 逆可落盘并按 LIFO 重放.
- NaiveWorldModel: 朴素逆规则 (move 反向 / grasp↔release / dispense↔aspirate /
  不可逆返回 None).
"""

from __future__ import annotations

from pathlib import Path

from huginn.security import revertible
from huginn.security.revertible import RevertibleContext, recover_from
from huginn.security.world_model import NaiveWorldModel, PhysicalAction


# ── OP_ACTION 可逆性 (时间可组合) ─────────────────────────────────
def test_track_world_action_reverts_via_executor(monkeypatch) -> None:
    ctx = RevertibleContext()
    executed: list[dict] = []
    monkeypatch.setattr(
        revertible, "_PHYSICAL_EXECUTOR", lambda inv: executed.append(inv)
    )

    ctx.track_world_action(
        {"gripper": "open"},
        {"type": "move", "params": {"start": "A", "target": "B"}},
        {"type": "move", "params": {"start": "B", "target": "A"}},
    )
    assert ctx.depth == 1

    ctx.revert_all()
    assert executed == [{"type": "move", "params": {"start": "B", "target": "A"}}]
    assert ctx.depth == 0


def test_world_action_is_lifo(monkeypatch) -> None:
    ctx = RevertibleContext()
    order: list[str] = []
    monkeypatch.setattr(revertible, "_PHYSICAL_EXECUTOR", lambda inv: order.append(inv["type"]))
    ctx.track_world_action({}, {"type": "grasp"}, {"type": "release"})
    ctx.track_world_action({}, {"type": "move"}, {"type": "move"})
    ctx.revert_all()
    assert order == ["move", "release"], "物理逆应 LIFO: 后登记的逆先执行"


def test_world_action_journal_replay(tmp_path: Path, monkeypatch) -> None:
    """OP_ACTION 逆可落盘 journal, 崩溃后按 LIFO 重放 (跨崩溃可逆)."""
    journal = tmp_path / "journal.json"
    ctx = RevertibleContext(journal_path=journal)
    ctx.track_world_action({}, {"type": "grasp"}, {"type": "release"})
    assert "physical_action" in journal.read_text(encoding="utf-8")

    # 模拟"进程崩溃": 新进程, 无 ctx 内存态, 只靠 journal 重放.
    executed: list[dict] = []
    monkeypatch.setattr(
        revertible, "_PHYSICAL_EXECUTOR", lambda inv: executed.append(inv)
    )
    n = recover_from(journal)
    assert n == 1
    assert executed == [{"type": "release"}]
    assert not journal.exists(), "重放后应清理 journal"


def test_world_action_without_executor_is_best_effort(monkeypatch) -> None:
    """未注册物理执行器时, 回滚不抛异常 (best-effort, 与 dispose 一致)."""
    ctx = RevertibleContext()
    monkeypatch.setattr(revertible, "_PHYSICAL_EXECUTOR", None)
    ctx.track_world_action({}, {"type": "grasp"}, {"type": "release"})
    ctx.revert_all()  # 不应抛异常
    assert ctx.depth == 0


# ── NaiveWorldModel 朴素逆规则 ────────────────────────────────────
def test_naive_world_model_move_inverse() -> None:
    wm = NaiveWorldModel()
    inv = wm.infer_inverse(
        {}, PhysicalAction("move", {"start": "A", "target": "B"})
    )
    assert inv == PhysicalAction("move", {"start": "B", "target": "A"})


def test_naive_world_model_grasp_release() -> None:
    wm = NaiveWorldModel()
    assert wm.infer_inverse({}, PhysicalAction("grasp", {"obj": "tube"})) == (
        PhysicalAction("release", {"obj": "tube"})
    )
    assert wm.infer_inverse({}, PhysicalAction("release", {"obj": "tube"})) == (
        PhysicalAction("grasp", {"obj": "tube"})
    )


def test_naive_world_model_dispense_aspirate() -> None:
    wm = NaiveWorldModel()
    assert wm.infer_inverse({}, PhysicalAction("dispense", {"vol": 5})) == (
        PhysicalAction("aspirate", {"vol": 5})
    )


def test_naive_world_model_irreversible() -> None:
    wm = NaiveWorldModel()
    assert wm.infer_inverse({}, PhysicalAction("stir", {})) is None
    # move 缺 start/target 无法反向
    assert wm.infer_inverse({}, PhysicalAction("move", {"target": "B"})) is None


# ── PhysicalAction 可序列化 ───────────────────────────────────────
def test_physical_action_roundtrip() -> None:
    a = PhysicalAction("move", {"start": "A", "target": "B"})
    b = PhysicalAction.from_dict(a.to_dict())
    assert b == a
