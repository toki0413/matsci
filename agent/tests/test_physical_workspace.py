"""PhysicalWorkspace + 实验协议演示测试 (核心的物理第三次实例化).

验证:
- 空间可组合: 依赖链可用 → 步骤激活; 后端缺失 → 下游自动停用 (degrade).
- 时间可组合: 协议作为事务, 任一步失败 → 物理逆 LIFO 回滚, 工作台恢复.
- 感知确认: 动作执行但状态未达预期 → 抛异常触发回滚.
"""

from __future__ import annotations

import pytest

from huginn.security.experiment_protocol import (
    C_ALIQUOT,
    C_ASPIRATE,
    C_DISPENSE,
    C_MIX,
    K_ALIQUOTS,
    K_MIXED,
    build_pipette_workflow,
    run_pipette_protocol,
)
from huginn.security.workspace import (
    MockExecutor,
    PhysicalWorkspace,
    WorkspaceConfirmError,
)
from huginn.security.world_model import NaiveWorldModel, PhysicalAction


# ── 空间可组合: 依赖链激活 / 停用 ────────────────────────────────
def test_workflow_all_steps_active() -> None:
    ex = MockExecutor()
    wa = build_pipette_workflow(ex)
    assert wa.is_active(C_ASPIRATE)
    assert wa.is_active(C_DISPENSE)
    assert wa.is_active(C_MIX)
    assert wa.is_active(C_ALIQUOT)


def test_workflow_degrades_when_mixer_unavailable() -> None:
    """混合器缺失 → mix/aliquot 自动停用; aspirate/dispense 仍激活."""
    ex = MockExecutor()
    wa = build_pipette_workflow(ex, mixer_available=False)
    assert wa.is_active(C_ASPIRATE)
    assert wa.is_active(C_DISPENSE)
    assert not wa.is_active(C_MIX), "mix 依赖 mixer, 应停用"
    assert not wa.is_active(C_ALIQUOT), "aliquot 依赖 mixture.mixed, 应级联停用"


def test_workflow_degrades_when_pipette_unavailable() -> None:
    """移液枪缺失 → 整条链 (依赖 pipette) 停用."""
    ex = MockExecutor()
    wa = build_pipette_workflow(ex)
    wa.set_available("pipette.ready", False)
    assert not wa.is_active(C_ASPIRATE)
    assert not wa.is_active(C_DISPENSE)


# ── 成功协议: 全步骤执行 + 逆登记 ────────────────────────────────
def test_pipette_protocol_success() -> None:
    ex = MockExecutor()
    wa = build_pipette_workflow(ex)
    run_pipette_protocol(wa)
    types = [a.type for a in ex.log]
    assert types == ["aspirate", "dispense", "mix", "aliquot"]
    # 逆被登记: aspirate↔dispense 可逆各登记 1 个; mix/aliquot 在朴素模型中
    # 不可逆 (混合无法自动"拆开"), 不登记. 事务正常退出保留 → depth=2.
    assert wa.revertible.depth == 2


def test_pipette_protocol_skips_inactive_steps() -> None:
    """混合器缺失 → 协议只执行激活步骤 (aspirate/dispense), 不跑 mix/aliquot."""
    ex = MockExecutor()
    wa = build_pipette_workflow(ex, mixer_available=False)
    run_pipette_protocol(wa)
    types = [a.type for a in ex.log]
    assert types == ["aspirate", "dispense"]


# ── 时间可组合: 失败回滚 ─────────────────────────────────────────
def test_rollback_on_execution_failure() -> None:
    """dispense 执行失败 → 事务回滚, 触发 aspirate 的逆 (dispense) 执行.

    aspirate 的逆也是 dispense, 而 MockExecutor 对 dispense 一律 fail → 逆执行
    也失败, 被 best-effort 吞掉不中断回滚 (物理故障时逆同样可能失败)."""
    ex = MockExecutor(fail_on={"dispense"})
    wa = build_pipette_workflow(ex)
    with pytest.raises(RuntimeError, match="dispense"):
        run_pipette_protocol(wa)
    # 日志: aspirate 正向执行; 回滚时逆 dispense 尝试执行但因同因失败被吞.
    assert [a.type for a in ex.log] == ["aspirate"]
    assert wa.revertible.depth == 0, "事务回滚后逆栈应清空"


def test_rollback_lifo_order() -> None:
    ex = MockExecutor()
    wa = build_pipette_workflow(ex)
    with pytest.raises(RuntimeError, match="boom"):
        with wa.transaction():
            wa.execute(PhysicalAction("grasp", {"obj": "tube"}))
            wa.execute(PhysicalAction("move", {"start": "A", "target": "B"}))
            raise RuntimeError("boom")
    # LIFO: 先 move 逆 (反向 move), 再 grasp 逆 (release).
    types = [a.type for a in ex.log]
    assert types == ["grasp", "move", "move", "release"]
    assert ex.log[2].params == {"start": "B", "target": "A"}
    assert wa.revertible.depth == 0


# ── 感知确认 ──────────────────────────────────────────────────────
def test_confirm_failure_rolls_back() -> None:
    """感知确认失败 (动作执行但状态未达预期) → 抛 WorkspaceConfirmError 并回滚."""
    ex = MockExecutor()
    wa = build_pipette_workflow(ex)
    # 人为让 dispense 的确认永远失败.
    wa.confirm("tube.filled", lambda: False)
    with pytest.raises(WorkspaceConfirmError, match="tube.filled"):
        run_pipette_protocol(wa)
    # aspirate 与 dispense 都已执行并登记逆; 回滚 LIFO 依次执行两者的逆:
    # dispense 逆=aspirate, aspirate 逆=dispense.
    types = [a.type for a in ex.log]
    assert types == ["aspirate", "dispense", "aspirate", "dispense"]
    assert wa.revertible.depth == 0


# ── 不可逆动作 ────────────────────────────────────────────────────
def test_irreversible_action_executes_without_inverse() -> None:
    ex = MockExecutor()
    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    wa.execute(PhysicalAction("stir", {}))  # 朴素模型认为 stir 不可逆
    assert wa.revertible.depth == 0, "不可逆动作不登记逆"
    assert [a.type for a in ex.log] == ["stir"]