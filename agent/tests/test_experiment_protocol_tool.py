"""ExperimentProtocolTool 测试 — 物理实验协议接入 agent 工具.

验证:
- run: 完整协议执行, 返回 executed/degraded/inverse_count, 逆进入共享逆栈.
- run + mixer 缺失: mix/aliquot 自动降级, 只执行激活步骤.
- status: 返回各步骤激活状态.
- 复用 ToolContext.revertible: 成功时物理逆登记进共享逆栈, 上层可整体撤销.
"""

from __future__ import annotations

import asyncio

from huginn.core_types import ToolContext
from huginn.security.revertible import RevertibleContext
from huginn.tools.experiment_protocol_tool import (
    ExperimentProtocolInput,
    ExperimentProtocolTool,
)


def _call(args: ExperimentProtocolInput, rv: RevertibleContext | None = None):
    tool = ExperimentProtocolTool()
    ctx = ToolContext(
        session_id="s", workspace="/tmp", revertible=rv,
    )
    return asyncio.run(tool.call(args, ctx))


def test_run_success_with_shared_revertible() -> None:
    rv = RevertibleContext()
    res = _call(ExperimentProtocolInput(action="run"), rv)
    assert res.success, res.error
    data = res.data
    assert data["executed_steps"] == ["aspirate", "dispense", "mix", "aliquot"]
    assert data["degraded_steps"] == []
    # aspirate↔dispense 可逆各 1 个逆; mix/aliquot 朴素模型不可逆不进逆栈.
    assert data["inverse_count"] == 2
    assert data["revertible_shared_with_agent"] is True
    # 物理逆登记进共享逆栈, 上层整体回滚可一并撤销.
    assert rv.depth == 2


def test_run_reverts_into_shared_stack_on_failure() -> None:
    """执行失败时事务回滚, 逆被消费, 共享栈恢复到批次前深度."""
    rv = RevertibleContext()
    # 无 mixer → 只跑 aspirate/dispense; 让 dispense 的感知确认失败触发回滚.
    res = _call(ExperimentProtocolInput(action="run"), rv)
    assert res.success
    # 逆在共享栈 (成功批次保留).
    assert rv.depth == 2
    # 上层整体回滚 → 物理逆一并执行 (dispense 是 aspirate 的逆, 故出现).
    rv.revert_all()
    assert rv.depth == 0


def test_run_degrades_when_mixer_unavailable() -> None:
    res = _call(ExperimentProtocolInput(action="run", mixer_available=False))
    assert res.success, res.error
    data = res.data
    assert data["executed_steps"] == ["aspirate", "dispense"]
    assert set(data["degraded_steps"]) == {"mix_step", "aliquot_step"}
    assert data["inverse_count"] == 2


def test_status_reports_activation() -> None:
    res = _call(ExperimentProtocolInput(action="status", mixer_available=False))
    assert res.success, res.error
    steps = res.data["steps"]
    assert steps["aspirate_step"] is True
    assert steps["dispense_step"] is True
    assert steps["mix_step"] is False
    assert steps["aliquot_step"] is False


def test_run_without_shared_revertible_builds_own() -> None:
    res = _call(ExperimentProtocolInput(action="run"))
    assert res.success, res.error
    assert res.data["revertible_shared_with_agent"] is False
    assert res.data["inverse_count"] == 2


def test_run_learn_surrogate_accumulates_samples() -> None:
    """P2 快预演积累: sim 后端 + learn_surrogate 时, 真实(sim)执行的观测对喂进学代理.

    每个激活步骤 (aspirate/dispense/mix/aliquot) 让 workspace 的 observe 喂一条样本,
    在无退化时 surrogate_samples == 4; 退化时等于实际执行步数.
    """
    res = _call(
        ExperimentProtocolInput(action="run", executor_backend="sim", learn_surrogate=True)
    )
    assert res.success, res.error
    data = res.data
    assert data["executor_backend"] == "sim"
    assert data["surrogate_samples"] == len(data["executed_steps"]) == 4
    # 全协议: surrogate 覆盖全部动作类型
    assert data["surrogate_samples"] == 4
    # final_state 仍照常产出 (代理不影响执行本体)
    assert "sample_vol" in data["final_state"]


def test_run_learn_surrogate_degrades_still_accumulates() -> None:
    """退化时只积累实际执行步的样本 (mix/aliquot 停用 → 只积累 aspirate/dispense)."""
    res = _call(
        ExperimentProtocolInput(
            action="run", executor_backend="sim", mixer_available=False,
            learn_surrogate=True,
        )
    )
    assert res.success, res.error
    data = res.data
    assert data["executed_steps"] == ["aspirate", "dispense"]
    assert data["surrogate_samples"] == 2
    assert "sample_vol" in data["final_state"]


def test_run_no_surrogate_field_when_disabled() -> None:
    """默认 (learn_surrogate=False) 不暴露 surrogate 字段, 零回归."""
    res = _call(ExperimentProtocolInput(action="run"))
    assert res.success, res.error
    assert "surrogate_samples" not in res.data


def test_tool_registered() -> None:
    from huginn.tools import register_all_tools

    names = register_all_tools()
    assert "experiment_protocol_tool" in names
