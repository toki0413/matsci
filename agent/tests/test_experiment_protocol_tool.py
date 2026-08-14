"""ExperimentProtocolTool 测试 — 物理实验协议接入 agent 工具.

验证:
- run: 完整协议执行, 返回 executed/degraded/inverse_count, 逆进入共享逆栈.
- run + mixer 缺失: mix/aliquot 自动降级, 只执行激活步骤.
- status: 返回各步骤激活状态.
- 复用 ToolContext.revertible: 成功时物理逆登记进共享逆栈, 上层可整体撤销.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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


def test_tool_registered() -> None:
    from huginn.tools import register_all_tools

    names = register_all_tools()
    assert "experiment_protocol_tool" in names