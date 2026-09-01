"""通用外部计算工具适配器 + 占位测试.

证明"真实计算工具(外部进程/远程作业)"能以同一执行器骨架接入: 建作业→子进程
运行→解析成 state; 世界模型(快代理)与真实子进程(真计算)必须一致 (view-consistency),
是"世界模型 == 仪表"契约的软件兑现.
"""

from __future__ import annotations

import pytest

from huginn.security.behavior_lifecycle import BehaviorLifecycle
from huginn.security.compute_adapter import (
    ExternalComputeExecutor,
    ShellComputeTool,
    ShellComputeWorldModel,
    ToolInvocationError,
    build_shell_compute_artifact,
    shell_executor_from_artifact,
)
from huginn.security.physics_schema import matches_state
from huginn.security.tool_registry import (
    install_tool,
    make_components,
    registered_tools,
)
from huginn.security.workspace import PhysicalWorkspace
from huginn.security.world_model import PhysicalAction

_CV = 1.5 * 8.31446261815324


def test_external_executor_runs_real_subprocess():
    """外部执行器真跑子进程: 解析出能量 = n·Cv·T."""
    ex = shell_executor_from_artifact(build_shell_compute_artifact(1))
    ex.execute(PhysicalAction("shell_compute", {"n": 2.0, "T": 300.0}))
    assert ex.observe()["energy"] == pytest.approx(2.0 * _CV * 300.0)


def test_world_model_matches_external_executor():
    """快代理(世界模型) == 真实子进程(执行器) → view-consistency 契约成立."""
    ex = shell_executor_from_artifact(build_shell_compute_artifact(1))
    wm = ShellComputeWorldModel()
    a = PhysicalAction("shell_compute", {"n": 1.0, "T": 400.0})
    pred = wm.predict({}, a)
    ex.execute(a)
    assert matches_state(pred, ex.observe(), tolerance=1e-9)


def test_external_tool_wired_into_workspace_and_registry():
    """外部工具经同界面解析并跑进同一 PhysicalWorkspace (确认闭环), 且已注册."""
    assert "external_shell_compute" in registered_tools()
    exe, wm = make_components("external_shell_compute", build_shell_compute_artifact(1))
    assert isinstance(exe, ExternalComputeExecutor)
    wa = PhysicalWorkspace(wm, exe)
    wa.execute(PhysicalAction("shell_compute", {"n": 1.0, "T": 300.0}), preflight=True)
    assert wa.state["energy"] == pytest.approx(_CV * 300.0)


def test_external_tool_install_health_gate(tmp_path):
    """经 registry 安装: 外部子进程健康门控通过并成为 current."""
    lc = BehaviorLifecycle(tmp_path)
    r = install_tool(lc, "external_shell_compute", build_shell_compute_artifact(1))
    assert r.healthy and lc.current_version() == 1


def test_tool_failure_is_surface():
    """外部工具返回非 0 → ToolInvocationError (健康门控/执行层可捕获)."""
    tool = ShellComputeTool()
    with pytest.raises(ToolInvocationError):
        tool.parse_output("", 1)
