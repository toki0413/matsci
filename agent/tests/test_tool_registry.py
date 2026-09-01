"""方案A M5(软件部分): 工具注册/交换机制 — "同一接口换后端".

范德华 vs 理想气体: 两个真实(软件)物理工具注册进 ToolRegistry, 通过同一
``PhysicalWorkspace`` / ``BehaviorLifecycle`` / 界面解析/安装, 证明换 `_forward`
物理定律 = 换后端, 无需改上层. 真实 HPC 工具 (VASP) 未来以同种 ToolSpec 注册接入.
"""

from __future__ import annotations

import pytest

from huginn.security.behavior_lifecycle import BehaviorLifecycle
from huginn.security.mechanics_oscillator import (
    OscillatorExecutor,
    OscillatorWorldModel,
)
from huginn.security.thermo_system import ThermoExecutor, build_thermo_artifact
from huginn.security.tool_registry import (
    UnknownToolError,
    get_tool,
    install_tool,
    make_components,
    registered_tools,
)
from huginn.security.van_der_waals import _INITIAL as VDW_INITIAL
from huginn.security.van_der_waals import (
    VanDerWaalsExecutor,
    VanDerWaalsWorldModel,
    build_vdw_artifact,
)
from huginn.security.workspace import PhysicalWorkspace
from huginn.security.world_model import PhysicalAction

_R = 8.31446261815324


def test_registry_has_multi_domain_tools():
    """多物理域工具已注册 (热力学 + 力学 + 外部计算占位) → 同界面可解析不同后端."""
    assert set(registered_tools()) == {
        "ideal_gas",
        "van_der_waals",
        "oscillator",
        "external_shell_compute",
    }


def test_make_components_swaps_backend():
    """make_components: 同一接口, 不同工具名 → 不同执行器与世界模型 (换后端)."""
    ig_exec, ig_wm = make_components("ideal_gas", build_thermo_artifact(1))
    vd_exec, vd_wm = make_components("van_der_waals", build_vdw_artifact(1))
    assert isinstance(ig_exec, ThermoExecutor)
    assert isinstance(vd_exec, VanDerWaalsExecutor)
    assert isinstance(vd_wm, VanDerWaalsWorldModel)
    # 同一 T,V 下两套物理定律给出不同 p → 证明 forward 真被替换.
    ideal = {
        "p": _R * 300.0 / 0.05,
        "V": 0.05,
        "T": 300.0,
        "n": 1.0,
    }
    s_ig = ig_wm.predict(ideal, PhysicalAction("heat", {"dq": 0.0}))
    s_vd = vd_wm.predict(dict(VDW_INITIAL), PhysicalAction("heat", {"dq": 0.0}))
    assert s_ig["p"] != pytest.approx(s_vd["p"])


def test_both_backends_run_in_same_workspace():
    """同一 PhysicalWorkspace 可驱动两个不同物理后端 (接口不变)."""
    ig_exec, ig_wm = make_components("ideal_gas", build_thermo_artifact(1))
    wa_ig = PhysicalWorkspace(ig_wm, ig_exec)
    wa_ig.execute(PhysicalAction("heat", {"dq": 1000.0}), preflight=True)

    vd_exec, vd_wm = make_components("van_der_waals", build_vdw_artifact(1))
    wa_vd = PhysicalWorkspace(vd_wm, vd_exec)
    wa_vd.execute(PhysicalAction("heat", {"dq": 1000.0}), preflight=True)
    assert wa_ig.state["T"] > 0 and wa_vd.state["T"] > 0


def test_install_vdw_good_then_rollback_invalid(tmp_path):
    """经 registry 安装制品: 好 vdW 制品通过物理健康门控; 非法制品(V<nb)回滚."""
    lc = BehaviorLifecycle(tmp_path)

    good = build_vdw_artifact(1, initial=dict(VDW_INITIAL))
    r1 = install_tool(lc, "van_der_waals", good)
    assert r1.healthy and lc.current_version() == 1

    # V < nb → vdW 非物理 → 健康检查失败 → 回滚到 v1.
    bad = build_vdw_artifact(2, initial={**VDW_INITIAL, "V": 1e-6})
    assert get_tool("van_der_waals").health_check(bad) is False
    r2 = install_tool(lc, "van_der_waals", bad)
    assert (not r2.healthy) and r2.rolled_back_to == 1 and lc.current_version() == 1


def test_unknown_tool_raises(tmp_path):
    with pytest.raises(UnknownToolError):
        install_tool(BehaviorLifecycle(tmp_path), "vasp", build_thermo_artifact(1))


def test_cross_domain_mechanics_via_same_interface(tmp_path):
    """不止 VASP / 不只热力学: 力学域振子经同一接口/同一 workspace 运行并健康门控."""
    from huginn.security.mechanics_oscillator import _INITIAL as OSC_INITIAL
    from huginn.security.mechanics_oscillator import build_osc_artifact

    # 同一接口解析出力学后端 (不同领域).
    exe, wm = make_components("oscillator", build_osc_artifact(1))
    assert isinstance(exe, OscillatorExecutor)
    assert isinstance(wm, OscillatorWorldModel)

    # 同一 PhysicalWorkspace 驱动力学域动作 (kick / displace) 感知确认通过.
    wa = PhysicalWorkspace(wm, exe)
    wa.execute(PhysicalAction("kick", {"dv": 3.0}), preflight=True)
    wa.execute(PhysicalAction("displace", {"dx": 2.0}), preflight=True)
    assert wa.state["x"] == pytest.approx(2.0) and wa.state["v"] == pytest.approx(3.0)

    # 经 registry 安装: 好制品通过力学健康门控; 超界制品 (x 越界) 回滚.
    lc = BehaviorLifecycle(tmp_path)
    good = build_osc_artifact(1, initial=dict(OSC_INITIAL))
    assert install_tool(lc, "oscillator", good).healthy and lc.current_version() == 1
    bad = build_osc_artifact(2, initial={"x": 500.0, "v": 0.0})  # |x|>x_max=100
    assert get_tool("oscillator").health_check(bad) is False
    r2 = install_tool(lc, "oscillator", bad)
    assert (not r2.healthy) and r2.rolled_back_to == 1 and lc.current_version() == 1
