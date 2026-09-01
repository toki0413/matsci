"""方案A M1+M2: 理想气体前向真值执行器 + 感知确认闭环.

覆盖: 解析前向真值 (pV=nRT 恒成立) / gauge 传感器 systematic+标定 /
可逆动作精确逆 / PhysicalWorkspace 预演-执行-sensor_view 同视角确认全闭环.
"""

from __future__ import annotations

import pytest

from huginn.security.actuator_model import ErrorModel
from huginn.security.physics_schema import matches_state
from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor, forward
from huginn.security.workspace import PhysicalWorkspace
from huginn.security.world_model import PhysicalAction

_R = 8.31446261815324
# 初始态须 EOS 自洽 (p = nRT/V), 否则逆还原会落在 EOS 一致值而非字面量.
_S0 = {"p": _R * 273.15 / 0.022414, "V": 0.022414, "T": 273.15, "n": 1.0}


def test_forward_heat_raises_t_isochoric():
    """等容加热: T 升、V 不变, 且 pV = nRT 恒成立."""
    s1 = forward(_S0, PhysicalAction("heat", {"dq": 1000.0}))
    assert s1["T"] > _S0["T"]
    assert s1["V"] == pytest.approx(_S0["V"])
    assert s1["p"] * s1["V"] == pytest.approx(_R * s1["T"] * s1["n"])


def test_forward_move_isothermal():
    """准静态等温变体积: T 不变, p 反比于 V."""
    s1 = forward(_S0, PhysicalAction("move", {"v": 0.0448}))
    assert s1["T"] == pytest.approx(_S0["T"])
    assert s1["p"] * s1["V"] == pytest.approx(_R * s1["T"] * s1["n"])


def test_sensor_gauge_systematic_and_calibrate():
    """gauge 传感器: 同一幅值抬高压强计/温度计读数; 标定消除 systematic."""
    em = ErrorModel(systematic=1.0, sigma=0.0)
    ex = ThermoExecutor(error_model=em, seed=0)
    a = PhysicalAction("heat", {"dq": 1000.0})
    ex.execute(a)
    ideal = forward(_S0, a)
    assert ex.observe()["T"] == pytest.approx(ideal["T"] + 1.0)
    assert ex.observe()["p"] == pytest.approx(ideal["p"] + 1.0)
    assert ex.calibrate() == pytest.approx(1.0)
    assert em.systematic == 0.0


def test_inverse_of_heat_and_move_exact():
    """可逆动作的精确逆: heat 逆 = 反向 heat, move 逆 = 移回原体积, 精确还原."""
    wm = IdealGasWorldModel()
    for action in (
        PhysicalAction("heat", {"dq": 1000.0}),
        PhysicalAction("move", {"v": 0.0448}),
    ):
        after = wm.predict(_S0, action)
        inv = wm.infer_inverse(_S0, action)
        assert inv is not None
        restored = wm.predict(after, inv)
        assert restored["T"] == pytest.approx(_S0["T"])
        assert restored["V"] == pytest.approx(_S0["V"])
        assert restored["p"] == pytest.approx(_S0["p"])


def test_m2_view_consistent_confirm_closed_loop():
    """物理闭环: 预演(世界模型) → 执行 → sensor_view 同视角确认 通过.

    偏置 +1.0 下, naive 理想预测直比实测会误判; sensor_view 同视角则判定一致,
    从而"偏置但正确"的动作不被错误回滚 (microduck 编码器读穿输出侧).
    """
    em = ErrorModel(systematic=1.0, sigma=0.0)
    ex = ThermoExecutor(error_model=em, seed=0)
    wa = PhysicalWorkspace(IdealGasWorldModel(), ex)
    a = PhysicalAction("heat", {"dq": 1000.0})
    ideal = IdealGasWorldModel().predict(dict(_S0), a)

    wa.execute(a, preflight=True)  # 内部: expected 经 sensor_view 同视角, 不抛.
    obs = wa.state
    assert obs["T"] == pytest.approx(ideal["T"] + 1.0)
    # naive: 理想预测直接对比实测 → 严格容差失败 (误判).
    assert not matches_state(ideal, obs, tolerance=1e-9)
    # 同视角: sensor_view(ideal) == 实测 → 判定一致.
    assert matches_state(ex.sensor_view(ideal, a.type), obs, tolerance=1e-9)
