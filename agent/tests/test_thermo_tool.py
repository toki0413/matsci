"""Smoke tests for thermo_tool registration + routing.

md_thermo 的物理数学在 test_md_thermo.py 里覆盖, 这里只管:
  - 纯组分物性查询能跑通 (thermo 库装了的前提)
  - md_thermo / phase_diagram 两个 action 不依赖 thermo 库, 路由要在
    可用性检查之前分出去 (否则没装 thermo 时这俩也挂)
"""

from __future__ import annotations

import asyncio
import math
import typing

import pytest

from huginn.tools import thermo_tool as thermo_mod
from huginn.tools.thermo_tool import ThermoTool, ThermoToolInput
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


def _run(args: ThermoToolInput):
    return asyncio.run(ThermoTool().call(args, CTX))


# ── 纯组分物性查询 (需要 thermo 库) ─────────────────────────────────
def test_water_basic_properties():
    pytest.importorskip("thermo")  # heavy dep, skip if not installed
    res = _run(ThermoToolInput(action="properties", compound="water"))
    assert res.success, res.error
    props = res.data["properties"]
    # 水的分子量 ~18.015, 沸点 ~373K, 离线查得到就行
    assert math.isclose(props["MW"], 18.015, abs_tol=0.1)
    assert math.isclose(props["Tb"], 373.15, abs_tol=1.0)
    assert res.data["conditions"]["compound"] == "water"


# ── md_thermo 在 action 枚举里 ───────────────────────────────────────
def test_md_thermo_in_action_enum():
    schema = ThermoToolInput.model_json_schema()
    actions = schema["properties"]["action"]["enum"]
    assert "md_thermo" in actions
    # typing.get_args 也能直接拿到 Literal 的取值
    literal_args = typing.get_args(ThermoToolInput.model_fields["action"].annotation)
    assert "md_thermo" in literal_args


# ── md_thermo 不依赖 thermo 库: 路由在可用性检查之前 ─────────────────
def test_md_thermo_bypasses_thermo_availability(monkeypatch):
    # 假装 thermo 没装, md_thermo 照样得能跑 (它只用 numpy)
    monkeypatch.setattr(thermo_mod, "_THERMO_AVAILABLE", False)

    series = {
        "temperature": [300.0, 301.0, 299.0, 300.5, 299.5],
        "kinetic_energy": [2.0, 2.01, 1.99, 2.005, 1.995],
        "potential_energy": [-8.0, -8.02, -7.98, -8.01, -7.99],
    }
    res = _run(ThermoToolInput(
        action="md_thermo", md_time_series=series, n_atoms=10
    ))
    assert res.success, res.error
    props = res.data["properties"]
    assert props["cv_eV_K"] is not None and props["cv_eV_K"] > 0
    assert props["method"] == "fluctuation_formula"


# ── phase_diagram 也该绕过 thermo 可用性检查 (回归) ──────────────────
def test_phase_diagram_bypasses_thermo_availability(monkeypatch):
    monkeypatch.setattr(thermo_mod, "_THERMO_AVAILABLE", False)
    # 给个最小 entries, 不验证相图正确性, 只验证没被 thermo 检查拦下
    res = _run(ThermoToolInput(
        action="phase_diagram",
        entries=[{"composition": "Li", "energy": 0.0},
                {"composition": "O", "energy": 0.0},
                {"composition": "Li2O", "energy": -10.0}],
    ))
    # pymatgen 没装时会返回明确的安装提示, 但绝不是 thermo 缺失的错误
    if not res.success:
        assert "pymatgen" in (res.error or ""), res.error
    else:
        assert res.data["action"] == "phase_diagram"
