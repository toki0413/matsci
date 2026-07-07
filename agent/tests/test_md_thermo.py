"""Tests for thermo_tool md_thermo action (LAMMPS thermo -> 热力学量).

打通 LAMMPS MD 输出 -> 统计力学涨落公式这条管道. 不依赖 thermo 库,
md_thermo 走的是 numpy 路径, 所以这里不 mock thermo.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest
from pydantic import ValidationError

from huginn.tools.thermo_tool import ThermoTool, ThermoToolInput
from huginn.types import ToolContext

# 跟 _md_thermo 里保持一致的物理常数, 用来算期望值
KB = 8.617333e-5  # eV/K
EV_TO_J_MOL = 96485.0

CTX = ToolContext(session_id="test", workspace=".")


def _run(args: ThermoToolInput):
    tool = ThermoTool()
    return asyncio.run(tool.call(args, CTX))


def _make_series(
    n: int = 1000,
    T: float = 310.0,
    e_sigma: float = 0.05,
    ke0: float = 2.0,
    pe0: float = -5.0,
    seed: int = 0,
) -> dict:
    """造一组 KE/PE/T 时序, 总能量叠一层高斯噪声 (方差已知好对账)."""
    rng = np.random.default_rng(seed)
    temp = np.full(n, T)
    noise = rng.normal(0.0, e_sigma, n)
    total = pe0 + ke0 + noise
    ke = np.full(n, ke0)  # KE 恒定, 模拟 NVT thermostat
    pe = total - ke
    return {
        "time": np.arange(n, dtype=float).tolist(),
        "temperature": temp.tolist(),
        "kinetic_energy": ke.tolist(),
        "potential_energy": pe.tolist(),
    }


# ── 1. 基本热容计算 ──────────────────────────────────────────────────
def test_basic_cv_from_energy_fluctuations():
    n_atoms = 50
    series = _make_series(n=2000, T=310.0, e_sigma=0.05, seed=1)

    res = _run(ThermoToolInput(
        action="md_thermo",
        md_time_series=series,
        n_atoms=n_atoms,
    ))
    assert res.success, res.error
    props = res.data["properties"]

    # 用同样的 numpy 算一遍期望值, 跟工具输出对账
    total = (
        np.asarray(series["kinetic_energy"])
        + np.asarray(series["potential_energy"])
    )
    var_e = float(np.var(total))  # ddof=0, 跟工具一致
    expected_cv = var_e / (KB * 310.0 ** 2 * n_atoms)

    assert props["cv_eV_K"] is not None
    assert math.isclose(props["cv_eV_K"], expected_cv, rel_tol=1e-9)
    assert math.isclose(props["cv_j_mol_k"], expected_cv * EV_TO_J_MOL, rel_tol=1e-9)

    # 合理性: 每原子热容 > 0, 摩尔热容在同量级 (理想气体 ~12 J/mol/K 附近)
    assert props["cv_eV_K"] > 0
    assert 0 < props["cv_j_mol_k"] < 200

    # 均值/涨落对账
    assert math.isclose(props["avg_total"], float(np.mean(total)), rel_tol=1e-9)
    assert math.isclose(props["avg_kinetic"], 2.0)
    assert math.isclose(props["avg_potential"], float(np.mean(total - 2.0)), rel_tol=1e-9)
    assert math.isclose(props["avg_temperature"], 310.0)
    assert math.isclose(props["std_temperature"], 0.0, abs_tol=1e-12)
    assert math.isclose(props["std_total"], float(np.std(total)), rel_tol=1e-9)
    assert props["n_samples"] == 2000
    assert props["n_atoms"] == n_atoms
    assert props["method"] == "fluctuation_formula"
    assert "energy fluctuations" in props["note"]

    # T != T_ref (300K) 时 Helmholtz 应该算得出来
    assert props["helmholtz_free_energy_eV"] is not None
    assert math.isfinite(props["helmholtz_free_energy_eV"])


# ── 2. LAMMPS thermo_data 格式输入, 字段映射 ────────────────────────
def test_lammps_thermo_data_field_mapping():
    # 直接拿 LAMMPS _parse_log 输出的 lower-case 列名喂进来
    lammps_data = {
        "step": [0, 100, 200, 300, 400],
        "temp": [298.0, 300.0, 302.0, 301.0, 299.0],
        "kineng": [3.80, 3.85, 3.90, 3.87, 3.82],
        "poteng": [-10.00, -10.10, -9.95, -10.05, -10.00],
        "n_atoms": 64,
    }

    res = _run(ThermoToolInput(
        action="md_thermo",
        md_time_series=lammps_data,
    ))
    assert res.success, res.error
    props = res.data["properties"]

    temp = np.asarray(lammps_data["temp"], dtype=float)
    ke = np.asarray(lammps_data["kineng"], dtype=float)
    pe = np.asarray(lammps_data["poteng"], dtype=float)

    # 别名 temp/kineng/poteng 都正确映射到了对应的物理量
    assert math.isclose(props["avg_temperature"], float(np.mean(temp)))
    assert math.isclose(props["avg_kinetic"], float(np.mean(ke)))
    assert math.isclose(props["avg_potential"], float(np.mean(pe)))
    # 没给 toteng, 工具用 KE+PE 拼总能量
    assert math.isclose(props["avg_total"], float(np.mean(ke + pe)))
    assert props["n_samples"] == 5
    assert props["n_atoms"] == 64  # 从 md_time_series 里取的
    assert props["cv_eV_K"] is not None and props["cv_eV_K"] > 0


# ── 3. 能量涨落越大, Cv 越大 ─────────────────────────────────────────
def test_cv_scales_with_energy_fluctuation():
    n_atoms = 50
    # 同样均值温度, 一组能量噪声小, 一组大 (方差比 100:1)
    small = _make_series(n=3000, T=300.0, e_sigma=0.01, seed=2)
    large = _make_series(n=3000, T=300.0, e_sigma=0.10, seed=3)

    r_small = _run(ThermoToolInput(
        action="md_thermo", md_time_series=small, n_atoms=n_atoms
    ))
    r_large = _run(ThermoToolInput(
        action="md_thermo", md_time_series=large, n_atoms=n_atoms
    ))
    assert r_small.success and r_large.success

    cv_small = r_small.data["properties"]["cv_eV_K"]
    cv_large = r_large.data["properties"]["cv_eV_K"]
    assert cv_small is not None and cv_large is not None
    # 大涨落 -> 大 Cv; 噪声 std 差 10x, 方差差 100x, Cv 也应差 ~100x
    assert cv_large > cv_small
    assert cv_large > 10.0 * cv_small


# ── 4. 缺少 md_time_series 时优雅失败 ────────────────────────────────
def test_missing_md_time_series_fails_gracefully():
    # 完全不给 md_time_series -> schema 层直接 ValidationError
    with pytest.raises(ValidationError, match="md_time_series"):
        ThermoToolInput(action="md_thermo")

    # 给了空 dict 也算缺数据 -> schema 层拦截
    with pytest.raises(ValidationError, match="md_time_series"):
        ThermoToolInput(action="md_thermo", md_time_series={})

    # 给了 dict 但没有温度序列 -> 运行期优雅报错, 不抛异常
    res = _run(ThermoToolInput(
        action="md_thermo",
        md_time_series={"step": [0, 1, 2]},  # 只有 step, 没温度
    ))
    assert res.success is False
    assert res.data is None
    assert "temperature" in res.error
