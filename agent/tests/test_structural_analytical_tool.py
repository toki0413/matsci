"""Tests for structural_analytical_tool — beam/plate/shell/stress_concentration.

直接调底层函数 (beam_static / plate_modal / ...) 而非 async call(),
因为 call() 只是 lazy import + 分发, 逻辑层全在同步函数里.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest
from pydantic import ValidationError

from huginn.tools.structural_analytical import (
    StructuralAnalyticalInput,
    StructuralAnalyticalTool,
)
from huginn.tools.structural_analytical.beams import (
    beam_buckling,
    beam_modal,
    beam_static,
)
from huginn.tools.structural_analytical.plates import (
    plate_buckling,
    plate_modal,
    plate_static,
)
from huginn.tools.structural_analytical.shells import shell_buckling, shell_modal
from huginn.tools.structural_analytical.stress_concentration import (
    stress_concentration,
)


# ── beam ──

def test_beam_static_cantilever_point():
    """悬臂梁端部集中力: δ_tip = P L³ / (3 E I)."""
    args = StructuralAnalyticalInput(
        action="beam_static",
        theory="euler_bernoulli",
        boundary="cantilever",
        beam={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "length": 1.0,
            "section_type": "rectangular",
            "section_dims": {"b": 0.02, "h": 0.04},
            "density": 7850.0,
        },
        loads=[{"type": "point", "value": -100.0, "position": 1.0}],
        n_points=101,
    )
    result = beam_static(args)
    assert result.success is True

    E = 210e9
    I = 0.02 * 0.04**3 / 12.0
    L = 1.0
    P = 100.0
    expected_tip = P * L**3 / (3.0 * E * I)
    actual_tip = abs(result.data["max_deflection"])
    # 端部挠度应接近解析解 (数值积分有误差, 容差 5%)
    assert actual_tip == pytest.approx(expected_tip, rel=0.05)


def test_beam_modal_simply_supported():
    """简支梁一阶频率: ω₁ = π² √(EI/(ρ A L⁴))."""
    args = StructuralAnalyticalInput(
        action="beam_modal",
        theory="euler_bernoulli",
        boundary="simply_supported",
        n_modes=3,
        beam={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "length": 2.0,
            "section_type": "rectangular",
            "section_dims": {"b": 0.02, "h": 0.04},
            "density": 7850.0,
        },
    )
    result = beam_modal(args)
    assert result.success is True
    assert len(result.data["frequencies_hz"]) == 3

    E = 210e9
    I = 0.02 * 0.04**3 / 12.0
    A = 0.02 * 0.04
    rho = 7850.0
    L = 2.0
    expected_omega1 = math.pi**2 * math.sqrt(E * I / (rho * A * L**4))
    expected_f1 = expected_omega1 / (2 * math.pi)
    assert result.data["frequencies_hz"][0] == pytest.approx(expected_f1, rel=0.01)


def test_beam_buckling_euler():
    """简支梁 Euler 屈曲: P_cr = π² E I / L² (K=1)."""
    args = StructuralAnalyticalInput(
        action="beam_buckling",
        boundary="simply_supported",
        beam={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "length": 2.0,
            "section_type": "rectangular",
            "section_dims": {"b": 0.02, "h": 0.04},
            "density": 7850.0,
        },
    )
    result = beam_buckling(args)
    assert result.success is True

    E = 210e9
    I = 0.02 * 0.04**3 / 12.0
    L = 2.0
    expected_Pcr = math.pi**2 * E * I / L**2
    assert result.data["critical_load"] == pytest.approx(expected_Pcr, rel=0.01)
    assert result.data["k_factor"] == pytest.approx(1.0)


# ── plate ──

def test_plate_static_rectangular_uniform():
    """简支矩形板均布载荷: w_max ≈ 0.00406 q a⁴ / D (Timoshenko Table 8)."""
    args = StructuralAnalyticalInput(
        action="plate_static",
        theory="kirchhoff",
        plate={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "thickness": 0.01,
            "density": 7850.0,
        },
        plate_shape="rectangular",
        plate_dims={"a": 0.5, "b": 0.5},
        plate_bc="simply_supported",
        plate_load={"type": "uniform", "magnitude": 1000.0},
    )
    result = plate_static(args)
    assert result.success is True

    E = 210e9
    nu = 0.3
    h = 0.01
    a = 0.5
    q = 1000.0
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    # Timoshenko Table 8, 简支方板均布: w_max = 0.00406 q a^4 / D
    expected_wmax = 0.00406 * q * a**4 / D
    assert abs(result.data["max_deflection"]) == pytest.approx(expected_wmax, rel=0.05)


def test_plate_modal_fundamental():
    """简支方板一阶频率: ω₁₁ = π² √(D/(ρh)) · (1/a² + 1/b²)."""
    args = StructuralAnalyticalInput(
        action="plate_modal",
        theory="kirchhoff",
        n_modes=5,
        plate={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "thickness": 0.01,
            "density": 7850.0,
        },
        plate_shape="rectangular",
        plate_dims={"a": 0.5, "b": 0.5},
        plate_bc="simply_supported",
    )
    result = plate_modal(args)
    assert result.success is True

    E = 210e9
    nu = 0.3
    h = 0.01
    rho = 7850.0
    a = 0.5
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    expected_omega11 = math.pi**2 * math.sqrt(D / (rho * h)) * (1.0 / a**2 + 1.0 / a**2)
    expected_f11 = expected_omega11 / (2 * math.pi)
    assert result.data["frequencies_hz"][0] == pytest.approx(expected_f11, rel=0.02)


# ── shell ──

def test_shell_buckling_axial():
    """圆柱壳轴向屈曲: σ_cr = knockdown · E h / (R √(3(1-ν²)))."""
    args = StructuralAnalyticalInput(
        action="shell_buckling",
        theory="donnell",
        shell={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "radius": 0.5,
            "length": 2.0,
            "thickness": 0.002,
            "density": 7850.0,
        },
        shell_load_type="axial",
    )
    result = shell_buckling(args)
    assert result.success is True

    E = 210e9
    nu = 0.3
    R = 0.5
    h = 0.002
    knockdown = 0.65
    expected_sigma = knockdown * E * h / (R * math.sqrt(3.0 * (1.0 - nu**2)))
    assert result.data["critical_stress"] == pytest.approx(expected_sigma, rel=0.01)
    assert result.data["knockdown_factor"] == pytest.approx(0.65)


def test_shell_buckling_external_pressure():
    """外压屈曲: 扫 (m,n) 求最小 p_cr, 应返回正值和模态."""
    args = StructuralAnalyticalInput(
        action="shell_buckling",
        theory="donnell",
        shell={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "radius": 0.5,
            "length": 2.0,
            "thickness": 0.002,
            "density": 7850.0,
        },
        shell_load_type="external_pressure",
    )
    result = shell_buckling(args)
    assert result.success is True
    assert result.data["critical_pressure"] > 0
    assert "buckling_mode" in result.data
    assert result.data["buckling_mode"]["n"] >= 2


def test_shell_modal_fundamental():
    """圆柱壳模态: 应返回正频率, m≥1, n≥0."""
    args = StructuralAnalyticalInput(
        action="shell_modal",
        theory="donnell",
        n_modes=5,
        shell={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "radius": 0.5,
            "length": 2.0,
            "thickness": 0.002,
            "density": 7850.0,
        },
    )
    result = shell_modal(args)
    assert result.success is True
    assert len(result.data["frequencies_hz"]) == 5
    assert all(f > 0 for f in result.data["frequencies_hz"])
    # 频率应升序
    freqs = result.data["frequencies_hz"]
    assert freqs == sorted(freqs)


# ── stress concentration ──

def test_stress_concentration_hole():
    """圆孔单向拉伸: Kt=3 (无限大板)."""
    args = StructuralAnalyticalInput(
        action="stress_concentration",
        notch_type="hole",
        notch_geometry={"d": 0.01, "D": 1.0},  # d/D=0.01 接近无限大板
        notch_load="tension",
    )
    result = stress_concentration(args)
    assert result.success is True
    # d/D 很小时 Kt 接近 3
    assert result.data["stress_concentration_factor"] == pytest.approx(3.0, abs=0.1)


def test_stress_concentration_elliptical():
    """椭圆孔: Kt = 1 + 2(a/b)."""
    args = StructuralAnalyticalInput(
        action="stress_concentration",
        notch_type="elliptical_hole",
        notch_geometry={"a": 0.02, "b": 0.01},  # a/b=2, Kt=5
        notch_load="tension",
    )
    result = stress_concentration(args)
    assert result.success is True
    expected_Kt = 1.0 + 2.0 * (0.02 / 0.01)
    assert result.data["stress_concentration_factor"] == pytest.approx(expected_Kt)


# ── input validation ──

def test_input_validation_missing_spec():
    """缺 beam spec 应报错."""
    with pytest.raises(ValidationError):
        StructuralAnalyticalInput(action="beam_static", boundary="cantilever")


def test_input_validation_theory_mismatch():
    """beam action 配 kirchhoff theory 应报错."""
    with pytest.raises(ValidationError):
        StructuralAnalyticalInput(
            action="beam_static",
            theory="kirchhoff",  # beam 只接受 euler_bernoulli/timoshenko
            boundary="cantilever",
            beam={
                "youngs_modulus": 210e9,
                "length": 1.0,
                "section_type": "rectangular",
                "section_dims": {"b": 0.02, "h": 0.04},
            },
            loads=[{"type": "point", "value": -100.0, "position": 1.0}],
        )


def test_input_validation_shell_theory():
    """shell action 配 euler_bernoulli theory 应报错."""
    with pytest.raises(ValidationError):
        StructuralAnalyticalInput(
            action="shell_buckling",
            theory="euler_bernoulli",  # shell 只接受 donnell/flugge
            shell={
                "youngs_modulus": 210e9,
                "radius": 0.5,
                "length": 2.0,
                "thickness": 0.002,
            },
            shell_load_type="axial",
        )


# ── tool registration & async call smoke ──

def test_tool_registration():
    """工具类基本属性."""
    tool = StructuralAnalyticalTool()
    assert tool.name == "structural_analytical_tool"
    assert tool.category == "sim"
    assert tool.read_only is True
    assert tool.destructive is False
    cost = tool.estimate_cost(None)
    assert cost is not None
    assert cost["cpu_hours"] == 0.0


def test_async_call_dispatch():
    """async call() 能正确分发到 beam_buckling."""
    tool = StructuralAnalyticalTool()
    args = StructuralAnalyticalInput(
        action="beam_buckling",
        boundary="simply_supported",
        beam={
            "youngs_modulus": 210e9,
            "poissons_ratio": 0.3,
            "length": 2.0,
            "section_type": "rectangular",
            "section_dims": {"b": 0.02, "h": 0.04},
            "density": 7850.0,
        },
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert "critical_load" in result.data
