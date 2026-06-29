"""Tests for specialty_analysis_tool — buckling/modal/fatigue/fracture."""

import asyncio
import math

import pytest
from pydantic import ValidationError

from huginn.tools.specialty_analysis import (
    SpecialtyAnalysisInput,
    SpecialtyAnalysisTool,
)
from huginn.tools.specialty_analysis.buckling import eigenvalue_buckling
from huginn.tools.specialty_analysis.fatigue import (
    fatigue_crack_growth,
    fatigue_sn,
)
from huginn.tools.specialty_analysis.fracture import fracture_lefm
from huginn.tools.specialty_analysis.modal import modal_lanczos


# ── eigenvalue_buckling ──


def test_eigenvalue_buckling_simple_column():
    """K=[[1,0],[0,1]], KG=[[0.1,0],[0,0.1]] → λ_cr = 0.1.

    eigh(KG, K) 解 (KG) φ = λ (K) φ. KG=0.1 I, K=I → λ = 0.1.
    物理解释: KG 代表单位载荷下的几何刚度, λ_cr=0.1 表示 0.1 倍单位载荷即屈曲.
    """
    args = SpecialtyAnalysisInput(
        action="eigenvalue_buckling",
        stiffness_matrix=[[1.0, 0.0], [0.0, 1.0]],
        geometric_stiffness=[[0.1, 0.0], [0.0, 0.1]],
        num_modes=2,
    )
    result = eigenvalue_buckling(args)
    assert result.success is True
    factors = result.data["critical_load_factors"]
    assert len(factors) == 2
    assert factors[0] == pytest.approx(0.1, rel=1e-6)
    assert factors[1] == pytest.approx(0.1, rel=1e-6)


def test_eigenvalue_buckling_shape_mismatch():
    args = SpecialtyAnalysisInput(
        action="eigenvalue_buckling",
        stiffness_matrix=[[1.0, 0.0], [0.0, 1.0]],
        geometric_stiffness=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],  # 2x3 错误
        num_modes=1,
    )
    result = eigenvalue_buckling(args)
    assert result.success is False
    assert "shape" in result.error.lower()


# ── modal_lanczos ──


def test_modal_lanczos_2dof():
    """K=[[2,-1],[-1,1]], M=I → ω² = (3±√5)/2 ≈ {0.382, 2.618}.

    经典 2-DOF 弹簧-质量系统, 特征方程 λ² - 3λ + 1 = 0.
    """
    args = SpecialtyAnalysisInput(
        action="modal_lanczos",
        stiffness_matrix=[[2.0, -1.0], [-1.0, 1.0]],
        mass_matrix=[[1.0, 0.0], [0.0, 1.0]],
        num_modes=2,
    )
    result = modal_lanczos(args)
    assert result.success is True
    omegas = result.data["angular_frequencies_rad_s"]
    assert len(omegas) == 2
    expected_omega1_sq = (3.0 - math.sqrt(5.0)) / 2.0
    expected_omega1 = math.sqrt(expected_omega1_sq)
    assert omegas[0] == pytest.approx(expected_omega1, rel=1e-4)
    expected_omega2_sq = (3.0 + math.sqrt(5.0)) / 2.0
    expected_omega2 = math.sqrt(expected_omega2_sq)
    assert omegas[1] == pytest.approx(expected_omega2, rel=1e-4)


# ── fatigue_sn ──


def test_fatigue_sn_basquin():
    """Basquin 零均值: σ_f'=1100MPa, b=-0.1, σ_a=200MPa."""
    args = SpecialtyAnalysisInput(
        action="fatigue_sn",
        sn_params={"sigma_f_prime": 1100e6, "b": -0.1},
        stress_amplitude=200e6,
        mean_stress=0.0,
        mean_stress_theory="goodman",
        cycles_limit=10**6,
    )
    result = fatigue_sn(args)
    assert result.success is True
    n_f = result.data["fatigue_life_cycles"]
    expected = 0.5 * (200.0 / 1100.0) ** (1.0 / -0.1)
    assert n_f == pytest.approx(expected, rel=1e-6)
    assert n_f > 0


def test_fatigue_sn_goodman_reduces_life():
    """Goodman 修正: σ_m > 0 → σ_a,eq < σ_a → N_f 增大."""
    args_zero = SpecialtyAnalysisInput(
        action="fatigue_sn",
        sn_params={"sigma_f_prime": 1100e6, "b": -0.1},
        stress_amplitude=200e6,
        mean_stress=0.0,
        mean_stress_theory="goodman",
        material_uts=800e6,
        cycles_limit=10**6,
    )
    args_mean = SpecialtyAnalysisInput(
        action="fatigue_sn",
        sn_params={"sigma_f_prime": 1100e6, "b": -0.1},
        stress_amplitude=200e6,
        mean_stress=200e6,
        mean_stress_theory="goodman",
        material_uts=800e6,
        cycles_limit=10**6,
    )
    n_zero = fatigue_sn(args_zero).data["fatigue_life_cycles"]
    n_mean = fatigue_sn(args_mean).data["fatigue_life_cycles"]
    assert n_mean > n_zero
    assert fatigue_sn(args_mean).data["correction_factor"] == pytest.approx(0.75, rel=1e-6)


# ── fatigue_crack_growth ──


def test_fatigue_crack_growth_paris():
    """C=1e-12, m=3, ΔK=10 MPa√m, a: 0.5→5mm. 恒 ΔK 解析解."""
    args = SpecialtyAnalysisInput(
        action="fatigue_crack_growth",
        paris_params={"C": 1e-12, "m": 3.0},
        dk_range=10.0e6,
        a_init=0.5e-3,
        a_final=5.0e-3,
    )
    result = fatigue_crack_growth(args)
    assert result.success is True
    n_f = result.data["fatigue_life_cycles"]
    expected_growth = 1e-12 * (10.0e6) ** 3.0
    expected_n = (5.0e-3 - 0.5e-3) / expected_growth
    assert n_f == pytest.approx(expected_n, rel=1e-6)
    assert n_f > 0


def test_fatigue_crack_growth_a_final_le_a_init():
    with pytest.raises(ValidationError):
        SpecialtyAnalysisInput(
            action="fatigue_crack_growth",
            paris_params={"C": 1e-12, "m": 3.0},
            dk_range=10.0e6,
            a_init=5.0e-3,
            a_final=0.5e-3,
        )


# ── fracture_lefm ──


def test_fracture_lefm_edge_crack():
    """Y=1.12, σ=100MPa, a=10mm → K_I ≈ 19.9 MPa√m, safe vs K_IC=80."""
    args = SpecialtyAnalysisInput(
        action="fracture_lefm",
        crack_type="edge",
        crack_length=0.01,
        applied_stress=100e6,
        geometry_factor=1.12,
        k_ic=80e6,
        youngs_modulus=210e9,
        poissons_ratio=0.3,
    )
    result = fracture_lefm(args)
    assert result.success is True
    k_i = result.data["stress_intensity_factor_ki"]
    expected_ki = 1.12 * 100e6 * math.sqrt(math.pi * 0.01)
    assert k_i == pytest.approx(expected_ki, rel=1e-6)
    assert result.data["assessment"] == "safe"
    assert result.data["safety_factor"] == pytest.approx(80e6 / expected_ki, rel=1e-6)
    e_prime = 210e9 / (1.0 - 0.3**2)
    expected_j = expected_ki**2 / e_prime
    assert result.data["j_integral"] == pytest.approx(expected_j, rel=1e-6)


def test_fracture_lefm_unstable():
    """K_IC < K_I → unstable_propagation."""
    args = SpecialtyAnalysisInput(
        action="fracture_lefm",
        crack_type="edge",
        crack_length=0.05,
        applied_stress=200e6,
        geometry_factor=1.12,
        k_ic=30e6,
    )
    result = fracture_lefm(args)
    assert result.success is True
    assert result.data["assessment"] == "unstable_propagation"


# ── input validation ──


def test_input_validation_missing_matrix():
    with pytest.raises(ValidationError):
        SpecialtyAnalysisInput(action="eigenvalue_buckling", num_modes=3)


def test_input_validation_missing_sn_params():
    with pytest.raises(ValidationError):
        SpecialtyAnalysisInput(
            action="fatigue_sn", stress_amplitude=100e6
        )


def test_input_validation_paris_a_final_le_a_init():
    with pytest.raises(ValidationError):
        SpecialtyAnalysisInput(
            action="fatigue_crack_growth",
            paris_params={"C": 1e-12, "m": 3.0},
            dk_range=10.0,
            a_init=1.0,
            a_final=0.5,
        )


# ── tool registration & async dispatch ──


def test_tool_registration():
    """specialty_analysis_tool 注册到 registry."""
    from huginn.tools import register_all_tools
    registered = register_all_tools()
    assert "specialty_analysis_tool" in registered or any(
        "specialty" in r for r in registered
    )


def test_async_call_dispatch():
    """async call 走 SpecialtyAnalysisTool.call 应该正确分发到 fracture_lefm."""
    tool = SpecialtyAnalysisTool()
    args = SpecialtyAnalysisInput(
        action="fracture_lefm",
        crack_type="interior",
        crack_length=0.02,
        applied_stress=50e6,
        geometry_factor=1.0,
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    expected = 1.0 * 50e6 * math.sqrt(math.pi * 0.02)
    assert result.data["stress_intensity_factor_ki"] == pytest.approx(expected, rel=1e-6)
