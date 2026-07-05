"""Tests for mechanical_tool — stress / PM / thermal / fatigue / rolling.

Calls the private methods directly (they're synchronous) plus a couple of
async call() smoke tests to make sure dispatch works end-to-end.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from pydantic import ValidationError

from huginn.tools.sim.mechanical_tool import (
    MechanicalTool,
    MechanicalToolInput,
)


# ── stress_analysis ──────────────────────────────────────────────

def test_stress_analysis_uniaxial():
    """Cylinder uniaxial: σ = F/A, ε = σ/E."""
    args = MechanicalToolInput(
        action="stress_analysis",
        geometry_type="cylinder",
        dimensions={"d": 0.02, "L": 0.1},
        material_props={"E": 210e9, "nu": 0.3, "yield_strength": 250e6},
        load_type="uniaxial",
        load_value=10000.0,
    )
    tool = MechanicalTool()
    result = tool._stress_analysis(args)
    assert result.success is True

    A = math.pi * 0.02**2 / 4.0
    expected_sigma = 10000.0 / A
    expected_strain = expected_sigma / 210e9
    expected_def = expected_strain * 0.1

    assert result.data["max_stress"] == pytest.approx(expected_sigma, rel=1e-6)
    assert result.data["max_strain"] == pytest.approx(expected_strain, rel=1e-6)
    assert result.data["deformation"] == pytest.approx(expected_def, rel=1e-6)
    assert result.data["safety_factor"] == pytest.approx(250e6 / expected_sigma, rel=1e-6)


def test_stress_analysis_bending_plate():
    """Plate bending: σ = 6M/(w·t²)."""
    args = MechanicalToolInput(
        action="stress_analysis",
        geometry_type="plate",
        dimensions={"w": 0.05, "t": 0.01, "L": 0.2},
        material_props={"E": 210e9, "nu": 0.3, "yield_strength": 250e6},
        load_type="bending",
        load_value=50.0,
    )
    tool = MechanicalTool()
    result = tool._stress_analysis(args)
    assert result.success is True

    expected_sigma = 6.0 * 50.0 / (0.05 * 0.01**2)
    assert result.data["max_stress"] == pytest.approx(expected_sigma, rel=1e-6)
    assert result.data["safety_factor"] == pytest.approx(250e6 / expected_sigma, rel=1e-6)


def test_stress_analysis_torsion():
    """Cylinder torsion: τ = T·r/J = 16T/(πd³)."""
    args = MechanicalToolInput(
        action="stress_analysis",
        geometry_type="cylinder",
        dimensions={"d": 0.03, "L": 0.5},
        material_props={"E": 210e9, "nu": 0.3, "yield_strength": 250e6},
        load_type="torsion",
        load_value=500.0,
    )
    tool = MechanicalTool()
    result = tool._stress_analysis(args)
    assert result.success is True

    d = 0.03
    T = 500.0
    J = math.pi * d**4 / 32.0
    expected_tau = T * (d / 2.0) / J
    G = 210e9 / (2.0 * (1.0 + 0.3))
    expected_theta = T * 0.5 / (G * J)

    assert result.data["max_stress"] == pytest.approx(expected_tau, rel=1e-6)
    assert result.data["twist_angle"] == pytest.approx(expected_theta, rel=1e-6)
    assert result.data["polar_moment"] == pytest.approx(J, rel=1e-6)


# ── powder_metallurgy ─────────────────────────────────────────────

def test_powder_metallurgy_green_density():
    """Iron powder: compute green density from press force."""
    args = MechanicalToolInput(
        action="powder_metallurgy",
        powder_material="iron",
        die_diameter=0.02,
        die_height=0.04,
        press_force=200000.0,
        friction_coefficient=0.15,
    )
    tool = MechanicalTool()
    result = tool._powder_metallurgy(args)
    assert result.success is True

    A = math.pi * 0.02**2 / 4.0
    P_applied = 200000.0 / A
    P_green = P_applied / (1.0 + 0.15 * (0.04 / 0.02))
    K, D0 = 2.5e-9, 0.45
    D_green = 1.0 - (1.0 - D0) * math.exp(-K * P_green)
    rho_green = D_green * 7870.0

    assert result.data["applied_pressure"] == pytest.approx(P_applied, rel=1e-6)
    assert result.data["green_pressure"] == pytest.approx(P_green, rel=1e-6)
    assert result.data["green_density"] == pytest.approx(rho_green, rel=1e-6)
    assert result.data["relative_density"] == pytest.approx(D_green, rel=1e-6)
    # Sanity: green density should be between fill and theoretical
    assert D0 < D_green < 1.0


def test_powder_metallurgy_ejection_force():
    """Ejection force = μ · P_radial · A_wall."""
    args = MechanicalToolInput(
        action="powder_metallurgy",
        powder_material="iron",
        die_diameter=0.02,
        die_height=0.04,
        press_force=200000.0,
        friction_coefficient=0.15,
    )
    tool = MechanicalTool()
    result = tool._powder_metallurgy(args)
    assert result.success is True

    P_green = result.data["green_pressure"]
    P_radial = 0.4 * P_green
    A_wall = math.pi * 0.02 * 0.04
    expected_F_eject = 0.15 * P_radial * A_wall

    assert result.data["die_wall_pressure"] == pytest.approx(P_radial, rel=1e-6)
    assert result.data["ejection_force"] == pytest.approx(expected_F_eject, rel=1e-6)
    assert result.data["ejection_force"] > 0


def test_powder_metallurgy_target_density():
    """Target density given → compute required pressure via inverse Heckel."""
    args = MechanicalToolInput(
        action="powder_metallurgy",
        powder_material="iron",
        die_diameter=0.02,
        die_height=0.04,
        target_density=7000.0,
        friction_coefficient=0.15,
    )
    tool = MechanicalTool()
    result = tool._powder_metallurgy(args)
    assert result.success is True

    D_target = 7000.0 / 7870.0
    K, D0 = 2.5e-9, 0.45
    expected_P_green = -(1.0 / K) * math.log((1.0 - D_target) / (1.0 - D0))
    expected_P_applied = expected_P_green * (1.0 + 0.15 * (0.04 / 0.02))

    assert result.data["green_pressure"] == pytest.approx(expected_P_green, rel=1e-6)
    assert result.data["required_pressure"] == pytest.approx(expected_P_applied, rel=1e-6)
    assert result.data["green_density"] == pytest.approx(7000.0)


# ── thermal_stress ───────────────────────────────────────────────

def test_thermal_stress_plane_strain():
    """Constrained plate: σ = E·α·ΔT / (1-ν)."""
    args = MechanicalToolInput(
        action="thermal_stress",
        material_props={"E": 210e9, "nu": 0.3, "alpha": 12e-6},
        temperature_gradient=100.0,
        constraint_type="plane_strain",
        dimensions={"t": 0.01, "L": 0.5},
    )
    tool = MechanicalTool()
    result = tool._thermal_stress(args)
    assert result.success is True

    E, alpha, nu = 210e9, 12e-6, 0.3
    dT = 100.0
    expected_sigma = E * alpha * dT / (1.0 - nu)
    expected_eps = alpha * dT

    assert result.data["thermal_stress"] == pytest.approx(expected_sigma, rel=1e-6)
    assert result.data["thermal_strain"] == pytest.approx(expected_eps, rel=1e-6)

    # Buckling check
    sigma_cr = 4.0 * math.pi**2 * E / (12.0 * (1.0 - nu**2)) * (0.01 / 0.5) ** 2
    assert result.data["critical_buckling_stress"] == pytest.approx(sigma_cr, rel=1e-6)
    assert result.data["buckling_check"] == "likely"  # 360 MPa > ~304 MPa


# ── fatigue_life ──────────────────────────────────────────────────

def test_fatigue_life_goodman_infinite():
    """Goodman safety factor, infinite life when σ_a < S_e."""
    args = MechanicalToolInput(
        action="fatigue_life",
        fatigue_params={
            "S_ut": 900e6,
            "S_e": 450e6,
            "stress_amplitude": 200e6,
            "mean_stress": 200e6,
            "Kf": 1.0,
        },
        fatigue_criterion="goodman",
    )
    tool = MechanicalTool()
    result = tool._fatigue_life(args)
    assert result.success is True

    sa, sm = 200e6, 200e6
    expected_n = 1.0 / (sa / 450e6 + sm / 900e6)
    assert result.data["safety_factor"] == pytest.approx(expected_n, rel=1e-6)
    assert result.data["fatigue_life"] == float("inf")
    assert "infinite" in result.data["life_regime"]


def test_fatigue_life_finite_basquin():
    """Finite life via Basquin: N_f = 0.5·(σ_a/σ'_f)^(1/b)."""
    args = MechanicalToolInput(
        action="fatigue_life",
        fatigue_params={
            "S_ut": 900e6,
            "S_e": 300e6,
            "stress_amplitude": 600e6,
            "mean_stress": 0.0,
            "Kf": 1.0,
            "sigma_f_prime": 1800e6,
            "b": -0.085,
        },
        fatigue_criterion="asme_elliptical",
    )
    tool = MechanicalTool()
    result = tool._fatigue_life(args)
    assert result.success is True

    # ASME elliptical with σ_m=0: n = S_e/σ_a
    expected_n = 1.0 / math.sqrt((600e6 / 300e6) ** 2 + 0.0)
    assert result.data["safety_factor"] == pytest.approx(expected_n, rel=1e-6)

    # Basquin: σ_a = σ'_f·(2N_f)^b → N_f = 0.5·(σ_a/σ'_f)^(1/b)
    expected_N = 0.5 * (600e6 / 1800e6) ** (1.0 / -0.085)
    assert result.data["fatigue_life"] == pytest.approx(expected_N, rel=1e-6)
    assert result.data["life_regime"] == "finite"


# ── rolling_force ─────────────────────────────────────────────────

def test_rolling_force_basic():
    """Rolling force: F = σ·L·w·Q_p, L = √(R·Δh)."""
    args = MechanicalToolInput(
        action="rolling_force",
        roll_radius=0.4,
        reduction=0.002,
        width=1.0,
        flow_stress=300e6,
        friction_coefficient=0.15,
        roll_speed_rpm=100.0,
        initial_thickness=0.01,
    )
    tool = MechanicalTool()
    result = tool._rolling_force(args)
    assert result.success is True

    R, dh, w, sigma, mu = 0.4, 0.002, 1.0, 300e6, 0.15
    L = math.sqrt(R * dh)
    h_avg = (0.01 + (0.01 - dh)) / 2.0
    Q_p = 1.0 + mu * L / (2.0 * h_avg)
    expected_F = sigma * L * w * Q_p
    expected_torque = expected_F * L
    expected_power = expected_torque * 2.0 * math.pi * 100.0 / 60.0

    assert result.data["rolling_force"] == pytest.approx(expected_F, rel=1e-6)
    assert result.data["contact_length"] == pytest.approx(L, rel=1e-6)
    assert result.data["torque"] == pytest.approx(expected_torque, rel=1e-6)
    assert result.data["power"] == pytest.approx(expected_power, rel=1e-6)
    assert result.data["min_rollable_thickness"] > 0
    # Force should be in the MN range for a steel rolling pass
    assert 1e6 < result.data["rolling_force"] < 100e6


# ── input validation ─────────────────────────────────────────────

def test_unknown_action_raises():
    """Invalid action string should be rejected by pydantic Literal."""
    with pytest.raises(ValidationError):
        MechanicalToolInput(action="not_a_real_action")


def test_missing_required_fields_stress():
    """stress_analysis without geometry_type should fail validation."""
    with pytest.raises(ValidationError):
        MechanicalToolInput(
            action="stress_analysis",
            material_props={"E": 210e9},
            load_type="uniaxial",
            load_value=1000.0,
        )


def test_missing_required_fields_rolling():
    """rolling_force without flow_stress should fail validation."""
    with pytest.raises(ValidationError):
        MechanicalToolInput(
            action="rolling_force",
            roll_radius=0.4,
            reduction=0.002,
            width=1.0,
            friction_coefficient=0.15,
        )


# ── async call() smoke tests ─────────────────────────────────────

def test_async_call_stress_analysis():
    """async call() dispatches to _stress_analysis correctly."""
    tool = MechanicalTool()
    args = MechanicalToolInput(
        action="stress_analysis",
        geometry_type="cylinder",
        dimensions={"d": 0.02, "L": 0.1},
        material_props={"E": 210e9, "nu": 0.3, "yield_strength": 250e6},
        load_type="uniaxial",
        load_value=10000.0,
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert "max_stress" in result.data


def test_async_call_rolling_force():
    """async call() dispatches to _rolling_force correctly."""
    tool = MechanicalTool()
    args = MechanicalToolInput(
        action="rolling_force",
        roll_radius=0.4,
        reduction=0.002,
        width=1.0,
        flow_stress=300e6,
        friction_coefficient=0.15,
    )
    result = asyncio.run(tool.call(args, context=None))
    assert result.success is True
    assert "rolling_force" in result.data


# ── tool registration ─────────────────────────────────────────────

def test_tool_metadata():
    """Tool class has correct static attributes."""
    tool = MechanicalTool()
    assert tool.name == "mechanical_tool"
    assert tool.category == "sim"
    assert tool.read_only is True
    cost = tool.estimate_cost(None)
    assert cost is not None
    assert cost["cpu_hours"] == 0.0
