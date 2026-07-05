"""Mechanical engineering calculator tool.

Quick analytical calculators for stress analysis, powder metallurgy,
thermal stress, fatigue life, and rolling force. No FEM — these are
closed-form solutions for back-of-envelope design checks.

Formula references:
- Shigley's Mechanical Engineering Design (fatigue, stress)
- German, Powder Metallurgy Science (compaction)
- Bland & Ford, Cold Rolling with Strip Tension (rolling force)
- Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells (thermal)
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult


# Common powder materials: theoretical density (kg/m³), compressibility K (Pa⁻¹),
# fill relative density D0. K comes from fitting D = 1-(1-D0)·exp(-K·P) to
# typical industrial compaction curves. Rough but usable for a first estimate.
_POWDER_DATA: dict[str, dict[str, float]] = {
    "iron": {"rho": 7870.0, "K": 2.5e-9, "D0": 0.45},
    "copper": {"rho": 8960.0, "K": 3.0e-9, "D0": 0.50},
    "aluminum": {"rho": 2700.0, "K": 4.0e-9, "D0": 0.45},
    "titanium": {"rho": 4500.0, "K": 2.0e-9, "D0": 0.40},
    "tungsten": {"rho": 19300.0, "K": 1.5e-9, "D0": 0.30},
    "steel": {"rho": 7850.0, "K": 2.5e-9, "D0": 0.45},
    "nickel": {"rho": 8908.0, "K": 2.8e-9, "D0": 0.42},
}

# Steel rolls by default (E in Pa)
_ROLL_E = 2.1e11


class MechanicalToolInput(BaseModel):
    """Input for all mechanical engineering calculations.

    Fields are grouped by action. Only the fields relevant to the
    chosen action need to be populated.
    """

    action: Literal[
        "stress_analysis",
        "powder_metallurgy",
        "thermal_stress",
        "fatigue_life",
        "rolling_force",
    ] = Field(..., description="Which calculation to run")

    # ── stress_analysis ──
    geometry_type: Literal["cylinder", "plate", "beam"] | None = None
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="cylinder: {d, L}; plate: {w, t, L}; beam: {b, h, L} (SI: m)",
    )
    material_props: dict[str, float] = Field(
        default_factory=dict,
        description="E (Pa), nu, yield_strength (Pa)",
    )
    load_type: Literal["uniaxial", "biaxial", "bending", "torsion"] | None = None
    load_value: float | None = Field(
        default=None,
        description="N (uniaxial), Pa (biaxial), N·m (bending/torsion)",
    )

    # ── powder_metallurgy ──
    powder_material: str | None = None
    theoretical_density: float | None = Field(
        default=None, gt=0, description="kg/m³, overrides lookup if given"
    )
    target_density: float | None = Field(
        default=None, gt=0, description="kg/m³ — green density target"
    )
    die_diameter: float | None = Field(default=None, gt=0, description="m")
    die_height: float | None = Field(
        default=None, gt=0, description="m, compaction height (fill height)"
    )
    press_force: float | None = Field(default=None, gt=0, description="N")
    friction_coefficient: float | None = Field(
        default=None, ge=0, le=1, description="μ, die wall friction"
    )

    # ── thermal_stress ──
    temperature_gradient: float | None = Field(
        default=None, description="ΔT in K (positive = heating)"
    )
    constraint_type: Literal[
        "fully_constrained", "one_direction", "plane_strain"
    ] | None = None

    # ── fatigue_life ──
    fatigue_params: dict[str, float] = Field(
        default_factory=dict,
        description="S_ut, S_e, stress_amplitude, mean_stress, Kf, S_y, sigma_f_prime, b",
    )
    fatigue_criterion: Literal[
        "goodman", "soderberg", "asme_elliptical"
    ] = "goodman"

    # ── rolling_force ──
    roll_radius: float | None = Field(default=None, gt=0, description="m")
    reduction: float | None = Field(default=None, gt=0, description="Δh, m")
    width: float | None = Field(default=None, gt=0, description="strip width, m")
    flow_stress: float | None = Field(default=None, gt=0, description="Pa")
    roll_speed_rpm: float | None = Field(default=None, ge=0, description="rpm")
    initial_thickness: float | None = Field(
        default=None, gt=0, description="h0, m — needed for friction hill factor"
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "MechanicalToolInput":
        """Validate that the action has the fields it needs."""
        if self.action == "stress_analysis":
            missing = []
            if self.geometry_type is None:
                missing.append("geometry_type")
            if not self.dimensions:
                missing.append("dimensions")
            if not self.material_props or "E" not in self.material_props:
                missing.append("material_props (needs E)")
            if self.load_type is None:
                missing.append("load_type")
            if self.load_value is None:
                missing.append("load_value")
            if missing:
                raise ValueError(
                    f"stress_analysis missing required fields: {', '.join(missing)}"
                )

        elif self.action == "powder_metallurgy":
            missing = []
            if self.powder_material is None and self.theoretical_density is None:
                missing.append("powder_material or theoretical_density")
            if self.die_diameter is None:
                missing.append("die_diameter")
            if self.die_height is None:
                missing.append("die_height")
            if self.friction_coefficient is None:
                missing.append("friction_coefficient")
            if self.press_force is None and self.target_density is None:
                missing.append("press_force or target_density")
            if missing:
                raise ValueError(
                    f"powder_metallurgy missing required fields: {', '.join(missing)}"
                )

        elif self.action == "thermal_stress":
            missing = []
            if not self.material_props or "E" not in self.material_props:
                missing.append("material_props (needs E)")
            if not self.material_props or "alpha" not in self.material_props:
                missing.append("material_props (needs alpha)")
            if self.temperature_gradient is None:
                missing.append("temperature_gradient")
            if self.constraint_type is None:
                missing.append("constraint_type")
            if missing:
                raise ValueError(
                    f"thermal_stress missing required fields: {', '.join(missing)}"
                )

        elif self.action == "fatigue_life":
            fp = self.fatigue_params
            missing = []
            if "S_ut" not in fp:
                missing.append("fatigue_params.S_ut")
            if "S_e" not in fp:
                missing.append("fatigue_params.S_e")
            if "stress_amplitude" not in fp:
                missing.append("fatigue_params.stress_amplitude")
            if missing:
                raise ValueError(
                    f"fatigue_life missing required fields: {', '.join(missing)}"
                )

        elif self.action == "rolling_force":
            missing = []
            if self.roll_radius is None:
                missing.append("roll_radius")
            if self.reduction is None:
                missing.append("reduction")
            if self.width is None:
                missing.append("width")
            if self.flow_stress is None:
                missing.append("flow_stress")
            if self.friction_coefficient is None:
                missing.append("friction_coefficient")
            if missing:
                raise ValueError(
                    f"rolling_force missing required fields: {', '.join(missing)}"
                )

        return self


class MechanicalTool(HuginnTool):
    """Analytical calculators for mechanical engineering design checks.

    All formulas are closed-form — no FEM, no iteration. Good for
    quick sizing and sanity checks before running a full simulation.
    """

    name = "mechanical_tool"
    category = "sim"
    profile = ToolProfile(constraint_scope="mechanical")
    description = (
        "Mechanical engineering calculator: stress analysis (uniaxial/bending/"
        "torsion), powder metallurgy compaction, thermal stress, fatigue life "
        "(Goodman/Soderberg/ASME), and rolling/extrusion force. Pure analytical "
        "formulas, no external solver."
    )
    read_only = True
    input_schema = MechanicalToolInput

    def estimate_cost(self, args: MechanicalToolInput) -> dict[str, float] | None:
        return {"cpu_hours": 0.0, "walltime_hours": 0.01}

    async def call(
        self, args: MechanicalToolInput, context: ToolContext
    ) -> ToolResult:
        dispatch = {
            "stress_analysis": self._stress_analysis,
            "powder_metallurgy": self._powder_metallurgy,
            "thermal_stress": self._thermal_stress,
            "fatigue_life": self._fatigue_life,
            "rolling_force": self._rolling_force,
        }
        handler = dispatch.get(args.action)
        if handler is None:
            return ToolResult(
                data=None, success=False,
                error=f"Unknown action: {args.action}",
            )
        try:
            return handler(args)
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"{args.action} failed: {exc}",
            )

    # ── stress_analysis ───────────────────────────────────────────

    def _stress_analysis(self, args: MechanicalToolInput) -> ToolResult:
        """Simple geometry stress: uniaxial, biaxial, bending, torsion."""
        import numpy as np

        E = args.material_props["E"]
        nu = args.material_props.get("nu", 0.3)
        sy = args.material_props.get("yield_strength", 0.0)
        gtype = args.geometry_type
        dims = args.dimensions
        ltype = args.load_type
        val = args.load_value

        result: dict[str, Any] = {
            "geometry": gtype,
            "load_type": ltype,
            "formula": "",
        }

        if ltype == "uniaxial":
            if gtype == "cylinder":
                d = dims["d"]
                A = math.pi * d**2 / 4.0
            elif gtype == "plate":
                A = dims["w"] * dims["t"]
            elif gtype == "beam":
                A = dims["b"] * dims["h"]
            else:
                return _err(f"uniaxial not supported for {gtype}")
            sigma = val / A
            strain = sigma / E
            L = dims.get("L", 0.0)
            deformation = strain * L
            result["formula"] = "σ = F/A"
            result.update({
                "area": A,
                "max_stress": sigma,
                "max_strain": strain,
                "deformation": deformation,
            })

        elif ltype == "biaxial":
            # load_value is the applied stress (Pa) in both directions
            sigma = val
            # von Mises for equal biaxial: σ_eq = σ
            result["formula"] = "σ_vm = σ (equal biaxial)"
            result.update({
                "max_stress": sigma,
                "max_strain": sigma / E,
                "deformation": 0.0,
            })

        elif ltype == "bending":
            if gtype == "plate":
                w = dims.get("w", 1.0)
                t = dims["t"]
                # σ = 6M/(w·t²) — total moment with plate width
                sigma = 6.0 * val / (w * t**2)
                result["formula"] = "σ = 6M/(w·t²)"
            elif gtype == "beam":
                b = dims["b"]
                h = dims["h"]
                # σ = 6M/(b·h²) — rectangular cross-section
                sigma = 6.0 * val / (b * h**2)
                result["formula"] = "σ = 6M/(b·h²)"
            elif gtype == "cylinder":
                d = dims["d"]
                # Solid circular: σ = 32M/(π·d³)
                sigma = 32.0 * val / (math.pi * d**3)
                result["formula"] = "σ = 32M/(π·d³)"
            else:
                return _err(f"bending not supported for {gtype}")
            strain = sigma / E
            result.update({
                "max_stress": sigma,
                "max_strain": strain,
                "deformation": 0.0,
            })

        elif ltype == "torsion":
            if gtype != "cylinder":
                return _err("torsion only supported for cylinder geometry")
            d = dims["d"]
            L = dims.get("L", 0.0)
            T = val
            # Polar moment J = πd⁴/32, τ_max = T·r/J = 16T/(πd³)
            J = math.pi * d**4 / 32.0
            r = d / 2.0
            tau = T * r / J
            G = E / (2.0 * (1.0 + nu))
            theta = T * L / (G * J)  # twist angle, radians
            result["formula"] = "τ = T·r/J, θ = TL/(GJ)"
            result.update({
                "polar_moment": J,
                "max_stress": tau,
                "max_strain": tau / G,
                "twist_angle": theta,
                "deformation": theta,
            })

        # safety factor
        if sy > 0 and result.get("max_stress", 0) > 0:
            result["safety_factor"] = sy / result["max_stress"]
        else:
            result["safety_factor"] = None

        return ToolResult(data=result, success=True)

    # ── powder_metallurgy ──────────────────────────────────────────

    def _powder_metallurgy(self, args: MechanicalToolInput) -> ToolResult:
        """Uniaxial die compaction: pressure, green density, ejection force.

        Compaction model: D = 1 - (1-D0)·exp(-K·P)  (modified Heckel)
        Friction loss: P_applied = P_green·(1 + μ·L/D)
        """
        import numpy as np  # noqa: F401 — kept for consistency with other actions

        # Material data
        if args.theoretical_density is not None:
            rho_th = args.theoretical_density
            K = 2.5e-9  # default compressibility (iron-like)
            D0 = 0.45
        else:
            mat = (args.powder_material or "").lower()
            data = _POWDER_DATA.get(mat)
            if data is None:
                return _err(
                    f"Unknown powder material '{args.powder_material}'. "
                    f"Known: {list(_POWDER_DATA)}"
                )
            rho_th = data["rho"]
            K = data["K"]
            D0 = data["D0"]

        d = args.die_diameter
        H = args.die_height  # fill height
        mu = args.friction_coefficient

        A_die = math.pi * d**2 / 4.0
        L_over_D = H / d  # aspect ratio for friction

        result: dict[str, Any] = {
            "powder_material": args.powder_material,
            "theoretical_density": rho_th,
            "die_diameter": d,
            "fill_height": H,
            "friction_coefficient": mu,
        }

        # Case 1: target density given → compute required pressure
        if args.target_density is not None:
            D_target = args.target_density / rho_th
            if D_target >= 1.0:
                return _err("target_density exceeds theoretical density")
            if D_target <= D0:
                return _err(
                    f"target relative density {D_target:.3f} <= fill density D0={D0:.3f}"
                )
            # Inverse Heckel: P = -(1/K)·ln((1-D)/(1-D0))
            P_green = -(1.0 / K) * math.log((1.0 - D_target) / (1.0 - D0))
            P_applied = P_green * (1.0 + mu * L_over_D)
            F_required = P_applied * A_die
            result["required_pressure"] = P_applied
            result["green_pressure"] = P_green
            result["required_force"] = F_required
            result["green_density"] = args.target_density
            result["relative_density"] = D_target
        else:
            P_applied = args.press_force / A_die
            # Friction loss: effective green pressure
            P_green = P_applied / (1.0 + mu * L_over_D)
            D_green = 1.0 - (1.0 - D0) * math.exp(-K * P_green)
            rho_green = D_green * rho_th
            result["applied_pressure"] = P_applied
            result["green_pressure"] = P_green
            result["green_density"] = rho_green
            result["relative_density"] = D_green

        # Die wall (radial) pressure — lateral pressure coefficient
        # For metal powders, radial ≈ 0.3-0.5 × axial green pressure
        P_green_eff = result["green_pressure"]
        P_radial = 0.4 * P_green_eff  # simplified lateral ratio
        A_wall = math.pi * d * H  # die wall contact area

        F_eject = mu * P_radial * A_wall

        result["die_wall_pressure"] = P_radial
        result["ejection_force"] = F_eject
        result["die_area"] = A_die
        result["formula"] = "P_applied = P_green·(1 + μ·L/D), F_eject = μ·P_radial·A_wall"

        return ToolResult(data=result, success=True)

    # ── thermal_stress ─────────────────────────────────────────────

    def _thermal_stress(self, args: MechanicalToolInput) -> ToolResult:
        """Thermal stress and deformation for constrained bodies.

        σ = E·α·ΔT / (1-ν)  for plane strain (task spec)
        σ = E·α·ΔT         for 1D constrained
        σ = E·α·ΔT / (1-2ν) for fully constrained (3D)
        """
        import numpy as np  # noqa: F401

        E = args.material_props["E"]
        alpha = args.material_props["alpha"]
        nu = args.material_props.get("nu", 0.3)
        dT = args.temperature_gradient
        ct = args.constraint_type

        eps_th = alpha * dT  # free thermal strain

        if ct == "one_direction":
            sigma = E * alpha * dT
            formula = "σ = E·α·ΔT"
        elif ct == "plane_strain":
            sigma = E * alpha * dT / (1.0 - nu)
            formula = "σ = E·α·ΔT / (1-ν)"
        elif ct == "fully_constrained":
            sigma = E * alpha * dT / (1.0 - 2.0 * nu)
            formula = "σ = E·α·ΔT / (1-2ν)"
        else:
            return _err(f"Unknown constraint_type: {ct}")

        result: dict[str, Any] = {
            "thermal_stress": sigma,
            "thermal_strain": eps_th,
            "constraint_type": ct,
            "temperature_gradient": dT,
            "formula": formula,
        }

        # Buckling check for plates with thickness and length given
        dims = args.dimensions
        if "t" in dims and "L" in dims:
            t = dims["t"]
            L = dims["L"]
            # Critical buckling stress for simply supported plate (k=4)
            sigma_cr = 4.0 * math.pi**2 * E / (
                12.0 * (1.0 - nu**2)
            ) * (t / L) ** 2
            result["critical_buckling_stress"] = sigma_cr
            result["buckling_ratio"] = abs(sigma) / sigma_cr if sigma_cr > 0 else None
            result["buckling_check"] = (
                "likely" if abs(sigma) > sigma_cr else "safe"
            )

        sy = args.material_props.get("yield_strength", 0.0)
        if sy > 0:
            result["safety_factor"] = sy / abs(sigma) if sigma != 0 else None

        return ToolResult(data=result, success=True)

    # ── fatigue_life ───────────────────────────────────────────────

    def _fatigue_life(self, args: MechanicalToolInput) -> ToolResult:
        """Fatigue life and safety factor using mean-stress criteria.

        Goodman:      σ_a/S_e + σ_m/S_ut = 1/n
        Soderberg:    σ_a/S_e + σ_m/S_y  = 1/n
        ASME ellip.: (σ_a/S_e)² + (σ_m/S_ut)² = 1/n²
        Basquin:      σ_a = σ'_f·(2N_f)^b
        """
        import numpy as np  # noqa: F401
        import sympy as sp

        fp = args.fatigue_params
        S_ut = fp["S_ut"]
        S_e = fp["S_e"]
        sigma_a = fp["stress_amplitude"]
        sigma_m = fp.get("mean_stress", 0.0)
        Kf = fp.get("Kf", 1.0)
        criterion = args.fatigue_criterion

        # Apply stress concentration
        sa = sigma_a * Kf
        sm = sigma_m * Kf

        # Symbolic expression for the chosen criterion
        sa_s, sm_s, Se_s, Sut_s, Sy_s = sp.symbols(
            "sigma_a sigma_m S_e S_ut S_y", positive=True
        )
        if criterion == "goodman":
            n_sym = 1.0 / (sa_s / Se_s + sm_s / Sut_s)
            formula_str = "n = 1 / (σ_a/S_e + σ_m/S_ut)"
        elif criterion == "soderberg":
            Sy = fp.get("S_y", 0.6 * S_ut)
            n_sym = 1.0 / (sa_s / Se_s + sm_s / Sy_s)
            formula_str = "n = 1 / (σ_a/S_e + σ_m/S_y)"
        else:  # asme_elliptical
            n_sym = 1.0 / sp.sqrt((sa_s / Se_s) ** 2 + (sm_s / Sut_s) ** 2)
            formula_str = "n = 1 / √((σ_a/S_e)² + (σ_m/S_ut)²)"

        # Numerical safety factor
        if criterion == "goodman":
            denom = sa / S_e + sm / S_ut
            n = 1.0 / denom if denom > 0 else float("inf")
        elif criterion == "soderberg":
            Sy = fp.get("S_y", 0.6 * S_ut)
            denom = sa / S_e + sm / Sy
            n = 1.0 / denom if denom > 0 else float("inf")
        else:
            n = 1.0 / math.sqrt((sa / S_e) ** 2 + (sm / S_ut) ** 2)

        # Fatigue life via Basquin: σ_a = σ'_f·(2N_f)^b
        sigma_f = fp.get("sigma_f_prime", S_ut)
        b_exp = fp.get("b", -0.085)

        if sa <= S_e:
            N_f = float("inf")
            life_regime = "infinite (N > 10⁶)"
        elif sa >= S_ut:
            N_f = 0.5  # less than 1 reversal — LCF
            life_regime = "low-cycle (σ_a ≥ S_ut)"
        else:
            N_f = 0.5 * (sa / sigma_f) ** (1.0 / b_exp)
            life_regime = "finite"

        result: dict[str, Any] = {
            "safety_factor": n,
            "fatigue_life": N_f,
            "life_regime": life_regime,
            "criterion": criterion,
            "effective_stress_amplitude": sa,
            "effective_mean_stress": sm,
            "stress_concentration_factor": Kf,
            "formula": formula_str,
            "basquin_params": {
                "sigma_f_prime": sigma_f,
                "b": b_exp,
            },
        }

        return ToolResult(data=result, success=True)

    # ── rolling_force ──────────────────────────────────────────────

    def _rolling_force(self, args: MechanicalToolInput) -> ToolResult:
        """Rolling force, torque, power, minimum thickness.

        F = σ_flow · L · w · Q_p
        L = √(R·Δh)          contact arc length
        Q_p = 1 + μ·L/(2·h_avg)  friction hill multiplier
        M = F·L              torque (both rolls, lever arm ≈ L/2)
        P = M·ω              power
        h_min = C·μ·R·σ_flow/E_roll  (Hitchcock roll flattening)
        """
        import numpy as np  # noqa: F401

        R = args.roll_radius
        dh = args.reduction
        w = args.width
        sigma = args.flow_stress
        mu = args.friction_coefficient
        N_rpm = args.roll_speed_rpm or 0.0
        h0 = args.initial_thickness

        # Contact arc length
        L = math.sqrt(R * dh)

        # Friction hill factor
        if h0 is not None:
            h_out = h0 - dh
            h_avg = (h0 + h_out) / 2.0
            Q_p = 1.0 + mu * L / (2.0 * h_avg)
        else:
            # Without initial thickness, skip friction hill correction
            h_avg = None
            Q_p = 1.0

        # Rolling force
        F = sigma * L * w * Q_p

        # Torque: lever arm ≈ L/2, both rolls
        lever_arm = 0.5 * L
        torque = 2.0 * F * lever_arm  # = F·L

        # Power
        omega = 2.0 * math.pi * N_rpm / 60.0
        power = torque * omega

        # Minimum rollable thickness (Hitchcock, C≈7.84)
        h_min = 7.84 * mu * R * sigma / _ROLL_E

        result: dict[str, Any] = {
            "rolling_force": F,
            "contact_length": L,
            "friction_hill_factor": Q_p,
            "torque": torque,
            "power": power,
            "roll_speed_rpm": N_rpm,
            "min_rollable_thickness": h_min,
            "formula": "F = σ·L·w·Q_p, M = F·L, h_min = 7.84·μ·R·σ/E_roll",
        }
        if h_avg is not None:
            result["average_thickness"] = h_avg
        else:
            result["warning"] = (
                "initial_thickness not given, Q_p set to 1.0 (no friction hill)"
            )

        return ToolResult(data=result, success=True)


def _err(msg: str) -> ToolResult:
    return ToolResult(data=None, success=False, error=msg)
