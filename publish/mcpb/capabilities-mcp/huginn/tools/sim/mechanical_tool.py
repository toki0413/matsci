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

import logging
import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ToolProfile

logger = logging.getLogger(__name__)


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


# ── 每个 action 的必填字段检查 ────────────────────────────────────────


def _missing_stress_fields(args: MechanicalToolInput) -> list[str]:
    """stress_analysis 必填字段."""
    missing = []
    if args.geometry_type is None:
        missing.append("geometry_type")
    if not args.dimensions:
        missing.append("dimensions")
    if not args.material_props or "E" not in args.material_props:
        missing.append("material_props (needs E)")
    if args.load_type is None:
        missing.append("load_type")
    if args.load_value is None:
        missing.append("load_value")
    return missing


def _missing_powder_fields(args: MechanicalToolInput) -> list[str]:
    """powder_metallurgy 必填字段."""
    missing = []
    if args.powder_material is None and args.theoretical_density is None:
        missing.append("powder_material or theoretical_density")
    if args.die_diameter is None:
        missing.append("die_diameter")
    if args.die_height is None:
        missing.append("die_height")
    if args.friction_coefficient is None:
        missing.append("friction_coefficient")
    if args.press_force is None and args.target_density is None:
        missing.append("press_force or target_density")
    return missing


def _missing_thermal_fields(args: MechanicalToolInput) -> list[str]:
    """thermal_stress 必填字段."""
    missing = []
    if not args.material_props or "E" not in args.material_props:
        missing.append("material_props (needs E)")
    if not args.material_props or "alpha" not in args.material_props:
        missing.append("material_props (needs alpha)")
    if args.temperature_gradient is None:
        missing.append("temperature_gradient")
    if args.constraint_type is None:
        missing.append("constraint_type")
    return missing


def _missing_fatigue_fields(args: MechanicalToolInput) -> list[str]:
    """fatigue_life 必填字段."""
    fp = args.fatigue_params
    missing = []
    if "S_ut" not in fp:
        missing.append("fatigue_params.S_ut")
    if "S_e" not in fp:
        missing.append("fatigue_params.S_e")
    if "stress_amplitude" not in fp:
        missing.append("fatigue_params.stress_amplitude")
    return missing


def _missing_rolling_fields(args: MechanicalToolInput) -> list[str]:
    """rolling_force 必填字段."""
    missing = []
    if args.roll_radius is None:
        missing.append("roll_radius")
    if args.reduction is None:
        missing.append("reduction")
    if args.width is None:
        missing.append("width")
    if args.flow_stress is None:
        missing.append("flow_stress")
    if args.friction_coefficient is None:
        missing.append("friction_coefficient")
    return missing


# action → 必填字段检查函数
_REQUIRED_FIELD_CHECKS: dict[str, Callable[[Any], list[str]]] = {
    "stress_analysis": _missing_stress_fields,
    "powder_metallurgy": _missing_powder_fields,
    "thermal_stress": _missing_thermal_fields,
    "fatigue_life": _missing_fatigue_fields,
    "rolling_force": _missing_rolling_fields,
}


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
    constraint_type: (
        Literal["fully_constrained", "one_direction", "plane_strain"] | None
    ) = None

    # ── fatigue_life ──
    fatigue_params: dict[str, float] = Field(
        default_factory=dict,
        description="S_ut, S_e, stress_amplitude, mean_stress, Kf, S_y, sigma_f_prime, b",
    )
    fatigue_criterion: Literal["goodman", "soderberg", "asme_elliptical"] = "goodman"

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
    def _check_required_fields(self) -> MechanicalToolInput:
        """Validate that the action has the fields it needs."""
        check = _REQUIRED_FIELD_CHECKS.get(self.action)
        if check is not None:
            missing = check(self)
            if missing:
                raise ValueError(
                    f"{self.action} missing required fields: {', '.join(missing)}"
                )
        return self


# ── stress_analysis ───────────────────────────────────────────────────


def _bending_sigma_plate(dims: dict[str, float], val: float) -> float:
    """σ = 6M/(w·t²)."""
    return 6.0 * val / (dims.get("w", 1.0) * dims["t"] ** 2)


def _bending_sigma_beam(dims: dict[str, float], val: float) -> float:
    """σ = 6M/(b·h²)."""
    return 6.0 * val / (dims["b"] * dims["h"] ** 2)


def _bending_sigma_cylinder(dims: dict[str, float], val: float) -> float:
    """实心圆截面 σ = 32M/(π·d³)."""
    return 32.0 * val / (math.pi * dims["d"] ** 3)


# geometry → (σ 计算函数, 公式文本)
_BENDING_MODELS: dict[str, tuple[Callable[[dict[str, float], float], float], str]] = {
    "plate": (_bending_sigma_plate, "σ = 6M/(w·t²)"),
    "beam": (_bending_sigma_beam, "σ = 6M/(b·h²)"),
    "cylinder": (_bending_sigma_cylinder, "σ = 32M/(π·d³)"),
}


def _stress_uniaxial(args: MechanicalToolInput) -> dict[str, Any] | ToolResult:
    """σ = F/A, 面积按截面形状取."""
    gtype = args.geometry_type
    dims = args.dimensions
    if gtype == "cylinder":
        area = math.pi * dims["d"] ** 2 / 4.0
    elif gtype == "plate":
        area = dims["w"] * dims["t"]
    elif gtype == "beam":
        area = dims["b"] * dims["h"]
    else:
        return _err(f"uniaxial not supported for {gtype}")
    sigma = args.load_value / area
    strain = sigma / args.material_props["E"]
    return {
        "formula": "σ = F/A",
        "area": area,
        "max_stress": sigma,
        "max_strain": strain,
        "deformation": strain * dims.get("L", 0.0),
    }


def _stress_biaxial(args: MechanicalToolInput) -> dict[str, Any]:
    """等双轴: load_value 即施加应力, von Mises 等于该值."""
    sigma = args.load_value
    return {
        "formula": "σ_vm = σ (equal biaxial)",
        "max_stress": sigma,
        "max_strain": sigma / args.material_props["E"],
        "deformation": 0.0,
    }


def _stress_bending(args: MechanicalToolInput) -> dict[str, Any] | ToolResult:
    """弯曲应力, 按截面形状选公式."""
    model = _BENDING_MODELS.get(args.geometry_type or "")
    if model is None:
        return _err(f"bending not supported for {args.geometry_type}")
    sigma_fn, formula = model
    sigma = sigma_fn(args.dimensions, args.load_value)
    return {
        "formula": formula,
        "max_stress": sigma,
        "max_strain": sigma / args.material_props["E"],
        "deformation": 0.0,
    }


def _stress_torsion(args: MechanicalToolInput) -> dict[str, Any] | ToolResult:
    """实心圆轴扭转: τ = T·r/J, θ = TL/(GJ)."""
    if args.geometry_type != "cylinder":
        return _err("torsion only supported for cylinder geometry")
    dims = args.dimensions
    E = args.material_props["E"]
    nu = args.material_props.get("nu", 0.3)
    d = dims["d"]
    J = math.pi * d**4 / 32.0
    tau = args.load_value * (d / 2.0) / J
    G = E / (2.0 * (1.0 + nu))
    theta = args.load_value * dims.get("L", 0.0) / (G * J)
    return {
        "formula": "τ = T·r/J, θ = TL/(GJ)",
        "polar_moment": J,
        "max_stress": tau,
        "max_strain": tau / G,
        "twist_angle": theta,
        "deformation": theta,
    }


# load_type → 分支处理函数
_STRESS_LOAD_HANDLERS: dict[
    str, Callable[[MechanicalToolInput], dict[str, Any] | ToolResult]
] = {
    "uniaxial": _stress_uniaxial,
    "biaxial": _stress_biaxial,
    "bending": _stress_bending,
    "torsion": _stress_torsion,
}


def _safety_factor(yield_strength: float, max_stress: float) -> float | None:
    """安全系数 = σ_y/σ_max; 无屈服强度或无应力时为 None."""
    if yield_strength > 0 and max_stress > 0:
        return yield_strength / max_stress
    return None


# ── powder_metallurgy ─────────────────────────────────────────────────


def _powder_material_data(args: MechanicalToolInput) -> dict[str, float] | ToolResult:
    """查粉末材料表; 显式 theoretical_density 覆盖时用缺省 K/D0."""
    if args.theoretical_density is not None:
        return {"rho": args.theoretical_density, "K": 2.5e-9, "D0": 0.45}
    data = _POWDER_DATA.get((args.powder_material or "").lower())
    if data is None:
        return _err(
            f"Unknown powder material '{args.powder_material}'. "
            f"Known: {list(_POWDER_DATA)}"
        )
    return data


def _powder_geometry(args: MechanicalToolInput) -> dict[str, float]:
    """压制几何量: 模具面积与高径比."""
    d = args.die_diameter
    H = args.die_height
    return {
        "d": d,
        "H": H,
        "mu": args.friction_coefficient,
        "A_die": math.pi * d**2 / 4.0,
        "L_over_D": H / d,
    }


def _powder_from_target(
    args: MechanicalToolInput, mat: dict[str, float], geo: dict[str, float]
) -> dict[str, Any] | ToolResult:
    """给定目标密度 → 逆 Heckel 求所需压制压力."""
    K, D0 = mat["K"], mat["D0"]
    D_target = args.target_density / mat["rho"]
    if D_target >= 1.0:
        return _err("target_density exceeds theoretical density")
    if D_target <= D0:
        return _err(
            f"target relative density {D_target:.3f} <= fill density D0={D0:.3f}"
        )
    P_green = -(1.0 / K) * math.log((1.0 - D_target) / (1.0 - D0))
    P_applied = P_green * (1.0 + geo["mu"] * geo["L_over_D"])
    return {
        "required_pressure": P_applied,
        "green_pressure": P_green,
        "required_force": P_applied * geo["A_die"],
        "green_density": args.target_density,
        "relative_density": D_target,
    }


def _powder_from_press(
    args: MechanicalToolInput, mat: dict[str, float], geo: dict[str, float]
) -> dict[str, Any]:
    """给定压制力 → 正 Heckel 求生坯密度."""
    P_applied = args.press_force / geo["A_die"]
    P_green = P_applied / (1.0 + geo["mu"] * geo["L_over_D"])
    D_green = 1.0 - (1.0 - mat["D0"]) * math.exp(-mat["K"] * P_green)
    return {
        "applied_pressure": P_applied,
        "green_pressure": P_green,
        "green_density": D_green * mat["rho"],
        "relative_density": D_green,
    }


# ── thermal_stress ────────────────────────────────────────────────────


def _thermal_sigma_one_direction(E, alpha, dT, nu) -> tuple[float, str]:
    """1D 约束: σ = E·α·ΔT."""
    return E * alpha * dT, "σ = E·α·ΔT"


def _thermal_sigma_plane_strain(E, alpha, dT, nu) -> tuple[float, str]:
    """平面应变: σ = E·α·ΔT/(1-ν)."""
    return E * alpha * dT / (1.0 - nu), "σ = E·α·ΔT / (1-ν)"


def _thermal_sigma_fully_constrained(E, alpha, dT, nu) -> tuple[float, str]:
    """三维全约束: σ = E·α·ΔT/(1-2ν)."""
    return E * alpha * dT / (1.0 - 2.0 * nu), "σ = E·α·ΔT / (1-2ν)"


# constraint_type → (σ 计算函数, 公式文本)
_THERMAL_SIGMA_MODELS: dict[
    str, Callable[[float, float, float, float], tuple[float, str]]
] = {
    "one_direction": _thermal_sigma_one_direction,
    "plane_strain": _thermal_sigma_plane_strain,
    "fully_constrained": _thermal_sigma_fully_constrained,
}


def _buckling_fields(
    sigma: float, E: float, nu: float, dims: dict[str, float]
) -> dict[str, Any]:
    """薄板屈曲校核 — 仅当 dims 同时给了 t 与 L."""
    if "t" not in dims or "L" not in dims:
        return {}
    t = dims["t"]
    L = dims["L"]
    # 简支板临界屈曲应力 (k=4)
    sigma_cr = 4.0 * math.pi**2 * E / (12.0 * (1.0 - nu**2)) * (t / L) ** 2
    return {
        "critical_buckling_stress": sigma_cr,
        "buckling_ratio": abs(sigma) / sigma_cr if sigma_cr > 0 else None,
        "buckling_check": "likely" if abs(sigma) > sigma_cr else "safe",
    }


# ── fatigue_life ──────────────────────────────────────────────────────


def _fatigue_formula(criterion: str, sp) -> str:
    """所选平均应力准则的符号表达式 (文档用途) 与公式文本."""
    sa_s, sm_s, Se_s, Sut_s, Sy_s = sp.symbols(
        "sigma_a sigma_m S_e S_ut S_y", positive=True
    )
    if criterion == "goodman":
        1.0 / (sa_s / Se_s + sm_s / Sut_s)
        return "n = 1 / (σ_a/S_e + σ_m/S_ut)"
    if criterion == "soderberg":
        1.0 / (sa_s / Se_s + sm_s / Sy_s)
        return "n = 1 / (σ_a/S_e + σ_m/S_y)"
    # asme_elliptical
    1.0 / sp.sqrt((sa_s / Se_s) ** 2 + (sm_s / Sut_s) ** 2)
    return "n = 1 / √((σ_a/S_e)² + (σ_m/S_ut)²)"


def _fatigue_safety(
    sa: float, sm: float, S_e: float, S_ut: float, criterion: str, fp: dict[str, float]
) -> float:
    """数值安全系数 — Goodman / Soderberg / ASME 椭圆."""
    if criterion == "goodman":
        denom = sa / S_e + sm / S_ut
    elif criterion == "soderberg":
        denom = sa / S_e + sm / fp.get("S_y", 0.6 * S_ut)
    else:
        return 1.0 / math.sqrt((sa / S_e) ** 2 + (sm / S_ut) ** 2)
    return 1.0 / denom if denom > 0 else float("inf")


def _fatigue_basquin(
    sa: float, S_e: float, S_ut: float, fp: dict[str, float]
) -> tuple[float, str, dict[str, float]]:
    """Basquin 寿命 N_f 与所在区段."""
    sigma_f = fp.get("sigma_f_prime", S_ut)
    b_exp = fp.get("b", -0.085)
    params = {"sigma_f_prime": sigma_f, "b": b_exp}
    if sa <= S_e:
        return float("inf"), "infinite (N > 10⁶)", params
    if sa >= S_ut:
        return 0.5, "low-cycle (σ_a ≥ S_ut)", params
    return 0.5 * (sa / sigma_f) ** (1.0 / b_exp), "finite", params


# ── rolling_force ─────────────────────────────────────────────────────


def _friction_hill(
    mu: float, arc: float, h0: float | None, dh: float
) -> tuple[float, float | None]:
    """摩擦丘系数 Q_p 与平均厚度; 无 h0 时 Q_p=1.0, h_avg=None."""
    if h0 is None:
        return 1.0, None
    h_out = h0 - dh
    h_avg = (h0 + h_out) / 2.0
    return 1.0 + mu * arc / (2.0 * h_avg), h_avg


def _rolling_result(
    force: float,
    arc: float,
    Q_p: float,
    N_rpm: float,
    h_min: float,
    h_avg: float | None,
) -> dict[str, Any]:
    """汇总轧制力/力矩/功率/最小可轧厚度."""
    torque = force * arc
    result: dict[str, Any] = {
        "rolling_force": force,
        "contact_length": arc,
        "friction_hill_factor": Q_p,
        "torque": torque,
        "power": torque * 2.0 * math.pi * N_rpm / 60.0,
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
    return result


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

    async def call(self, args: MechanicalToolInput, context: ToolContext) -> ToolResult:
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
                data=None,
                success=False,
                error=f"Unknown action: {args.action}",
            )
        try:
            result = handler(args)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"{args.action} failed: {exc}",
            )

        # Physics audit — stress vs yield, strain, safety factor, fatigue life
        if result.data:
            try:
                from huginn.execution.physics_auditor import PhysicsAuditor

                auditor = PhysicsAuditor()
                audit_report = auditor.audit(
                    "mechanical_tool", args.action, result.data, args.model_dump()
                )
                result.data["physics_audit"] = audit_report.to_dict()
            except Exception:  # 防御: 审计失败不阻断结果返回
                logger.debug("audit failure can't block result delivery", exc_info=True)

        return result

    # ── stress_analysis ───────────────────────────────────────────

    def _stress_analysis(self, args: MechanicalToolInput) -> ToolResult:
        """Simple geometry stress: uniaxial, biaxial, bending, torsion."""

        handler = _STRESS_LOAD_HANDLERS.get(args.load_type or "")
        if handler is None:
            return _err(f"Unknown load_type: {args.load_type}")

        updates = handler(args)
        if isinstance(updates, ToolResult):
            return updates

        result: dict[str, Any] = {
            "geometry": args.geometry_type,
            "load_type": args.load_type,
            "formula": "",
        }
        result.update(updates)
        result["safety_factor"] = _safety_factor(
            args.material_props.get("yield_strength", 0.0),
            result.get("max_stress", 0),
        )

        return ToolResult(data=result, success=True)

    # ── powder_metallurgy ──────────────────────────────────────────

    def _powder_metallurgy(self, args: MechanicalToolInput) -> ToolResult:
        """Uniaxial die compaction: pressure, green density, ejection force.

        Compaction model: D = 1 - (1-D0)·exp(-K·P)  (modified Heckel)
        Friction loss: P_applied = P_green·(1 + μ·L/D)
        """
        import numpy as np  # noqa: F401 — kept for consistency with other actions

        mat = _powder_material_data(args)
        if isinstance(mat, ToolResult):
            return mat
        geo = _powder_geometry(args)

        result: dict[str, Any] = {
            "powder_material": args.powder_material,
            "theoretical_density": mat["rho"],
            "die_diameter": geo["d"],
            "fill_height": geo["H"],
            "friction_coefficient": geo["mu"],
        }

        # Case 1: target density given → pressure; else press force → density
        if args.target_density is not None:
            updates = _powder_from_target(args, mat, geo)
        else:
            updates = _powder_from_press(args, mat, geo)
        if isinstance(updates, ToolResult):
            return updates
        result.update(updates)

        # Die wall (radial) pressure — lateral pressure coefficient
        # For metal powders, radial ≈ 0.3-0.5 × axial green pressure
        P_radial = 0.4 * result["green_pressure"]
        A_wall = math.pi * geo["d"] * geo["H"]  # die wall contact area

        result["die_wall_pressure"] = P_radial
        result["ejection_force"] = geo["mu"] * P_radial * A_wall
        result["die_area"] = geo["A_die"]
        result["formula"] = (
            "P_applied = P_green·(1 + μ·L/D), F_eject = μ·P_radial·A_wall"
        )

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

        model = _THERMAL_SIGMA_MODELS.get(ct or "")
        if model is None:
            return _err(f"Unknown constraint_type: {ct}")
        sigma, formula = model(E, alpha, dT, nu)

        result: dict[str, Any] = {
            "thermal_stress": sigma,
            "thermal_strain": alpha * dT,  # free thermal strain
            "constraint_type": ct,
            "temperature_gradient": dT,
            "formula": formula,
        }

        # Buckling check for plates with thickness and length given
        result.update(_buckling_fields(sigma, E, nu, args.dimensions))

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
        Kf = fp.get("Kf", 1.0)
        criterion = args.fatigue_criterion

        # Apply stress concentration
        sa = fp["stress_amplitude"] * Kf
        sm = fp.get("mean_stress", 0.0) * Kf

        formula_str = _fatigue_formula(criterion, sp)
        n = _fatigue_safety(sa, sm, S_e, S_ut, criterion, fp)
        N_f, life_regime, basquin = _fatigue_basquin(sa, S_e, S_ut, fp)

        result: dict[str, Any] = {
            "safety_factor": n,
            "fatigue_life": N_f,
            "life_regime": life_regime,
            "criterion": criterion,
            "effective_stress_amplitude": sa,
            "effective_mean_stress": sm,
            "stress_concentration_factor": Kf,
            "formula": formula_str,
            "basquin_params": basquin,
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

        # Contact arc length
        L = math.sqrt(R * dh)

        # Friction hill factor
        Q_p, h_avg = _friction_hill(mu, L, args.initial_thickness, dh)

        # Rolling force
        F = sigma * L * w * Q_p

        # Minimum rollable thickness (Hitchcock, C≈7.84)
        h_min = 7.84 * mu * R * sigma / _ROLL_E

        return ToolResult(
            data=_rolling_result(F, L, Q_p, N_rpm, h_min, h_avg),
            success=True,
        )


def _err(msg: str) -> ToolResult:
    return ToolResult(data=None, success=False, error=msg)
