"""Unit conversion tool — unified access to the pint-based unit system.

Lets the agent convert quantities, check dimensional consistency, and
normalize values to SI. Falls back to a lightweight registry when pint
is not installed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult, ValidationResult
from huginn.tools.base import HuginnTool
from huginn.utils.units import convert, format_quantity, is_pint_available, to_si

# Conversion factors to SI for natural / atomic / CGS unit systems.
# Each entry maps a physical quantity to per-system {factor, unit} pairs
# where *factor* converts one unit of that system to the SI base.
_NATURAL_UNIT_TABLE: dict[str, dict[str, dict[str, float | str]]] = {
    "energy": {
        "si": {"factor": 1.0, "unit": "J"},
        "atomic": {"factor": 4.3597447222071e-18, "unit": "Hartree"},
        "cgs": {"factor": 1e-7, "unit": "erg"},
    },
    "length": {
        "si": {"factor": 1.0, "unit": "m"},
        "atomic": {"factor": 5.29177210903e-11, "unit": "Bohr"},
        "cgs": {"factor": 1e-2, "unit": "cm"},
    },
    "mass": {
        "si": {"factor": 1.0, "unit": "kg"},
        "atomic": {"factor": 9.1093837015e-31, "unit": "m_e"},
        "cgs": {"factor": 1e-3, "unit": "g"},
    },
    "velocity": {
        "si": {"factor": 1.0, "unit": "m/s"},
        "atomic": {"factor": 2.18769126364e6, "unit": "a.u."},
        "cgs": {"factor": 1e-2, "unit": "cm/s"},
    },
    "time": {
        "si": {"factor": 1.0, "unit": "s"},
        "atomic": {"factor": 2.4188843265857e-17, "unit": "a.u."},
        "cgs": {"factor": 1.0, "unit": "s"},
    },
    "momentum": {
        "si": {"factor": 1.0, "unit": "kg*m/s"},
        "atomic": {"factor": 1.99285191410e-24, "unit": "a.u."},
        "cgs": {"factor": 1e-5, "unit": "g*cm/s"},
    },
    "force": {
        "si": {"factor": 1.0, "unit": "N"},
        "atomic": {"factor": 8.2387234983e-8, "unit": "Hartree/Bohr"},
        "cgs": {"factor": 1e-5, "unit": "dyn"},
    },
    "pressure": {
        "si": {"factor": 1.0, "unit": "Pa"},
        "atomic": {"factor": 2.9421015697e13, "unit": "Hartree/Bohr^3"},
        "cgs": {"factor": 0.1, "unit": "Ba"},
    },
    "charge": {
        "si": {"factor": 1.0, "unit": "C"},
        "atomic": {"factor": 1.602176634e-19, "unit": "e"},
        "cgs": {"factor": 3.33564095198152e-10, "unit": "statC"},
    },
    "action": {
        "si": {"factor": 1.0, "unit": "J*s"},
        "atomic": {"factor": 1.054571817e-34, "unit": "hbar"},
        "cgs": {"factor": 1e-7, "unit": "erg*s"},
    },
}


class UnitToolInput(BaseModel):
    action: Literal[
        "convert",
        "to_si",
        "check_dimension",
        "list_units",
        "infer_dimension",
        "unit_arithmetic",
        "natural_units",
    ] = Field(default="convert")
    value: float = Field(default=1.0, description="Numeric value to convert")
    from_unit: str = Field(default="", description="Source unit")
    to_unit: str = Field(default="", description="Target unit for convert action")
    dimension: str | None = Field(
        default=None,
        description="Expected dimension for check_dimension (e.g. energy, length, pressure)",
    )
    # Fields for infer_dimension
    expression: str | None = Field(
        default=None,
        description="Expression to evaluate for infer_dimension "
        "(e.g. 'm * a', '0.5 * m * v**2').",
    )
    variables: dict[str, str] | None = Field(
        default=None,
        description="Variable name to unit mapping for infer_dimension "
        "(e.g. {'m': 'kg', 'a': 'm/s**2'}).",
    )
    # Fields for unit_arithmetic
    operation: Literal["add", "subtract", "multiply", "divide"] | None = Field(
        default=None,
        description="Arithmetic operation for unit_arithmetic.",
    )
    value1: float | None = Field(default=None, description="First value for unit_arithmetic")
    unit1: str = Field(default="", description="Unit of first value for unit_arithmetic")
    value2: float | None = Field(default=None, description="Second value for unit_arithmetic")
    unit2: str = Field(default="", description="Unit of second value for unit_arithmetic")
    # Fields for natural_units
    from_system: str | None = Field(
        default=None,
        description="Source unit system for natural_units (si, atomic, cgs).",
    )
    to_system: str | None = Field(
        default=None,
        description="Target unit system for natural_units (si, atomic, cgs).",
    )
    quantity: str | None = Field(
        default=None,
        description="Physical quantity for natural_units "
        "(energy, length, mass, velocity, time, momentum, force, "
        "pressure, charge, action).",
    )


_DIM_LABEL_TO_DIMENSION: dict[str, tuple[str, ...]] = {
    "energy": ("M", "L", "T"),
    "length": ("L",),
    "pressure": ("M", "L", "T"),
    "temperature": ("Theta",),
    "time": ("T",),
    "force": ("M", "L", "T"),
    "mass": ("M",),
    "density": ("M", "L"),
    "frequency": ("T",),
    "volume": ("L",),
    "area": ("L",),
    "velocity": ("L", "T"),
}


class UnitTool(HuginnTool):
    """Convert units, normalize to SI, and check physical dimensions."""

    name = "unit_tool"
    category = "sci"
    description = (
        "Convert physical quantities between units (e.g. eV ↔ J, Å ↔ m, "
        "GPa ↔ Pa) and check dimensional consistency. Uses pint when available."
    )
    input_schema = UnitToolInput

    def is_read_only(self, args: UnitToolInput) -> bool:
        return True

    async def validate_input(
        self, args: UnitToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "convert":
            if not args.from_unit or not args.to_unit:
                return ValidationResult(
                    result=False,
                    message="convert requires both from_unit and to_unit.",
                )
        elif args.action == "to_si":
            if not args.from_unit:
                return ValidationResult(
                    result=False, message="to_si requires from_unit."
                )
        elif args.action == "check_dimension":
            if not args.from_unit or not args.dimension:
                return ValidationResult(
                    result=False,
                    message="check_dimension requires from_unit and dimension.",
                )
        elif args.action == "infer_dimension":
            if not args.expression or not args.variables:
                return ValidationResult(
                    result=False,
                    message="infer_dimension requires expression and variables.",
                )
        elif args.action == "unit_arithmetic":
            if (
                not args.operation
                or args.value1 is None
                or args.value2 is None
                or not args.unit1
                or not args.unit2
            ):
                return ValidationResult(
                    result=False,
                    message="unit_arithmetic requires operation, value1, "
                    "unit1, value2, and unit2.",
                )
        elif args.action == "natural_units" and (
            not args.from_system or not args.to_system or not args.quantity
        ):
            return ValidationResult(
                result=False,
                message="natural_units requires from_system, to_system, "
                "and quantity.",
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        from huginn.utils.units import Q

        input_data = UnitToolInput(**args)

        try:
            if input_data.action == "convert":
                quantity = Q(input_data.value, input_data.from_unit)
                converted = convert(quantity, input_data.to_unit)
                return ToolResult(
                    data={
                        "value": input_data.value,
                        "from_unit": input_data.from_unit,
                        "to_unit": input_data.to_unit,
                        "result": (
                            float(converted.magnitude)
                            if hasattr(converted, "magnitude")
                            else converted.magnitude
                        ),
                        "formatted": format_quantity(converted),
                    },
                    success=True,
                )

            elif input_data.action == "to_si":
                quantity = Q(input_data.value, input_data.from_unit)
                si_value = to_si(quantity)
                return ToolResult(
                    data={
                        "value": input_data.value,
                        "from_unit": input_data.from_unit,
                        "si_value": si_value,
                    },
                    success=True,
                )

            elif input_data.action == "check_dimension":
                from huginn.research.external_validator import (
                    resolve_unit_dimension,
                )

                # 契约层单轨: 解析 from_unit 的量纲签名, 与期望维度字符串比对.
                sig = resolve_unit_dimension(input_data.from_unit)
                if sig is None:
                    return ToolResult(
                        data={
                            "value": input_data.value,
                            "from_unit": input_data.from_unit,
                            "dimension": input_data.dimension,
                            "is_valid": False,
                            "note": f"unit '{input_data.from_unit}' 契约 registry 无法解析(量纲未知)",
                        },
                        success=True,
                    )
                is_valid = (
                    input_data.dimension is not None
                    and self._dim_matches(sig, input_data.dimension)
                )
                return ToolResult(
                    data={
                        "value": input_data.value,
                        "from_unit": input_data.from_unit,
                        "dimension": input_data.dimension,
                        "is_valid": is_valid,
                        "dimension_signature": sig,
                    },
                    success=True,
                )

            elif input_data.action == "list_units":
                from huginn.utils.units import DFT_UNITS, MD_UNITS, SI_UNITS

                return ToolResult(
                    data={
                        "pint_available": is_pint_available(),
                        "presets": {
                            "dft": DFT_UNITS,
                            "md": MD_UNITS,
                            "si": SI_UNITS,
                        },
                    },
                    success=True,
                )

            elif input_data.action == "infer_dimension":
                return self._infer_dimension(
                    input_data.expression or "",
                    input_data.variables or {},
                )

            elif input_data.action == "unit_arithmetic":
                return self._unit_arithmetic(
                    input_data.operation or "add",
                    input_data.value1 if input_data.value1 is not None else 0.0,
                    input_data.unit1,
                    input_data.value2 if input_data.value2 is not None else 0.0,
                    input_data.unit2,
                )

            elif input_data.action == "natural_units":
                return self._natural_units(
                    input_data.value,
                    input_data.from_system or "",
                    input_data.to_system or "",
                    input_data.quantity or "",
                )

            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown action: {input_data.action}",
            )

        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unit operation failed: {e}",
            )

    # ── Compound dimension inference ──────────────────────────────────

    @staticmethod
    def _dim_matches(signature: str, dimension_label: str) -> bool:
        """契约层量纲签名(如 ``M1·L1·T-2``)是否属于请求的物理量纲标签.

        ``dimension_label`` 取能量/长度/压力/温度/时间/力/质量/密度/频率/体积/面积/速度.
        用维度字母是否出现(指数非 0)来判定, 与契约层 UnitRegistry 的 7 基维对应.
        未识别标签 → False(不硬判).
        """
        label = (dimension_label or "").strip().lower()
        inv = _DIM_LABEL_TO_DIMENSION
        want = inv.get(label)
        if want is None:
            return False
        # want 比如 "M L T" 的子集; 检查签名里这些维度的指数非零即可(签名格式 L{d}g 可能带负).
        present = signature.split("·")
        have = set()
        for chunk in present:
            if not chunk:
                continue
            letter = chunk[0]
            have.add(letter)
        return set(want).issubset(have)

    def _infer_dimension(
        self, expression: str, variables: dict[str, str]
    ) -> ToolResult:
        """Infer the dimension of a compound expression (契约层单轨).

        不再依赖 pint 的维度字符串 —— 把 ``variables`` 当符号单位表, 交给
        ``external_validator.check_expression_dimensions``(sympy + 契约层 UnitRegistry)
        推断, 与 FEM/Lean/结构解析等引擎用同一判据. ``expression`` 是纯无量纲的
        量纲推断式(无目标单位), 故这里只看推断量纲, 不比对目标.
        """
        from huginn.research.external_validator import (
            _to_symbol,
            check_expression_dimensions,
        )

        sym_units = {var: _to_symbol(str(unit)) for var, unit in (variables or {}).items()}
        if not sym_units:
            return ToolResult(
                data=None, success=False,
                error="infer_dimension requires at least one variable with unit.",
            )
        try:
            # 无量纲推断: 给一个 dummy 目标等于推断结果, 仅取 inferred 签名.
            out = check_expression_dimensions(expression, sym_units, expected_unit="1")
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                data=None, success=False,
                error=f"Dimension inference failed: {e}",
            )
        if out.get("error") and out.get("inferred") is None:
            # 解析/量纲引擎不可用 → 如实报不可用(诚实边界), 不硬判.
            return ToolResult(
                data=None, success=False,
                error=out["error"],
            )
        return ToolResult(
            data={
                "result_dimension": out["inferred"],
                "result_unit": out["inferred"],
                "consistent": True,
            },
            success=True,
        )

    def _unit_arithmetic(
        self,
        operation: str,
        value1: float,
        unit1: str,
        value2: float,
        unit2: str,
    ) -> ToolResult:
        """Perform arithmetic on two quantities with full unit tracking.

        add/subtract check dimension consistency and convert automatically;
        multiply/divide compute the resulting compound unit.
        """
        try:
            from huginn.utils.units import Q

            q1 = Q(value1, unit1)
            q2 = Q(value2, unit2)

            if operation == "add":
                result = q1 + q2
            elif operation == "subtract":
                result = q1 - q2
            elif operation == "multiply":
                result = q1 * q2
            elif operation == "divide":
                result = q1 / q2
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

            if is_pint_available():
                result_value = float(result.magnitude)
                result_unit = str(result.units)
                result_dim = str(result.dimensionality)
            else:
                # Fallback quantity — only add/subtract carry full info
                result_value = float(result.magnitude)
                result_unit = result.unit
                result_dim = result.dimensionality

            return ToolResult(
                data={
                    "result_value": result_value,
                    "result_unit": result_unit,
                    "result_dimension": result_dim,
                },
                success=True,
            )
        except Exception as e:
            if type(e).__name__ == "DimensionalityError":
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Dimension mismatch: cannot {operation} "
                    f"{unit1} and {unit2}. {e}",
                )
            return ToolResult(
                data=None,
                success=False,
                error=f"Unit arithmetic failed: {e}",
            )

    def _natural_units(
        self,
        value: float,
        from_system: str,
        to_system: str,
        quantity: str,
    ) -> ToolResult:
        """Convert between unit systems (SI, atomic, CGS).

        Uses a lookup table of fundamental conversion factors.  The value
        is first converted to SI, then to the target system.
        """
        try:
            quantity = quantity.lower()
            from_system = from_system.lower()
            to_system = to_system.lower()

            if quantity not in _NATURAL_UNIT_TABLE:
                available = ", ".join(sorted(_NATURAL_UNIT_TABLE.keys()))
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown quantity '{quantity}'. Supported: {available}",
                )

            table = _NATURAL_UNIT_TABLE[quantity]
            if from_system not in table:
                systems = ", ".join(sorted(table.keys()))
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown source system '{from_system}'. "
                    f"Supported: {systems}",
                )
            if to_system not in table:
                systems = ", ".join(sorted(table.keys()))
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown target system '{to_system}'. "
                    f"Supported: {systems}",
                )

            from_entry = table[from_system]
            to_entry = table[to_system]

            # Convert through SI: value * from_factor -> SI -> / to_factor
            si_value = value * float(from_entry["factor"])
            result_value = si_value / float(to_entry["factor"])

            return ToolResult(
                data={
                    "value": result_value,
                    "unit": to_entry["unit"],
                    "from_system": from_system,
                    "to_system": to_system,
                    "quantity": quantity,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Natural unit conversion failed: {e}",
            )
