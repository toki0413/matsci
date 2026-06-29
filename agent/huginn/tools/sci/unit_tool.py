"""Unit conversion tool — unified access to the pint-based unit system.

Lets the agent convert quantities, check dimensional consistency, and
normalize values to SI. Falls back to a lightweight registry when pint
is not installed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult
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
                from huginn.utils.units import check_units

                quantity = Q(input_data.value, input_data.from_unit)
                is_valid = check_units(quantity, input_data.dimension or "")
                return ToolResult(
                    data={
                        "value": input_data.value,
                        "from_unit": input_data.from_unit,
                        "dimension": input_data.dimension,
                        "is_valid": is_valid,
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

    def _infer_dimension(
        self, expression: str, variables: dict[str, str]
    ) -> ToolResult:
        """Infer the dimension of a compound expression.

        Each variable is turned into a unit quantity (magnitude 1) so that
        the resulting dimensionality reflects the expression's structure.
        """
        try:
            from huginn.security.safe_eval import safe_eval
            from huginn.utils.units import Q

            if not is_pint_available():
                return ToolResult(
                    data=None,
                    success=False,
                    error="pint is required for dimension inference.",
                )

            namespace = {var: Q(1.0, unit) for var, unit in variables.items()}
            result = safe_eval(expression, namespace)

            if hasattr(result, "dimensionality"):
                return ToolResult(
                    data={
                        "result_dimension": str(result.dimensionality),
                        "result_unit": str(result.units),
                        "consistent": True,
                    },
                    success=True,
                )

            # Plain number — dimensionless result
            return ToolResult(
                data={
                    "result_dimension": "dimensionless",
                    "result_unit": "dimensionless",
                    "consistent": True,
                },
                success=True,
            )
        except Exception as e:
            # pint raises DimensionalityError when adding incompatible units
            if type(e).__name__ == "DimensionalityError":
                return ToolResult(
                    data={
                        "result_dimension": "inconsistent",
                        "result_unit": "unknown",
                        "consistent": False,
                        "error": str(e),
                    },
                    success=True,
                )
            return ToolResult(
                data=None,
                success=False,
                error=f"Dimension inference failed: {e}",
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
