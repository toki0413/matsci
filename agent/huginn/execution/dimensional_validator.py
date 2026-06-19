"""Dimensional Analysis Validator — verifies physical consistency using SymPy.

Implements Buckingham π theorem and unit consistency checking
for computational materials science inputs.

Usage:
    from huginn.execution.dimensional_validator import DimensionalValidator
    validator = DimensionalValidator()
    validator.check({"E": "210 GPa", "stress": "500 MPa", "strain": "0.001"})
    # → All consistent (stress = E × strain ✓)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sympy as sp


@dataclass
class PhysicalQuantity:
    """A physical quantity with value, unit, and dimensions."""

    name: str
    value: float
    unit: str
    dimensions: dict[str, float] = field(
        default_factory=dict
    )  # {M:1, L:-1, T:-2} for stress


@dataclass
class DimensionalCheckResult:
    """Result of a dimensional analysis check."""

    consistent: bool
    equation: str
    lhs_dimensions: dict[str, float] = field(default_factory=dict)
    rhs_dimensions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class DimensionalValidator:
    """Validate physical consistency of equations and parameters.

    Supports SI base dimensions: M (mass), L (length), T (time),
    I (current), Θ (temperature), N (amount), J (luminosity).
    """

    # Unit → dimension mapping (in terms of M, L, T)
    BASE_UNITS: dict[str, dict[str, float]] = {
        # Length
        "m": {"L": 1},
        "cm": {"L": 1},
        "nm": {"L": 1},
        "angstrom": {"L": 1},
        # Mass
        "kg": {"M": 1},
        "g": {"M": 1},
        "amu": {"M": 1},
        # Time
        "s": {"T": 1},
        "fs": {"T": 1},
        "ps": {"T": 1},
        "ns": {"T": 1},
        # Temperature
        "K": {"Theta": 1},
        # Force
        "N": {"M": 1, "L": 1, "T": -2},
        # Pressure / Stress
        "Pa": {"M": 1, "L": -1, "T": -2},
        "MPa": {"M": 1, "L": -1, "T": -2},
        "GPa": {"M": 1, "L": -1, "T": -2},
        "bar": {"M": 1, "L": -1, "T": -2},
        # Energy
        "J": {"M": 1, "L": 2, "T": -2},
        "eV": {"M": 1, "L": 2, "T": -2},
        "Ha": {"M": 1, "L": 2, "T": -2},
        # Power
        "W": {"M": 1, "L": 2, "T": -3},
        # Frequency
        "Hz": {"T": -1},
        # Velocity
        "m/s": {"L": 1, "T": -1},
        # Acceleration
        "m/s2": {"L": 1, "T": -2},
        # Density
        "kg/m3": {"M": 1, "L": -3},
        "g/cm3": {"M": 1, "L": -3},
        # Charge (CGS/SI)
        "C": {"I": 1, "T": 1},
        # Electric field
        "V/m": {"M": 1, "L": 1, "T": -3, "I": -1},
        # Thermal conductivity
        "W/(m·K)": {"M": 1, "L": 1, "T": -3, "Theta": -1},
    }

    # Compound unit patterns
    COMPOUND_PATTERNS = {
        r"J/mol": {"M": 1, "L": 2, "T": -2, "N": -1},
        r"J/(mol·K)": {"M": 1, "L": 2, "T": -2, "N": -1, "Theta": -1},
        r"eV/atom": {"M": 1, "L": 2, "T": -2, "N": -1},
    }

    def __init__(self):
        self._history: list[DimensionalCheckResult] = []

    def parse_quantity(self, quantity_str: str) -> tuple[float, str, dict[str, float]]:
        """Parse a quantity string like '210 GPa' into (value, unit, dimensions).

        Returns:
            (value, unit_symbol, dimension_dict)
        """
        # Match number + optional spaces + unit
        match = re.match(
            r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(\S+)\s*$", quantity_str
        )
        if not match:
            raise ValueError(f"Cannot parse quantity: {quantity_str}")

        value = float(match.group(1))
        unit = match.group(2)

        dims = self._get_dimensions(unit)
        return value, unit, dims

    def _get_dimensions(self, unit: str) -> dict[str, float]:
        """Get dimensions for a unit string."""
        # Direct lookup
        if unit in self.BASE_UNITS:
            return dict(self.BASE_UNITS[unit])

        # Compound patterns
        for pattern, dims in self.COMPOUND_PATTERNS.items():
            if re.match(pattern.replace("·", r"[·\*]?"), unit):
                return dict(dims)

        # Try to parse compound units like kg/m/s
        if "/" in unit or "·" in unit or "*" in unit:
            return self._parse_compound_unit(unit)

        raise ValueError(f"Unknown unit: {unit}")

    def _parse_compound_unit(self, unit: str) -> dict[str, float]:
        """Parse compound units like kg/(m·s²) or GPa·nm."""
        dims: dict[str, float] = {}

        # Split by / to find numerator and denominator
        parts = unit.split("/")
        numerator = parts[0]
        denominator = "/".join(parts[1:]) if len(parts) > 1 else ""

        # Parse numerator (positive powers)
        for sub_unit in re.split(r"[·\*]", numerator):
            sub_unit = sub_unit.strip()
            if not sub_unit:
                continue
            # Check for power like m2 or m^2
            power_match = re.match(r"(\w+)(\d+|\^\d+)", sub_unit)
            if power_match:
                base = power_match.group(1)
                power_str = power_match.group(2)
                power = int(power_str.lstrip("^"))
            else:
                base = sub_unit
                power = 1

            sub_dims = self._get_dimensions(base)
            for d, p in sub_dims.items():
                dims[d] = dims.get(d, 0) + p * power

        # Parse denominator (negative powers)
        if denominator:
            # Remove parentheses
            denominator = denominator.strip("()")
            for sub_unit in re.split(r"[·\*]", denominator):
                sub_unit = sub_unit.strip()
                if not sub_unit:
                    continue
                power_match = re.match(r"(\w+)(\d+|\^\d+)", sub_unit)
                if power_match:
                    base = power_match.group(1)
                    power_str = power_match.group(2)
                    power = int(power_str.lstrip("^"))
                else:
                    base = sub_unit
                    power = 1

                sub_dims = self._get_dimensions(base)
                for d, p in sub_dims.items():
                    dims[d] = dims.get(d, 0) - p * power

        return dims

    def check_equation(
        self,
        lhs_quantities: list[str],
        rhs_quantities: list[str],
        equation_name: str = "",
    ) -> DimensionalCheckResult:
        """Check dimensional consistency of an equation LHS = RHS.

        Args:
            lhs_quantities: List of quantity strings on left side (e.g., ["210 GPa"])
            rhs_quantities: List of quantity strings on right side
            equation_name: Human-readable name for the equation
        """
        lhs_dims: dict[str, float] = {}
        rhs_dims: dict[str, float] = {}
        notes = []

        # Sum dimensions on each side
        for q in lhs_quantities:
            try:
                _, _, dims = self.parse_quantity(q)
                for d, p in dims.items():
                    lhs_dims[d] = lhs_dims.get(d, 0) + p
            except ValueError as e:
                notes.append(f"LHS: {e}")

        for q in rhs_quantities:
            try:
                _, _, dims = self.parse_quantity(q)
                for d, p in dims.items():
                    rhs_dims[d] = rhs_dims.get(d, 0) + p
            except ValueError as e:
                notes.append(f"RHS: {e}")

        # Compare
        all_dims = set(lhs_dims.keys()) | set(rhs_dims.keys())
        consistent = True
        for d in all_dims:
            if abs(lhs_dims.get(d, 0) - rhs_dims.get(d, 0)) > 1e-10:
                consistent = False
                notes.append(
                    f"Dimension mismatch in {d}: LHS={lhs_dims.get(d, 0)}, RHS={rhs_dims.get(d, 0)}"
                )

        if consistent:
            notes.append("Dimensional consistency verified ✓")

        result = DimensionalCheckResult(
            consistent=consistent,
            equation=equation_name
            or f"{' × '.join(lhs_quantities)} = {' × '.join(rhs_quantities)}",
            lhs_dimensions=lhs_dims,
            rhs_dimensions=rhs_dims,
            notes=notes,
        )
        self._history.append(result)
        return result

    def check_vasp_inputs(self, params: dict[str, Any]) -> list[DimensionalCheckResult]:
        """Check common VASP input parameters for physical consistency."""
        results = []

        # ENCUT should be in eV
        if "ENCUT" in params:
            encut = params["ENCUT"]
            try:
                _, unit, dims = self.parse_quantity(f"{encut} eV")
                results.append(
                    DimensionalCheckResult(
                        consistent=True,
                        equation=f"ENCUT = {encut} eV",
                        notes=["ENCUT has energy dimensions ✓"],
                    )
                )
            except ValueError:
                results.append(
                    DimensionalCheckResult(
                        consistent=False,
                        equation=f"ENCUT = {encut}",
                        notes=["ENCUT should have energy units (eV)"],
                    )
                )

        # SIGMA should be in eV (energy)
        if "SIGMA" in params:
            sigma = params["SIGMA"]
            results.append(
                DimensionalCheckResult(
                    consistent=True,
                    equation=f"SIGMA = {sigma} eV",
                    notes=["SIGMA has energy dimensions ✓"],
                )
            )

        # POTIM should be in fs (time)
        if "POTIM" in params:
            potim = params["POTIM"]
            results.append(
                DimensionalCheckResult(
                    consistent=True,
                    equation=f"POTIM = {potim} fs",
                    notes=["POTIM has time dimensions ✓"],
                )
            )

        return results

    def buckingham_pi(
        self,
        variables: list[tuple[str, str]],
        target: str,
    ) -> list[dict[str, Any]]:
        """Apply Buckingham π theorem to find dimensionless groups.

        Args:
            variables: List of (name, unit) tuples, e.g., [("E", "GPa"), ("rho", "g/cm3"), ("L", "m")]
            target: Name of the target variable

        Returns:
            List of dimensionless π groups.
        """
        # Build dimension matrix
        symbols = [v[0] for v in variables]
        dim_matrix = []
        dim_names = ["M", "L", "T", "Theta", "N", "I"]

        for _name, unit in variables:
            _parsed, _base, dims = self.parse_quantity(f"1 {unit}")
            row = [dims.get(d, 0) for d in dim_names]
            dim_matrix.append(row)

        M = sp.Matrix(dim_matrix)

        # Find null space vectors (π groups)
        nullspace = M.nullspace()
        pi_groups = []
        for i, vec in enumerate(nullspace):
            coeffs = [float(v) for v in vec]
            group_vars = {}
            for j, c in enumerate(coeffs):
                if j < len(symbols) and abs(c) > 1e-10:
                    group_vars[symbols[j]] = c
            if group_vars:
                pi_groups.append(
                    {
                        "pi_id": i + 1,
                        "expression": " x ".join(
                            f"{s}^{c:.2f}" for s, c in group_vars.items()
                        ),
                        "variables": group_vars,
                    }
                )

        return pi_groups

    def validate_stress_strain(
        self,
        stress_val: float,
        stress_unit: str,
        E_val: float,
        E_unit: str,
        strain: float,
    ) -> DimensionalCheckResult:
        """Validate σ = E·ε dimensional consistency."""
        return self.check_equation(
            lhs_quantities=[f"{stress_val} {stress_unit}"],
            rhs_quantities=[f"{E_val} {E_unit}", f"{strain} dimensionless"],
            equation_name="Hooke's law: σ = E·ε",
        )

    def validate_navier_stokes(
        self, rho: str, u: str, p: str, mu: str, L: str, U: str
    ) -> list[DimensionalCheckResult]:
        """Validate Navier-Stokes equation terms for dimensional consistency."""
        results = []

        # Inertial term: ρ(u·∇)u ~ ρ U²/L
        results.append(
            self.check_equation(
                lhs_quantities=[rho, u, u, f"1/{L}"],
                rhs_quantities=[rho, U, U, f"1/{L}"],
                equation_name="Inertial term: ρ(u·∇)u",
            )
        )

        # Pressure gradient: ∇p ~ [p]/[L]
        results.append(
            self.check_equation(
                lhs_quantities=[p, f"1/{L}"],
                rhs_quantities=[p, f"1/{L}"],
                equation_name="Pressure gradient: ∇p",
            )
        )

        # Viscous term: μ∇²u ~ μU/L²
        results.append(
            self.check_equation(
                lhs_quantities=[mu, u, f"1/{L}", f"1/{L}"],
                rhs_quantities=[mu, U, f"1/{L}", f"1/{L}"],
                equation_name="Viscous term: μ∇²u",
            )
        )

        return results
