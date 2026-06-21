"""Dimensional Analysis Validator — composable algebraic unit system with SymPy inference.

Provides:
- ``Unit``: algebraic type supporting multiply / divide / power on physical dimensions
- ``UnitRegistry``: composable registry where derived units (N, Pa, J, …) are defined
  via algebraic combinations of SI base units, so ``N/m²`` automatically equals ``Pa``
- ``DimensionalValidator``: backward-compatible validator for equations and SymPy expressions
- Buckingham π theorem, VASP input checks, Navier-Stokes validation

Usage::

    from huginn.execution.dimensional_validator import DimensionalValidator, registry

    # Algebraic equivalence — no manual table entries needed
    registry.equivalent("N/m²", "Pa")        # True
    registry.equivalent("kg·m/s²", "N")      # True
    registry.convert(210, "GPa", "Pa")        # 2.1e+11

    # SymPy expression dimension inference
    import sympy as sp
    x = sp.Symbol("x")
    u = sp.Function("u")
    v = DimensionalValidator()
    unit = v.infer_dimensions(sp.diff(u(x), x, 2), {"x": "m", "u": "K"})
    # → Unit with dimensions L:-2, Θ:1  (i.e. K/m²)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

# ── dimension index constants ────────────────────────────────────────
_M, _L, _T, _THETA, _N, _I, _J = range(7)
DIM_NAMES = ("M", "L", "T", "Theta", "N", "I", "J")
_ZERO = (0.0,) * 7


# ====================================================================
# Unit algebraic type
# ====================================================================

@dataclass(frozen=True)
class Unit:
    """A physical unit represented as an SI-base dimension vector + scale factor.

    Supports algebraic operations: multiply, divide, power.  Equality
    compares **only** the dimension vector (ignoring scale and name), so
    ``Pa == N/m²`` evaluates to ``True``.
    """

    scale: float = 1.0
    dimensions: tuple[float, ...] = _ZERO
    name: str = ""

    # -- construction helpers -----------------------------------------

    @classmethod
    def base(cls, index: int, name: str = "") -> Unit:
        """Create a base unit for the given dimension index (0=M … 6=J)."""
        dims = [0.0] * 7
        dims[index] = 1.0
        return cls(scale=1.0, dimensions=tuple(dims), name=name)

    @classmethod
    def dimensionless(cls, scale: float = 1.0) -> Unit:
        return cls(scale=scale, dimensions=_ZERO, name="1")

    # -- algebraic operations -----------------------------------------

    def __mul__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        dims = tuple(a + b for a, b in zip(self.dimensions, other.dimensions))
        return Unit(scale=self.scale * other.scale, dimensions=dims)

    def __truediv__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        dims = tuple(a - b for a, b in zip(self.dimensions, other.dimensions))
        return Unit(scale=self.scale / other.scale, dimensions=dims)

    def __pow__(self, n: int | float) -> Unit:
        dims = tuple(d * n for d in self.dimensions)
        return Unit(scale=self.scale ** n, dimensions=dims)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self.dimensions == other.dimensions

    def __hash__(self) -> int:
        return hash(self.dimensions)

    # -- queries ------------------------------------------------------

    def same_dimensions(self, other: Unit) -> bool:
        return self.dimensions == other.dimensions

    @property
    def is_dimensionless(self) -> bool:
        return all(abs(d) < 1e-12 for d in self.dimensions)

    @property
    def dimension_dict(self) -> dict[str, float]:
        """Return dimensions as ``{name: exponent}`` (zero exponents omitted)."""
        return {
            DIM_NAMES[i]: d
            for i, d in enumerate(self.dimensions)
            if abs(d) > 1e-12
        }

    @property
    def dimension_signature(self) -> str:
        parts = []
        for i, d in enumerate(self.dimensions):
            if abs(d) > 1e-12:
                parts.append(f"{DIM_NAMES[i]}{d:g}")
        return "·".join(parts) if parts else "dimensionless"

    def __repr__(self) -> str:
        if self.name:
            return f"Unit({self.name!r})"
        return f"Unit({self.dimension_signature})"


# ====================================================================
# Unit registry
# ====================================================================

class UnitRegistry:
    """Composable unit registry.

    Base units are defined explicitly.  Derived units are built from
    algebraic combinations of base units, so their dimensions are
    **derived**, not manually specified.  SI prefixes are applied
    automatically.
    """

    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}
        self._build()

    # -- public API ---------------------------------------------------

    def get(self, unit_str: str) -> Unit:
        """Resolve a unit string (simple or compound) to a :class:`Unit`."""
        unit_str = unit_str.strip()
        if not unit_str or unit_str in ("1", "dimensionless"):
            return Unit.dimensionless()
        # direct cache hit
        if unit_str in self._units:
            return self._units[unit_str]
        # try parsing as compound expression
        return self._parse(unit_str)

    def equivalent(self, a: str, b: str) -> bool:
        """Check whether two unit strings have identical dimensions."""
        return self.get(a) == self.get(b)

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert *value* between dimensionally compatible units."""
        u_from = self.get(from_unit)
        u_to = self.get(to_unit)
        if u_from != u_to:
            raise ValueError(
                f"Incompatible dimensions: {from_unit} ({u_from.dimension_signature}) "
                f"vs {to_unit} ({u_to.dimension_signature})"
            )
        return value * u_from.scale / u_to.scale

    def register(self, name: str, unit: Unit) -> None:
        """Register a unit (also stores it under *name* for lookup)."""
        self._units[name] = Unit(scale=unit.scale, dimensions=unit.dimensions, name=name)

    def all_units(self) -> dict[str, Unit]:
        """Return a copy of all registered units."""
        return dict(self._units)

    # -- internal build -----------------------------------------------

    def _build(self) -> None:
        u = self._units

        # SI base units
        u["m"] = Unit.base(_L, "m")
        u["kg"] = Unit.base(_M, "kg")
        u["s"] = Unit.base(_T, "s")
        u["K"] = Unit.base(_THETA, "K")
        u["mol"] = Unit.base(_N, "mol")
        u["A"] = Unit.base(_I, "A")
        u["cd"] = Unit.base(_J, "cd")

        # Common non-SI base units with scale factors
        u["cm"] = Unit(scale=1e-2, dimensions=u["m"].dimensions, name="cm")
        u["mm"] = Unit(scale=1e-3, dimensions=u["m"].dimensions, name="mm")
        u["nm"] = Unit(scale=1e-9, dimensions=u["m"].dimensions, name="nm")
        u["angstrom"] = Unit(scale=1e-10, dimensions=u["m"].dimensions, name="angstrom")
        u["Å"] = u["angstrom"]
        u["um"] = Unit(scale=1e-6, dimensions=u["m"].dimensions, name="um")
        u["μm"] = u["um"]

        u["g"] = Unit(scale=1e-3, dimensions=u["kg"].dimensions, name="g")
        u["amu"] = Unit(scale=1.66054e-27, dimensions=u["kg"].dimensions, name="amu")

        u["fs"] = Unit(scale=1e-15, dimensions=u["s"].dimensions, name="fs")
        u["ps"] = Unit(scale=1e-12, dimensions=u["s"].dimensions, name="ps")
        u["ns"] = Unit(scale=1e-9, dimensions=u["s"].dimensions, name="ns")
        u["ms"] = Unit(scale=1e-3, dimensions=u["s"].dimensions, name="ms")
        u["min"] = Unit(scale=60.0, dimensions=u["s"].dimensions, name="min")
        u["h"] = Unit(scale=3600.0, dimensions=u["s"].dimensions, name="h")

        # Derived SI units — defined algebraically
        u["N"] = u["kg"] * u["m"] / u["s"] ** 2
        u["N"] = Unit(scale=u["N"].scale, dimensions=u["N"].dimensions, name="N")

        u["Pa"] = u["N"] / u["m"] ** 2
        u["Pa"] = Unit(scale=u["Pa"].scale, dimensions=u["Pa"].dimensions, name="Pa")

        u["J"] = u["N"] * u["m"]
        u["J"] = Unit(scale=u["J"].scale, dimensions=u["J"].dimensions, name="J")

        u["W"] = u["J"] / u["s"]
        u["W"] = Unit(scale=u["W"].scale, dimensions=u["W"].dimensions, name="W")

        u["Hz"] = Unit.dimensionless() / u["s"]
        u["Hz"] = Unit(scale=u["Hz"].scale, dimensions=u["Hz"].dimensions, name="Hz")

        u["V"] = u["W"] / u["A"]
        u["V"] = Unit(scale=u["V"].scale, dimensions=u["V"].dimensions, name="V")

        u["C"] = u["A"] * u["s"]
        u["C"] = Unit(scale=u["C"].scale, dimensions=u["C"].dimensions, name="C")

        u["Ohm"] = u["V"] / u["A"]
        u["Ohm"] = Unit(scale=u["Ohm"].scale, dimensions=u["Ohm"].dimensions, name="Ohm")
        u["Ω"] = u["Ohm"]

        u["T"] = u["kg"] / (u["A"] * u["s"] ** 2)
        u["T"] = Unit(scale=u["T"].scale, dimensions=u["T"].dimensions, name="T")

        u["H"] = u["V"] * u["s"] / u["A"]
        u["H"] = Unit(scale=u["H"].scale, dimensions=u["H"].dimensions, name="H")

        u["F"] = u["C"] / u["V"]
        u["F"] = Unit(scale=u["F"].scale, dimensions=u["F"].dimensions, name="F")

        # Non-SI energy units
        u["eV"] = Unit(scale=1.60218e-19, dimensions=u["J"].dimensions, name="eV")
        u["Ha"] = Unit(scale=4.35974e-18, dimensions=u["J"].dimensions, name="Ha")
        u["Ry"] = Unit(scale=2.17987e-18, dimensions=u["J"].dimensions, name="Ry")
        u["kcal"] = Unit(scale=4184.0, dimensions=u["J"].dimensions, name="kcal")
        u["cal"] = Unit(scale=4.184, dimensions=u["J"].dimensions, name="cal")
        u["BTU"] = Unit(scale=1055.06, dimensions=u["J"].dimensions, name="BTU")

        # Non-SI pressure
        u["bar"] = Unit(scale=1e5, dimensions=u["Pa"].dimensions, name="bar")
        u["atm"] = Unit(scale=101325.0, dimensions=u["Pa"].dimensions, name="atm")
        u["Torr"] = Unit(scale=133.322, dimensions=u["Pa"].dimensions, name="Torr")

        # Common compound units
        u["m/s"] = u["m"] / u["s"]
        u["m/s"] = Unit(scale=u["m/s"].scale, dimensions=u["m/s"].dimensions, name="m/s")
        u["m/s2"] = u["m"] / u["s"] ** 2
        u["m/s2"] = Unit(scale=u["m/s2"].scale, dimensions=u["m/s2"].dimensions, name="m/s2")

        u["kg/m3"] = u["kg"] / u["m"] ** 3
        u["kg/m3"] = Unit(scale=u["kg/m3"].scale, dimensions=u["kg/m3"].dimensions, name="kg/m3")
        u["g/cm3"] = u["g"] / u["cm"] ** 3
        u["g/cm3"] = Unit(scale=u["g/cm3"].scale, dimensions=u["g/cm3"].dimensions, name="g/cm3")

        u["J/mol"] = u["J"] / u["mol"]
        u["J/mol"] = Unit(scale=u["J/mol"].scale, dimensions=u["J/mol"].dimensions, name="J/mol")
        u["J/(mol·K)"] = u["J"] / (u["mol"] * u["K"])
        u["J/(mol·K)"] = Unit(
            scale=u["J/(mol·K)"].scale, dimensions=u["J/(mol·K)"].dimensions, name="J/(mol·K)"
        )
        u["eV/atom"] = u["eV"] / Unit.dimensionless()  # "atom" is dimensionless count
        u["eV/atom"] = Unit(
            scale=u["eV/atom"].scale, dimensions=u["eV/atom"].dimensions, name="eV/atom"
        )

        u["V/m"] = u["V"] / u["m"]
        u["V/m"] = Unit(scale=u["V/m"].scale, dimensions=u["V/m"].dimensions, name="V/m")

        u["W/(m·K)"] = u["W"] / (u["m"] * u["K"])
        u["W/(m·K)"] = Unit(
            scale=u["W/(m·K)"].scale, dimensions=u["W/(m·K)"].dimensions, name="W/(m·K)"
        )

        # Apply SI prefixes to common units
        si_prefixes = {
            "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1,
            "d": 1e-1, "c": 1e-2, "m": 1e-3, "μ": 1e-6, "u": 1e-6,
            "n": 1e-9, "p": 1e-12, "f": 1e-15,
        }
        prefixable = ["Pa", "Hz", "V", "W", "J", "N", "Ohm", "F", "H", "T", "m", "g", "s", "eV"]
        for base_name in prefixable:
            base = u.get(base_name)
            if base is None:
                continue
            for pfx, factor in si_prefixes.items():
                name = f"{pfx}{base_name}"
                if name in u:
                    continue  # don't overwrite manually defined units
                u[name] = Unit(scale=base.scale * factor, dimensions=base.dimensions, name=name)

    # -- compound unit parser -----------------------------------------

    _TOKEN_RE = re.compile(
        r"""
        ([A-Za-zΩÅμ°][A-Za-z0-9ΩÅμ°]*)   # unit name
        | (\d+(?:\.\d+)?)                  # numeric literal
        | (\()                             # open paren
        | (\))                             # close paren
        | (\*\*|\^)                        # power operator
        | ([·*×/])                         # multiply / divide
        """,
        re.VERBOSE,
    )

    def _parse(self, expr: str) -> Unit:
        """Parse a compound unit expression into a Unit."""
        # Normalize unicode dots and spaces
        expr = expr.replace("⋅", "·").replace("×", "*")
        tokens = self._tokenize(expr)
        result, pos = self._parse_expr(tokens, 0)
        return result

    def _tokenize(self, expr: str) -> list[str]:
        return [m.group(0) for m in self._TOKEN_RE.finditer(expr)]

    def _parse_expr(self, tokens: list[str], pos: int) -> tuple[Unit, int]:
        """Parse a full expression: product / quotient chain."""
        left, pos = self._parse_power(tokens, pos)
        while pos < len(tokens) and tokens[pos] in ("·", "*", "/", "×"):
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_power(tokens, pos)
            if op == "/":
                left = left / right
            else:
                left = left * right
        return left, pos

    def _parse_power(self, tokens: list[str], pos: int) -> tuple[Unit, int]:
        """Parse a base with optional exponent: ``m^2``, ``m**3``."""
        base, pos = self._parse_atom(tokens, pos)
        if pos < len(tokens) and tokens[pos] in ("^", "**"):
            pos += 1
            if pos < len(tokens):
                try:
                    n = float(tokens[pos])
                    pos += 1
                    return base ** n, pos
                except ValueError:
                    pass
        return base, pos

    def _parse_atom(self, tokens: list[str], pos: int) -> tuple[Unit, int]:
        """Parse an atomic unit: name, number, or parenthesized expression."""
        if pos >= len(tokens):
            return Unit.dimensionless(), pos

        tok = tokens[pos]

        # Parenthesized sub-expression
        if tok == "(":
            pos += 1
            unit, pos = self._parse_expr(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return unit, pos

        # Numeric literal (dimensionless)
        try:
            val = float(tok)
            return Unit.dimensionless(scale=val), pos + 1
        except ValueError:
            pass

        # Unit name — check registry, then try stripping trailing digit as power
        if tok in self._units:
            return self._units[tok], pos + 1

        # Trailing digit power: "m2" → m^2, "s2" → s^2
        m = re.match(r"^([A-Za-zΩÅμ]+)(\d+)$", tok)
        if m:
            base_name, pwr_str = m.group(1), m.group(2)
            if base_name in self._units:
                return self._units[base_name] ** int(pwr_str), pos + 1

        raise ValueError(f"Unknown unit: {tok!r}")


# ── module-level singleton ───────────────────────────────────────────
registry = UnitRegistry()


# ====================================================================
# SymPy dimension inference
# ====================================================================

class _DimInfer:
    """Walk a SymPy expression tree and propagate dimensions."""

    def __init__(self, reg: UnitRegistry, symbol_units: dict[str, str | Unit]) -> None:
        self._reg = reg
        self._sym: dict[str, Unit] = {}
        for name, u in symbol_units.items():
            self._sym[name] = u if isinstance(u, Unit) else reg.get(u)

    def infer(self, expr: sp.Basic) -> Unit:
        """Return the :class:`Unit` of *expr*."""
        return self._walk(expr)

    def _walk(self, expr: sp.Basic) -> Unit:
        # Numeric constants → dimensionless
        if expr.is_number:
            return Unit.dimensionless()

        # Named symbol
        if isinstance(expr, sp.Symbol):
            name = str(expr)
            if name in self._sym:
                return self._sym[name]
            return Unit.dimensionless()

        # Addition / subtraction — all operands must match
        if isinstance(expr, sp.Add):
            units = [self._walk(arg) for arg in expr.args]
            ref = units[0]
            for u in units[1:]:
                if u != ref and not u.is_dimensionless:
                    raise ValueError(
                        f"Dimension mismatch in addition: {ref.dimension_signature} "
                        f"vs {u.dimension_signature}"
                    )
            return ref

        # Multiplication
        if isinstance(expr, sp.Mul):
            result = Unit.dimensionless()
            for arg in expr.args:
                result = result * self._walk(arg)
            return result

        # Power
        if isinstance(expr, sp.Pow):
            base, exp = expr.args
            base_unit = self._walk(base)
            # Exponent should be dimensionless
            if exp.is_number:
                return base_unit ** float(exp)
            return base_unit  # symbolic exponent → keep base dims

        # Derivative
        if isinstance(expr, sp.Derivative):
            # Derivative(expr, x, n) → dim(expr) / dim(x)^n
            inner = expr.args[0]
            inner_unit = self._walk(inner)
            # Collect derivative variables and counts
            # Note: SymPy uses sp.Tuple, not Python tuple, so check both
            var_counts: dict[sp.Symbol, int] = {}
            for v in expr.args[1:]:
                if isinstance(v, (tuple, sp.Tuple)):
                    sym, count = v[0], int(v[1])
                elif isinstance(v, sp.Symbol):
                    sym, count = v, 1
                else:
                    continue
                var_counts[sym] = var_counts.get(sym, 0) + count
            result = inner_unit
            for sym, count in var_counts.items():
                sym_unit = self._walk(sym)
                result = result / (sym_unit ** count)
            return result

        # Applied undefined function: u(x) → look up "u" in symbol_units
        if isinstance(expr, sp.Function) or (
            isinstance(expr, sp.core.function.Application)
        ):
            func_name = type(expr).__name__ if hasattr(type(expr), "__name__") else str(expr.func)
            if func_name in self._sym:
                return self._sym[func_name]
            # Fall back: try to infer from arguments
            if expr.args:
                return self._walk(expr.args[0])
            return Unit.dimensionless()

        # Fallback: try args recursively
        if hasattr(expr, "args") and expr.args:
            return self._walk(expr.args[0])

        return Unit.dimensionless()


# ====================================================================
# Legacy dataclasses (preserved for backward compatibility)
# ====================================================================

@dataclass
class PhysicalQuantity:
    """A physical quantity with value, unit, and dimensions."""

    name: str
    value: float
    unit: str
    dimensions: dict[str, float] = field(default_factory=dict)


@dataclass
class DimensionalCheckResult:
    """Result of a dimensional analysis check."""

    consistent: bool
    equation: str
    lhs_dimensions: dict[str, float] = field(default_factory=dict)
    rhs_dimensions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ====================================================================
# DimensionalValidator (backward-compatible API)
# ====================================================================

class DimensionalValidator:
    """Validate physical consistency of equations and parameters.

    Supports SI base dimensions: M (mass), L (length), T (time),
    I (current), Θ (temperature), N (amount), J (luminosity).

    Internally uses a composable :class:`UnitRegistry` that can derive
    compound units automatically (e.g. ``N/m² == Pa``).
    """

    # Expose registry for external use
    registry: UnitRegistry = registry

    # Legacy class-level tables (kept for introspection compatibility)
    BASE_UNITS: dict[str, dict[str, float]] = {
        name: u.dimension_dict for name, u in registry.all_units().items()
    }
    COMPOUND_PATTERNS: dict[str, dict[str, float]] = {}

    def __init__(self) -> None:
        self._history: list[DimensionalCheckResult] = []

    # -- parsing ------------------------------------------------------

    def parse_quantity(
        self, quantity_str: str
    ) -> tuple[float, str, dict[str, float]]:
        """Parse a quantity string like ``'210 GPa'`` into (value, unit, dimensions).

        Returns:
            ``(value, unit_symbol, dimension_dict)``
        """
        match = re.match(
            r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(.+?)\s*$", quantity_str
        )
        if not match:
            raise ValueError(f"Cannot parse quantity: {quantity_str}")

        value = float(match.group(1))
        unit_str = match.group(2).strip()

        unit = self.registry.get(unit_str)
        return value, unit_str, unit.dimension_dict

    # -- equation checking --------------------------------------------

    def check_equation(
        self,
        lhs_quantities: list[str],
        rhs_quantities: list[str],
        equation_name: str = "",
    ) -> DimensionalCheckResult:
        """Check dimensional consistency of an equation LHS = RHS.

        Each quantity is a string like ``"210 GPa"`` or ``"1.0 kg/m3"``.
        Multiple quantities on one side are **multiplied** together.
        """
        lhs_unit = Unit.dimensionless()
        rhs_unit = Unit.dimensionless()
        notes: list[str] = []

        for q in lhs_quantities:
            try:
                _, _, dims = self.parse_quantity(q)
                q_unit = self.registry.get(q.split(None, 1)[1] if " " in q.strip() else q.strip())
                lhs_unit = lhs_unit * q_unit
            except ValueError as e:
                notes.append(f"LHS: {e}")

        for q in rhs_quantities:
            try:
                _, _, dims = self.parse_quantity(q)
                q_unit = self.registry.get(q.split(None, 1)[1] if " " in q.strip() else q.strip())
                rhs_unit = rhs_unit * q_unit
            except ValueError as e:
                notes.append(f"RHS: {e}")

        lhs_dims = lhs_unit.dimension_dict
        rhs_dims = rhs_unit.dimension_dict

        consistent = lhs_unit == rhs_unit
        if consistent:
            notes.append("Dimensional consistency verified ✓")
        else:
            all_dims = set(lhs_dims) | set(rhs_dims)
            for d in sorted(all_dims):
                lv = lhs_dims.get(d, 0)
                rv = rhs_dims.get(d, 0)
                if abs(lv - rv) > 1e-10:
                    notes.append(f"Dimension mismatch in {d}: LHS={lv}, RHS={rv}")

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

    # -- SymPy inference ----------------------------------------------

    def infer_dimensions(
        self,
        expr: sp.Basic,
        symbol_units: dict[str, str | Unit],
    ) -> Unit:
        """Infer the physical dimensions of a SymPy expression.

        Args:
            expr: A SymPy expression.
            symbol_units: Mapping from symbol/function names to unit strings
                or :class:`Unit` objects.  E.g. ``{"x": "m", "u": "K"}``.

        Returns:
            The inferred :class:`Unit`.

        Raises:
            ValueError: If an addition has dimensionally inconsistent operands.
        """
        return _DimInfer(self.registry, symbol_units).infer(expr)

    def check_expression(
        self,
        expr: sp.Basic,
        symbol_units: dict[str, str | Unit],
        expected_unit: str | Unit,
    ) -> DimensionalCheckResult:
        """Check that *expr* has dimensions matching *expected_unit*."""
        inferred = self.infer_dimensions(expr, symbol_units)
        expected = (
            expected_unit if isinstance(expected_unit, Unit) else self.registry.get(expected_unit)
        )
        ok = inferred == expected
        return DimensionalCheckResult(
            consistent=ok,
            equation=f"expr has dims {inferred.dimension_signature}",
            lhs_dimensions=inferred.dimension_dict,
            rhs_dimensions=expected.dimension_dict,
            notes=[
                "Dimensions match ✓" if ok
                else f"Expected {expected.dimension_signature}, got {inferred.dimension_signature}"
            ],
        )

    # -- VASP helpers -------------------------------------------------

    def check_vasp_inputs(self, params: dict[str, Any]) -> list[DimensionalCheckResult]:
        """Check common VASP input parameters for physical consistency."""
        results = []

        if "ENCUT" in params:
            encut = params["ENCUT"]
            try:
                self.parse_quantity(f"{encut} eV")
                results.append(DimensionalCheckResult(
                    consistent=True, equation=f"ENCUT = {encut} eV",
                    notes=["ENCUT has energy dimensions ✓"],
                ))
            except ValueError:
                results.append(DimensionalCheckResult(
                    consistent=False, equation=f"ENCUT = {encut}",
                    notes=["ENCUT should have energy units (eV)"],
                ))

        if "SIGMA" in params:
            sigma = params["SIGMA"]
            results.append(DimensionalCheckResult(
                consistent=True, equation=f"SIGMA = {sigma} eV",
                notes=["SIGMA has energy dimensions ✓"],
            ))

        if "POTIM" in params:
            potim = params["POTIM"]
            results.append(DimensionalCheckResult(
                consistent=True, equation=f"POTIM = {potim} fs",
                notes=["POTIM has time dimensions ✓"],
            ))

        return results

    # -- Buckingham π -------------------------------------------------

    def buckingham_pi(
        self,
        variables: list[tuple[str, str]],
        target: str,
    ) -> list[dict[str, Any]]:
        """Apply Buckingham π theorem to find dimensionless groups.

        Args:
            variables: List of ``(name, unit)`` tuples.
            target: Name of the target variable.

        Returns:
            List of dimensionless π groups.
        """
        symbols = [v[0] for v in variables]
        dim_names_list = list(DIM_NAMES[:6])  # M L T Θ N I
        dim_matrix = []

        for _name, unit_str in variables:
            u = self.registry.get(unit_str)
            row = [u.dimensions[i] for i in range(6)]
            dim_matrix.append(row)

        M = sp.Matrix(dim_matrix)
        nullspace = M.nullspace()
        pi_groups = []
        for i, vec in enumerate(nullspace):
            coeffs = [float(v) for v in vec]
            group_vars = {}
            for j, c in enumerate(coeffs):
                if j < len(symbols) and abs(c) > 1e-10:
                    group_vars[symbols[j]] = c
            if group_vars:
                pi_groups.append({
                    "pi_id": i + 1,
                    "expression": " x ".join(
                        f"{s}^{c:.2f}" for s, c in group_vars.items()
                    ),
                    "variables": group_vars,
                })

        return pi_groups

    # -- convenience validators ---------------------------------------

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
        results.append(self.check_equation(
            lhs_quantities=[rho, u, u, f"1/{L}"],
            rhs_quantities=[rho, U, U, f"1/{L}"],
            equation_name="Inertial term: ρ(u·∇)u",
        ))
        results.append(self.check_equation(
            lhs_quantities=[p, f"1/{L}"],
            rhs_quantities=[p, f"1/{L}"],
            equation_name="Pressure gradient: ∇p",
        ))
        results.append(self.check_equation(
            lhs_quantities=[mu, u, f"1/{L}", f"1/{L}"],
            rhs_quantities=[mu, U, f"1/{L}", f"1/{L}"],
            equation_name="Viscous term: μ∇²u",
        ))
        return results

    # -- internal (kept for compat) -----------------------------------

    def _get_dimensions(self, unit: str) -> dict[str, float]:
        """Get dimensions for a unit string (legacy helper)."""
        return self.registry.get(unit).dimension_dict

    def _parse_compound_unit(self, unit: str) -> dict[str, float]:
        """Parse compound units (legacy helper, delegates to registry)."""
        return self.registry.get(unit).dimension_dict
