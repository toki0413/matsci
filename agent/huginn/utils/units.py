"""Unit management for materials science computations.

Provides a unified interface for unit conversion, dimension checking, and
quantity arithmetic. Uses ``pint`` when available; otherwise falls back to
a lightweight built-in registry covering the most common materials science
units (energy, length, pressure, temperature, time, force, density).

Usage:
    from huginn.utils.units import ureg, Q, convert, check_units

    energy = Q(5.0, "eV")          # 5 eV
    energy_j = convert(energy, "J") # 8.01e-19 J
    lattice = Q(4.05, "angstrom")   # 4.05 Å
    pressure = Q(10, "GPa")         # 10 GPa
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Try pint first ────────────────────────────────────────────────────
_pint_available = False
try:
    import pint

    ureg = pint.UnitRegistry()
    ureg.formatter.default_format = "~P"  # compact pretty format

    # Common materials science unit aliases
    ureg.define("@alias electronvolt = eV")
    ureg.define("@alias angstrom = ang = Å")
    _pint_available = True
except ImportError:
    logger.debug("pint not available — using lightweight fallback unit registry")
    ureg = None  # type: ignore[assignment]


# ── Fallback conversion table ─────────────────────────────────────────
# Base SI units: J, m, Pa, K, s, N, kg
# All conversions are to the SI base unit.
_FALLBACK_CONVERSIONS: dict[str, tuple[float, str]] = {
    # Energy → Joule
    "ev": (1.602176634e-19, "J"),
    "electronvolt": (1.602176634e-19, "J"),
    "hartree": (4.3597447222071e-18, "J"),
    "ha": (4.3597447222071e-18, "J"),
    "rydberg": (2.1798723611035e-18, "J"),
    "ry": (2.1798723611035e-18, "J"),
    "kj/mol": (1.0 / 6.02214076e23 * 1000, "J"),
    "kcal/mol": (1.0 / 6.02214076e23 * 4184, "J"),
    "mev": (1.602176634e-22, "J"),
    # Length → meter
    "angstrom": (1e-10, "m"),
    "ang": (1e-10, "m"),
    "a": (1e-10, "m"),  # ambiguous but common in crystallography
    "bohr": (5.29177210903e-11, "m"),
    "nm": (1e-9, "m"),
    "pm": (1e-12, "m"),
    "au_length": (5.29177210903e-11, "m"),
    # Pressure → Pascal
    "gpa": (1e9, "Pa"),
    "mpa": (1e6, "Pa"),
    "kpa": (1e3, "Pa"),
    "bar": (1e5, "Pa"),
    "atm": (101325.0, "Pa"),
    # Temperature → Kelvin (no offset conversions in fallback)
    "kelvin": (1.0, "K"),
    "k": (1.0, "K"),
    # Time → second
    "fs": (1e-15, "s"),
    "ps": (1e-12, "s"),
    "ns": (1e-9, "s"),
    "us": (1e-6, "s"),
    "ms": (1e-3, "s"),
    # Force → Newton
    "n": (1.0, "N"),
    "kn": (1e3, "N"),
    # Mass → kilogram
    "kg": (1.0, "kg"),
    "g": (1e-3, "kg"),
    "amu": (1.66053906660e-27, "kg"),
    "u": (1.66053906660e-27, "kg"),
    # Density → kg/m³
    "g/cm3": (1000.0, "kg/m3"),
    "g/cm^3": (1000.0, "kg/m3"),
    # Frequency → Hz
    "thz": (1e12, "Hz"),
    "ghz": (1e9, "Hz"),
    # SI base units (identity)
    "j": (1.0, "J"),
    "joule": (1.0, "J"),
    "m": (1.0, "m"),
    "meter": (1.0, "m"),
    "pa": (1.0, "Pa"),
    "pascal": (1.0, "Pa"),
    "s": (1.0, "s"),
    "second": (1.0, "s"),
    "hz": (1.0, "Hz"),
}

# Dimension categories for type checking
_DIMENSIONS: dict[str, set[str]] = {
    "energy": {"ev", "electronvolt", "hartree", "ha", "rydberg", "ry", "kj/mol", "kcal/mol", "mev", "j", "joule"},
    "length": {"angstrom", "ang", "a", "bohr", "nm", "pm", "au_length", "m", "meter"},
    "pressure": {"gpa", "mpa", "kpa", "bar", "atm", "pa", "pascal"},
    "temperature": {"kelvin", "k"},
    "time": {"fs", "ps", "ns", "us", "ms", "s", "second"},
    "force": {"n", "kn"},
    "mass": {"kg", "g", "amu", "u"},
    "density": {"g/cm3", "g/cm^3", "kg/m3"},
    "frequency": {"thz", "ghz", "hz"},
}


def _normalize_unit(unit: str) -> str:
    """Normalize a unit string for lookup."""
    return unit.strip().lower().replace(" ", "")


def _get_dimension(unit: str) -> str | None:
    """Return the dimension name for a unit, or None if unknown."""
    norm = _normalize_unit(unit)
    for dim, units in _DIMENSIONS.items():
        if norm in units:
            return dim
    return None


# ── Public API ────────────────────────────────────────────────────────

class _FallbackQuantity:
    """Lightweight quantity for when pint is unavailable."""

    __slots__ = ("magnitude", "unit")

    def __init__(self, magnitude: float, unit: str) -> None:
        self.magnitude = float(magnitude)
        self.unit = _normalize_unit(unit)

    def to(self, target_unit: str) -> _FallbackQuantity:
        target = _normalize_unit(target_unit)
        if self.unit == target:
            return _FallbackQuantity(self.magnitude, target)

        src_conv = _FALLBACK_CONVERSIONS.get(self.unit)
        dst_conv = _FALLBACK_CONVERSIONS.get(target)

        if src_conv is None:
            raise ValueError(f"Unknown source unit: {self.unit}")
        if dst_conv is None:
            raise ValueError(f"Unknown target unit: {target}")

        src_factor, src_base = src_conv
        dst_factor, dst_base = dst_conv

        if src_base != dst_base:
            raise ValueError(
                f"Cannot convert {self.unit} ({src_base}) to {target_unit} ({dst_base})"
            )

        # Convert to base, then to target
        base_value = self.magnitude * src_factor
        return _FallbackQuantity(base_value / dst_factor, target)

    @property
    def dimensionality(self) -> str:
        dim = _get_dimension(self.unit)
        return dim or "unknown"

    def __repr__(self) -> str:
        return f"{self.magnitude} {self.unit}"

    def __str__(self) -> str:
        return f"{self.magnitude:g} {self.unit}"

    def __add__(self, other: _FallbackQuantity) -> _FallbackQuantity:
        if self.unit != other.unit:
            other = other.to(self.unit)
        return _FallbackQuantity(self.magnitude + other.magnitude, self.unit)

    def __sub__(self, other: _FallbackQuantity) -> _FallbackQuantity:
        if self.unit != other.unit:
            other = other.to(self.unit)
        return _FallbackQuantity(self.magnitude - other.magnitude, self.unit)

    def __mul__(self, scalar: float) -> _FallbackQuantity:
        return _FallbackQuantity(self.magnitude * scalar, self.unit)

    def __truediv__(self, scalar: float) -> _FallbackQuantity:
        return _FallbackQuantity(self.magnitude / scalar, self.unit)


def Q(value: float, unit: str) -> Any:
    """Create a quantity. Uses pint if available, else fallback."""
    if _pint_available:
        return ureg.Quantity(value, unit)
    return _FallbackQuantity(value, unit)


def convert(quantity: Any, target_unit: str) -> Any:
    """Convert a quantity to the target unit."""
    if _pint_available:
        return quantity.to(target_unit)
    if isinstance(quantity, _FallbackQuantity):
        return quantity.to(target_unit)
    raise TypeError(f"Cannot convert {type(quantity)} — expected a quantity")


def check_units(quantity: Any, expected_dimension: str) -> bool:
    """Check if a quantity has the expected dimension.

    Args:
        quantity: A pint Quantity or _FallbackQuantity.
        expected_dimension: One of 'energy', 'length', 'pressure',
            'temperature', 'time', 'force', 'mass', 'density', 'frequency'.
    """
    if _pint_available:
        dim_map = {
            "energy": "[energy]",
            "length": "[length]",
            "pressure": "[pressure]",
            "temperature": "[temperature]",
            "time": "[time]",
            "force": "[force]",
            "mass": "[mass]",
            "density": "[mass] / [length] ** 3",
            "frequency": "1 / [time]",
        }
        expected = dim_map.get(expected_dimension)
        if expected is None:
            return False
        return str(quantity.dimensionality) == expected

    if isinstance(quantity, _FallbackQuantity):
        return quantity.dimensionality == expected_dimension
    return False


def to_si(quantity: Any) -> float:
    """Extract the numeric value in the SI base unit."""
    if _pint_available:
        base = quantity.to_base_units()
        return float(base.magnitude)
    if isinstance(quantity, _FallbackQuantity):
        conv = _FALLBACK_CONVERSIONS.get(quantity.unit)
        if conv is None:
            raise ValueError(f"Unknown unit: {quantity.unit}")
        return quantity.magnitude * conv[0]
    raise TypeError(f"Cannot extract SI value from {type(quantity)}")


def format_quantity(quantity: Any, precision: int = 6) -> str:
    """Format a quantity for display."""
    if _pint_available:
        return f"{quantity:.{precision}g~P}"
    if isinstance(quantity, _FallbackQuantity):
        return f"{quantity.magnitude:.{precision}g} {quantity.unit}"
    return str(quantity)


def is_pint_available() -> bool:
    """Return True if pint is installed and active."""
    return _pint_available


# ── Common materials science unit presets ─────────────────────────────

# Standard unit sets for different simulation types
DFT_UNITS = {
    "energy": "eV",
    "length": "angstrom",
    "pressure": "GPa",
    "force": "eV/angstrom",
}

MD_UNITS = {
    "energy": "eV",
    "length": "angstrom",
    "time": "fs",
    "temperature": "K",
    "velocity": "angstrom/fs",
}

SI_UNITS = {
    "energy": "J",
    "length": "m",
    "pressure": "Pa",
    "temperature": "K",
    "time": "s",
    "force": "N",
    "mass": "kg",
}
