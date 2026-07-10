"""Unit normalization and dimensional consistency checking.

Materials science tools use wildly different unit systems:
  - VASP:    eV, Å, GPa, kB
  - QE:      Rydberg, Bohr, kbar
  - CP2K:    Hartree, Angstrom
  - GROMACS: kJ/mol, nm, bar
  - LAMMPS:  depends on `units` command (real/metal/si/cgs)

This module provides:
  1. UnitNormalizer: convert between common materials science units
  2. PhysicalRange: expected value ranges for sanity checking
  3. dimensional_consistency_hook: POST_TOOL_USE hook that warns when
     tool outputs have physically impossible values or unit mismatches

No sympy dependency — just a lookup table + conversion factors.
Keeping it dead simple because the unit space is finite and well-known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from huginn.hooks import POST_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)

# ── Conversion factors to SI base units ──────────────────────────
# Format: (target_si, factor, offset)
# value_si = value * factor + offset
# Grouped by physical quantity.

_UNIT_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "energy": {
        # → Joule
        "ev": (1.602176634e-19, 0.0),
        "rydberg": (2.179872361e-18, 0.0),
        "hartree": (4.359744722e-18, 0.0),
        "kj/mol": (1.660539069e-21, 0.0),  # kJ/mol per particle
        "kcal/mol": (6.9477e-21, 0.0),
        "joule": (1.0, 0.0),
    },
    "length": {
        # → meter
        "angstrom": (1e-10, 0.0),
        "bohr": (5.291772109e-11, 0.0),
        "nm": (1e-9, 0.0),
        "pm": (1e-12, 0.0),
        "meter": (1.0, 0.0),
    },
    "pressure": {
        # → Pascal
        "gpa": (1e9, 0.0),
        "mpa": (1e6, 0.0),
        "kbar": (1e8, 0.0),
        "bar": (1e5, 0.0),
        "atm": (1.01325e5, 0.0),
        "pascal": (1.0, 0.0),
    },
    "temperature": {
        # → Kelvin
        "k": (1.0, 0.0),
        "celsius": (1.0, 273.15),
        "fahrenheit": (5.0 / 9.0, 459.67 * 5.0 / 9.0),
    },
    "force": {
        # → Newton
        "ev/angstrom": (1.602176634e-9, 0.0),
        "hartree/bohr": (8.2387e-8, 0.0),
        "rydberg/bohr": (4.1193e-8, 0.0),
        "newton": (1.0, 0.0),
    },
    "time": {
        # → second
        "fs": (1e-15, 0.0),
        "ps": (1e-12, 0.0),
        "ns": (1e-9, 0.0),
        "au_time": (2.4188843e-17, 0.0),  # atomic unit of time
        "second": (1.0, 0.0),
    },
}


def convert(value: float, from_unit: str, to_unit: str, quantity: str) -> float:
    """Convert a value between two units of the same quantity.

    Args:
        value: numeric value in from_unit
        from_unit: source unit (e.g. "ev", "angstrom", "gpa")
        to_unit: target unit
        quantity: physical quantity ("energy", "length", "pressure", etc.)

    Returns:
        value in to_unit

    Raises:
        KeyError if units not in table
    """
    table = _UNIT_TABLE.get(quantity)
    if table is None:
        raise KeyError(f"Unknown quantity: {quantity}")
    from_factor, from_offset = table[from_unit.lower()]
    to_factor, to_offset = table[to_unit.lower()]
    # value → SI → target
    si_value = value * from_factor + from_offset
    return (si_value - to_offset) / to_factor


def to_si(value: float, unit: str, quantity: str) -> float:
    """Convert to SI base unit."""
    table = _UNIT_TABLE.get(quantity)
    if table is None:
        raise KeyError(f"Unknown quantity: {quantity}")
    factor, offset = table[unit.lower()]
    return value * factor + offset


def from_si(si_value: float, target_unit: str, quantity: str) -> float:
    """Convert from SI base unit to target unit."""
    table = _UNIT_TABLE.get(quantity)
    if table is None:
        raise KeyError(f"Unknown quantity: {quantity}")
    factor, offset = table[target_unit.lower()]
    return (si_value - offset) / factor


# ── Physical value range checking ───────────────────────────────

@dataclass
class PhysicalRange:
    """Expected range for a physical property."""
    property_name: str
    min_si: float
    max_si: float
    unit: str  # SI unit name
    quantity: str  # energy/length/pressure/...
    warn_msg: str = ""

    def check(self, value: float, unit: str | None = None) -> tuple[bool, str]:
        """Check if value is within expected range. Returns (ok, message)."""
        try:
            si_val = to_si(value, unit or self.unit, self.quantity)
        except (KeyError, TypeError):
            return True, ""  # can't check, pass
        if si_val < self.min_si or si_val > self.max_si:
            return False, (
                f"{self.property_name}={value} {unit or self.unit} "
                f"is outside expected range "
                f"[{from_si(self.min_si, unit or self.unit, self.quantity):.4g}, "
                f"{from_si(self.max_si, unit or self.unit, self.quantity):.4g}] "
                f"{self.unit}. {self.warn_msg}"
            )
        return True, ""


# Common physical ranges (SI units)
_RANGES: list[PhysicalRange] = [
    PhysicalRange("band_gap", 0.0, 10.0 * 1.602e-19, "ev", "energy",
                  "Band gaps are typically 0-10 eV. >10 eV is an insulator wide gap."),
    PhysicalRange("lattice_constant", 1e-10, 5e-9, "angstrom", "length",
                  "Lattice constants are typically 1-50 Å."),
    PhysicalRange("bulk_modulus", 1e9, 1000e9, "gpa", "pressure",
                  "Bulk modulus typically 1-500 GPa."),
    PhysicalRange("youngs_modulus", 0.1e9, 2000e9, "gpa", "pressure",
                  "Young's modulus typically 1-2000 GPa."),
    PhysicalRange("melting_point", 0.0, 5000.0, "k", "temperature",
                  "Melting points typically 0-5000 K."),
    PhysicalRange("timestep_md", 0.1e-15, 10e-15, "fs", "time",
                  "MD timesteps are typically 0.1-5 fs."),
    PhysicalRange("temperature", 0.0, 10000.0, "k", "temperature",
                  "Simulation temperatures typically 0-10000 K."),
    PhysicalRange("pressure", -1e8, 1e12, "gpa", "pressure",
                  "Pressures typically -0.1 to 1000 GPa."),
    PhysicalRange("energy_per_atom", -20e-19, 5e-19, "ev", "energy",
                  "DFT energy per atom typically -20 to 5 eV."),
    PhysicalRange("density", 1.0, 25000.0, "kg/m^3", "length",
                  "Density typically 1-25000 kg/m³."),
]

# Map property names (case-insensitive) to range checkers
_RANGE_MAP: dict[str, PhysicalRange] = {
    r.property_name.lower(): r for r in _RANGES
}

# Common aliases
_RANGE_ALIASES: dict[str, str] = {
    "e_gap": "band_gap",
    "gap": "band_gap",
    "a": "lattice_constant",
    "k": "bulk_modulus",
    "e": "youngs_modulus",
    "t_melt": "melting_point",
    "dt": "timestep_md",
    "temp": "temperature",
    "rho": "density",
    "energy": "energy_per_atom",
}


def check_value(name: str, value: float, unit: str | None = None) -> tuple[bool, str]:
    """Check a named property value against expected physical range.

    Returns (ok, warning_message). ok=True means value is fine.
    """
    key = name.lower().strip()
    if key in _RANGE_ALIASES:
        key = _RANGE_ALIASES[key]
    rng = _RANGE_MAP.get(key)
    if rng is None:
        return True, ""  # no range data, pass
    return rng.check(value, unit)


# ── POST_TOOL_USE hook: dimensional consistency ─────────────────

async def dimensional_consistency_hook(ctx: HookContext) -> None:
    """POST_TOOL_USE hook: check tool output values for physical plausibility.

    Scans ctx.result for key_properties with numeric values and checks them
    against known physical ranges. Does NOT block — only warns via metadata.
    """
    if ctx.result is None:
        return

    # Extract key_properties from result
    props: dict[str, Any] = {}
    if isinstance(ctx.result, dict):
        props = ctx.result.get("key_properties", {})
        if not props:
            # Try nested data
            data = ctx.result.get("data", {})
            if isinstance(data, dict):
                props = data.get("key_properties", {})
    elif hasattr(ctx.result, "key_properties"):
        props = getattr(ctx.result, "key_properties", {})

    if not props:
        return

    warnings: list[str] = []
    for name, val in props.items():
        if not isinstance(val, (int, float)):
            continue
        # Skip very small values that are probably flags/indices
        if isinstance(val, int) and abs(val) < 1000:
            continue
        ok, msg = check_value(name, float(val))
        if not ok:
            warnings.append(msg)

    if warnings:
        ctx.metadata["dimensional_warnings"] = warnings
        logger.warning(
            "Dimensional consistency check: %d warning(s) for %s: %s",
            len(warnings), ctx.tool_name, "; ".join(warnings),
        )


def register_dimensional_hook(hm: HookManager) -> None:
    """Register the dimensional consistency hook."""
    if getattr(hm, "_dimensional_hook_registered", False):
        return
    hm.register(POST_TOOL_USE, dimensional_consistency_hook)
    hm._dimensional_hook_registered = True
    logger.info("Dimensional consistency hook registered (POST_TOOL_USE)")
