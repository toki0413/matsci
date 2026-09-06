"""Constraint adapter — maps reference-domain constraints to task contexts.

This is the concrete realization of the "homeomorphism" in the framework:
abstract safety/quality invariants are translated into checks against actual
tool outputs (DFT energies, MD trajectories, phonon frequencies, etc.).
"""

from __future__ import annotations

from typing import Any

from huginn.constraints.operators import QualityOperator, SafetyOperator
from huginn.constraints.reference import Constraint, ConstraintResult


def _result(
    name: str,
    passed: bool,
    value: Any,
    expected: str,
    tolerance: float,
    message: str,
    severity: str = "warn",
) -> ConstraintResult:
    return ConstraintResult(
        name=name,
        passed=passed,
        value=value,
        expected=expected,
        tolerance=tolerance,
        message=message,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Quality invariants for materials science calculations
# ---------------------------------------------------------------------------


def _energy_sign(data: dict[str, Any]) -> ConstraintResult:
    energy = data.get("energy")
    if energy is None:
        return _result("energy_sign", True, None, "negative", 0, "Energy not available")
    passed = energy < 0
    return _result(
        "energy_sign",
        passed,
        energy,
        "negative",
        0,
        (
            "Energy is positive — unusual for a stable structure"
            if not passed
            else "Energy sign OK"
        ),
    )


def _force_convergence(data: dict[str, Any]) -> ConstraintResult:
    max_force = data.get("max_force")
    if max_force is None:
        return _result(
            "force_convergence", True, None, "< 0.01", 0, "Max force not available"
        )
    threshold = 0.01
    passed = max_force < threshold
    return _result(
        "force_convergence",
        passed,
        max_force,
        f"< {threshold}",
        threshold,
        (
            f"Max force {max_force:.4f} eV/Å exceeds threshold {threshold}"
            if not passed
            else "Force convergence OK"
        ),
    )


def _band_gap(data: dict[str, Any]) -> ConstraintResult:
    gap = data.get("band_gap")
    if gap is None:
        return _result("band_gap", True, None, ">= 0", 0, "Band gap not available")
    passed = gap >= 0
    return _result(
        "band_gap",
        passed,
        gap,
        ">= 0",
        0,
        f"Negative band gap: {gap:.3f} eV" if not passed else f"Band gap: {gap:.3f} eV",
    )


def _volume_positive(data: dict[str, Any]) -> ConstraintResult:
    volume = data.get("volume")
    if volume is None:
        return _result("volume_positive", True, None, "> 0", 0, "Volume not available")
    passed = volume > 0
    return _result(
        "volume_positive",
        passed,
        volume,
        "> 0",
        0,
        "Volume is non-positive" if not passed else "Volume OK",
    )


_REFERENCE_MAGNETIC_MOMENTS = {
    "Fe": 2.2,
    "Co": 1.7,
    "Ni": 0.6,
    "Mn": 3.5,
    "Cr": 3.0,
    "V": 2.5,
}


def _magnetic_moments(data: dict[str, Any]) -> ConstraintResult:
    moments = data.get("magnetic_moments", {})
    if not moments:
        return _result(
            "magnetic_moments", True, None, "consistent", 0, "No magnetic moments"
        )
    warnings = []
    for element, moment in moments.items():
        ref = _REFERENCE_MAGNETIC_MOMENTS.get(element)
        if ref and abs(moment) > ref * 3:
            warnings.append(f"{element}: {moment:.2f} μB >> reference {ref} μB")
    passed = len(warnings) == 0
    return _result(
        "magnetic_moments",
        passed,
        moments,
        "consistent with reference",
        0,
        "; ".join(warnings) if warnings else "Magnetic moments look reasonable",
    )


def _energy_conservation(data: dict[str, Any]) -> ConstraintResult:
    drift = data.get("energy_drift_per_atom")
    if drift is None:
        return _result(
            "energy_conservation",
            True,
            None,
            "< 0.001",
            0,
            "Energy drift not available",
        )
    threshold = 0.001
    passed = abs(drift) < threshold
    return _result(
        "energy_conservation",
        passed,
        drift,
        f"< {threshold}",
        threshold,
        (
            f"Energy drift {drift:.6f} eV/atom/ps exceeds threshold"
            if not passed
            else "Energy conservation OK"
        ),
    )


def _temperature_stability(data: dict[str, Any]) -> ConstraintResult:
    temp_std = data.get("temperature_std")
    target_temp = data.get("target_temperature", 300)
    if temp_std is None:
        return _result(
            "temperature_stability",
            True,
            None,
            "stable",
            0,
            "Temperature data not available",
        )
    threshold = target_temp * 0.1
    passed = temp_std < threshold
    return _result(
        "temperature_stability",
        passed,
        temp_std,
        f"< {threshold:.1f} K",
        threshold,
        (
            f"Temperature std {temp_std:.1f} K too high"
            if not passed
            else "Temperature stable"
        ),
    )


def _atom_count(data: dict[str, Any]) -> ConstraintResult:
    initial = data.get("initial_atom_count")
    final = data.get("final_atom_count")
    if initial is None or final is None:
        return _result("atom_count", True, None, "equal", 0, "Atom count not available")
    passed = initial == final
    return _result(
        "atom_count",
        passed,
        final,
        str(initial),
        0,
        (
            f"Lost {initial - final} atoms during simulation!"
            if not passed
            else "All atoms accounted for"
        ),
    )


def _density_reasonable(data: dict[str, Any]) -> ConstraintResult:
    density = data.get("density")
    if density is None:
        return _result("density", True, None, "0.5-25", 0, "Density not available")
    passed = 0.5 < density < 25
    return _result(
        "density",
        passed,
        density,
        "0.5-25 g/cm³",
        0,
        (
            f"Density {density:.2f} g/cm³ outside reasonable range"
            if not passed
            else "Density reasonable"
        ),
    )


def _no_imaginary_modes(data: dict[str, Any]) -> ConstraintResult:
    freqs = data.get("frequencies", [])
    if not freqs:
        return _result(
            "imaginary_modes", True, None, "none", 0, "Frequencies not available"
        )
    imaginary = [f for f in freqs if f < -0.1]
    passed = len(imaginary) == 0
    return _result(
        "imaginary_modes",
        passed,
        len(imaginary),
        "0",
        0,
        (
            f"{len(imaginary)} imaginary modes found!"
            if not passed
            else "No imaginary modes"
        ),
    )


# ---------------------------------------------------------------------------
# Safety rules (minimal examples — can be extended)
# ---------------------------------------------------------------------------


def _no_nan_values(data: dict[str, Any]) -> ConstraintResult:
    """Safety rule: numerical results must not contain NaN/Inf."""
    import math

    values = [v for v in data.values() if isinstance(v, (int, float))]
    bad = [v for v in values if not math.isfinite(v)]
    passed = len(bad) == 0
    return _result(
        "finite_values",
        passed,
        len(bad),
        "0",
        0,
        (
            f"{len(bad)} non-finite numerical values detected"
            if bad
            else "All values finite"
        ),
        severity="block",
    )


# ---------------------------------------------------------------------------
# Default constraint library
# ---------------------------------------------------------------------------


def build_default_library() -> tuple[SafetyOperator, QualityOperator]:
    """Return the default (SafetyOperator, QualityOperator) pair."""
    safety = SafetyOperator(
        [
            Constraint("finite_values", "*", "safety", "block", _no_nan_values),
        ]
    )
    quality = QualityOperator(
        [
            # DFT
            Constraint("energy_sign", "dft", "quality", "warn", _energy_sign),
            Constraint(
                "force_convergence", "dft", "quality", "warn", _force_convergence
            ),
            Constraint("band_gap", "dft", "quality", "warn", _band_gap),
            Constraint("volume_positive", "dft", "quality", "block", _volume_positive),
            Constraint("magnetic_moments", "dft", "quality", "warn", _magnetic_moments),
            # MD
            Constraint(
                "energy_conservation", "md", "quality", "warn", _energy_conservation
            ),
            Constraint(
                "temperature_stability", "md", "quality", "warn", _temperature_stability
            ),
            Constraint("atom_count", "md", "quality", "block", _atom_count),
            Constraint("density", "md", "quality", "warn", _density_reasonable),
            # Phonon
            Constraint(
                "imaginary_modes", "phonon", "quality", "block", _no_imaginary_modes
            ),
        ]
    )
    return safety, quality


class ConstraintAdapter:
    """Maps reference-domain constraints to a concrete task scope and evaluates them."""

    def __init__(
        self,
        safety: SafetyOperator | None = None,
        quality: QualityOperator | None = None,
    ):
        self.safety = safety or SafetyOperator()
        self.quality = quality or QualityOperator()

    @classmethod
    def default(cls) -> ConstraintAdapter:
        """Create an adapter with the built-in materials-science constraint library."""
        safety, quality = build_default_library()
        return cls(safety=safety, quality=quality)

    def evaluate(
        self, scope: str, data: dict[str, Any]
    ) -> tuple[list[ConstraintResult], list[ConstraintResult]]:
        """Evaluate safety + quality constraints for ``scope`` against ``data``.

        Returns ``(safety_results, quality_results)``.
        """
        safety_results = self.safety.evaluate(data, scope=scope)
        quality_results = self.quality.evaluate(data, scope=scope)
        # Also run global safety constraints that apply to every scope.
        safety_results += [
            r
            for r in self.safety.evaluate(data)
            if r.name not in {x.name for x in safety_results}
        ]
        return safety_results, quality_results

    def evaluate_all(self, scope: str, data: dict[str, Any]) -> list[ConstraintResult]:
        """Evaluate all constraints and return a flat list."""
        safety, quality = self.evaluate(scope, data)
        return safety + quality
