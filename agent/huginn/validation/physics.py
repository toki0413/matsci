"""Physical validation system — check calculation results for physical reasonableness.

Validates that computed quantities obey known physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    value: Any
    expected: Any
    tolerance: float
    message: str
    # 数值奖励通道: None 时由聚合层从 passed 推导 (passed→1.0, failed→0.0)
    score: float | None = None


class PhysicsValidator:
    """Validates physical reasonableness of calculation results."""

    # Reference magnetic moments (Bohr magnetons)
    REFERENCE_MAGNETIC_MOMENTS = {
        "Fe": 2.2,
        "Co": 1.7,
        "Ni": 0.6,
        "Mn": 3.5,
        "Cr": 3.0,
        "V": 2.5,
    }

    def validate_dft_result(self, result: dict[str, Any]) -> list[ValidationCheck]:
        """Run all DFT-specific validations."""
        checks = []

        # 1. Energy sign check
        checks.append(self._check_energy_sign(result))

        # 2. Force convergence check
        checks.append(self._check_force_convergence(result))

        # 3. Band gap check
        checks.append(self._check_band_gap(result))

        # 4. Volume check (positive)
        checks.append(self._check_volume_positive(result))

        # 5. Magnetic moment consistency
        checks.append(self._check_magnetic_moments(result))

        return checks

    def validate_md_result(self, result: dict[str, Any]) -> list[ValidationCheck]:
        """Run all MD-specific validations."""
        checks = []

        # 1. Energy conservation
        checks.append(self._check_energy_conservation(result))

        # 2. Temperature stability
        checks.append(self._check_temperature_stability(result))

        # 3. No lost atoms
        checks.append(self._check_atom_count(result))

        # 4. Density reasonableness
        checks.append(self._check_density_reasonable(result))

        return checks

    def validate_phonon_result(self, result: dict[str, Any]) -> list[ValidationCheck]:
        """Run phonon-specific validations."""
        checks = []

        # 1. No imaginary modes at gamma (for stable structures)
        checks.append(self._check_no_imaginary_modes(result))

        return checks

    def _check_energy_sign(self, result: dict) -> ValidationCheck:
        energy = result.get("energy")
        if energy is None:
            return ValidationCheck(
                "energy_sign", True, None, "negative", 0, "Energy not available"
            )

        passed = energy < 0
        return ValidationCheck(
            name="energy_sign",
            passed=passed,
            value=energy,
            expected="negative",
            tolerance=0,
            message=(
                "Energy is positive — unusual for a stable structure"
                if not passed
                else "Energy sign OK"
            ),
        )

    def _check_force_convergence(self, result: dict) -> ValidationCheck:
        max_force = result.get("max_force")
        if max_force is None:
            return ValidationCheck(
                "force_convergence", True, None, "< 0.01", 0, "Max force not available"
            )

        threshold = 0.01
        passed = max_force < threshold
        return ValidationCheck(
            name="force_convergence",
            passed=passed,
            value=max_force,
            expected=f"< {threshold}",
            tolerance=threshold,
            message=(
                f"Max force {max_force:.4f} eV/Å exceeds threshold {threshold}"
                if not passed
                else "Force convergence OK"
            ),
        )

    def _check_band_gap(self, result: dict) -> ValidationCheck:
        gap = result.get("band_gap")
        if gap is None:
            return ValidationCheck(
                "band_gap", True, None, "> 0", 0, "Band gap not available"
            )

        passed = gap >= 0
        return ValidationCheck(
            name="band_gap",
            passed=passed,
            value=gap,
            expected=">= 0",
            tolerance=0,
            message=(
                f"Negative band gap: {gap:.3f} eV"
                if not passed
                else f"Band gap: {gap:.3f} eV"
            ),
        )

    def _check_volume_positive(self, result: dict) -> ValidationCheck:
        volume = result.get("volume")
        if volume is None:
            return ValidationCheck(
                "volume_positive", True, None, "> 0", 0, "Volume not available"
            )

        passed = volume > 0
        return ValidationCheck(
            name="volume_positive",
            passed=passed,
            value=volume,
            expected="> 0",
            tolerance=0,
            message="Volume is non-positive" if not passed else "Volume OK",
        )

    def _check_magnetic_moments(self, result: dict) -> ValidationCheck:
        moments = result.get("magnetic_moments", {})
        if not moments:
            return ValidationCheck(
                "magnetic_moments", True, None, "consistent", 0, "No magnetic moments"
            )

        warnings = []
        for element, moment in moments.items():
            ref = self.REFERENCE_MAGNETIC_MOMENTS.get(element)
            if ref and abs(moment) > ref * 3:
                warnings.append(f"{element}: {moment:.2f} μB >> reference {ref} μB")

        passed = len(warnings) == 0
        return ValidationCheck(
            name="magnetic_moments",
            passed=passed,
            value=moments,
            expected="consistent with reference",
            tolerance=0,
            message=(
                "; ".join(warnings) if warnings else "Magnetic moments look reasonable"
            ),
        )

    def _check_energy_conservation(self, result: dict) -> ValidationCheck:
        drift = result.get("energy_drift_per_atom")
        if drift is None:
            return ValidationCheck(
                "energy_conservation",
                True,
                None,
                "< 0.001",
                0,
                "Energy drift not available",
            )

        threshold = 0.001  # eV/atom/ps
        passed = abs(drift) < threshold
        return ValidationCheck(
            name="energy_conservation",
            passed=passed,
            value=drift,
            expected=f"< {threshold}",
            tolerance=threshold,
            message=(
                f"Energy drift {drift:.6f} eV/atom/ps exceeds threshold"
                if not passed
                else "Energy conservation OK"
            ),
        )

    def _check_temperature_stability(self, result: dict) -> ValidationCheck:
        temp_std = result.get("temperature_std")
        target_temp = result.get("target_temperature", 300)
        if temp_std is None:
            return ValidationCheck(
                "temperature_stability",
                True,
                None,
                "stable",
                0,
                "Temperature data not available",
            )

        # Allow 10% fluctuation
        threshold = target_temp * 0.1
        passed = temp_std < threshold
        return ValidationCheck(
            name="temperature_stability",
            passed=passed,
            value=temp_std,
            expected=f"< {threshold:.1f} K",
            tolerance=threshold,
            message=(
                f"Temperature std {temp_std:.1f} K too high"
                if not passed
                else "Temperature stable"
            ),
        )

    def _check_atom_count(self, result: dict) -> ValidationCheck:
        initial = result.get("initial_atom_count")
        final = result.get("final_atom_count")
        if initial is None or final is None:
            return ValidationCheck(
                "atom_count", True, None, "equal", 0, "Atom count not available"
            )

        passed = initial == final
        return ValidationCheck(
            name="atom_count",
            passed=passed,
            value=final,
            expected=initial,
            tolerance=0,
            message=(
                f"Lost {initial - final} atoms during simulation!"
                if not passed
                else "All atoms accounted for"
            ),
        )

    def _check_density_reasonable(self, result: dict) -> ValidationCheck:
        density = result.get("density")
        if density is None:
            return ValidationCheck(
                "density", True, None, "0.5-25", 0, "Density not available"
            )

        # Reasonable density range for condensed matter: 0.5-25 g/cm³
        passed = 0.5 < density < 25
        return ValidationCheck(
            name="density",
            passed=passed,
            value=density,
            expected="0.5-25 g/cm³",
            tolerance=0,
            message=(
                f"Density {density:.2f} g/cm³ outside reasonable range"
                if not passed
                else "Density reasonable"
            ),
        )

    def _check_no_imaginary_modes(self, result: dict) -> ValidationCheck:
        freqs = result.get("frequencies", [])
        if not freqs:
            return ValidationCheck(
                "imaginary_modes", True, None, "none", 0, "Frequencies not available"
            )

        imaginary = [
            f for f in freqs if f < -0.1
        ]  # cm^-1, small negative = numerical noise
        passed = len(imaginary) == 0
        return ValidationCheck(
            name="imaginary_modes",
            passed=passed,
            value=len(imaginary),
            expected=0,
            tolerance=0,
            message=(
                f"{len(imaginary)} imaginary modes found!"
                if not passed
                else "No imaginary modes"
            ),
        )


# Re-export execution/physics_auditor.py 的审计器, 统一从 validation 包导入
from huginn.execution.physics_auditor import (  # noqa: E402
    AuditReport,
    PhysicsAuditor,
    PhysicsFinding,
)
