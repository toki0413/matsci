"""PhysicsAuditor — checks if computational results are physically reasonable.

Catches "successful but unphysical" results that AutoFixLoop misses.
AutoFixLoop only triggers on returncode != 0; PhysicsAuditor runs after
successful execution and flags values that violate physics常识.

Symmetric with RedTeamReviewer: RedTeam reviews research logic (hypothesis
falsifiability, confounders), PhysicsAuditor reviews numerical results
(energy ranges, convergence quality, thermodynamic consistency).

Integration: called after result parsing in VaspTool/LammpsTool, before
returning ToolResult. Findings are attached to data["physics_audit"].
Error-level findings can trigger AutoFixLoop retry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Severity = Literal["info", "warning", "error"]


@dataclass
class PhysicsFinding:
    """A single physics plausibility finding."""

    severity: Severity
    category: str  # unphysical_value | convergence_suspicious | parameter_mismatch | thermodynamic_violation
    message: str
    field: str
    value: Any = None
    expected_range: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "field": self.field,
            "value": self.value,
            "expected_range": self.expected_range,
        }


@dataclass
class AuditReport:
    """Full audit report for one calculation result."""

    tool_name: str
    action: str
    findings: list[PhysicsFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action": self.action,
            "findings": [f.to_dict() for f in self.findings],
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
        }


class PhysicsAuditor:
    """Audits computational results for physical plausibility.

    Usage:
        auditor = PhysicsAuditor()
        report = auditor.audit("vasp_tool", "relax", parsed, input_params)
        if report.has_errors:
            # trigger AutoFixLoop or mark result as suspicious
    """

    def audit(
        self,
        tool_name: str,
        action: str,
        parsed: dict[str, Any],
        input_params: dict[str, Any] | None = None,
    ) -> AuditReport:
        """Audit a parsed result dict. Returns AuditReport."""
        input_params = input_params or {}
        report = AuditReport(tool_name=tool_name, action=action)

        if tool_name == "vasp_tool":
            self._audit_vasp(report, action, parsed, input_params)
        elif tool_name == "lammps_tool":
            self._audit_lammps(report, action, parsed, input_params)
        elif tool_name in ("qe_tool", "cp2k_tool"):
            # QE/CP2K share similar energy/convergence semantics with VASP
            self._audit_vasp(report, action, parsed, input_params)
        else:
            # Unknown tool, skip
            pass

        if report.has_errors:
            logger.warning(
                "physics audit found errors in %s/%s: %s",
                tool_name,
                action,
                [f.message for f in report.findings if f.severity == "error"],
            )
        return report

    # ── VASP / DFT checks ─────────────────────────────────────────

    def _audit_vasp(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        energy = parsed.get("energy")
        n_atoms = self._count_atoms(params)
        band_gap = parsed.get("band_gap")
        converged = parsed.get("converged", False)
        nelm = parsed.get("nelm")
        ispin = parsed.get("ispin", 1)
        mag_moments = parsed.get("magnetic_moments", [])
        volume = parsed.get("volume")

        # 1. Energy per atom — too positive is unphysical
        if energy is not None and n_atoms > 0:
            e_per_atom = energy / n_atoms
            # Typical range: -200 to 0 eV/atom for stable materials.
            # Positive energy means atoms are repelling — usually wrong sign or unbound.
            if e_per_atom > 0:
                report.findings.append(
                    PhysicsFinding(
                        severity="error",
                        category="unphysical_value",
                        message=f"Energy per atom is positive ({e_per_atom:.2f} eV/atom) — material is unbound, likely an error",
                        field="energy",
                        value=e_per_atom,
                        expected_range="-200 to 0 eV/atom",
                    )
                )
            elif e_per_atom < -200:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="unphysical_value",
                        message=f"Energy per atom very negative ({e_per_atom:.2f} eV/atom) — check pseudopotentials",
                        field="energy",
                        value=e_per_atom,
                        expected_range="-200 to 0 eV/atom",
                    )
                )

        # 2. Band gap — typical range 0-15 eV
        if band_gap is not None:
            if band_gap < 0:
                report.findings.append(
                    PhysicsFinding(
                        severity="error",
                        category="unphysical_value",
                        message=f"Negative band gap ({band_gap:.2f} eV) — check calculation method",
                        field="band_gap",
                        value=band_gap,
                        expected_range="0 to 15 eV",
                    )
                )
            elif band_gap > 15:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="unphysical_value",
                        message=f"Very large band gap ({band_gap:.2f} eV) — possible insulator but verify",
                        field="band_gap",
                        value=band_gap,
                        expected_range="0 to 15 eV",
                    )
                )

        # 3. Convergence suspicious: converged=True but NELM was hit.
        # We can't know exact steps from OUTCAR easily, but if NELM is
        # suspiciously high (like 200+) and converged, the SCF probably struggled.
        if converged and nelm is not None:
            if nelm >= 200:
                report.findings.append(
                    PhysicsFinding(
                        severity="info",
                        category="convergence_suspicious",
                        message=f"NELM={nelm} is high — SCF may have struggled to converge",
                        field="nelm",
                        value=nelm,
                        expected_range="typically 40-100 for well-behaved systems",
                    )
                )

        # 4. Not converged but action expects converged result
        if not converged and action in ("relax", "scf", "band", "dos"):
            report.findings.append(
                PhysicsFinding(
                    severity="error",
                    category="convergence_suspicious",
                    message=f"Calculation did not converge for action={action}, results unreliable",
                    field="converged",
                    value=converged,
                    expected_range="True for reliable results",
                )
            )

        # 5. Magnetic moments — unusually large for non-f-elements
        if mag_moments and ispin == 2:
            max_mag = max(abs(m) for m in mag_moments) if mag_moments else 0
            if max_mag > 15:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="unphysical_value",
                        message=f"Magnetic moment {max_mag:.2f} μB exceeds typical range — check for f-electrons or calculation error",
                        field="magnetic_moments",
                        value=max_mag,
                        expected_range="0-15 μB/atom (except lanthanides/actinides)",
                    )
                )

        # 6. Volume — negative or zero is impossible
        if volume is not None and volume <= 0:
            report.findings.append(
                PhysicsFinding(
                    severity="error",
                    category="unphysical_value",
                    message=f"Volume is non-positive ({volume:.2f} Å³) — structure parsing error",
                    field="volume",
                    value=volume,
                    expected_range="> 0 Å³",
                )
            )

        # 7. Parameter mismatch: ISIF=2 but action=relax claims cell optimized
        isif = params.get("isif")
        if isif == 2 and action == "relax":
            report.findings.append(
                PhysicsFinding(
                    severity="info",
                    category="parameter_mismatch",
                    message="ISIF=2 only relaxes ions, cell shape/volume unchanged — full relax needs ISIF=3",
                    field="isif",
                    value=isif,
                    expected_range="3 for full cell relaxation",
                )
            )

    # ── LAMMPS / MD checks ────────────────────────────────────────

    def _audit_lammps(
        self,
        report: AuditReport,
        action: str,
        data: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        thermo = data.get("thermo_data", {})
        final_energy = data.get("final_energy")

        # 1. Temperature — absurd values indicate bad thermostat or explosion
        temps = thermo.get("temp", [])
        if temps:
            max_temp = max(temps)
            if max_temp > 10000:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="unphysical_value",
                        message=f"Temperature reached {max_temp:.0f} K — system may have destabilized (unless plasma)",
                        field="temp",
                        value=max_temp,
                        expected_range="< 10000 K for condensed matter",
                    )
                )

            # Temperature spike: sudden jump > 3x in last few steps
            if len(temps) >= 20:
                recent = temps[-5:]
                baseline = sum(temps[-20:-5]) / 15
                if baseline > 0 and max(recent) / baseline > 3:
                    report.findings.append(
                        PhysicsFinding(
                            severity="warning",
                            category="thermodynamic_violation",
                            message=f"Temperature spike: {max(recent):.0f} K vs baseline {baseline:.0f} K — possible instability",
                            field="temp",
                            value=max(recent),
                            expected_range="stable around target temperature",
                        )
                    )

        # 2. Energy drift — NVE should conserve energy
        energies = thermo.get("toteng", [])
        if len(energies) >= 20:
            # Check drift in last half of trajectory
            half = len(energies) // 2
            late = energies[half:]
            if late:
                energy_range = max(late) - min(late)
                avg_energy = sum(late) / len(late)
                if avg_energy != 0 and abs(energy_range / avg_energy) > 0.01:
                    report.findings.append(
                        PhysicsFinding(
                            severity="warning",
                            category="thermodynamic_violation",
                            message=f"Energy drift {energy_range:.4f} ({abs(energy_range/avg_energy)*100:.2f}% of average) — check timestep",
                            field="toteng",
                            value=energy_range,
                            expected_range="< 1% drift for NVE",
                        )
                    )

        # 3. Pressure — extreme values for condensed matter
        presses = thermo.get("press", [])
        if presses:
            max_press = max(abs(p) for p in presses)
            if max_press > 100000:  # 100 GPa in bar (LAMMPS uses bar)
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="unphysical_value",
                        message=f"Pressure reached {max_press:.0f} bar ({max_press/1000:.1f} GPa) — extremely high",
                        field="press",
                        value=max_press,
                        expected_range="< 100000 bar for typical simulations",
                    )
                )

        # 4. Final energy — None when log parsing failed
        if final_energy is None and action in ("run", "minimize", "equilibrate"):
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="convergence_suspicious",
                    message="No final energy extracted — log file may be incomplete",
                    field="final_energy",
                    value=None,
                    expected_range="numeric energy value",
                )
            )

    # ── helpers ────────────────────────────────────────────────────

    def _count_atoms(self, params: dict[str, Any]) -> int:
        """Extract atom count from VASP input params.

        Tries a direct n_atoms field first, then falls back to parsing
        the structure string (XYZ header or POSCAR counts line).
        """
        # Direct field
        if "n_atoms" in params:
            try:
                return int(params["n_atoms"])
            except (ValueError, TypeError):
                pass
        # From structure string — XYZ format has atom count on the first line
        struct = params.get("structure") or params.get("poscar")
        if isinstance(struct, str):
            lines = struct.strip().split("\n")
            if len(lines) > 2 and lines[0].strip().isdigit():
                try:
                    return int(lines[0].strip())
                except ValueError:
                    pass
            # POSCAR: line 7 (index 6) holds per-element counts
            if len(lines) > 6:
                counts = lines[6].strip().split()
                try:
                    return sum(int(x) for x in counts)
                except ValueError:
                    pass
        return 0
