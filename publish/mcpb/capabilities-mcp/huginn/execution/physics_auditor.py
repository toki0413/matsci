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
        elif tool_name == "gaussian_tool":
            self._audit_gaussian(report, action, parsed, input_params)
        elif tool_name == "orca_tool":
            self._audit_orca(report, action, parsed, input_params)
        elif tool_name == "gromacs_tool":
            self._audit_gromacs(report, action, parsed, input_params)
        elif tool_name == "abaqus_tool":
            self._audit_abaqus(report, action, parsed, input_params)
        elif tool_name == "comsol_tool":
            self._audit_comsol(report, action, parsed, input_params)
        elif tool_name == "openfoam_tool":
            self._audit_openfoam(report, action, parsed, input_params)
        elif tool_name == "fenics_tool":
            self._audit_fenics(report, action, parsed, input_params)
        elif tool_name == "elmer_tool":
            self._audit_elmer(report, action, parsed, input_params)
        elif tool_name == "transolver_tool":
            self._audit_transolver(report, action, parsed, input_params)
        elif tool_name == "mechanical_tool":
            self._audit_mechanical(report, action, parsed, input_params)
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
        """VASP 输出物理合理性审计 (7 项检查)."""
        self._vasp_check_energy(
            report, parsed.get("energy"), self._count_atoms(params)
        )
        self._vasp_check_band_gap(report, parsed.get("band_gap"))
        self._vasp_check_convergence(
            report, parsed.get("converged", False), parsed.get("nelm"), action
        )
        self._vasp_check_mag(
            report, parsed.get("magnetic_moments", []), parsed.get("ispin", 1)
        )
        self._vasp_check_volume(report, parsed.get("volume"))
        self._vasp_check_isif(report, params, action)

    def _vasp_check_energy(
        self, report: AuditReport, energy: Any, n_atoms: int
    ) -> None:
        """1. Energy per atom — too positive is unphysical."""
        if energy is None or n_atoms <= 0:
            return
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

    def _vasp_check_band_gap(
        self, report: AuditReport, band_gap: Any
    ) -> None:
        """2. Band gap — typical range 0-15 eV."""
        if band_gap is None:
            return
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

    def _vasp_check_convergence(
        self, report: AuditReport, converged: Any, nelm: Any, action: str
    ) -> None:
        """3+4. 收敛性: 高 NELM 提示 / 未收敛按 action 分级."""
        if converged and nelm is not None and nelm >= 200:
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

        # Not converged but action expects converged result.
        # Only flag as error for relax (ionic convergence) — scf/band/dos use
        # electronic convergence (audit_20260717/14 P1-5). The tool's parser
        # already applies the right criterion; if it reports converged=False
        # here, that's a real failure. But for scf/band/dos the criterion is
        # softer — demote to warning so agent sees the issue without
        # triggering AutoFixLoop's "soft failure" 2× retry.
        if not converged and action == "relax":
            report.findings.append(
                PhysicsFinding(
                    severity="error",
                    category="convergence_suspicious",
                    message=f"Ionic relaxation did not converge for action={action}, results unreliable",
                    field="converged",
                    value=converged,
                    expected_range="True for reliable results",
                )
            )
        elif not converged and action in ("scf", "band", "dos"):
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="convergence_suspicious",
                    message=(
                        f"Electronic convergence not reached for action={action} "
                        f"(NELM hit or no EDIFF marker). Results may be usable "
                        f"for qualitative inspection but not for quantitative claims."
                    ),
                    field="converged",
                    value=converged,
                    expected_range="True for reliable results",
                )
            )

    def _vasp_check_mag(
        self, report: AuditReport, mag_moments: Any, ispin: Any
    ) -> None:
        """5. Magnetic moments — unusually large for non-f-elements."""
        if not mag_moments or ispin != 2:
            return
        max_mag = max(abs(m) for m in mag_moments)
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

    def _vasp_check_volume(
        self, report: AuditReport, volume: Any
    ) -> None:
        """6. Volume — negative or zero is impossible."""
        if volume is None or volume > 0:
            return
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

    def _vasp_check_isif(
        self, report: AuditReport, params: dict[str, Any], action: str
    ) -> None:
        """7. Parameter mismatch: ISIF=2 but action=relax claims cell optimized."""
        isif = params.get("isif")
        if isif != 2 or action != "relax":
            return
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
        """LAMMPS 输出物理合理性审计 (4 项检查)."""
        thermo = data.get("thermo_data", {})
        self._lmp_check_temperature(report, thermo.get("temp", []))
        self._lmp_check_energy_drift(report, thermo.get("toteng", []))
        self._lmp_check_pressure(report, thermo.get("press", []))
        self._lmp_check_final_energy(report, data.get("final_energy"), action)

    def _lmp_check_temperature(
        self, report: AuditReport, temps: Any
    ) -> None:
        """1. Temperature — absurd values indicate bad thermostat or explosion."""
        if not temps:
            return
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
        if len(temps) < 20:
            return
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

    def _lmp_check_energy_drift(
        self, report: AuditReport, energies: Any
    ) -> None:
        """2. Energy drift — NVE should conserve energy."""
        if len(energies) < 20:
            return
        # Check drift in last half of trajectory
        half = len(energies) // 2
        late = energies[half:]
        if not late:
            return
        energy_range = max(late) - min(late)
        avg_energy = sum(late) / len(late)
        if avg_energy == 0 or abs(energy_range / avg_energy) <= 0.01:
            return
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

    def _lmp_check_pressure(
        self, report: AuditReport, presses: Any
    ) -> None:
        """3. Pressure — extreme values for condensed matter."""
        if not presses:
            return
        max_press = max(abs(p) for p in presses)
        if max_press <= 100000:  # 100 GPa in bar (LAMMPS uses bar)
            return
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

    def _lmp_check_final_energy(
        self, report: AuditReport, final_energy: Any, action: str
    ) -> None:
        """4. Final energy — None when log parsing failed."""
        if final_energy is not None or action not in ("run", "minimize", "equilibrate"):
            return
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

    # ── Gaussian / quantum chemistry checks ───────────────────────

    def _audit_gaussian(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """Gaussian 输出物理合理性审计 (5 项检查)."""
        try:
            self._gauss_check_scf(report, parsed.get("scf_convergence_failure", False))
            self._qc_check_freq(report, parsed.get("frequencies", []), action)
            self._qc_check_energy(report, parsed.get("energy"))
            self._gauss_check_multiplicity(
                report, parsed.get("multiplicity", params.get("multiplicity"))
            )
            self._gauss_check_charge(
                report, parsed.get("charge", params.get("charge"))
            )
        except Exception:
            logger.debug("gaussian audit failed", exc_info=True)

    def _gauss_check_scf(
        self, report: AuditReport, scf_failure: Any
    ) -> None:
        """1. SCF non-convergence."""
        if not scf_failure:
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message="SCF failed to converge — electronic structure is unreliable",
                field="scf_convergence",
                value=True,
                expected_range="converged SCF",
            )
        )

    def _qc_check_freq(
        self, report: AuditReport, frequencies: Any, action: str
    ) -> None:
        """Imaginary frequencies (negative eigenvalues in freq calc)."""
        # Small negative (~< 50 cm^-1) is often numerical noise; large
        # negative means the structure is a saddle point, not a minimum.
        if not frequencies or action != "freq":
            return
        imaginary = [f for f in frequencies if f < 0]
        if not imaginary:
            return
        most_negative = min(imaginary)
        if most_negative < -100:
            report.findings.append(
                PhysicsFinding(
                    severity="error",
                    category="unphysical_value",
                    message=(
                        f"Large imaginary frequency ({most_negative:.1f} cm^-1) — "
                        "structure is a saddle point, not a true minimum"
                    ),
                    field="frequencies",
                    value=most_negative,
                    expected_range="> 0 cm^-1 for a true minimum",
                )
            )
        elif most_negative < -10:
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="unphysical_value",
                    message=(
                        f"Small imaginary frequency ({most_negative:.1f} cm^-1) — "
                        "likely numerical noise or incomplete optimization"
                    ),
                    field="frequencies",
                    value=most_negative,
                    expected_range="> 0 cm^-1 for a true minimum",
                )
            )

    def _qc_check_energy(
        self, report: AuditReport, energy: Any, label: str = "SCF energy"
    ) -> None:
        """Energy magnitude sanity — Hartrees for quantum chemistry."""
        # Bound molecules should have negative energy; extremely negative
        # suggests wrong pseudopotential/basis or wrong atom count.
        if energy is None:
            return
        if energy > 0:
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="unphysical_value",
                    message=(
                        f"{label} is positive ({energy:.4f} Hartree) — "
                        "check charge/multiplicity or basis set"
                    ),
                    field="energy",
                    value=energy,
                    expected_range="< 0 Hartree for bound molecules",
                )
            )
        elif energy < -50000:
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="unphysical_value",
                    message=(
                        f"{label} extremely negative ({energy:.4f} Hartree) — "
                        "check basis set or atom count"
                    ),
                    field="energy",
                    value=energy,
                    expected_range="-2000 to 0 Hartree for typical molecules",
                )
            )

    def _gauss_check_multiplicity(
        self, report: AuditReport, multiplicity: Any
    ) -> None:
        """4a. Charge/multiplicity consistency — multiplicity."""
        if multiplicity is None:
            return
        try:
            m = int(multiplicity)
            if m < 1:
                report.findings.append(
                    PhysicsFinding(
                        severity="error",
                        category="parameter_mismatch",
                        message=f"Multiplicity {m} is invalid — must be >= 1",
                        field="multiplicity",
                        value=m,
                        expected_range=">= 1",
                    )
                )
            elif m > 7:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="parameter_mismatch",
                        message=(
                            f"High spin multiplicity ({m}) — verify intended "
                            "(e.g. transition metal cluster)"
                        ),
                        field="multiplicity",
                        value=m,
                        expected_range="1-7 for typical molecules",
                    )
                )
        except (ValueError, TypeError):
            logger.debug("best-effort op failed", exc_info=True)

    def _gauss_check_charge(
        self, report: AuditReport, charge: Any
    ) -> None:
        """4b. Charge/multiplicity consistency — charge."""
        if charge is None:
            return
        try:
            q = int(charge)
            if abs(q) > 10:
                report.findings.append(
                    PhysicsFinding(
                        severity="warning",
                        category="parameter_mismatch",
                        message=f"Extreme charge ({q:+d}) — verify charge state is intended",
                        field="charge",
                        value=q,
                        expected_range="-10 to +10 for typical molecules",
                    )
                )
        except (ValueError, TypeError):
            logger.debug("best-effort op failed", exc_info=True)

    # ── ORCA / quantum chemistry checks ───────────────────────────

    def _audit_orca(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """ORCA 输出物理合理性审计 (4 项检查)."""
        try:
            energy = parsed.get("energy")
            self._orca_check_energy_presence(report, energy, action)
            self._qc_check_freq(report, parsed.get("frequencies", []), action)
            self._qc_check_energy(report, energy, label="Energy")
            self._orca_check_opt_steps(
                report, action, parsed.get("converged", False),
                parsed.get("optimization_steps", 0),
            )
        except Exception:
            logger.debug("orca audit failed", exc_info=True)

    def _orca_check_energy_presence(
        self, report: AuditReport, energy: Any, action: str
    ) -> None:
        """1. SCF non-convergence — no FINAL SINGLE POINT ENERGY means SCF failed."""
        if energy is not None or action not in ("sp", "opt", "freq"):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message="No FINAL SINGLE POINT ENERGY — SCF likely did not converge",
                field="energy",
                value=None,
                expected_range="finite energy value",
            )
        )

    def _orca_check_opt_steps(
        self, report: AuditReport, action: str, converged: Any, opt_steps: Any
    ) -> None:
        """4. Optimization suspicious — too many steps without converging."""
        if action != "opt" or converged or opt_steps <= 50:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    f"Optimization ran {opt_steps} steps without converging — "
                    "check initial geometry or convergence thresholds"
                ),
                field="optimization_steps",
                value=opt_steps,
                expected_range="< 50 steps for typical optimizations",
            )
        )

    # ── GROMACS / MD checks ────────────────────────────────────────

    def _audit_gromacs(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """GROMACS 输出物理合理性审计 (6 项检查)."""
        try:
            md_log = parsed.get("md_log_data", {})
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            full_text = (stdout + "\n" + stderr).lower()
            self._gmx_check_constraints(
                report, md_log.get("lincs_warnings", 0),
                md_log.get("shake_warnings", 0),
            )
            self._gmx_check_neighbor_list(
                report, md_log.get("neighbor_list_warnings", 0)
            )
            self._gmx_check_temperature(report, md_log.get("temperatures", []))
            self._gmx_check_pressure(report, md_log.get("pressures", []))
            self._gmx_check_energy_drift(report, md_log.get("energies", []))
            self._gmx_check_nan(
                report, md_log.get("has_nan", False), full_text
            )
        except Exception:
            logger.debug("gromacs audit failed", exc_info=True)

    def _gmx_check_constraints(
        self, report: AuditReport, lincs: Any, shake: Any
    ) -> None:
        """1. LINCS/SHAKE constraint warnings."""
        if lincs > 0:
            severity = "error" if lincs > 10 else "warning"
            report.findings.append(
                PhysicsFinding(
                    severity=severity,
                    category="convergence_suspicious",
                    message=(
                        f"{lincs} LINCS warning(s) — constraint violations, "
                        "timestep may be too large"
                    ),
                    field="lincs_warnings",
                    value=lincs,
                    expected_range="0",
                )
            )

        if shake > 0:
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="convergence_suspicious",
                    message=f"{shake} SHAKE warning(s) — constraint violations",
                    field="shake_warnings",
                    value=shake,
                    expected_range="0",
                )
            )

    def _gmx_check_neighbor_list(
        self, report: AuditReport, nl_warnings: Any
    ) -> None:
        """2. Neighbor list warnings."""
        if nl_warnings <= 0:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    f"{nl_warnings} neighbor list warning(s) — "
                    "check nstlist or cutoff distance"
                ),
                field="neighbor_list_warnings",
                value=nl_warnings,
                expected_range="0",
            )
        )

    def _gmx_check_temperature(
        self, report: AuditReport, temps: Any
    ) -> None:
        """3. Temperature spikes."""
        if not temps:
            return
        max_temp = max(temps)
        if max_temp <= 10000:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    f"Temperature reached {max_temp:.0f} K — "
                    "system likely destabilized"
                ),
                field="temperature",
                value=max_temp,
                expected_range="< 10000 K for condensed matter",
            )
        )

    def _gmx_check_pressure(
        self, report: AuditReport, presses: Any
    ) -> None:
        """4. Pressure extremes."""
        if not presses:
            return
        max_press = max(abs(p) for p in presses)
        if max_press <= 100000:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    f"Pressure reached {max_press:.0f} bar — extremely high"
                ),
                field="pressure",
                value=max_press,
                expected_range="< 100000 bar for typical simulations",
            )
        )

    def _gmx_check_energy_drift(
        self, report: AuditReport, energies: Any
    ) -> None:
        """5. Energy drift — NVE should conserve total energy.

        Compare the average of the first half to the second half; a
        shift > 1% is a red flag for timestep or thermostat issues.
        """
        if len(energies) < 10:
            return
        half = len(energies) // 2
        early_avg = sum(energies[:half]) / half
        late_avg = sum(energies[half:]) / (len(energies) - half)
        drift = abs(late_avg - early_avg)
        if early_avg == 0 or abs(drift / early_avg) <= 0.01:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="thermodynamic_violation",
                message=(
                    f"Energy drift {drift:.4f} kJ/mol "
                    f"({abs(drift / early_avg) * 100:.2f}% of average) — "
                    "check timestep or thermostat"
                ),
                field="total_energy",
                value=drift,
                expected_range="< 1% drift for NVE",
            )
        )

    def _gmx_check_nan(
        self, report: AuditReport, has_nan: Any, full_text: str
    ) -> None:
        """6. NaN/Inf blowup."""
        if not has_nan and "nan" not in full_text and "inf" not in full_text:
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="NaN or Inf detected in output — simulation has blown up",
                field="output",
                value="NaN/Inf",
                expected_range="finite numbers",
            )
        )

    # ── Abaqus / FEA checks ────────────────────────────────────────

    def _audit_abaqus(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """Abaqus 输出物理合理性审计 (5 项检查)."""
        try:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).upper()
            self._abaqus_check_convergence(report, text)
            self._abaqus_check_plastic(report, text)
            self._abaqus_check_hourglass(report, text)
            self._abaqus_check_contact(report, text)
            self._abaqus_check_iterations(report, text)
        except Exception:
            logger.debug("abaqus audit failed", exc_info=True)

    def _abaqus_check_convergence(
        self, report: AuditReport, text: str
    ) -> None:
        """1. Non-convergence — Abaqus prints these in .msg/stdout."""
        if not any(kw in text for kw in [
            "CONVERGENCE IS JUDGED UNSATISFACTORY",
            "HAS NOT CONVERGED",
            "TOO MANY ATTEMPTS",
            "ABORTED",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message=(
                    "Abaqus analysis did not converge — check step controls, "
                    "mesh, or material model"
                ),
                field="convergence",
                value="failed",
                expected_range="converged",
            )
        )

    def _abaqus_check_plastic(
        self, report: AuditReport, text: str
    ) -> None:
        """2. Excessive plastic strain — material curve exceeded."""
        if "PLASTIC STRAIN" not in text or "EXCEED" not in text:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    "Plastic strain exceeds material limit — check material "
                    "data or load magnitude"
                ),
                field="plastic_strain",
                value="exceeded",
                expected_range="within material curve",
            )
        )

    def _abaqus_check_hourglass(
        self, report: AuditReport, text: str
    ) -> None:
        """3. Zero-energy modes / hourglass — under-integrated elements."""
        if not any(kw in text for kw in ["HOURGLASS", "ZERO ENERGY MODE"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    "Zero-energy (hourglass) modes detected — consider "
                    "enhanced strain formulation or refined mesh"
                ),
                field="hourglass",
                value="detected",
                expected_range="none",
            )
        )

    def _abaqus_check_contact(
        self, report: AuditReport, text: str
    ) -> None:
        """4. Contact penetration anomalies."""
        if "OVERCLOSURE" not in text or "WARNING" not in text:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    "Contact overclosure warnings — check contact stiffness "
                    "or penalty settings"
                ),
                field="contact",
                value="overclosure",
                expected_range="no excessive overclosure",
            )
        )

    def _abaqus_check_iterations(
        self, report: AuditReport, text: str
    ) -> None:
        """5. Excessive iterations / cutbacks — near limit point."""
        if "TOO MANY ITERATIONS" not in text and "CUT BACK" not in text:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    "Excessive iterations or cutbacks — solution may be "
                    "near a limit point"
                ),
                field="iterations",
                value="excessive",
                expected_range="within step limits",
            )
        )

    # ── COMSOL / FEA checks ────────────────────────────────────────

    def _audit_comsol(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """COMSOL 输出物理合理性审计 (4 项检查)."""
        try:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).lower()
            self._comsol_check_convergence(report, text)
            self._comsol_check_mesh(report, text)
            self._comsol_check_divergence(report, text)
            self._comsol_check_nan(report, text)
        except Exception:
            logger.debug("comsol audit failed", exc_info=True)

    def _comsol_check_convergence(
        self, report: AuditReport, text: str
    ) -> None:
        """1. Non-convergence."""
        if not any(kw in text for kw in [
            "did not converge",
            "not converged",
            "convergence failure",
            "failed to converge",
            "non-converged",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message=(
                    "COMSOL solver did not converge — check mesh, BCs, "
                    "or solver settings"
                ),
                field="convergence",
                value="failed",
                expected_range="converged",
            )
        )

    def _comsol_check_mesh(
        self, report: AuditReport, text: str
    ) -> None:
        """2. Mesh quality warnings."""
        if not any(kw in text for kw in [
            "mesh quality",
            "inverted element",
            "mesh element quality",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    "Mesh quality warnings — solution accuracy may be compromised"
                ),
                field="mesh",
                value="poor quality",
                expected_range="good mesh quality",
            )
        )

    def _comsol_check_divergence(
        self, report: AuditReport, text: str
    ) -> None:
        """3. Solution divergence."""
        if not any(kw in text for kw in ["diverged", "divergence"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message=(
                    "Solution divergence detected — check solver stability "
                    "or physics settings"
                ),
                field="solution",
                value="diverged",
                expected_range="converged",
            )
        )

    def _comsol_check_nan(
        self, report: AuditReport, text: str
    ) -> None:
        """4. NaN/Inf in solution."""
        if not any(kw in text for kw in ["nan", "inf", "not-a-number"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="NaN or Inf detected in COMSOL output — solution is invalid",
                field="solution",
                value="NaN/Inf",
                expected_range="finite numbers",
            )
        )

    # ── OpenFOAM / CFD checks ─────────────────────────────────────

    def _audit_openfoam(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """OpenFOAM 输出物理合理性审计 (6 项检查)."""
        try:
            log_data = parsed.get("parsed", {})
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).lower()
            self._of_check_courant(report, self._of_parse_courant(text))
            self._of_check_nan(report, text)
            self._of_check_crash(report, text)
            self._of_check_residuals(report, log_data.get("final_residuals", {}))
            self._of_check_mesh(report, text)
            self._of_check_endtime(report, log_data.get("converged", False), action)
        except Exception:
            logger.debug("openfoam audit failed", exc_info=True)

    def _of_parse_courant(self, text: str) -> list[float]:
        """从日志行解析 Courant 数 max 值."""
        courant_maxes: list[float] = []
        for line in text.splitlines():
            if "courant number mean:" not in line or "max:" not in line:
                continue
            try:
                val = line.split("max:")[-1].strip().split()[0]
                courant_maxes.append(float(val))
            except (ValueError, IndexError):
                logger.debug("best-effort op failed", exc_info=True)
        return courant_maxes

    def _of_check_courant(
        self, report: AuditReport, courant_maxes: list[float]
    ) -> None:
        """1. Courant number too high — transient solvers need Co < 1."""
        # OpenFOAM logs print "Courant Number mean: X max: Y" each step.
        if not courant_maxes:
            return
        max_co = max(courant_maxes)
        if max_co <= 1.0:
            return
        report.findings.append(
            PhysicsFinding(
                severity="error" if max_co > 10 else "warning",
                category="convergence_suspicious",
                message=(
                    f"Courant number max={max_co:.2f} exceeds 1 — "
                    "timestep too large for transient solver"
                ),
                field="courant_number",
                value=max_co,
                expected_range="< 1 for explicit, < 10 for implicit",
            )
        )

    def _of_check_nan(self, report: AuditReport, text: str) -> None:
        """2. NaN / Inf / divergence in solution."""
        if not any(kw in text for kw in ["nan", "inf", "divergence", "diverged"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="NaN/Inf or divergence detected in OpenFOAM output",
                field="solution",
                value="NaN/Inf/divergence",
                expected_range="finite, converged",
            )
        )

    def _of_check_crash(self, report: AuditReport, text: str) -> None:
        """3. Solver crashed — FOAM FATAL ERROR or segfault."""
        if not any(kw in text for kw in [
            "foam fatal error",
            "segmentation fault",
            "aborted",
            "floating point exception",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message="OpenFOAM solver crashed — check case setup",
                field="solver",
                value="crashed",
                expected_range="clean exit",
            )
        )

    def _of_check_residuals(
        self, report: AuditReport, residuals: Any
    ) -> None:
        """4. Residuals not decreasing — final residuals still high."""
        for var, val in residuals.items():
            try:
                fv = float(val)
            except (ValueError, TypeError):
                logger.debug("best-effort op failed", exc_info=True)
                continue
            if fv <= 0.1:
                continue
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="convergence_suspicious",
                    message=(
                        f"Final residual for {var} is high ({fv:.2e}) — "
                        "solution may not be converged"
                    ),
                    field=f"residual_{var}",
                    value=fv,
                    expected_range="< 1e-4 for converged solution",
                )
            )

    def _of_check_mesh(self, report: AuditReport, text: str) -> None:
        """5. Mesh quality issues."""
        if not any(kw in text for kw in [
            "skewness",
            "non-orthogonality",
            "mesh quality",
            "high aspect ratio",
            "negative volume",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    "Mesh quality issues detected — solution accuracy "
                    "may be compromised"
                ),
                field="mesh",
                value="poor quality",
                expected_range="good mesh quality",
            )
        )

    def _of_check_endtime(
        self, report: AuditReport, converged: Any, action: str
    ) -> None:
        """6. Did not reach endTime."""
        if converged or action != "run":
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    "OpenFOAM solver did not reach endTime — check log "
                    "for early termination"
                ),
                field="converged",
                value=False,
                expected_range="True",
            )
        )

    # ── FEniCS / FEM checks ───────────────────────────────────────

    def _audit_fenics(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """FEniCS 输出物理合理性审计 (4 项检查)."""
        try:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).lower()
            self._fenics_check_nan(report, text)
            self._fenics_check_newton(report, text)
            self._fenics_check_bcs(report, text)
            self._fenics_check_convergence(report, action, parsed)
        except Exception:
            logger.debug("fenics audit failed", exc_info=True)

    def _fenics_check_nan(self, report: AuditReport, text: str) -> None:
        """1. NaN/Inf in solution — solver diverged."""
        if not any(kw in text for kw in ["nan", "inf", "not a number"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="NaN or Inf detected in FEniCS output — solution diverged",
                field="solution",
                value="NaN/Inf",
                expected_range="finite numbers",
            )
        )

    def _fenics_check_newton(self, report: AuditReport, text: str) -> None:
        """2. Newton solver non-convergence."""
        if not any(kw in text for kw in [
            "newton solver did not converge",
            "max iterations",
            "failed to converge",
            "did not converge",
        ]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message=(
                    "Newton solver did not converge — check nonlinearity, "
                    "BCs, or mesh"
                ),
                field="convergence",
                value="failed",
                expected_range="converged",
            )
        )

    def _fenics_check_bcs(self, report: AuditReport, text: str) -> None:
        """3. Boundary condition inconsistency."""
        if not any(kw in text for kw in [
            "boundary condition",
            "dirichletbc",
            "overlapping",
            "inconsistent",
        ]) or ("warning" not in text and "error" not in text):
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="parameter_mismatch",
                message="Boundary condition issues detected — check BC definitions",
                field="boundary_conditions",
                value="inconsistent",
                expected_range="consistent BCs",
            )
        )

    def _fenics_check_convergence(
        self, report: AuditReport, action: str, parsed: dict[str, Any]
    ) -> None:
        """4. convergence_check action — differences not below tolerance."""
        if action != "convergence_check":
            return
        diffs = parsed.get("differences", [])
        converged = parsed.get("converged", False)
        if diffs and not converged:
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="convergence_suspicious",
                    message=(
                        "Mesh convergence check failed — solution may not "
                        "be mesh-independent"
                    ),
                    field="converged",
                    value=converged,
                    expected_range="True (differences < 1e-2)",
                )
            )
        # NaN in differences means file format / load error
        if any(d != d for d in diffs):
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="unphysical_value",
                    message=(
                        "NaN in convergence differences — solution file "
                        "format may be wrong"
                    ),
                    field="differences",
                    value="NaN",
                    expected_range="finite numbers",
                )
            )

    # ── Elmer / FEM checks ────────────────────────────────────────

    def _audit_elmer(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).lower()
            self._elmer_check_convergence(report, text)
            self._elmer_check_singular(report, text)
            self._elmer_check_nan(report, text)
        except Exception:
            logger.debug("elmer audit failed", exc_info=True)

    def _elmer_check_convergence(self, report: AuditReport, text: str) -> None:
        keywords = [
            "did not converge",
            "not converged",
            "convergence failed",
            "failed to converge",
            "no convergence",
        ]
        if not any(kw in text for kw in keywords):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message=(
                    "Elmer solver did not converge — check mesh, BCs, "
                    "or solver settings"
                ),
                field="convergence",
                value="failed",
                expected_range="converged",
            )
        )

    def _elmer_check_singular(self, report: AuditReport, text: str) -> None:
        keywords = [
            "singular",
            "ill-conditioned",
            "zero pivot",
            "numerically singular",
        ]
        if not any(kw in text for kw in keywords):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="convergence_suspicious",
                message=(
                    "Singular or ill-conditioned matrix — check BCs or mesh quality"
                ),
                field="matrix",
                value="singular",
                expected_range="well-conditioned",
            )
        )

    def _elmer_check_nan(self, report: AuditReport, text: str) -> None:
        if not any(kw in text for kw in ["nan", "inf", "not-a-number"]):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="NaN or Inf detected in Elmer output — solution is invalid",
                field="solution",
                value="NaN/Inf",
                expected_range="finite numbers",
            )
        )

    # ── Transolver / neural surrogate checks ──────────────────────

    def _audit_transolver(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            train_loss = parsed.get("train_loss", [])
            if train_loss:
                self._tvv_check_loss_nan(report, train_loss)
                self._tvv_check_loss_inf(report, train_loss)
                self._tvv_check_loss_trend(report, train_loss)
                self._tvv_check_grad_explosion(report, train_loss)
            self._tvv_check_predictions_nan(report, parsed.get("predictions", []))
            self._tvv_check_checkpoint_warnings(report, parsed.get("warnings", []))
        except Exception:
            logger.debug("transolver audit failed", exc_info=True)

    def _tvv_check_loss_nan(self, report: AuditReport, train_loss: list[Any]) -> None:
        if not any(v != v for v in train_loss):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message=(
                    "NaN detected in training loss — gradient explosion or bad data"
                ),
                field="train_loss",
                value="NaN",
                expected_range="finite numbers",
            )
        )

    def _tvv_check_loss_inf(self, report: AuditReport, train_loss: list[Any]) -> None:
        if not any(abs(v) == float("inf") for v in train_loss):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message="Inf detected in training loss — gradient explosion",
                field="train_loss",
                value="Inf",
                expected_range="finite numbers",
            )
        )

    def _tvv_check_loss_trend(self, report: AuditReport, train_loss: list[Any]) -> None:
        if len(train_loss) < 4:
            return
        half = len(train_loss) // 2
        early_avg = sum(train_loss[:half]) / half
        late_avg = sum(train_loss[half:]) / (len(train_loss) - half)
        if not (early_avg > 0 and late_avg > early_avg * 1.5):
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="convergence_suspicious",
                message=(
                    f"Training loss not decreasing (early={early_avg:.4e}, "
                    f"late={late_avg:.4e}) — check learning rate or data"
                ),
                field="train_loss",
                value=late_avg,
                expected_range="decreasing over epochs",
            )
        )

    def _tvv_check_grad_explosion(
        self, report: AuditReport, train_loss: list[Any]
    ) -> None:
        finite_losses = [
            abs(v) for v in train_loss if v == v and abs(v) != float("inf")
        ]
        if not finite_losses:
            return
        worst = max(finite_losses)
        if worst <= 1e6:
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message=(
                    f"Loss reached {worst:.2e} — likely gradient explosion, "
                    "add gradient clipping"
                ),
                field="train_loss",
                value=worst,
                expected_range="< 1e6",
            )
        )

    def _tvv_check_predictions_nan(
        self, report: AuditReport, predictions: list[Any]
    ) -> None:
        if not predictions:
            return
        flat = [
            v for row in predictions for v in (row if isinstance(row, list) else [row])
        ]
        if not any(v != v for v in flat):
            return
        report.findings.append(
            PhysicsFinding(
                severity="error",
                category="unphysical_value",
                message=(
                    "NaN in model predictions — checkpoint may be corrupted or "
                    "input out of distribution"
                ),
                field="predictions",
                value="NaN",
                expected_range="finite numbers",
            )
        )

    def _tvv_check_checkpoint_warnings(
        self, report: AuditReport, warnings: list[Any]
    ) -> None:
        for w in warnings:
            wl = str(w).lower()
            hit = ("checkpoint" in wl) or ("mismatch" in wl) or ("missing keys" in wl)
            if not hit:
                continue
            report.findings.append(
                PhysicsFinding(
                    severity="warning",
                    category="parameter_mismatch",
                    message=f"Checkpoint issue: {w}",
                    field="checkpoint",
                    value=w,
                    expected_range="matching checkpoint",
                )
            )

    # ── Mechanical / analytical checks ────────────────────────────

    def _audit_mechanical(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            self._mech_check_stress(report, parsed, params)
            self._mech_check_strain(report, parsed)
            self._mech_check_safety_factor(report, parsed)
            self._mech_check_fatigue(report, parsed)
        except Exception:
            logger.debug("mechanical audit failed", exc_info=True)

    def _mech_check_stress(
        self, report: AuditReport, parsed: dict[str, Any], params: dict[str, Any]
    ) -> None:
        max_stress = parsed.get("max_stress") or parsed.get("thermal_stress")
        yield_strength = params.get("material_props", {}).get("yield_strength", 0.0)
        if max_stress is None or yield_strength <= 0:
            return
        if abs(max_stress) <= yield_strength:
            return
        ratio = abs(max_stress) / yield_strength
        report.findings.append(
            PhysicsFinding(
                severity="error" if ratio > 2 else "warning",
                category="unphysical_value",
                message=(
                    f"Stress ({abs(max_stress):.2e} Pa) exceeds yield "
                    f"strength ({yield_strength:.2e} Pa) — material will yield"
                ),
                field="max_stress",
                value=max_stress,
                expected_range=f"< {yield_strength:.2e} Pa (yield)",
            )
        )

    def _mech_check_strain(self, report: AuditReport, parsed: dict[str, Any]) -> None:
        max_strain = parsed.get("max_strain")
        if max_strain is None or abs(max_strain) <= 0.2:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    f"Strain ({abs(max_strain):.4f}) exceeds typical failure "
                    "threshold (0.2) — material failure likely"
                ),
                field="max_strain",
                value=max_strain,
                expected_range="< 0.2 (20%)",
            )
        )

    def _mech_check_safety_factor(
        self, report: AuditReport, parsed: dict[str, Any]
    ) -> None:
        safety_factor = parsed.get("safety_factor")
        if safety_factor is None or safety_factor >= 1.0:
            return
        report.findings.append(
            PhysicsFinding(
                severity="error" if safety_factor < 0.5 else "warning",
                category="unphysical_value",
                message=(
                    f"Safety factor {safety_factor:.2f} < 1 — design will fail "
                    "under load"
                ),
                field="safety_factor",
                value=safety_factor,
                expected_range="> 1.0 for safe design",
            )
        )

    def _mech_check_fatigue(self, report: AuditReport, parsed: dict[str, Any]) -> None:
        fatigue_life = parsed.get("fatigue_life")
        if fatigue_life is None or fatigue_life == float("inf"):
            return
        if fatigue_life >= 1000:
            return
        report.findings.append(
            PhysicsFinding(
                severity="warning",
                category="unphysical_value",
                message=(
                    f"Fatigue life {fatigue_life:.0f} cycles is very low — "
                    "low-cycle fatigue regime"
                ),
                field="fatigue_life",
                value=fatigue_life,
                expected_range="> 1000 for high-cycle fatigue",
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
                logger.debug("best-effort op failed", exc_info=True)
        # From structure string — XYZ format has atom count on the first line
        struct = params.get("structure") or params.get("poscar")
        if isinstance(struct, str):
            lines = struct.strip().split("\n")
            if len(lines) > 2 and lines[0].strip().isdigit():
                try:
                    return int(lines[0].strip())
                except ValueError:
                    logger.debug("best-effort op failed", exc_info=True)
            # POSCAR: line 7 (index 6) holds per-element counts
            if len(lines) > 6:
                counts = lines[6].strip().split()
                try:
                    return sum(int(x) for x in counts)
                except ValueError:
                    logger.debug("best-effort op failed", exc_info=True)
        return 0


if __name__ == "__main__":
    # Smoke test — each new audit method should catch its target pathology.
    a = PhysicsAuditor()

    # OpenFOAM: high Courant + NaN
    r = a.audit("openfoam_tool", "run", {
        "parsed": {"converged": False, "final_residuals": {"Ux": 0.5}},
        "stdout": "Courant Number mean: 0.5 max: 15.3\nnan detected",
    })
    assert r.has_errors, "openfoam: should flag high Courant + NaN"

    # FEniCS: Newton divergence + NaN
    r = a.audit("fenics_tool", "solve_pde", {
        "stdout": "Newton solver did not converge",
        "stderr": "RuntimeError: nan in solution",
    })
    assert r.has_errors, "fenics: should flag Newton non-convergence + NaN"

    # Elmer: singular matrix
    r = a.audit("elmer_tool", "solve_sif", {
        "stdout": "Matrix is singular",
        "stderr": "",
    })
    assert r.has_errors, "elmer: should flag singular matrix"

    # Transolver: NaN in loss
    r = a.audit("transolver_tool", "train", {
        "train_loss": [1.0, float("nan"), 2.0],
        "warnings": [],
    })
    assert r.has_errors, "transolver: should flag NaN in loss"

    # Transolver: gradient explosion
    r = a.audit("transolver_tool", "train", {
        "train_loss": [1.0, 1e3, 2e6],
        "warnings": [],
    })
    assert r.has_errors, "transolver: should flag gradient explosion"

    # Mechanical: stress exceeds yield (3x → error severity)
    r = a.audit("mechanical_tool", "stress_analysis", {
        "max_stress": 7.5e8,
        "max_strain": 0.0025,
    }, {"material_props": {"yield_strength": 2.5e8}})
    assert r.has_errors, "mechanical: should flag stress > yield"

    # Mechanical: safety factor < 1
    r = a.audit("mechanical_tool", "fatigue_life", {
        "safety_factor": 0.4,
        "fatigue_life": 500,
    })
    assert r.has_errors, "mechanical: should flag safety factor < 0.5"

    # Clean results should produce no errors
    r = a.audit("openfoam_tool", "run", {
        "parsed": {"converged": True, "final_residuals": {"Ux": 1e-6}},
        "stdout": "End\n",
    })
    assert not r.has_errors, "openfoam: clean run should have no errors"

    print("All physics audit self-checks passed.")
