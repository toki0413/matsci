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

    # ── Gaussian / quantum chemistry checks ───────────────────────

    def _audit_gaussian(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            energy = parsed.get("energy")
            frequencies = parsed.get("frequencies", [])
            scf_failure = parsed.get("scf_convergence_failure", False)
            charge = parsed.get("charge", params.get("charge"))
            multiplicity = parsed.get("multiplicity", params.get("multiplicity"))

            # 1. SCF non-convergence
            if scf_failure:
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

            # 2. Imaginary frequencies (negative eigenvalues in freq calc).
            # Small negative (~< 50 cm^-1) is often numerical noise; large
            # negative means the structure is a saddle point, not a minimum.
            if frequencies and action == "freq":
                imaginary = [f for f in frequencies if f < 0]
                if imaginary:
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

            # 3. Energy magnitude sanity — Gaussian reports Hartrees.
            # Bound molecules should have negative energy; extremely negative
            # suggests wrong pseudopotential/basis or wrong atom count.
            if energy is not None:
                if energy > 0:
                    report.findings.append(
                        PhysicsFinding(
                            severity="warning",
                            category="unphysical_value",
                            message=(
                                f"SCF energy is positive ({energy:.4f} Hartree) — "
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
                                f"SCF energy extremely negative ({energy:.4f} Hartree) — "
                                "check basis set or atom count"
                            ),
                            field="energy",
                            value=energy,
                            expected_range="-2000 to 0 Hartree for typical molecules",
                        )
                    )

            # 4. Charge/multiplicity consistency
            if multiplicity is not None:
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
                    pass

            if charge is not None:
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
                    pass
        except Exception:
            logger.debug("gaussian audit failed", exc_info=True)

    # ── ORCA / quantum chemistry checks ───────────────────────────

    def _audit_orca(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            energy = parsed.get("energy")
            frequencies = parsed.get("frequencies", [])
            converged = parsed.get("converged", False)
            opt_steps = parsed.get("optimization_steps", 0)

            # 1. SCF non-convergence — no FINAL SINGLE POINT ENERGY means SCF failed
            if energy is None and action in ("sp", "opt", "freq"):
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

            # 2. Imaginary frequencies (same logic as Gaussian)
            if frequencies and action == "freq":
                imaginary = [f for f in frequencies if f < 0]
                if imaginary:
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

            # 3. Energy magnitude sanity (Hartrees, same range as Gaussian)
            if energy is not None:
                if energy > 0:
                    report.findings.append(
                        PhysicsFinding(
                            severity="warning",
                            category="unphysical_value",
                            message=(
                                f"Energy is positive ({energy:.4f} Hartree) — "
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
                                f"Energy extremely negative ({energy:.4f} Hartree) — "
                                "check basis set or atom count"
                            ),
                            field="energy",
                            value=energy,
                            expected_range="-2000 to 0 Hartree for typical molecules",
                        )
                    )

            # 4. Optimization suspicious — too many steps without converging
            if action == "opt" and not converged and opt_steps > 50:
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
        except Exception:
            logger.debug("orca audit failed", exc_info=True)

    # ── GROMACS / MD checks ────────────────────────────────────────

    def _audit_gromacs(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            md_log = parsed.get("md_log_data", {})
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            full_text = (stdout + "\n" + stderr).lower()

            # 1. LINCS/SHAKE constraint warnings
            lincs = md_log.get("lincs_warnings", 0)
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

            shake = md_log.get("shake_warnings", 0)
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

            # 2. Neighbor list warnings
            nl_warnings = md_log.get("neighbor_list_warnings", 0)
            if nl_warnings > 0:
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

            # 3. Temperature spikes
            temps = md_log.get("temperatures", [])
            if temps:
                max_temp = max(temps)
                if max_temp > 10000:
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

            # 4. Pressure extremes
            presses = md_log.get("pressures", [])
            if presses:
                max_press = max(abs(p) for p in presses)
                if max_press > 100000:
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

            # 5. Energy drift — NVE should conserve total energy.
            # Compare the average of the first half to the second half; a
            # shift > 1% is a red flag for timestep or thermostat issues.
            energies = md_log.get("energies", [])
            if len(energies) >= 10:
                half = len(energies) // 2
                early_avg = sum(energies[:half]) / half
                late_avg = sum(energies[half:]) / (len(energies) - half)
                drift = abs(late_avg - early_avg)
                if early_avg != 0 and abs(drift / early_avg) > 0.01:
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

            # 6. NaN/Inf blowup
            if md_log.get("has_nan", False) or "nan" in full_text or "inf" in full_text:
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
        except Exception:
            logger.debug("gromacs audit failed", exc_info=True)

    # ── Abaqus / FEA checks ────────────────────────────────────────

    def _audit_abaqus(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).upper()

            # 1. Non-convergence — Abaqus prints these in .msg/stdout
            if any(kw in text for kw in [
                "CONVERGENCE IS JUDGED UNSATISFACTORY",
                "HAS NOT CONVERGED",
                "TOO MANY ATTEMPTS",
                "ABORTED",
            ]):
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

            # 2. Excessive plastic strain — material curve exceeded
            if "PLASTIC STRAIN" in text and "EXCEED" in text:
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

            # 3. Zero-energy modes / hourglass — under-integrated elements
            if any(kw in text for kw in ["HOURGLASS", "ZERO ENERGY MODE"]):
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

            # 4. Contact penetration anomalies
            if "OVERCLOSURE" in text and "WARNING" in text:
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

            # 5. Excessive iterations / cutbacks — near limit point
            if "TOO MANY ITERATIONS" in text or "CUT BACK" in text:
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
        except Exception:
            logger.debug("abaqus audit failed", exc_info=True)

    # ── COMSOL / FEA checks ────────────────────────────────────────

    def _audit_comsol(
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

            # 1. Non-convergence
            if any(kw in text for kw in [
                "did not converge",
                "not converged",
                "convergence failure",
                "failed to converge",
                "non-converged",
            ]):
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

            # 2. Mesh quality warnings
            if any(kw in text for kw in [
                "mesh quality",
                "inverted element",
                "mesh element quality",
            ]):
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

            # 3. Solution divergence
            if any(kw in text for kw in ["diverged", "divergence"]):
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

            # 4. NaN/Inf in solution
            if any(kw in text for kw in ["nan", "inf", "not-a-number"]):
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
        except Exception:
            logger.debug("comsol audit failed", exc_info=True)

    # ── OpenFOAM / CFD checks ─────────────────────────────────────

    def _audit_openfoam(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            log_data = parsed.get("parsed", {})
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            text = (stdout + "\n" + stderr).lower()

            # 1. Courant number too high — transient solvers need Co < 1.
            # OpenFOAM logs print "Courant Number mean: X max: Y" each step.
            courant_maxes: list[float] = []
            for line in text.splitlines():
                if "courant number mean:" in line and "max:" in line:
                    try:
                        val = line.split("max:")[-1].strip().split()[0]
                        courant_maxes.append(float(val))
                    except (ValueError, IndexError):
                        pass
            if courant_maxes:
                max_co = max(courant_maxes)
                if max_co > 1.0:
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

            # 2. NaN / Inf / divergence in solution
            if any(kw in text for kw in ["nan", "inf", "divergence", "diverged"]):
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

            # 3. Solver crashed — FOAM FATAL ERROR or segfault
            if any(kw in text for kw in [
                "foam fatal error",
                "segmentation fault",
                "aborted",
                "floating point exception",
            ]):
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

            # 4. Residuals not decreasing — final residuals still high
            residuals = log_data.get("final_residuals", {})
            for var, val in residuals.items():
                try:
                    fv = float(val)
                except (ValueError, TypeError):
                    continue
                if fv > 0.1:
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

            # 5. Mesh quality issues
            if any(kw in text for kw in [
                "skewness",
                "non-orthogonality",
                "mesh quality",
                "high aspect ratio",
                "negative volume",
            ]):
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

            # 6. Did not reach endTime
            if not log_data.get("converged", False) and action == "run":
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
        except Exception:
            logger.debug("openfoam audit failed", exc_info=True)

    # ── FEniCS / FEM checks ───────────────────────────────────────

    def _audit_fenics(
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

            # 1. NaN/Inf in solution — solver diverged
            if any(kw in text for kw in ["nan", "inf", "not a number"]):
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

            # 2. Newton solver non-convergence
            if any(kw in text for kw in [
                "newton solver did not converge",
                "max iterations",
                "failed to converge",
                "did not converge",
            ]):
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

            # 3. Boundary condition inconsistency
            if any(kw in text for kw in [
                "boundary condition",
                "dirichletbc",
                "overlapping",
                "inconsistent",
            ]) and ("warning" in text or "error" in text):
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

            # 4. convergence_check action — differences not below tolerance
            if action == "convergence_check":
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
        except Exception:
            logger.debug("fenics audit failed", exc_info=True)

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

            # 1. Non-convergence
            if any(kw in text for kw in [
                "did not converge",
                "not converged",
                "convergence failed",
                "failed to converge",
                "no convergence",
            ]):
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

            # 2. Singular / ill-conditioned matrix
            if any(kw in text for kw in [
                "singular",
                "ill-conditioned",
                "zero pivot",
                "numerically singular",
            ]):
                report.findings.append(
                    PhysicsFinding(
                        severity="error",
                        category="convergence_suspicious",
                        message=(
                            "Singular or ill-conditioned matrix — check BCs or "
                            "mesh quality"
                        ),
                        field="matrix",
                        value="singular",
                        expected_range="well-conditioned",
                    )
                )

            # 3. NaN/Inf in solution
            if any(kw in text for kw in ["nan", "inf", "not-a-number"]):
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
        except Exception:
            logger.debug("elmer audit failed", exc_info=True)

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

            # 1. NaN/Inf in loss (train action)
            if train_loss:
                if any(l != l for l in train_loss):
                    report.findings.append(
                        PhysicsFinding(
                            severity="error",
                            category="unphysical_value",
                            message=(
                                "NaN detected in training loss — gradient "
                                "explosion or bad data"
                            ),
                            field="train_loss",
                            value="NaN",
                            expected_range="finite numbers",
                        )
                    )
                if any(abs(l) == float("inf") for l in train_loss):
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

                # 2. Loss not decreasing — compare first half to second half
                if len(train_loss) >= 4:
                    half = len(train_loss) // 2
                    early_avg = sum(train_loss[:half]) / half
                    late_avg = sum(train_loss[half:]) / (len(train_loss) - half)
                    if early_avg > 0 and late_avg > early_avg * 1.5:
                        report.findings.append(
                            PhysicsFinding(
                                severity="warning",
                                category="convergence_suspicious",
                                message=(
                                    f"Training loss not decreasing (early="
                                    f"{early_avg:.4e}, late={late_avg:.4e}) — "
                                    "check learning rate or data"
                                ),
                                field="train_loss",
                                value=late_avg,
                                expected_range="decreasing over epochs",
                            )
                        )

                # 3. Gradient explosion — loss value itself blew up
                finite_losses = [abs(l) for l in train_loss if l == l and abs(l) != float("inf")]
                if finite_losses and max(finite_losses) > 1e6:
                    report.findings.append(
                        PhysicsFinding(
                            severity="error",
                            category="unphysical_value",
                            message=(
                                f"Loss reached {max(finite_losses):.2e} — likely "
                                "gradient explosion, add gradient clipping"
                            ),
                            field="train_loss",
                            value=max(finite_losses),
                            expected_range="< 1e6",
                        )
                    )

            # 4. NaN in predictions (predict action)
            predictions = parsed.get("predictions", [])
            if predictions:
                flat = [
                    v
                    for row in predictions
                    for v in (row if isinstance(row, list) else [row])
                ]
                if any(v != v for v in flat):
                    report.findings.append(
                        PhysicsFinding(
                            severity="error",
                            category="unphysical_value",
                            message=(
                                "NaN in model predictions — checkpoint may be "
                                "corrupted or input out of distribution"
                            ),
                            field="predictions",
                            value="NaN",
                            expected_range="finite numbers",
                        )
                    )

            # 5. Checkpoint mismatch — scan warnings
            warnings = parsed.get("warnings", [])
            for w in warnings:
                wl = str(w).lower()
                if "checkpoint" in wl or "mismatch" in wl or "missing keys" in wl:
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
        except Exception:
            logger.debug("transolver audit failed", exc_info=True)

    # ── Mechanical / analytical checks ────────────────────────────

    def _audit_mechanical(
        self,
        report: AuditReport,
        action: str,
        parsed: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        try:
            # 1. Stress exceeding material yield
            max_stress = parsed.get("max_stress") or parsed.get("thermal_stress")
            material_props = params.get("material_props", {})
            yield_strength = material_props.get("yield_strength", 0.0)

            if max_stress is not None and yield_strength > 0:
                if abs(max_stress) > yield_strength:
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

            # 2. Strain beyond failure threshold (20% is well past typical failure)
            max_strain = parsed.get("max_strain")
            if max_strain is not None and abs(max_strain) > 0.2:
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

            # 3. Safety factor < 1 — design will fail under load
            safety_factor = parsed.get("safety_factor")
            if safety_factor is not None and safety_factor < 1.0:
                report.findings.append(
                    PhysicsFinding(
                        severity="error" if safety_factor < 0.5 else "warning",
                        category="unphysical_value",
                        message=(
                            f"Safety factor {safety_factor:.2f} < 1 — design will "
                            "fail under load"
                        ),
                        field="safety_factor",
                        value=safety_factor,
                        expected_range="> 1.0 for safe design",
                    )
                )

            # 4. Fatigue life — very low cycles means LCF regime
            fatigue_life = parsed.get("fatigue_life")
            if fatigue_life is not None:
                # inf means infinite life, which is fine
                if fatigue_life != float("inf") and fatigue_life < 1000:
                    report.findings.append(
                        PhysicsFinding(
                            severity="warning",
                            category="unphysical_value",
                            message=(
                                f"Fatigue life {fatigue_life:.0f} cycles is very "
                                "low — low-cycle fatigue regime"
                            ),
                            field="fatigue_life",
                            value=fatigue_life,
                            expected_range="> 1000 for high-cycle fatigue",
                        )
                    )
        except Exception:
            logger.debug("mechanical audit failed", exc_info=True)

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
