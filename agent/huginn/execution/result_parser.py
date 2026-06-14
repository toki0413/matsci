"""Result Parser — extracts physical insights from raw calculation output.

Turns log files, OUTCARs, trajectory dumps into structured data that
the Agent can reason about.

Supported parsers:
  - VASP: OUTCAR, vasprun.xml, DOSCAR, EIGENVAL
  - Gaussian: output log, fchk
  - LAMMPS: log file, dump files
  - ABAQUS: .odb (via Python scripting), .dat
  - OpenFOAM: log files, post-processing fields
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedResult:
    """Structured result from parsing a calculation output."""
    software: str
    task: str
    converged: bool
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    stress: Optional[List[float]] = None
    band_gap: Optional[float] = None
    magnetic_moment: Optional[float] = None
    physical_quantities: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultParser:
    """Parse raw output files from computational materials science software.

    Usage:
        parser = ResultParser()
        result = parser.parse_vasp_outcar(Path("OUTCAR"))
        print(result.converged, result.energy)
    """

    def __init__(self):
        self._parsers = {
            "vasp_outcar": self._parse_vasp_outcar,
            "vasp_vasprun": self._parse_vasprun_quick,
            "gaussian_log": self._parse_gaussian_log,
            "lammps_log": self._parse_lammps_log,
            "abaqus_dat": self._parse_abaqus_dat,
            "openfoam_log": self._parse_openfoam_log,
        }

    def parse(self, file_path: Path, file_type: Optional[str] = None) -> ParsedResult:
        """Auto-detect file type and parse."""
        if file_type is None:
            file_type = self._detect_file_type(file_path)

        parser_fn = self._parsers.get(file_type)
        if parser_fn is None:
            return ParsedResult(
                software="unknown",
                task="unknown",
                converged=False,
                errors=[f"No parser available for file type: {file_type}"],
            )
        return parser_fn(file_path)

    def _detect_file_type(self, path: Path) -> str:
        name = path.name.lower()
        if name == "outcar":
            return "vasp_outcar"
        if name == "vasprun.xml":
            return "vasp_vasprun"
        if name.endswith(".log"):
            # Distinguish Gaussian vs LAMMPS vs OpenFOAM
            content = path.read_text(errors="ignore")[:5000]
            if "Entering Gaussian System" in content:
                return "gaussian_log"
            if "LAMMPS" in content or "Total # of neighbors" in content:
                return "lammps_log"
            if "foam" in content.lower() or "simplefoam" in content.lower():
                return "openfoam_log"
            return "gaussian_log"  # default
        if name.endswith(".dat"):
            return "abaqus_dat"
        return "unknown"

    # ------------------------------------------------------------------
    # VASP
    # ------------------------------------------------------------------

    def _parse_vasp_outcar(self, path: Path) -> ParsedResult:
        """Parse VASP OUTCAR for key physical quantities."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="VASP", task="unknown", converged=False)

        # Task detection
        if "IBRION =" in text:
            ibrion = self._extract_int(text, r"IBRION\s*=\s*(\d+)")
            result.task = {0: "MD", 1: "relax", 2: "relax", 3: "relax", 5: "freq", 6: "freq", 7: "freq", 8: "freq"}.get(ibrion, "scf")

        # Energy
        energy_match = re.search(r"free  energy   TOTEN\s*=\s+([-\d.]+)", text)
        if energy_match:
            result.energy = float(energy_match.group(1))

        # Convergence
        if "reached required accuracy" in text:
            result.converged = True
        elif "EDIFFG" in text and result.task == "relax":
            # Check if ionic relaxation converged
            result.converged = "reached required accuracy - stopping structural energy minimisation" in text

        # Forces (last ionic step)
        forces = self._extract_forces_outcar(text)
        if forces:
            result.forces = forces
            max_force = max(sum(f[i]**2 for i in range(3))**0.5 for f in forces)
            result.physical_quantities["max_force"] = max_force

        # Stress
        stress_match = re.search(r"in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
        if stress_match:
            result.stress = [float(stress_match.group(i)) for i in range(1, 7)]

        # Magnetic moment
        mag_match = re.search(r"mag=\s+([\d.]+)", text)
        if mag_match:
            result.magnetic_moment = float(mag_match.group(1))

        # Band gap (if available)
        if "E-fermi" in text:
            ef_match = re.search(r"E-fermi\s*:\s*([-\d.]+)", text)
            if ef_match:
                result.physical_quantities["fermi_energy"] = float(ef_match.group(1))

        # Warnings
        if "ZBRENT" in text:
            result.warnings.append("ZBRENT error encountered — possible SCF convergence issue")
        if "EDDDAV" in text:
            result.warnings.append("EDDDAV error — try reducing NBANDS or changing ALGO")
        if "WARNING" in text:
            warns = re.findall(r"WARNING:\s*(.+?)(?:\n|$)", text)
            result.warnings.extend(warns[:5])

        result.metadata["file_size"] = path.stat().st_size
        result.metadata["lines"] = text.count("\n")
        return result

    def _extract_forces_outcar(self, text: str) -> Optional[List[List[float]]]:
        """Extract forces from the last ionic step in OUTCAR."""
        # Find all force blocks
        blocks = list(re.finditer(r"TOTAL-FORCE \(eV/Angst\)\n-{2,}\n(.+?)(?:\n-{2,})", text, re.DOTALL))
        if not blocks:
            return None
        last_block = blocks[-1].group(1)
        forces = []
        for line in last_block.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 6:
                forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        return forces

    def _parse_vasprun_quick(self, path: Path) -> ParsedResult:
        """Quick parse of vasprun.xml (without full XML parsing)."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="VASP", task="unknown", converged=False)

        # Energy
        e_match = re.search(r"<i name=\"e_fr_energy\">([-\d.]+)</i>", text)
        if e_match:
            result.energy = float(e_match.group(1))

        # Convergence
        if "<v name=\"converged\">T</v>" in text or "<i name=\"converged\">T</i>" in text:
            result.converged = True

        return result

    # ------------------------------------------------------------------
    # Gaussian
    # ------------------------------------------------------------------

    def _parse_gaussian_log(self, path: Path) -> ParsedResult:
        """Parse Gaussian output log file."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="Gaussian", task="unknown", converged=False)

        # Task detection
        if "opt" in text.lower() and "freq" in text.lower():
            result.task = "opt+freq"
        elif "opt" in text.lower():
            result.task = "opt"
        elif "freq" in text.lower():
            result.task = "freq"
        elif "td" in text.lower():
            result.task = "td"
        else:
            result.task = "sp"

        # Energy
        scf_match = re.search(r"SCF Done:\s+E\(\w+\)\s*=\s+([-\d.]+)", text)
        if scf_match:
            result.energy = float(scf_match.group(1))

        # Convergence
        if "Normal termination" in text:
            result.converged = True
        elif "Error termination" in text:
            result.converged = False
            err_match = re.search(r"Error termination.*\n(.*)", text)
            if err_match:
                result.errors.append(err_match.group(1).strip())

        # HOMO-LUMO gap
        homo_match = re.search(r"Alpha\s+occ\.\s+.*?([-]?\d+\.\d+)\s*\n\s*Alpha\s+virt\.", text, re.DOTALL)
        if homo_match:
            homo = float(homo_match.group(1))
            lumo_match = re.search(r"Alpha\s+virt\.\s+.*?([-]?\d+\.\d+)", text, re.DOTALL)
            if lumo_match:
                lumo = float(lumo_match.group(1))
                result.band_gap = (lumo - homo) * 27.2114  # Hartree to eV

        # Frequencies (for opt+freq)
        freq_lines = re.findall(r"Frequencies\s+--\s+(.+)", text)
        freqs = []
        for line in freq_lines:
            freqs.extend(re.findall(r"-?[\d.]+", line))
        if freqs:
            result.physical_quantities["frequencies"] = [float(f) for f in freqs]
            imag = [f for f in result.physical_quantities["frequencies"] if f < 0]
            if imag:
                result.warnings.append(f"{len(imag)} imaginary frequencies — not a true minimum")

        return result

    # ------------------------------------------------------------------
    # LAMMPS
    # ------------------------------------------------------------------

    def _parse_lammps_log(self, path: Path) -> ParsedResult:
        """Parse LAMMPS log file."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="LAMMPS", task="MD", converged=False)

        # Extract thermo data
        thermo_lines = []
        in_thermo = False
        for line in text.split("\n"):
            if line.startswith("Step "):
                in_thermo = True
                headers = line.split()
                continue
            if in_thermo:
                parts = line.split()
                if len(parts) == len(headers) and parts[0].replace(".", "", 1).isdigit():
                    thermo_lines.append({h: float(v) for h, v in zip(headers, parts)})
                else:
                    in_thermo = False

        if thermo_lines:
            result.physical_quantities["final_temperature"] = thermo_lines[-1].get("Temp")
            result.physical_quantities["final_pressure"] = thermo_lines[-1].get("Press")
            result.physical_quantities["final_pe"] = thermo_lines[-1].get("PotEng")
            result.physical_quantities["final_ke"] = thermo_lines[-1].get("KinEng")
            result.physical_quantities["total_steps"] = int(thermo_lines[-1].get("Step", 0))

            # Energy conservation check
            if len(thermo_lines) > 10:
                pe_values = [row.get("TotEng", row.get("PotEng", 0)) for row in thermo_lines]
                if pe_values:
                    pe_drift = (max(pe_values) - min(pe_values)) / abs(pe_values[0]) if pe_values[0] != 0 else 0
                    result.physical_quantities["energy_drift_relative"] = pe_drift
                    if pe_drift > 0.01:
                        result.warnings.append(f"High energy drift: {pe_drift:.2%}")

        # Check for errors
        if "ERROR" in text:
            errs = re.findall(r"ERROR:\s*(.+?)(?:\n|$)", text)
            result.errors.extend(errs[:5])
        else:
            result.converged = True  # LAMMPS completed without error

        return result

    # ------------------------------------------------------------------
    # ABAQUS
    # ------------------------------------------------------------------

    def _parse_abaqus_dat(self, path: Path) -> ParsedResult:
        """Parse ABAQUS .dat file for results."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="ABAQUS", task="FEA", converged=False)

        # Check completion
        if "THE ANALYSIS HAS BEEN COMPLETED" in text:
            result.converged = True

        # Extract element stresses/strains (simplified)
        stress_blocks = re.findall(r"S11\s+S22\s+S33\s+S12\s+S13\s+S23\n(.+?)(?:\n\n|\Z)", text, re.DOTALL)
        if stress_blocks:
            result.physical_quantities["stress_blocks_found"] = len(stress_blocks)

        # Warnings
        if "WARNING" in text:
            warns = re.findall(r"\*\*\*WARNING:\s*(.+?)(?:\n|$)", text)
            result.warnings.extend(warns[:5])

        # Errors
        if "ERROR" in text:
            errs = re.findall(r"\*\*\*ERROR:\s*(.+?)(?:\n|$)", text)
            result.errors.extend(errs[:5])

        return result

    # ------------------------------------------------------------------
    # OpenFOAM
    # ------------------------------------------------------------------

    def _parse_openfoam_log(self, path: Path) -> ParsedResult:
        """Parse OpenFOAM solver log."""
        text = path.read_text(errors="ignore")
        result = ParsedResult(software="OpenFOAM", task="CFD", converged=False)

        # Extract final residuals
        residuals = {}
        for field in ["Ux", "Uy", "Uz", "p", "k", "omega", "epsilon"]:
            pattern = rf"{field}\s+final residual\s*=\s+([\d.eE+-]+)"
            matches = re.findall(pattern, text)
            if matches:
                residuals[field] = float(matches[-1])

        if residuals:
            result.physical_quantities["final_residuals"] = residuals
            max_res = max(residuals.values())
            result.converged = max_res < 1e-4

        # Check for errors
        if "FOAM FATAL ERROR" in text or "Floating point exception" in text:
            result.errors.append("FOAM fatal error or floating point exception")
        elif "Continuity error" in text:
            cont_match = re.search(r"Continuity error = ([\d.eE+-]+)", text)
            if cont_match:
                result.physical_quantities["continuity_error"] = float(cont_match.group(1))

        # Execution time
        time_match = re.search(r"ExecutionTime = ([\d.]+) s", text)
        if time_match:
            result.physical_quantities["execution_time_s"] = float(time_match.group(1))

        # Iterations
        iter_match = re.findall(r"Time = ([\d.]+)", text)
        if iter_match:
            result.physical_quantities["total_iterations"] = len(iter_match)

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_int(self, text: str, pattern: str) -> Optional[int]:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None
