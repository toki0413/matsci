"""CP2K DFT/MD tool — generate CP2K input and parse output.

When CP2K is not installed, the tool falls back to input-export mode.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool
from huginn.types import ToolResult, ToolContext


class Cp2kToolInput(BaseModel):
    action: Literal["generate", "run", "parse"] = Field(default="run")
    run_type: Literal["ENERGY_FORCE", "GEO_OPT", "MD", "CELL_OPT"] = Field(default="ENERGY_FORCE")
    method: str = Field(default="PBE")
    structure: dict = Field(
        default_factory=lambda: {
            "cell": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
            "species": ["Si", "Si"],
            "positions": [[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]],
            "units": "ANGSTROM",
        }
    )
    basis_set_file: str = Field(default="BASIS_SET")
    potential_file: str = Field(default="POTENTIAL")
    basis_set: str = Field(default="DZVP-MOLOPT-SR-GTH")
    potential: str = Field(default="GTH-PBE")
    cutoff: int = Field(default=400)
    rel_cutoff: int = Field(default=50)
    scf_eps: float = Field(default=1e-6)
    max_scf: int = Field(default=50)
    working_dir: str | None = Field(default=None)
    output_prefix: str = Field(default="cp2k_out")
    result_files: list[str] = Field(default_factory=list)


class Cp2kTool(HuginnTool):
    """Generate and run CP2K DFT and molecular dynamics calculations."""

    name = "cp2k_tool"
    description = (
        "Generate and run CP2K DFT/MD calculations. "
        "Falls back to exporting the input file when CP2K is not installed."
    )
    input_schema = Cp2kToolInput

    def __init__(self, cp2k_executable: str | None = None, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.cp2k_executable = cp2k_executable or self._find_cp2k()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_cp2k(self) -> str | None:
        env_path = os.environ.get("CP2K_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        for cmd in ["cp2k.popt", "cp2k.sopt", "cp2k"]:
            if shutil.which(cmd):
                return cmd
        return None

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = Cp2kToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "parse":
                return self._parse_results(input_data, work_dir)

            input_path = self._generate_input(input_data, work_dir, input_data.output_prefix)

            if input_data.action == "generate":
                return ToolResult(
                    data={
                        "input_path": str(input_path),
                        "cp2k_available": self.cp2k_executable is not None,
                        "message": "Generated CP2K input file.",
                    },
                    success=True,
                )

            return self._run_cp2k(input_data, work_dir, input_path)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"CP2K tool failed: {e}")

    def _generate_input(
        self, args: Cp2kToolInput, work_dir: Path, prefix: str
    ) -> Path:
        input_path = work_dir / f"{prefix}.inp"
        species = args.structure.get("species", [])
        positions = args.structure.get("positions", [])
        cell = args.structure.get("cell", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        units = args.structure.get("units", "ANGSTROM")

        coords = []
        for elem, pos in zip(species, positions):
            coords.append(f"  {elem:<4} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

        lines = [
            "&GLOBAL",
            f"  PROJECT {prefix}",
            f"  RUN_TYPE {args.run_type}",
            "  PRINT_LEVEL MEDIUM",
            "&END GLOBAL",
            "",
            "&FORCE_EVAL",
            "  METHOD Quickstep",
            "  &DFT",
            f"    BASIS_SET_FILE_NAME {args.basis_set_file}",
            f"    POTENTIAL_FILE_NAME {args.potential_file}",
            "    &QS",
            f"      EPS_DEFAULT 1.0E-12",
            "    &END QS",
            "    &MGRID",
            f"      CUTOFF {args.cutoff}",
            f"      REL_CUTOFF {args.rel_cutoff}",
            "    &END MGRID",
            "    &XC",
            "      &XC_FUNCTIONAL",
            f"        &{args.method} T",
            "        &END",
            "      &END XC_FUNCTIONAL",
            "    &END XC",
            "    &SCF",
            f"      EPS_SCF {args.scf_eps}",
            f"      MAX_SCF {args.max_scf}",
            "    &END SCF",
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
            f"      A {cell[0][0]:.8f} {cell[0][1]:.8f} {cell[0][2]:.8f}",
            f"      B {cell[1][0]:.8f} {cell[1][1]:.8f} {cell[1][2]:.8f}",
            f"      C {cell[2][0]:.8f} {cell[2][1]:.8f} {cell[2][2]:.8f}",
            f"      PERIODIC XYZ",
            "    &END CELL",
            "    &COORD",
            f"      UNIT {units}",
        ]
        lines.extend(coords)
        lines.extend([
            "    &END COORD",
            "    &KIND Si",
            f"      BASIS_SET {args.basis_set}",
            f"      POTENTIAL {args.potential}",
            "    &END KIND",
            "  &END SUBSYS",
            "&END FORCE_EVAL",
        ])

        if args.run_type == "MD":
            lines.extend([
                "",
                "&MOTION",
                "  &MD",
                "    ENSEMBLE NVT",
                "    TEMPERATURE 300.0",
                "    TIMESTEP 1.0",
                "    STEPS 100",
                "    &THERMOSTAT",
                "      TYPE NOSE",
                "      &NOSE",
                "        LENGTH 3",
                "        YOSHIDA 3",
                "        TIMECON 100.0",
                "        MTS 2",
                "      &END NOSE",
                "    &END THERMOSTAT",
                "  &END MD",
                "&END MOTION",
            ])

        input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return input_path

    def _run_cp2k(
        self, args: Cp2kToolInput, work_dir: Path, input_path: Path
    ) -> ToolResult:
        if not self.cp2k_executable:
            return ToolResult(
                data={
                    "input_path": str(input_path),
                    "cp2k_available": False,
                    "message": (
                        "CP2K executable not found. Input file exported; "
                        "run manually with: cp2k.popt -i " + input_path.name
                    ),
                },
                success=True,
            )

        output_path = work_dir / f"{args.output_prefix}.out"
        cmd = [self.cp2k_executable, "-i", str(input_path), "-o", str(output_path)]

        cfg = SandboxConfig(dry_run=False)
        result = self.sandbox.run(cmd, cwd=work_dir, config=cfg)

        parsed = self._parse_output_file(output_path)
        success = result.get("returncode", -1) == 0
        return ToolResult(
            data={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "cp2k_available": True,
                "parsed": parsed,
                "message": "CP2K execution completed." if success else "CP2K execution failed; see output.",
            },
            success=success,
        )

    def _parse_output_file(self, output_path: Path) -> dict[str, Any]:
        if not output_path.exists():
            return {"error": "Output file not found"}
        content = output_path.read_text(encoding="utf-8", errors="ignore")
        return self._parse_output(content)

    def _parse_output(self, content: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "energy": None,
            "converged": False,
            "forces": [],
            "stress": [],
            "n_scf_steps": 0,
        }

        lines = content.splitlines()

        # Energy
        for line in lines:
            if "Total energy:" in line or "ENERERGY| Total FORCE_EVAL" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        val = float(p)
                        # Accept values that look like total energies (negative or large)
                        if val < 0 or abs(val) > 1:
                            result["energy"] = val
                            break
                    except ValueError:
                        continue

        # Convergence
        result["converged"] = "SCF run converged" in content or "*** SCF run converged" in content

        # SCF steps
        result["n_scf_steps"] = content.count("SCF iteration")

        # Forces
        in_forces = False
        force_block: list[list[float]] = []
        for line in lines:
            if "ATOMIC FORCES in" in line or "FORCES" in line and "atom" in line.lower():
                in_forces = True
                force_block = []
                continue
            if in_forces:
                parts = line.split()
                if len(parts) >= 6 and parts[0].isdigit():
                    try:
                        force_block.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
                    except (ValueError, IndexError):
                        pass
                elif line.strip() == "":
                    in_forces = False
        if force_block:
            result["forces"] = force_block

        # Stress
        for i, line in enumerate(lines):
            if "STRESS|" in line or "Stress tensor" in line:
                stress = []
                for j in range(1, 4):
                    if i + j < len(lines):
                        parts = lines[i + j].split()
                        if len(parts) >= 3:
                            try:
                                stress.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
                            except (ValueError, IndexError):
                                pass
                if len(stress) == 3:
                    result["stress"] = stress

        return result

    def _parse_results(self, args: Cp2kToolInput, work_dir: Path) -> ToolResult:
        parsed: dict[str, Any] = {}
        for file_name in args.result_files:
            file_path = work_dir / file_name
            parsed[file_name] = self._parse_output_file(file_path)

        return ToolResult(
            data={"results": parsed, "message": f"Parsed {len(parsed)} CP2K output files."},
            success=True,
        )
