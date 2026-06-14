"""LAMMPS molecular dynamics tool — real execution via subprocess.

Uses the installed lmp.exe for actual MD simulations.
"""

from __future__ import annotations

import os
import re
from huginn.security import SandboxExecutor, SandboxConfig
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolResult, ToolContext

try:
    import huginn_ext
    _HAS_HUGINN_EXT = True
except ImportError:
    huginn_ext = None
    _HAS_HUGINN_EXT = False


class LammpsToolInput(BaseModel):
    action: Literal["run", "minimize", "equilibrate", "analyze_trajectory"] = Field(...)
    input_script: str = Field(default="", description="LAMMPS input script content or file path")
    structure_file: str | None = Field(default=None, description="Structure file path (data, xyz, etc.)")
    potentials: list[str] = Field(default_factory=list, description="List of potential file paths")
    trajectory_file: str | None = Field(default=None, description="Trajectory file to analyze (for analyze_trajectory)")
    output_prefix: str = Field(default="lammps_out")
    num_processes: int = Field(default=1, ge=1)
    working_dir: str | None = Field(default=None)
    fixes: dict[str, str] = Field(default_factory=dict, description="Auto-applied fixes from diagnosis (e.g., {'timestep': '0.5'})")


class LammpsToolOutput(BaseModel):
    log_path: str | None = None
    trajectory_path: str | None = None
    thermo_data: dict | None = None
    final_energy: float | None = None
    warnings: list[str] = []


class LammpsTool(HuginnTool):
    """Execute LAMMPS molecular dynamics simulations."""
    
    name = "lammps_tool"
    description = "Run LAMMPS molecular dynamics simulations (minimization, equilibration, production)"
    input_schema = LammpsToolInput
    
    def __init__(self, lammps_executable: str | None = None, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.lammps_executable = lammps_executable or self._find_lammps()
        self.sandbox = sandbox or SandboxExecutor()
    
    def _find_lammps(self) -> str | None:
        """Find LAMMPS executable on the system."""
        import glob
        
        # Check environment variable
        env_path = os.environ.get("LAMMPS_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        
        # Check PATH
        try:
            import shutil
            exe = shutil.which("lmp")
            if exe:
                return exe
        except Exception:
            pass
        
        # Check common Windows locations (with glob for unicode paths)
        patterns = [
            r"C:\Users\*\OneDrive\*\LAMMPS*\bin\lmp.exe",
            r"C:\Program Files*\LAMMPS*\bin\lmp.exe",
            r"C:\ProgramData\*\LAMMPS*\bin\lmp.exe",
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            for m in matches:
                if Path(m).exists():
                    return m
        
        return None
    
    def estimate_cost(self, args: LammpsToolInput) -> dict[str, float] | None:
        return {"cpu_hours": 2, "walltime_hours": 2}
    
    async def call(self, args: LammpsToolInput, context: ToolContext) -> ToolResult:
        # Handle trajectory analysis without running LAMMPS
        if args.action == "analyze_trajectory":
            traj_file = args.trajectory_file or args.input_script
            if not traj_file or not Path(traj_file).exists():
                return ToolResult(
                    data=None,
                    success=False,
                    error="Trajectory file not specified or not found"
                )
            analysis = self.parse_trajectory(traj_file)
            return ToolResult(
                data=analysis,
                success="error" not in analysis,
                error=analysis.get("error"),
            )
        
        if not self.lammps_executable:
            return ToolResult(
                data=None,
                success=False,
                error="LAMMPS executable not found. Set LAMMPS_EXECUTABLE environment variable or install LAMMPS."
            )
        
        # Determine working directory
        if args.working_dir:
            work_dir = Path(args.working_dir)
        else:
            work_dir = Path(context.workspace) / f"lammps_{args.output_prefix}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Write input script
        input_path = work_dir / "input.lammps"
        
        # Check if input_script is a file path or content
        script_path = Path(args.input_script)
        if script_path.exists():
            script_content = script_path.read_text(encoding="utf-8")
        else:
            script_content = args.input_script
        
        # Prepend structure read if structure_file provided
        if args.structure_file:
            structure_path = Path(args.structure_file)
            if structure_path.exists():
                # Detect format and prepend read command
                if structure_path.suffix in [".data", ".lmp"]:
                    prefix = f"read_data {structure_path}\n"
                elif structure_path.suffix == ".xyz":
                    prefix = f"read_xyz {structure_path}\n"
                else:
                    prefix = f"read_data {structure_path}\n"
                
                if "read_data" not in script_content and "read_xyz" not in script_content:
                    script_content = prefix + script_content
        
        # Apply auto-fixes from diagnosis to input script
        if args.fixes:
            script_content = self._apply_script_fixes(script_content, args.fixes)
        
        input_path.write_text(script_content, encoding="utf-8")
        
        # Copy potential files to working directory
        for pot in args.potentials:
            pot_path = Path(pot)
            if pot_path.exists():
                dest = work_dir / pot_path.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(pot_path, dest)
        
        # Resolve to absolute paths to avoid relative path issues on Windows
        work_dir_abs = work_dir.resolve()
        input_path_abs = input_path.resolve()
        log_path_abs = (work_dir_abs / "log.lammps").resolve()
        
        # Build command
        cmd = [self.lammps_executable, "-in", str(input_path_abs), "-log", str(log_path_abs)]
        if args.num_processes > 1:
            cmd = ["mpiexec", "-n", str(args.num_processes)] + cmd
        
        # Run LAMMPS
        try:
            sb_result = self.sandbox.run(
                cmd,
                cwd=str(work_dir_abs),
                timeout=3600,
            )
            result = sb_result
            
            # Parse log file for thermo data
            log_path = work_dir / "log.lammps"
            thermo_data, final_energy, warnings = self._parse_log(log_path)
            
            # Find trajectory file
            traj_path = None
            for ext in [".lammpstrj", ".dump", ".xyz"]:
                candidates = list(work_dir.glob(f"*{ext}"))
                if candidates:
                    traj_path = str(candidates[0])
                    break
            
            output = LammpsToolOutput(
                log_path=str(log_path),
                trajectory_path=traj_path,
                thermo_data=thermo_data,
                final_energy=final_energy,
                warnings=warnings,
            )
            
            success = result.returncode == 0
            error = result.stderr[:500] if not success else None
            
            data = output.model_dump()
            
            # Auto-parse trajectory if available
            if traj_path:
                traj_analysis = self.parse_trajectory(traj_path)
                if "error" not in traj_analysis:
                    data["trajectory_analysis"] = traj_analysis
            
            return ToolResult(
                data=data,
                success=success,
                error=error,
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None,
                success=False,
                error="LAMMPS execution timed out (3600s)"
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"LAMMPS execution failed: {e}"
            )
    
    def _parse_log(self, log_path: Path) -> tuple[dict, float | None, list[str]]:
        """Parse LAMMPS log file for thermodynamic data."""
        if not log_path.exists():
            return {}, None, ["Log file not found"]
        
        thermo_data = {}
        final_energy = None
        warnings = []
        
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            
            # Identify thermo columns from the header
            # Pattern: Step Temp Press TotEng ...
            header_match = re.search(r"^(Step\s+.*?)$", content, re.MULTILINE)
            columns = []
            if header_match:
                columns = header_match.group(1).split()
            
            # Extract all thermo data rows
            data_rows = []
            # Match lines that start with an integer step number followed by numeric values
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        # Verify most parts are numeric
                        numeric_count = sum(1 for p in parts if self._is_float(p))
                        if numeric_count >= len(parts) - 1:
                            data_rows.append([self._to_float_or_str(p) for p in parts])
                    except ValueError:
                        pass
            
            if data_rows and columns:
                # Transpose: columns[0] is Step, columns[1] is Temp, etc.
                for col_idx, col_name in enumerate(columns):
                    if col_idx < len(data_rows[0]):
                        values = [row[col_idx] for row in data_rows if col_idx < len(row)]
                        # Try to convert to float
                        float_values = []
                        for v in values:
                            if isinstance(v, float):
                                float_values.append(v)
                            elif isinstance(v, str) and self._is_float(v):
                                float_values.append(float(v))
                        if float_values:
                            thermo_data[col_name.lower()] = float_values
            
            # Extract final energy
            if "toteng" in thermo_data and thermo_data["toteng"]:
                final_energy = thermo_data["toteng"][-1]
            elif "toteng" not in thermo_data:
                # Fallback: search for TotEng explicitly
                energy_match = re.findall(r"TotEng\s+([-\d.eE]+)", content)
                if energy_match:
                    try:
                        final_energy = float(energy_match[-1])
                    except ValueError:
                        pass
            
            # Check for warnings
            if "WARNING" in content:
                warn_lines = [l.strip() for l in content.split("\n") if "WARNING" in l]
                warnings.extend(warn_lines[:5])
            
            # Check for errors
            if "ERROR" in content:
                err_lines = [l.strip() for l in content.split("\n") if "ERROR" in l]
                warnings.extend(err_lines[:3])
        
        except Exception as e:
            warnings.append(f"Failed to parse log: {e}")
        
        return thermo_data, final_energy, warnings
    
    def _apply_script_fixes(self, script: str, fixes: dict[str, str]) -> str:
        """Apply diagnosed fixes to LAMMPS input script.
        
        Replaces command parameters like 'timestep 1.0' with 'timestep 0.5'.
        """
        lines = script.split("\n")
        modified = []
        applied = set()
        
        for line in lines:
            stripped = line.strip().lower()
            # Skip comments and blank lines
            if not stripped or stripped.startswith("#"):
                modified.append(line)
                continue
            
            # Check each fix key
            for key, new_value in fixes.items():
                key_lower = key.lower()
                # Match command at start of line (allow leading whitespace)
                parts = stripped.split()
                if parts and parts[0] == key_lower:
                    # Replace the value part(s)
                    # e.g., 'timestep 1.0' → 'timestep 0.5'
                    # e.g., 'fix nvt all temp 300 300 0.1' → more complex
                    indent = line[:len(line) - len(line.lstrip())]
                    modified_line = f"{indent}{key} {new_value}"
                    modified.append(modified_line)
                    applied.add(key)
                    break
            else:
                modified.append(line)
        
        # If any fix wasn't applied, append it at the end
        for key, new_value in fixes.items():
            if key.lower() not in applied:
                modified.append(f"{key} {new_value}")
        
        return "\n".join(modified)
    
    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _to_float_or_str(self, s: str):
        try:
            return float(s)
        except ValueError:
            return s
    
    def parse_trajectory(self, traj_path: str | Path) -> dict[str, Any]:
        """Parse LAMMPS trajectory file and compute basic analyses.
        
        Supports .lammpstrj and .dump formats.
        Uses a Rust accelerator if available, falling back to pure Python.
        """
        from pathlib import Path
        import numpy as np
        
        traj_path = Path(traj_path)
        if not traj_path.exists():
            return {"error": "Trajectory file not found"}
        
        # Try Rust-accelerated parser first.
        if _HAS_HUGINN_EXT:
            try:
                result = huginn_ext.parse_lammps_dump(
                    str(traj_path),
                    compute_msd=True,
                    compute_rdf=True,
                    rdf_bins=100,
                    rdf_r_max=None,
                    include_frames=False,
                )
                if "error" not in result:
                    return result
            except Exception:
                pass
        
        return self._parse_trajectory_python(traj_path)
    
    def _parse_trajectory_python(self, traj_path: str | Path) -> dict[str, Any]:
        """Pure-Python LAMMPS trajectory parser (baseline/fallback)."""
        from pathlib import Path
        
        traj_path = Path(traj_path)
        
        result = {
            "n_frames": 0,
            "n_atoms": 0,
            "atom_types": set(),
            "box_bounds": [],
            "timesteps": [],
        }
        
        try:
            frames = []
            current_frame = None
            
            with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line == "ITEM: TIMESTEP":
                    if current_frame:
                        frames.append(current_frame)
                    current_frame = {"atoms": []}
                    i += 1
                    if i < len(lines):
                        current_frame["timestep"] = int(lines[i].strip())
                        result["timesteps"].append(current_frame["timestep"])
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    i += 1
                    if i < len(lines):
                        current_frame["n_atoms"] = int(lines[i].strip())
                        result["n_atoms"] = current_frame["n_atoms"]
                elif line.startswith("ITEM: BOX BOUNDS"):
                    bounds = []
                    for _ in range(3):
                        i += 1
                        if i < len(lines):
                            bounds.append([float(x) for x in lines[i].strip().split()])
                    current_frame["box"] = bounds
                    if not result["box_bounds"]:
                        result["box_bounds"] = bounds
                elif line.startswith("ITEM: ATOMS"):
                    # Parse atom data
                    atom_headers = line.replace("ITEM: ATOMS ", "").split()
                    atoms = []
                    for _ in range(current_frame.get("n_atoms", 0)):
                        i += 1
                        if i < len(lines):
                            parts = lines[i].strip().split()
                            atom = {}
                            for h, p in zip(atom_headers, parts):
                                try:
                                    atom[h] = float(p)
                                except ValueError:
                                    atom[h] = p
                            atoms.append(atom)
                            if "type" in atom:
                                result["atom_types"].add(int(atom["type"]))
                    current_frame["atoms"] = atoms
                i += 1
            
            if current_frame:
                frames.append(current_frame)
            
            result["n_frames"] = len(frames)
            result["atom_types"] = sorted(result["atom_types"])
            
            # Compute MSD if positions available
            if frames and len(frames) > 1 and all("x" in a for a in frames[0]["atoms"]):
                msd = self._compute_msd(frames)
                if msd:
                    result["msd"] = msd
            
            # Compute RDF if 2+ frames
            if frames and len(frames) >= 1 and all("x" in a for a in frames[0]["atoms"]):
                rdf = self._compute_rdf(frames[-1])
                if rdf:
                    result["rdf"] = rdf
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _compute_msd(self, frames: list[dict]) -> list[dict] | None:
        """Compute mean squared displacement across frames."""
        try:
            msd_data = []
            ref_positions = []
            for atom in frames[0]["atoms"]:
                ref_positions.append([atom["x"], atom["y"], atom["z"]])
            
            for frame in frames[1:]:
                displacements = []
                for i, atom in enumerate(frame["atoms"]):
                    dx = atom["x"] - ref_positions[i][0]
                    dy = atom["y"] - ref_positions[i][1]
                    dz = atom["z"] - ref_positions[i][2]
                    displacements.append(dx*dx + dy*dy + dz*dz)
                msd = sum(displacements) / len(displacements)
                msd_data.append({
                    "timestep": frame.get("timestep", 0),
                    "msd": msd,
                })
            return msd_data
        except Exception:
            return None
    
    def _compute_rdf(self, frame: dict, bins: int = 100, r_max: float | None = None) -> dict | None:
        """Compute radial distribution function for a single frame."""
        try:
            atoms = frame["atoms"]
            positions = []
            for atom in atoms:
                positions.append([atom["x"], atom["y"], atom["z"]])
            
            import math
            n = len(positions)
            
            # Estimate r_max from box
            box = frame.get("box", [[0, 10], [0, 10], [0, 10]])
            lx = box[0][1] - box[0][0]
            ly = box[1][1] - box[1][0]
            lz = box[2][1] - box[2][0]
            if r_max is None:
                r_max = min(lx, ly, lz) / 2
            
            dr = r_max / bins
            g = [0.0] * bins
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    # Minimum image convention
                    dx -= lx * round(dx / lx)
                    dy -= ly * round(dy / ly)
                    dz -= lz * round(dz / lz)
                    r = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if r < r_max:
                        idx = int(r / dr)
                        if idx < bins:
                            g[idx] += 2.0  # Pair contribution
            
            # Normalize
            volume = lx * ly * lz
            rho = n / volume
            for i in range(bins):
                r_inner = i * dr
                r_outer = (i + 1) * dr
                shell_vol = 4/3 * math.pi * (r_outer**3 - r_inner**3)
                if shell_vol > 0:
                    g[i] /= (n * rho * shell_vol)
            
            r_values = [(i + 0.5) * dr for i in range(bins)]
            return {"r": r_values, "g": g, "bins": bins, "r_max": r_max}
        except Exception:
            return None
