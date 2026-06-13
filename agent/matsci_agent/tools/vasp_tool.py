"""VASP DFT calculation tool.

Supports both real VASP execution (if available) and mock mode.
"""

from __future__ import annotations

import os
from matsci_agent.security import SandboxExecutor, SandboxConfig
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext

try:
    import matsci_ext
    _HAS_MATSCI_EXT = True
except ImportError:
    matsci_ext = None
    _HAS_MATSCI_EXT = False


class VaspToolInput(BaseModel):
    action: Literal["relax", "scf", "band", "dos", "md", "phonon"] = Field(...)
    working_dir: str = Field(..., description="Directory containing POSCAR/INCAR/POTCAR/KPOINTS")
    incar_overrides: dict = Field(default_factory=dict, description="Override specific INCAR tags")
    queue: Literal["debug", "normal", "gpu"] = Field(default="normal")
    walltime_hours: int = Field(default=24, ge=1, le=168)


class VaspToolOutput(BaseModel):
    job_id: str | None = None
    status: Literal["completed", "failed", "mock"] = "mock"
    energy: float | None = None
    converged: bool = False
    output_files: list[str] = []
    warnings: list[str] = []


class VaspTool(MatSciTool):
    """Submit and manage VASP DFT calculations."""
    
    name = "vasp_tool"
    description = "Run VASP DFT calculations (relaxation, SCF, band structure, DOS, MD, phonons)"
    input_schema = VaspToolInput
    
    def __init__(self, vasp_executable: str | None = None, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.vasp_executable = vasp_executable or self._find_vasp()
        self.sandbox = sandbox or SandboxExecutor()
    
    def _find_vasp(self) -> str | None:
        """Find VASP executable."""
        env_path = os.environ.get("VASP_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        
        # Check PATH
        try:
            import shutil
            for name in ["vasp", "vasp_std", "vasp_gam", "vasp_ncl"]:
                exe = shutil.which(name)
                if exe:
                    return exe
        except Exception:
            pass
        
        return None
    
    def estimate_cost(self, args: VaspToolInput) -> dict[str, float] | None:
        return {"cpu_hours": args.walltime_hours * 4, "walltime_hours": args.walltime_hours}
    
    async def call(self, args: VaspToolInput, context: ToolContext) -> ToolResult:
        work_dir = Path(args.working_dir)
        if not work_dir.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Working directory not found: {work_dir}"
            )
        
        # Check for required input files
        poscar = work_dir / "POSCAR"
        incar = work_dir / "INCAR"
        if not poscar.exists():
            return ToolResult(
                data=None,
                success=False,
                error="POSCAR not found in working directory"
            )
        
        # Apply INCAR overrides
        if args.incar_overrides and incar.exists():
            self._modify_incar(incar, args.incar_overrides)
        
        # If VASP is available, run it
        if self.vasp_executable:
            return await self._run_vasp(args, work_dir)
        
        # Mock mode: return synthetic results
        return self._mock_result(args, work_dir)
    
    async def _run_vasp(self, args: VaspToolInput, work_dir: Path) -> ToolResult:
        """Execute real VASP calculation."""
        try:
            cmd = [self.vasp_executable]
            sb_result = self.sandbox.run(
                cmd,
                cwd=str(work_dir),
                timeout=args.walltime_hours * 3600,
            )
            result = sb_result
            
            # Parse OUTCAR for comprehensive results
            outcar = work_dir / "OUTCAR"
            parsed = self._parse_outcar(outcar) if outcar.exists() else {}
            
            # Also try vasprun.xml for structured data
            vasprun = work_dir / "vasprun.xml"
            if vasprun.exists():
                parsed.update(self._parse_vasprun_quick(vasprun))
            
            output = VaspToolOutput(
                status="completed" if result.returncode == 0 else "failed",
                energy=parsed.get("energy"),
                converged=parsed.get("converged", False),
                output_files=[f.name for f in work_dir.iterdir() if f.suffix in [".OUTCAR", ".vasprun", ".CHG"]],
            )
            
            # Include parsed details in result
            data = output.model_dump()
            data["parsed"] = parsed
            
            return ToolResult(
                data=data,
                success=result.returncode == 0,
                error=result.stderr[:500] if result.returncode != 0 else None,
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None,
                success=False,
                error=f"VASP execution timed out ({args.walltime_hours}h)"
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"VASP execution failed: {e}"
            )
    
    def _parse_outcar(self, outcar_path: Path) -> dict[str, Any]:
        """Parse OUTCAR for key physical quantities.
        
        Uses the Rust accelerator when available and falls back to pure Python.
        """
        if _HAS_MATSCI_EXT:
            try:
                result = matsci_ext.parse_outcar(str(outcar_path))
                if "error" not in result:
                    return result
            except Exception:
                pass
        
        return self._parse_outcar_python(outcar_path)
    
    def _parse_outcar_python(self, outcar_path: Path) -> dict[str, Any]:
        """Pure-Python OUTCAR parser (baseline/fallback)."""
        import re
        
        result = {
            "energy": None,
            "converged": False,
            "forces": [],
            "magnetic_moments": [],
            "lattice_vectors": [],
            "volume": None,
            "band_gap": None,
            "encut": None,
            "kpoints": None,
            "nelm": None,
            "nelmin": None,
            "ispin": None,
        }
        
        try:
            content = outcar_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            
            # Energy
            energy_matches = re.findall(r"free  energy   TOTEN  =\s+([-\d.]+)", content)
            if energy_matches:
                result["energy"] = float(energy_matches[-1])
            
            # Convergence
            result["converged"] = "reached required accuracy" in content
            
            # ENCUT
            encut_match = re.search(r"ENCUT\s*=\s*([\d.]+)", content)
            if encut_match:
                result["encut"] = float(encut_match.group(1))
            
            # ISPIN
            ispin_match = re.search(r"ISPIN\s*=\s*(\d+)", content)
            if ispin_match:
                result["ispin"] = int(ispin_match.group(1))
            
            # NELM / NELMIN
            nelm_match = re.search(r"NELM\s*=\s*(\d+)", content)
            if nelm_match:
                result["nelm"] = int(nelm_match.group(1))
            nelmin_match = re.search(r"NELMIN\s*=\s*(\d+)", content)
            if nelmin_match:
                result["nelmin"] = int(nelmin_match.group(1))
            
            # K-points
            kpoint_match = re.search(r"k-points in units of 2pi/SCALE and weight:.*\n.*\n.*", content)
            if kpoint_match:
                result["kpoints"] = "found"  # Simplified
            
            # Lattice vectors (last occurrence)
            lattice_pattern = r"direct lattice vectors\s+reciprocal lattice vectors\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*"
            lattice_matches = re.findall(lattice_pattern, content)
            if lattice_matches:
                last = lattice_matches[-1]
                result["lattice_vectors"] = [
                    [float(last[0]), float(last[1]), float(last[2])],
                    [float(last[3]), float(last[4]), float(last[5])],
                    [float(last[6]), float(last[7]), float(last[8])],
                ]
            
            # Volume
            vol_match = re.findall(r"volume of cell :\s+([\d.]+)", content)
            if vol_match:
                result["volume"] = float(vol_match[-1])
            
            # Final forces
            force_section = re.findall(r"TOTAL-FORCE.*?\n(.*?)(?:\n\n|\n---)", content, re.DOTALL)
            if force_section:
                forces = []
                for line in force_section[-1].strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 6 and all(self._is_float(p) for p in parts[:6]):
                        forces.append({
                            "position": [float(parts[0]), float(parts[1]), float(parts[2])],
                            "force": [float(parts[3]), float(parts[4]), float(parts[5])],
                        })
                result["forces"] = forces
            
            # Magnetic moments
            mag_matches = re.findall(r"magnetization \(x\).*?\n(.*?)(?:\n\n|\n---)", content, re.DOTALL)
            if mag_matches:
                mag_moments = []
                for line in mag_matches[-1].strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and self._is_float(parts[-1]):
                        mag_moments.append(float(parts[-1]))
                result["magnetic_moments"] = mag_moments
            
            # Band gap (from E-fermi and band occupancies)
            efermi_match = re.search(r"E-fermi\s*:\s*([-\d.]+)", content)
            if efermi_match:
                result["efermi"] = float(efermi_match.group(1))
                # Rough gap estimate from highest occupied / lowest unoccupied
                band_matches = re.findall(r"k-point\s+\d+.*?\n(.*?)(?:\n\n|\nk-point)", content, re.DOTALL)
                # Simplified: just note if gap was calculated
                if "band No." in content:
                    result["band_gap"] = "see vasprun.xml or use py4vasp"
            
        except Exception as e:
            result["parse_error"] = str(e)
        
        return result
    
    def _parse_vasprun_quick(self, vasprun_path: Path) -> dict[str, Any]:
        """Quick-parse vasprun.xml for structured data."""
        import xml.etree.ElementTree as ET
        
        result = {"parse_source": "vasprun.xml"}
        
        try:
            tree = ET.parse(vasprun_path)
            root = tree.getroot()
            
            # Find calculation/energy/i
            for calc in root.findall(".//calculation"):
                energy_elem = calc.find(".//energy/i[@name='e_wo_entrp']")
                if energy_elem is not None and energy_elem.text:
                    result["energy_vasprun"] = float(energy_elem.text)
                
                # Forces
                varray = calc.find(".//varray[@name='forces']")
                if varray is not None:
                    forces = []
                    for v in varray.findall("v"):
                        forces.append([float(x) for x in v.text.split()])
                    result["forces_vasprun"] = forces
                break  # Only first calc for quick parse
            
            # K-points
            kpoints = root.find(".//kpoints")
            if kpoints is not None:
                varray = kpoints.find("varray[@name='kpointlist']")
                if varray is not None:
                    result["kpoint_count"] = len(varray.findall("v"))
            
        except Exception as e:
            result["parse_error"] = str(e)
        
        return result
    
    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _mock_result(self, args: VaspToolInput, work_dir: Path) -> ToolResult:
        """Generate mock results when VASP is not available."""
        import random
        
        mock_energies = {
            "relax": -150.0,
            "scf": -152.3,
            "band": -152.3,
            "dos": -152.3,
            "md": -148.5,
            "phonon": -152.3,
        }
        
        output = VaspToolOutput(
            status="mock",
            energy=mock_energies.get(args.action, -100.0) + random.uniform(-0.5, 0.5),
            converged=True,
            output_files=["OUTCAR", "vasprun.xml", "OSZICAR"],
            warnings=["VASP executable not found. Results are MOCK data for demonstration."]
        )
        
        return ToolResult(
            data=output.model_dump(),
            success=True,
            error=None,
        )
    
    def _modify_incar(self, incar_path: Path, overrides: dict) -> None:
        """Modify INCAR file with override values."""
        try:
            content = incar_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            modified = []
            overridden_keys = set()
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    modified.append(line)
                    continue
                
                # Check if this line defines a key we want to override
                for key in overrides:
                    if stripped.upper().startswith(key.upper() + " =") or stripped.upper().startswith(key.upper() + "="):
                        modified.append(f"{key} = {overrides[key]}")
                        overridden_keys.add(key)
                        break
                else:
                    modified.append(line)
            
            # Add any new keys that weren't in the original file
            for key, value in overrides.items():
                if key not in overridden_keys:
                    modified.append(f"{key} = {value}")
            
            incar_path.write_text("\n".join(modified), encoding="utf-8")
        
        except Exception as e:
            print(f"Warning: Failed to modify INCAR: {e}")
