"""AutoDock Vina docking tool — run molecular docking for drug discovery.

Uses the vina CLI executable (or Python bindings if available). When neither
is found, returns a needs_resolution flag so the agent can prompt the user
for an installation path — same flow as VASP / LAMMPS / GROMACS.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxError, SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.sim.executable_resolver import resolve_executable, ResolutionRequest
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class VinaToolInput(BaseModel):
    action: Literal["dock", "score_only", "prepare_ligand"] = Field(...)
    working_dir: str | None = Field(default=None)

    # Docking params
    receptor_pdbqt: str | None = Field(
        default=None, description="Path to receptor .pdbqt file"
    )
    ligand_pdbqt: str | None = Field(
        default=None, description="Path to ligand .pdbqt file"
    )
    center_x: float = Field(default=0.0, description="Grid box center X (Angstrom)")
    center_y: float = Field(default=0.0, description="Grid box center Y (Angstrom)")
    center_z: float = Field(default=0.0, description="Grid box center Z (Angstrom)")
    size_x: float = Field(default=22.5, description="Grid box size X (Angstrom)")
    size_y: float = Field(default=22.5, description="Grid box size Y (Angstrom)")
    size_z: float = Field(default=22.5, description="Grid box size Z (Angstrom)")
    exhaustiveness: int = Field(default=8, ge=1, le=32, description="Exhaustiveness of search")
    num_modes: int = Field(default=9, ge=1, le=100, description="Number of output binding modes")
    energy_range: float = Field(default=3.0, gt=0, description="Max energy difference from best mode (kcal/mol)")

    # Score-only params
    pose_pdbqt: str | None = Field(
        default=None, description="Path to pose .pdbqt for rescoring"
    )

    # Prepare ligand params
    smiles: str | None = Field(default=None, description="Ligand SMILES for prepare_ligand")
    input_sdf: str | None = Field(default=None, description="Input SDF for prepare_ligand")
    output_pdbqt: str | None = Field(default=None, description="Output .pdbqt path")


class VinaTool(HuginnTool):
    """Run AutoDock Vina molecular docking."""

    name = "vina_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="docking",
    )
    description = (
        "AutoDock Vina molecular docking: predict ligand-receptor binding "
        "poses and affinities. Also supports rescoring and ligand preparation "
        "(SMILES/SDF to PDBQT)."
    )
    input_schema = VinaToolInput

    def __init__(self, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.sandbox = sandbox or SandboxExecutor()

    def is_read_only(self, args: VinaToolInput) -> bool:
        return args.action == "score_only"

    def is_destructive(self, args: VinaToolInput) -> bool:
        return args.action in ("dock", "prepare_ligand")

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = VinaToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if input_data.action == "dock":
            return self._dock(input_data, work_dir)
        if input_data.action == "score_only":
            return self._score_only(input_data, work_dir)
        if input_data.action == "prepare_ligand":
            return self._prepare_ligand(input_data, work_dir)
        return ToolResult(data=None, success=False, error=f"Unknown action: {input_data.action}")

    # ── dock ─────────────────────────────────────────────────

    def _dock(self, inp: VinaToolInput, work_dir: Path) -> ToolResult:
        receptor = self._resolve_file(inp.receptor_pdbqt, work_dir)
        ligand = self._resolve_file(inp.ligand_pdbqt, work_dir)
        if receptor is None:
            return ToolResult(data=None, success=False, error=f"Receptor PDBQT not found: {inp.receptor_pdbqt}")
        if ligand is None:
            return ToolResult(data=None, success=False, error=f"Ligand PDBQT not found: {inp.ligand_pdbqt}")

        exe = self._resolve_vina()
        if isinstance(exe, ResolutionRequest):
            return ToolResult(
                data=None,
                success=False,
                error="AutoDock Vina executable not found.",
                metadata={"needs_resolution": True, "resolution_request": exe.to_dict()},
            )

        out_pdbqt = work_dir / "docked.pdbqt"
        cmd = [
            exe,
            "--receptor", str(receptor),
            "--ligand", str(ligand),
            "--out", str(out_pdbqt),
            "--center_x", str(inp.center_x),
            "--center_y", str(inp.center_y),
            "--center_z", str(inp.center_z),
            "--size_x", str(inp.size_x),
            "--size_y", str(inp.size_y),
            "--size_z", str(inp.size_z),
            "--exhaustiveness", str(inp.exhaustiveness),
            "--num_modes", str(inp.num_modes),
            "--energy_range", str(inp.energy_range),
        ]

        try:
            result = self.sandbox.run(
                cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=3600.0
            )
            if result.returncode != 0:
                return ToolResult(
                    data={"returncode": result.returncode, "stderr": result.stderr[-2000:]},
                    success=False,
                    error=f"Vina docking failed (exit {result.returncode})",
                )

            poses = self._parse_vina_output(result.stdout, out_pdbqt)

            # Physics audit — docking scores should be within physical range
            try:
                from huginn.execution.physics_auditor import PhysicsAuditor

                auditor = PhysicsAuditor()
                audit = auditor.audit("vina_tool", "dock", poses, inp.model_dump())
                poses["physics_audit"] = audit.to_dict()
            except Exception:
                logger.debug("audit failure can't block result delivery", exc_info=True)

            return ToolResult(data=poses)
        except SandboxError as e:
            return ToolResult(data=None, success=False, error=f"Docking blocked by sandbox: {e}")
        except subprocess.TimeoutExpired:
            return ToolResult(data=None, success=False, error="Docking timed out (3600s).")

    # ── score_only ───────────────────────────────────────────

    def _score_only(self, inp: VinaToolInput, work_dir: Path) -> ToolResult:
        pose = self._resolve_file(inp.pose_pdbqt, work_dir)
        if pose is None:
            return ToolResult(data=None, success=False, error=f"Pose PDBQT not found: {inp.pose_pdbqt}")

        exe = self._resolve_vina()
        if isinstance(exe, ResolutionRequest):
            return ToolResult(
                data=None,
                success=False,
                error="AutoDock Vina executable not found.",
                metadata={"needs_resolution": True, "resolution_request": exe.to_dict()},
            )

        cmd = [exe, "--score_only", "--ligand", str(pose)]
        try:
            result = self.sandbox.run(
                cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=300.0
            )
            success = result.returncode == 0
            # score_only prints to stdout, not a PDBQT file
            affinity = self._extract_score(result.stdout)
            return ToolResult(data={
                "action": "score_only",
                "affinity": affinity,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "message": "Rescoring complete." if success else f"Scoring failed (exit {result.returncode}).",
            }, success=success)
        except SandboxError as e:
            return ToolResult(data=None, success=False, error=f"Scoring blocked by sandbox: {e}")
        except subprocess.TimeoutExpired:
            return ToolResult(data=None, success=False, error="Scoring timed out.")

    # ── prepare_ligand ───────────────────────────────────────

    def _prepare_ligand(self, inp: VinaToolInput, work_dir: Path) -> ToolResult:
        """Convert SMILES or SDF to PDBQT using meeko (preferred) or obabel."""
        out = Path(inp.output_pdbqt) if inp.output_pdbqt else work_dir / "ligand.pdbqt"
        out.parent.mkdir(parents=True, exist_ok=True)

        # Try meeko first — it's the modern Vina ligand preparer
        try:
            return self._prepare_with_meeko(inp, out, work_dir)
        except FileNotFoundError:
            pass

        # Fall back to obabel
        try:
            return self._prepare_with_obabel(inp, out, work_dir)
        except FileNotFoundError:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "Neither meeko nor obabel found. Install one: "
                    "pip install meeko  or  conda install -c conda-forge openbabel"
                ),
            )

    def _prepare_with_meeko(self, inp: VinaToolInput, out: Path, work_dir: Path) -> ToolResult:
        """Prepare ligand using meeko (mk_prepare_ligand)."""
        import shutil as _sh

        if not _sh.which("mk_prepare_ligand"):
            raise FileNotFoundError("mk_prepare_ligand not on PATH")

        # Need an SDF input — if SMILES given, convert via rdkit first
        sdf_path = self._ensure_sdf(inp, work_dir)
        if sdf_path is None:
            return ToolResult(data=None, success=False, error="Could not obtain SDF for ligand preparation")

        cmd = ["mk_prepare_ligand", "-i", str(sdf_path), "-o", str(out)]
        result = self.sandbox.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=120.0)
        if result.returncode != 0:
            return ToolResult(data={"stderr": result.stderr}, success=False, error="meeko ligand preparation failed")
        return ToolResult(data={
            "action": "prepare_ligand",
            "output_pdbqt": str(out),
            "preparer": "meeko",
            "message": "Ligand prepared as PDBQT.",
        })

    def _prepare_with_obabel(self, inp: VinaToolInput, out: Path, work_dir: Path) -> ToolResult:
        """Fall back to obabel for PDBQT conversion."""
        sdf_path = self._ensure_sdf(inp, work_dir)
        if sdf_path is None:
            return ToolResult(data=None, success=False, error="Could not obtain SDF for ligand preparation")

        cmd = ["obabel", str(sdf_path), "-O", str(out), "--gen3d", "-xr"]
        result = self.sandbox.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=120.0)
        if result.returncode != 0:
            return ToolResult(data={"stderr": result.stderr}, success=False, error="obabel ligand preparation failed")
        return ToolResult(data={
            "action": "prepare_ligand",
            "output_pdbqt": str(out),
            "preparer": "obabel",
            "message": "Ligand prepared as PDBQT (obabel fallback).",
        })

    def _ensure_sdf(self, inp: VinaToolInput, work_dir: Path) -> Path | None:
        """Return path to an SDF file, converting from SMILES if needed."""
        if inp.input_sdf:
            p = Path(inp.input_sdf)
            return p if p.exists() else None
        if inp.smiles:
            sdf_out = work_dir / "ligand.sdf"
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                mol = Chem.MolFromSmiles(inp.smiles)
                if mol is None:
                    return None
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
                AllChem.MMFFOptimizeMolecule(mol)
                writer = Chem.SDWriter(str(sdf_out))
                writer.write(mol)
                writer.close()
                return sdf_out
            except ImportError:
                return None
        return None

    # ── helpers ──────────────────────────────────────────────

    def _resolve_file(self, path_str: str | None, work_dir: Path) -> Path | None:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = work_dir / p
        return p if p.exists() else None

    @staticmethod
    def _resolve_vina() -> str | ResolutionRequest:
        """Find vina executable — try Python package, then CLI resolver."""
        # Try the vina Python package first
        try:
            import vina  # noqa: F401
            return "__python__"
        except ImportError:
            pass
        return resolve_executable("autodock_vina")

    @staticmethod
    def _parse_vina_output(stdout: str, out_pdbqt: Path) -> dict[str, Any]:
        """Parse Vina stdout + output PDBQT into structured poses."""
        poses: list[dict[str, Any]] = []
        # Vina prints: "   1   -7.3   0.000   0.000"
        for m in re.finditer(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", stdout, re.MULTILINE):
            poses.append({
                "rank": int(m.group(1)),
                "affinity_kcal_mol": float(m.group(2)),
                "rmsd_lb": float(m.group(3)),
                "rmsd_ub": float(m.group(4)),
            })

        # Extract some Vina diagnostics
        info: dict[str, Any] = {
            "action": "dock",
            "poses": poses,
            "n_poses": len(poses),
            "best_affinity": poses[0]["affinity_kcal_mol"] if poses else None,
            "output_pdbqt": str(out_pdbqt) if out_pdbqt.exists() else None,
            "stdout_tail": stdout[-1500:] if stdout else "",
        }

        # Flag high-affinity drugs (< -9 kcal/mol is typically strong binding)
        if poses and poses[0]["affinity_kcal_mol"] < -9.0:
            info["binding_strength"] = "strong"
        elif poses and poses[0]["affinity_kcal_mol"] < -7.0:
            info["binding_strength"] = "moderate"
        elif poses:
            info["binding_strength"] = "weak"
        else:
            info["binding_strength"] = "no_binding"

        return info

    @staticmethod
    def _extract_score(stdout: str) -> float | None:
        """Pull the affinity from --score_only output."""
        m = re.search(r"Affinity:\s*(-?[\d.]+)", stdout)
        return float(m.group(1)) if m else None
