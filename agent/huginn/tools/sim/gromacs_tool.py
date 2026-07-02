"""GROMACS MD tool — run molecular dynamics via the gmx CLI.

When gmx is not installed, the tool validates input and returns a friendly
message. Supports md_run, energy_minimize, and trajectory analysis.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class GromacsToolInput(BaseModel):
    action: Literal["md_run", "energy_minimize", "analyze_traj"] = Field(...)
    working_dir: str | None = Field(default=None)
    # md_run / energy_minimize params
    tpr_file: str | None = Field(
        default=None, description="Path to .tpr run control file. Required for md_run/energy_minimize."
    )
    nsteps: int = Field(default=5000, ge=1, description="Number of simulation steps.")
    # analyze_traj params
    trajectory_file: str | None = Field(
        default=None, description="Path to .xtc/.trr trajectory file."
    )
    analysis_type: Literal["rms", "rdf", "gyrate", "rmsd"] = Field(
        default="rms", description="Type of trajectory analysis."
    )
    reference_file: str | None = Field(
        default=None, description="Reference structure for RMS/RMSD (.gro/.pdb)."
    )


def _gmx_available() -> bool:
    return shutil.which("gmx") is not None


class GromacsTool(HuginnTool):
    """Run GROMACS molecular dynamics simulations."""

    name = "gromacs_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="md",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Run GROMACS MD: energy minimization, MD, and trajectory analysis. "
        "Falls back to input validation when gmx is not installed."
    )
    input_schema = GromacsToolInput

    def __init__(self, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.sandbox = sandbox or SandboxExecutor()

    def is_read_only(self, args: GromacsToolInput) -> bool:
        return args.action == "analyze_traj"

    def is_destructive(self, args: GromacsToolInput) -> bool:
        return args.action in ("md_run", "energy_minimize")

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GromacsToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if input_data.action == "md_run":
            return self._md_run(input_data, work_dir)
        if input_data.action == "energy_minimize":
            return self._energy_minimize(input_data, work_dir)
        if input_data.action == "analyze_traj":
            return self._analyze_traj(input_data, work_dir)
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {input_data.action}"
        )

    def _resolve_file(self, path_str: str | None, work_dir: Path) -> Path | None:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = work_dir / p
        return p if p.exists() else None

    def _md_run(
        self, input_data: GromacsToolInput, work_dir: Path
    ) -> ToolResult:
        tpr = self._resolve_file(input_data.tpr_file, work_dir)
        if tpr is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"TPR file not found: {input_data.tpr_file}",
            )

        if not _gmx_available():
            return ToolResult(
                data={
                    "action": "md_run",
                    "status": "skipped",
                    "tpr_file": str(tpr),
                    "message": "gmx not installed. Run with: gmx mdrun -deffnm " + tpr.stem,
                },
                success=True,
            )

        try:
            result = subprocess.run(
                [
                    "gmx", "mdrun",
                    "-deffnm", str(tpr.with_suffix("")),
                    "-nsteps", str(input_data.nsteps),
                ],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=3600.0,
            )
            success = result.returncode == 0
            return ToolResult(
                data={
                    "action": "md_run",
                    "returncode": result.returncode,
                    "nsteps": input_data.nsteps,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                    "message": (
                        "MD run completed." if success
                        else f"MD run failed (exit {result.returncode})."
                    ),
                },
                success=success,
                error=None if success else f"gmx mdrun failed: {result.stderr[:300]}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="MD run timed out (3600s)."
            )

    def _energy_minimize(
        self, input_data: GromacsToolInput, work_dir: Path
    ) -> ToolResult:
        tpr = self._resolve_file(input_data.tpr_file, work_dir)
        if tpr is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"TPR file not found: {input_data.tpr_file}",
            )

        if not _gmx_available():
            return ToolResult(
                data={
                    "action": "energy_minimize",
                    "status": "skipped",
                    "tpr_file": str(tpr),
                    "message": "gmx not installed.",
                },
                success=True,
            )

        try:
            result = subprocess.run(
                [
                    "gmx", "mdrun",
                    "-deffnm", str(tpr.with_suffix("")),
                    "-nsteps", str(input_data.nsteps),
                ],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=3600.0,
            )
            success = result.returncode == 0
            return ToolResult(
                data={
                    "action": "energy_minimize",
                    "returncode": result.returncode,
                    "nsteps": input_data.nsteps,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "message": (
                        "Energy minimization completed." if success
                        else f"EM failed (exit {result.returncode})."
                    ),
                },
                success=success,
                error=None if success else f"gmx mdrun (EM) failed: {result.stderr[:300]}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="Energy minimization timed out."
            )

    def _analyze_traj(
        self, input_data: GromacsToolInput, work_dir: Path
    ) -> ToolResult:
        traj = self._resolve_file(input_data.trajectory_file, work_dir)
        if traj is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Trajectory file not found: {input_data.trajectory_file}",
            )

        if not _gmx_available():
            return ToolResult(
                data={
                    "action": "analyze_traj",
                    "status": "skipped",
                    "message": "gmx not installed.",
                },
                success=True,
            )

        # gmx rms/rdf/gyrate 都需要交互式选组, 用 echo 管道自动选 Backbone
        cmd_map = {
            "rms": ["gmx", "rms", "-s", str(traj), "-f", str(traj), "-o", "rms.xvg"],
            "rmsd": ["gmx", "rms", "-s", str(traj), "-f", str(traj), "-o", "rmsd.xvg"],
            "rdf": ["gmx", "rdf", "-f", str(traj), "-o", "rdf.xvg"],
            "gyrate": ["gmx", "gyrate", "-f", str(traj), "-o", "gyrate.xvg"],
        }
        cmd = cmd_map.get(input_data.analysis_type)
        if cmd is None:
            return ToolResult(
                data=None, success=False,
                error=f"Unknown analysis type: {input_data.analysis_type}",
            )

        try:
            # 自动选组 0 (System/Backbone)
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=300.0,
                input="0\n0\n",
            )
            success = result.returncode == 0
            output_file = work_dir / f"{input_data.analysis_type}.xvg"
            return ToolResult(
                data={
                    "action": "analyze_traj",
                    "analysis_type": input_data.analysis_type,
                    "returncode": result.returncode,
                    "output_file": str(output_file) if output_file.exists() else None,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "message": (
                        f"{input_data.analysis_type} analysis completed." if success
                        else f"Analysis failed (exit {result.returncode})."
                    ),
                },
                success=success,
                error=None if success else f"gmx {input_data.analysis_type} failed: {result.stderr[:300]}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="Trajectory analysis timed out."
            )
