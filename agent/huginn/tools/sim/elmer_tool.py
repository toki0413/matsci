"""ElmerFEM solver tool — solve PDEs via Elmer .sif input files.

When ElmerSolver is not installed, the tool validates and exports the .sif
file so the user can run it manually in an Elmer environment.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxError, SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ElmerToolInput(BaseModel):
    action: Literal["solve_sif", "validate_sif", "mesh_to_elmer"] = Field(...)
    working_dir: str | None = Field(default=None)
    # solve_sif / validate_sif params
    sif_content: str | None = Field(
        default=None, description="Elmer .sif file content. Required for solve_sif/validate_sif."
    )
    sif_file: str | None = Field(
        default=None, description="Path to existing .sif file. Alternative to sif_content."
    )
    # mesh_to_elmer params
    mesh_dir: str | None = Field(
        default=None, description="Path to mesh directory to convert to Elmer format.",
    )


def _elmer_available() -> bool:
    return shutil.which("ElmerSolver") is not None


def _elmergrid_available() -> bool:
    return shutil.which("ElmerGrid") is not None


class ElmerTool(HuginnTool):
    """Solve PDEs with ElmerFEM."""

    name = "elmer_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="medium",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="fem",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Solve PDEs using ElmerFEM. "
        "Falls back to file export when ElmerSolver is not installed."
    )
    input_schema = ElmerToolInput

    def __init__(self, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.sandbox = sandbox or SandboxExecutor()

    def is_read_only(self, args: ElmerToolInput) -> bool:
        return args.action == "validate_sif"

    def is_destructive(self, args: ElmerToolInput) -> bool:
        return args.action == "solve_sif"

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = ElmerToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if input_data.action == "solve_sif":
            return self._solve_sif(input_data, work_dir)
        if input_data.action == "validate_sif":
            return self._validate_sif(input_data, work_dir)
        if input_data.action == "mesh_to_elmer":
            return self._mesh_to_elmer(input_data, work_dir)
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {input_data.action}"
        )

    def _resolve_sif(
        self, input_data: ElmerToolInput, work_dir: Path
    ) -> Path | None:
        """Write sif_content to file or resolve existing sif_file path."""
        if input_data.sif_content:
            sif_path = work_dir / "case.sif"
            sif_path.write_text(input_data.sif_content, encoding="utf-8")
            return sif_path
        if input_data.sif_file:
            p = Path(input_data.sif_file)
            if not p.is_absolute():
                p = work_dir / p
            if p.exists():
                return p
        return None

    def _solve_sif(
        self, input_data: ElmerToolInput, work_dir: Path
    ) -> ToolResult:
        sif_path = self._resolve_sif(input_data, work_dir)
        if sif_path is None:
            return ToolResult(
                data=None,
                success=False,
                error="solve_sif requires 'sif_content' or 'sif_file'.",
            )

        if not _elmer_available():
            return ToolResult(
                data={
                    "action": "solve_sif",
                    "status": "sif_exported",
                    "sif_path": str(sif_path),
                    "message": (
                        "ElmerSolver not installed. SIF file saved — run with: "
                        f"ElmerSolver {sif_path.name}"
                    ),
                },
                success=True,
            )

        try:
            result = self.sandbox.run(
                ["ElmerSolver", str(sif_path)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=600.0,
            )
            success = result.returncode == 0
            data = {
                "action": "solve_sif",
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "sif_path": str(sif_path),
                "working_dir": str(work_dir),
                "message": (
                    "Elmer solve completed." if success
                    else f"Elmer solve failed (exit {result.returncode})."
                ),
            }

            # Physics audit — convergence, singular matrix, NaN/Inf
            try:
                from huginn.execution.physics_auditor import PhysicsAuditor

                auditor = PhysicsAuditor()
                audit_report = auditor.audit("elmer_tool", "solve_sif", data, {})
                data["physics_audit"] = audit_report.to_dict()
            except Exception:
                logger.debug("audit failure can't block result delivery", exc_info=True)

            return ToolResult(
                data=data,
                success=success,
                error=None if success else f"Elmer solve failed: {result.stderr[:300]}",
            )
        except SandboxError as e:
            return ToolResult(
                data=None, success=False, error=f"Elmer solve blocked by sandbox: {e}"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="Elmer solve timed out (600s)."
            )

    def _validate_sif(
        self, input_data: ElmerToolInput, work_dir: Path
    ) -> ToolResult:
        sif_path = self._resolve_sif(input_data, work_dir)
        if sif_path is None:
            return ToolResult(
                data=None,
                success=False,
                error="validate_sif requires 'sif_content' or 'sif_file'.",
            )

        content = sif_path.read_text(encoding="utf-8")
        issues: list[str] = []

        # 基本结构检查: Elmer SIF 必须有 Header 和 Simulation 区块
        if "Header" not in content:
            issues.append("Missing 'Header' section")
        if "Simulation" not in content:
            issues.append("Missing 'Simulation' section")
        if "Equation" not in content and "Solver" not in content:
            issues.append("Missing 'Equation' or 'Solver' section")

        # 检查必填关键字
        for keyword in ["Max Output Level", "Steady State Max", "Coordinate System"]:
            if keyword not in content:
                issues.append(f"Missing keyword: {keyword}")

        return ToolResult(
            data={
                "action": "validate_sif",
                "sif_path": str(sif_path),
                "valid": len(issues) == 0,
                "issues": issues,
                "message": "SIF valid." if not issues else f"{len(issues)} issue(s) found.",
            },
            success=True,
        )

    def _mesh_to_elmer(
        self, input_data: ElmerToolInput, work_dir: Path
    ) -> ToolResult:
        if not input_data.mesh_dir:
            return ToolResult(
                data=None, success=False, error="mesh_to_elmer requires 'mesh_dir'."
            )
        mesh_path = Path(input_data.mesh_dir)
        if not mesh_path.is_absolute():
            mesh_path = work_dir / mesh_path
        if not mesh_path.exists():
            return ToolResult(
                data=None, success=False, error=f"Mesh directory not found: {mesh_path}"
            )

        if not _elmergrid_available():
            return ToolResult(
                data={
                    "action": "mesh_to_elmer",
                    "status": "skipped",
                    "message": "ElmerGrid not installed. Cannot convert mesh.",
                },
                success=True,
            )

        try:
            # ElmerGrid 格式: ElmerGrid 14 2 <mesh_dir> [output_dir]
            result = self.sandbox.run(
                ["ElmerGrid", "14", "2", str(mesh_path), "2", str(work_dir / "elmer_mesh")],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            success = result.returncode == 0
            return ToolResult(
                data={
                    "action": "mesh_to_elmer",
                    "returncode": result.returncode,
                    "output_dir": str(work_dir / "elmer_mesh"),
                    "stdout": result.stdout[-1000:] if result.stdout else "",
                    "message": (
                        "Mesh converted to Elmer format." if success
                        else f"ElmerGrid failed (exit {result.returncode})."
                    ),
                },
                success=success,
                error=None if success else f"ElmerGrid failed: {result.stderr[:300]}",
            )
        except SandboxError as e:
            return ToolResult(
                data=None, success=False, error=f"ElmerGrid blocked by sandbox: {e}"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="ElmerGrid timed out (120s)."
            )
