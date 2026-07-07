"""FEniCS FEM solver tool — solve PDEs via the FEniCS/dolfin Python API.

When FEniCS is not installed, the tool falls back to generating a standalone
Python script that the user can run manually in a FEniCS environment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxError, SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class FenicsToolInput(BaseModel):
    action: Literal["solve_pde", "mesh_info", "convergence_check"] = Field(...)
    working_dir: str | None = Field(default=None)
    # solve_pde params
    script: str | None = Field(
        default=None,
        description="UFL/dolfin Python script to execute. Required for solve_pde.",
    )
    # mesh_info params
    mesh_file: str | None = Field(
        default=None, description="Path to mesh file (XML/HDF5) for mesh_info."
    )
    # convergence_check params
    solution_files: list[str] = Field(
        default_factory=list,
        description="List of solution files (different mesh sizes) for convergence check.",
    )


def _fenics_available(sandbox: SandboxExecutor) -> bool:
    """Check if dolfin (FEniCS) is importable."""
    try:
        result = sandbox.run(
            ["python", "-c", "import dolfin; print(dolfin.__version__)"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        return result.returncode == 0
    except SandboxError:
        return False
    except Exception:
        return False


class FenicsTool(HuginnTool):
    """Solve PDEs with FEniCS/dolfin."""

    name = "fenics_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="medium",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="fem",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Solve PDEs using FEniCS/dolfin. "
        "Falls back to script generation when FEniCS is not installed."
    )
    input_schema = FenicsToolInput

    def __init__(self, sandbox: SandboxExecutor | None = None):
        super().__init__()
        self.sandbox = sandbox or SandboxExecutor()

    def is_read_only(self, args: FenicsToolInput) -> bool:
        return args.action == "mesh_info"

    def is_destructive(self, args: FenicsToolInput) -> bool:
        return args.action == "solve_pde"

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = FenicsToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        if input_data.action == "solve_pde":
            return self._solve_pde(input_data, work_dir)
        if input_data.action == "mesh_info":
            return self._mesh_info(input_data, work_dir)
        if input_data.action == "convergence_check":
            return self._convergence_check(input_data, work_dir)
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {input_data.action}"
        )

    def _solve_pde(
        self, input_data: FenicsToolInput, work_dir: Path
    ) -> ToolResult:
        if not input_data.script:
            return ToolResult(
                data=None,
                success=False,
                error="solve_pde requires 'script' (UFL/dolfin Python code).",
            )

        script_path = work_dir / "fenics_solve.py"
        script_path.write_text(input_data.script, encoding="utf-8")

        if not _fenics_available(self.sandbox):
            return ToolResult(
                data={
                    "action": "solve_pde",
                    "status": "script_generated",
                    "script_path": str(script_path),
                    "message": (
                        "FEniCS not installed. Script saved — run it in a "
                        "FEniCS environment: python fenics_solve.py"
                    ),
                },
                success=True,
            )

        try:
            result = self.sandbox.run(
                ["python", str(script_path)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=300.0,
            )
            success = result.returncode == 0
            return ToolResult(
                data={
                    "action": "solve_pde",
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                    "working_dir": str(work_dir),
                    "message": (
                        "FEniCS solve completed." if success
                        else f"FEniCS solve failed (exit {result.returncode})."
                    ),
                },
                success=success,
                error=None if success else f"FEniCS solve failed: {result.stderr[:300]}",
            )
        except SandboxError as e:
            return ToolResult(
                data=None, success=False, error=f"FEniCS solve blocked by sandbox: {e}"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="FEniCS solve timed out (300s)."
            )

    def _mesh_info(
        self, input_data: FenicsToolInput, work_dir: Path
    ) -> ToolResult:
        if not input_data.mesh_file:
            return ToolResult(
                data=None, success=False, error="mesh_info requires 'mesh_file'."
            )
        mesh_path = Path(input_data.mesh_file)
        if not mesh_path.is_absolute():
            mesh_path = work_dir / mesh_path
        if not mesh_path.exists():
            return ToolResult(
                data=None, success=False, error=f"Mesh file not found: {mesh_path}"
            )

        info_script = textwrap.dedent(f"""
            from dolfin import Mesh
            mesh = Mesh("{mesh_path}")
            print("dim", mesh.geometry().dim())
            print("num_vertices", mesh.num_vertices())
            print("num_cells", mesh.num_cells())
        """)

        try:
            result = self.sandbox.run(
                ["python", "-c", info_script],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if result.returncode != 0:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"FEniCS mesh query failed: {result.stderr[:300]}",
                )
            info: dict[str, Any] = {"mesh_file": str(mesh_path)}
            for line in result.stdout.splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, val = parts
                    try:
                        info[key] = int(val)
                    except ValueError:
                        info[key] = val
            return ToolResult(
                data={"action": "mesh_info", **info},
                success=True,
            )
        except SandboxError as e:
            return ToolResult(
                data=None, success=False, error=f"FEniCS mesh query blocked by sandbox: {e}"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="FEniCS mesh query timed out."
            )

    def _convergence_check(
        self, input_data: FenicsToolInput, work_dir: Path
    ) -> ToolResult:
        if len(input_data.solution_files) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="convergence_check requires at least 2 solution_files.",
            )
        # Load each pair of consecutive solutions from XDMF and compute the
        # L2 norm of their difference.  Solutions may live on different meshes
        # (h-refinement), so fall back to relative L2 norm comparison when
        # errornorm can't be used directly.
        sol_files = input_data.solution_files
        diffs: list[float] = []
        method_used = "errornorm"
        for i in range(len(sol_files) - 1):
            script = textwrap.dedent(f"""
                import sys
                from dolfin import (
                    Mesh, Function, FunctionSpace, XDMFFile,
                    errornorm, norm,
                )

                def _load(path):
                    mesh = Mesh()
                    with XDMFFile(path) as f:
                        f.read(mesh)
                    V = FunctionSpace(mesh, "P", 1)
                    u = Function(V)
                    with XDMFFile(path) as f:
                        f.read_checkpoint(V, u, "u")
                    return mesh, V, u

                f1_path = {str(sol_files[i])!r}
                f2_path = {str(sol_files[i + 1])!r}
                try:
                    m1, V1, u1 = _load(f1_path)
                    m2, V2, u2 = _load(f2_path)
                except Exception as e:
                    print("error", str(e))
                    sys.exit(0)

                try:
                    d = errornorm(u1, u2, "L2")
                    print("diff", d)
                except Exception:
                    n1 = norm(u1, "L2")
                    n2 = norm(u2, "L2")
                    denom = max(n1, n2, 1e-15)
                    d = abs(n1 - n2) / denom
                    print("diff_rel", d)
            """)
            try:
                result = self.sandbox.run(
                    ["python", "-c", script],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                )
                parsed = False
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            val = float(parts[1])
                            if parts[0] == "diff_rel":
                                method_used = "relative_l2"
                            diffs.append(val)
                            parsed = True
                        except ValueError:
                            pass
                    elif parts and parts[0] == "error":
                        diffs.append(float("nan"))
                        parsed = True
                if not parsed:
                    diffs.append(float("nan"))
            except subprocess.TimeoutExpired:
                diffs.append(float("nan"))
            except Exception:
                diffs.append(float("nan"))

        valid = [d for d in diffs if d == d]
        converged = len(valid) > 0 and all(d < 0.01 for d in valid)
        return ToolResult(
            data={
                "action": "convergence_check",
                "n_solutions": len(sol_files),
                "differences": diffs,
                "method": method_used,
                "converged": converged,
                "message": (
                    "Convergence check complete — all L2 differences < 1e-2."
                    if converged
                    else "Convergence check complete — some differences exceed tolerance."
                    if valid
                    else "No valid differences computed (check solution file format)."
                ),
            },
            success=True,
        )
