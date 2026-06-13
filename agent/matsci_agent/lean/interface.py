"""Lean 4 interface — compile and verify Lean proofs via Lake.

Usage:
    from matsci_agent.lean.interface import LeanInterface
    lean = LeanInterface("lean/MatSciLean")
    result = lean.build()
    result = lean.verify_theorem("cauchy_stress_symmetry_iff_angular_momentum")
"""

from __future__ import annotations

import os
import re
import shutil
from matsci_agent.security import SandboxExecutor, SandboxConfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LeanResult:
    """Result of a Lean 4 compilation / verification."""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_seconds: float = 0.0


class LeanInterface:
    """Python interface to a Lake-based Lean 4 project.

    Provides:
      - `build()`: run `lake build`
      - `verify_theorem()`: check that a specific theorem compiles
      - `run_lean_code()`: compile a snippet of Lean 4 code in isolation
    """

    def __init__(self, project_path: str | Path, sandbox: SandboxExecutor | None = None):
        self.project_path = Path(project_path).resolve()
        if not (self.project_path / "lakefile.toml").exists():
            raise ValueError(f"No lakefile.toml found in {self.project_path}")
        self._lake_exe = shutil.which("lake")
        if self._lake_exe is None:
            raise RuntimeError("'lake' executable not found. Install Lean 4 via elan.")
        self.sandbox = sandbox or SandboxExecutor()

    # ------------------------------------------------------------------
    # Core commands
    # ------------------------------------------------------------------

    def build(self, target: str | None = None, quiet: bool = False) -> LeanResult:
        """Run `lake build` (or `lake build <target>`)."""
        cmd = [self._lake_exe, "build"]
        if target:
            cmd.append(target)
        if quiet:
            cmd.append("--quiet")

        return self._run(cmd, cwd=self.project_path)

    def clean(self) -> LeanResult:
        """Run `lake clean`."""
        return self._run([self._lake_exe, "clean"], cwd=self.project_path)

    def run_lean_code(
        self,
        code: str,
        imports: List[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify a snippet of Lean 4 code in a temporary file.

        Args:
            code: Lean 4 code to compile (without module header).
            imports: Additional imports beyond Mathlib.
            timeout: Maximum seconds to wait for compilation.
        """
        imports = imports or []
        header_lines = [f"import {imp}" for imp in imports]
        full_source = "\n".join(header_lines) + ("\n\n" if header_lines else "") + code + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
            dir=self.project_path / "MatSciLean",
            encoding="utf-8",
        ) as f:
            f.write(full_source)
            tmp_path = Path(f.name)

        try:
            # Build just this file
            module_name = f"MatSciLean.{tmp_path.stem}"
            result = self.build(target=module_name, quiet=True)
        finally:
            tmp_path.unlink(missing_ok=True)
            # Also clean up .olean if it was produced
            olean = tmp_path.with_suffix(".olean")
            olean.unlink(missing_ok=True)

        return result

    def eval_lean_code(
        self,
        code: str,
        imports: List[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Execute a snippet of Lean 4 code and capture printed output.

        The code is wrapped in a `main : IO Unit` function so that
        `lake env lean --run` can execute it.

        Args:
            code: Lean 4 code body (will be placed inside `main`).
            imports: Modules to import.
            timeout: Maximum seconds to wait.
        """
        imports = imports or []
        header_lines = [f"import {imp}" for imp in imports]
        full_source = (
            "\n".join(header_lines) + "\n\n"
            + code + "\n\n"
            + "def main : IO Unit := pure ()\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
            dir=self.project_path / "MatSciLean",
            encoding="utf-8",
        ) as f:
            f.write(full_source)
            tmp_path = Path(f.name)

        try:
            result = self._run(
                ["lake", "env", "lean", "--run", str(tmp_path)],
                cwd=self.project_path,
                timeout=timeout,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return result

    def verify_theorem(self, theorem_name: str, module: str = "MatSciLean.ContinuumMechanics") -> LeanResult:
        """Verify that a named theorem compiles successfully.

        This is a thin wrapper around `lake build <module>` plus a regex
        check that the theorem declaration exists in the source.
        """
        # 1. Build the module
        result = self.build(target=module, quiet=True)
        if not result.success:
            return result

        # 2. Check that the theorem name actually appears in the source
        src_file = (self.project_path / module.replace(".", "/")).with_suffix(".lean")
        if not src_file.exists():
            return LeanResult(
                success=False,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                elapsed_seconds=result.elapsed_seconds,
            )

        source = src_file.read_text(encoding="utf-8")
        if theorem_name not in source:
            return LeanResult(
                success=False,
                stdout="",
                stderr=f"Theorem '{theorem_name}' not found in {src_file}",
                returncode=-1,
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: List[str], cwd: Path, timeout: int = 300) -> LeanResult:
        import time
        start = time.time()
        try:
            sb = self.sandbox.run(
                cmd,
                cwd=str(cwd),
                timeout=timeout,
            )
            # Adapt SandboxResult to LeanResult fields
            class _Adapter:
                def __init__(self, sb):
                    self.stdout = sb.stdout
                    self.stderr = sb.stderr
                    self.returncode = sb.returncode
            proc = _Adapter(sb)
        except subprocess.TimeoutExpired as e:
            return LeanResult(
                success=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=-1,
                elapsed_seconds=time.time() - start,
            )

        elapsed = time.time() - start
        return LeanResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
        )

    def __repr__(self) -> str:
        return f"LeanInterface(project={self.project_path})"
