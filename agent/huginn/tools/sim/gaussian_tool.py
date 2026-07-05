"""Gaussian quantum chemistry tool.

Supports single-point energy, geometry optimization, and frequency calculations.
Parses .log output for SCF energy, forces, and convergence status.
Falls back to mock mode when Gaussian is not installed.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.security import SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator


# autofix 返回的 symbolic key -> Gaussian route section keyword
_GAUSSIAN_KEYWORD_MAP = {
    "scf": "SCF",
    "integral": "Int",
    "opt": "Opt",
    "maxcycle": "MaxCycle",
}


class GaussianToolInput(BaseModel):
    action: Literal["sp", "opt", "freq", "parse"] = Field(
        ...,
        description=(
            "sp: single point energy; opt: geometry optimization; "
            "freq: frequency analysis; parse: only parse existing .log"
        ),
    )
    working_dir: str = Field(
        ..., description="Directory containing the .gjf input file"
    )
    input_file: str | None = Field(
        default=None,
        description="Name of .gjf file (auto-detected if omitted)",
    )
    route_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Override route section keywords (e.g. {'scf': 'xqc'})",
    )
    max_auto_retries: int = Field(
        default=2, ge=0, le=5,
        description="On failure, auto-diagnose + patch route and retry up to N times",
    )
    timeout: int = Field(
        default=3600, ge=1,
        description="Max wall-clock seconds for a single Gaussian run",
    )

    @model_validator(mode="after")
    def _check_action_fields(self) -> "GaussianToolInput":
        if not self.working_dir:
            raise ValueError(f"action '{self.action}' requires 'working_dir'")
        return self


class GaussianToolOutput(BaseModel):
    status: Literal["completed", "failed", "mock"] = "mock"
    energy: float | None = None
    converged: bool = False
    forces: list[list[float]] = []
    output_files: list[str] = []
    warnings: list[str] = []


class GaussianTool(HuginnTool):
    """Run Gaussian quantum chemistry calculations."""

    name = "gaussian_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="qc",
        light_alternatives=(
            "materials_database_tool",
            "symbolic_math_tool",
            "numerical_tool",
        ),
    )
    description = (
        "Run Gaussian quantum chemistry calculations (single point, optimization, frequency). "
        "Parses .log output for SCF energy, forces, and convergence status. "
        "Supports auto-fix of common Gaussian errors (SCF convergence, optimization failure)."
    )
    input_schema = GaussianToolInput
    _init_kwargs_map = {"gaussian_executable": "gaussian_executable"}

    def __init__(
        self,
        gaussian_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        super().__init__()
        self.gaussian_executable = gaussian_executable or self._find_gaussian()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_gaussian(self) -> str | None:
        env_path = os.environ.get("GAUSSIAN_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        for name in ["g16", "g09", "g03"]:
            exe = shutil.which(name)
            if exe:
                return exe
        return None

    def estimate_cost(self, args: GaussianToolInput) -> dict[str, float] | None:
        return {"cpu_hours": 1.0, "walltime_hours": args.timeout / 3600.0}

    async def validate_input(
        self, args: GaussianToolInput, context: ToolContext
    ) -> ValidationResult:
        vr = HandleValidator.validate(
            HandleType.FILE_PATH, args.working_dir, context
        )
        if not vr.result:
            return ValidationResult(
                result=False,
                message=f"Working directory not found: {args.working_dir}",
                error_code=404,
            )
        return ValidationResult(result=True)

    async def call(
        self, args: GaussianToolInput, context: ToolContext
    ) -> ToolResult:
        work_dir = Path(args.working_dir)
        if not work_dir.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Working directory not found: {work_dir}",
            )

        gjf_file = self._find_gjf(work_dir, args.input_file)
        if gjf_file is None:
            return ToolResult(
                data=None,
                success=False,
                error="No .gjf input file found in working directory",
            )

        if args.route_overrides:
            self._apply_route_overrides(gjf_file, args.route_overrides)

        if args.action == "parse":
            return self._parse_and_return(work_dir, gjf_file)

        if self.gaussian_executable:
            return await self._run_gaussian(args, work_dir, gjf_file)

        return self._mock_result(args, work_dir)

    # ── execution ──────────────────────────────────────────────────

    async def _run_gaussian(
        self, args: GaussianToolInput, work_dir: Path, gjf_file: Path
    ) -> ToolResult:
        """Run Gaussian with auto-fix retry loop."""
        autoheal_log: list[dict[str, Any]] = []
        log_file = gjf_file.with_suffix(".log")

        for attempt in range(args.max_auto_retries + 1):
            cmd = [self.gaussian_executable, gjf_file.name]
            try:
                sb_result = self.sandbox.run(
                    cmd, cwd=str(work_dir), timeout=args.timeout
                )
            except Exception as e:
                return ToolResult(
                    data=None, success=False, error=f"Gaussian execution failed: {e}"
                )

            rc = self._get_returncode(sb_result)
            stderr = self._get_stderr(sb_result)

            if rc == 0:
                break

            # failed — try autofix if we have retries left
            if attempt < args.max_auto_retries:
                fixed = self._try_autofix(gjf_file, stderr or "")
                if fixed:
                    autoheal_log.append({
                        "attempt": attempt + 1,
                        "error": (stderr or "")[:300],
                        "fixes_applied": fixed["fixes"],
                        "reasoning": fixed["reasoning"],
                    })
                    continue
                break  # nothing to fix or no more retries

        # parse the log file
        parsed = self._parse_log(log_file) if log_file.exists() else {}

        output = GaussianToolOutput(
            status="completed" if rc == 0 else "failed",
            energy=parsed.get("energy"),
            converged=parsed.get("converged", False),
            forces=parsed.get("forces", []),
            output_files=[
                f.name for f in work_dir.iterdir()
                if f.suffix in [".log", ".chk"]
            ],
        )

        data = output.model_dump()
        data["parsed"] = parsed
        if autoheal_log:
            data["autoheal_attempts"] = autoheal_log

        return ToolResult(
            data=data,
            success=rc == 0,
            error=stderr[:500] if rc != 0 else None,
        )

    def _try_autofix(self, gjf_file: Path, stderr: str) -> dict[str, Any] | None:
        """Run AutoFixLoop against the route section. Returns fixes or None."""
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_route_params(gjf_file)
            fixed = AutoFixLoop().apply_fix("gaussian_tool", stderr, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)

            # translate symbolic fixes into route keywords we can actually write
            route_fixes: dict[str, Any] = {}
            for key, val in fixed.items():
                if key in _GAUSSIAN_KEYWORD_MAP:
                    route_fixes[key] = val
            if not route_fixes:
                return None

            self._apply_route_overrides(gjf_file, route_fixes)
            return {"fixes": route_fixes, "reasoning": reasoning}
        except Exception:
            return None

    # ── .gjf route section handling ────────────────────────────────

    def _read_route_params(self, gjf_path: Path) -> dict[str, Any]:
        """Parse the route line (starts with #) into a keyword dict."""
        params: dict[str, Any] = {}
        try:
            for line in gjf_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("#"):
                    # strip leading # and print-level letter (N/T/P)
                    rest = re.sub(r"^#+[nNtTpP]?\s*", "", s)
                    for token in rest.split():
                        if "=" in token:
                            key, val = token.split("=", 1)
                            params[key.lower()] = val.strip("()")
                        elif "/" in token:
                            parts = token.split("/")
                            params["method"] = parts[0]
                            if len(parts) > 1:
                                params["basis"] = parts[1]
                        else:
                            params[token.lower()] = True
                    break
        except Exception:
            pass
        return params

    def _apply_route_overrides(self, gjf_path: Path, overrides: dict) -> None:
        """Append or replace keywords in the route section."""
        try:
            lines = gjf_path.read_text(encoding="utf-8").split("\n")
            new_keywords: list[str] = []
            overridden: set[str] = set()

            for i, line in enumerate(lines):
                s = line.strip()
                if not s.startswith("#"):
                    continue
                for key, val in overrides.items():
                    kw = _GAUSSIAN_KEYWORD_MAP.get(key, key.upper())
                    # replace existing keyword if present
                    pat = re.compile(rf"\b{kw}(?:=\S+)?\b", re.IGNORECASE)
                    if pat.search(s):
                        replacement = self._format_keyword(kw, val)
                        s = pat.sub(replacement, s)
                        overridden.add(key)
                lines[i] = s
                break  # only first route line

            # append new keywords not already in the route
            for key, val in overrides.items():
                if key not in overridden:
                    kw = _GAUSSIAN_KEYWORD_MAP.get(key, key.upper())
                    new_keywords.append(self._format_keyword(kw, val))

            if new_keywords:
                for i, line in enumerate(lines):
                    if line.strip().startswith("#"):
                        lines[i] = line.rstrip() + " " + " ".join(new_keywords)
                        break

            gjf_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"Warning: failed to modify route section: {e}")

    @staticmethod
    def _format_keyword(kw: str, val: Any) -> str:
        if val is True or val is None:
            return kw
        if isinstance(val, str):
            return f"{kw}=({val.upper()})"
        return f"{kw}={val}"

    # ── .log parsing ───────────────────────────────────────────────

    def _parse_log(self, log_path: Path) -> dict[str, Any]:
        """Parse a Gaussian .log file for key results."""
        result: dict[str, Any] = {
            "energy": None,
            "converged": False,
            "forces": [],
            "optimization_completed": False,
            "frequencies": [],
            "normal_termination": False,
        }

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            result["parse_error"] = str(e)
            return result

        # SCF energy — last occurrence wins
        scf_matches = re.findall(
            r"SCF Done:\s+E\([^)]+\)\s+=\s+(-?[\d.]+)", content
        )
        if scf_matches:
            result["energy"] = float(scf_matches[-1])

        # normal termination
        result["normal_termination"] = "Normal termination" in content

        # optimization converged
        result["optimization_completed"] = (
            "Optimization completed" in content
            or "Stationary point found" in content
        )
        result["converged"] = result["optimization_completed"]

        # forces — last force block
        force_blocks = re.findall(
            r"Center\s+Atomic\s+Forces\s+\(Hartrees/Bohr\)\s*\n"
            r"((?:\s*\d+\s+\d+\s+\S+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n?)+)",
            content,
        )
        if force_blocks:
            forces: list[list[float]] = []
            for line in force_blocks[-1].strip().split("\n"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        forces.append([
                            float(parts[-3]), float(parts[-2]), float(parts[-1])
                        ])
                    except ValueError:
                        pass
            result["forces"] = forces

        # frequencies (from freq calculations)
        freq_matches = re.findall(r"Frequencies\s+--\s+(-?[\d.]+)", content)
        if freq_matches:
            result["frequencies"] = [float(f) for f in freq_matches]

        # SCF convergence failure
        if "Convergence failure" in content:
            result["converged"] = False
            result["scf_convergence_failure"] = True

        return result

    # ── helpers ────────────────────────────────────────────────────

    def _find_gjf(self, work_dir: Path, name: str | None) -> Path | None:
        if name:
            p = work_dir / name
            return p if p.exists() else None
        # auto-detect: prefer .gjf, fallback to .com
        for pattern in ["*.gjf", "*.com"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _parse_and_return(self, work_dir: Path, gjf_file: Path) -> ToolResult:
        log_file = gjf_file.with_suffix(".log")
        if not log_file.exists():
            # try a log with the same stem but different lookup
            log_file = work_dir / (gjf_file.stem + ".log")
        if not log_file.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"No .log file found for {gjf_file.name}",
            )
        parsed = self._parse_log(log_file)
        output = GaussianToolOutput(
            status="completed" if parsed.get("normal_termination") else "failed",
            energy=parsed.get("energy"),
            converged=parsed.get("converged", False),
            forces=parsed.get("forces", []),
        )
        data = output.model_dump()
        data["parsed"] = parsed
        return ToolResult(data=data, success=True)

    def _mock_result(
        self, args: GaussianToolInput, work_dir: Path
    ) -> ToolResult:
        """Return synthetic results when Gaussian is not installed."""
        import random

        mock_energies = {"sp": -76.0, "opt": -76.1, "freq": -76.1}
        output = GaussianToolOutput(
            status="mock",
            energy=mock_energies.get(args.action, -50.0) + random.uniform(-0.5, 0.5),
            converged=True,
            warnings=[
                "Gaussian executable not found. Results are MOCK data for demonstration."
            ],
        )
        return ToolResult(data=output.model_dump(), success=True)

    @staticmethod
    def _get_returncode(sb_result: Any) -> int:
        if hasattr(sb_result, "returncode"):
            return sb_result.returncode
        if isinstance(sb_result, dict):
            return sb_result.get("returncode", -1)
        return -1

    @staticmethod
    def _get_stderr(sb_result: Any) -> str:
        if hasattr(sb_result, "stderr"):
            return sb_result.stderr or ""
        if isinstance(sb_result, dict):
            return sb_result.get("stderr", "")
        return ""
