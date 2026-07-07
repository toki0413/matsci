"""ORCA quantum chemistry tool.

Supports single-point energy, geometry optimization, and frequency calculations.
Parses .out output for energy, optimization steps, and convergence status.
Falls back to mock mode when ORCA is not installed.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


# autofix symbolic key -> ORCA input keyword
_ORCA_KEYWORD_MAP = {
    "scf_conv": "SCF",
    "maxiter": "MAXITER",
    "opt_maxiter": "MAXITER",
    "grid": "Grid",
    "maxcore": "MAXCORE",
}


class OrcaToolInput(BaseModel):
    action: Literal["sp", "opt", "freq", "parse"] = Field(
        ...,
        description=(
            "sp: single point energy; opt: geometry optimization; "
            "freq: frequency analysis; parse: only parse existing .out"
        ),
    )
    working_dir: str = Field(
        ..., description="Directory containing the .inp input file"
    )
    input_file: str | None = Field(
        default=None,
        description="Name of .inp file (auto-detected if omitted)",
    )
    input_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Override ORCA input keywords (e.g. {'scf_conv': 'tight'})",
    )
    max_auto_retries: int = Field(
        default=2, ge=0, le=5,
        description="On failure, auto-diagnose + patch input and retry up to N times",
    )
    timeout: int = Field(
        default=3600, ge=1,
        description="Max wall-clock seconds for a single ORCA run",
    )

    @model_validator(mode="after")
    def _check_action_fields(self) -> "OrcaToolInput":
        if not self.working_dir:
            raise ValueError(f"action '{self.action}' requires 'working_dir'")
        return self


class OrcaToolOutput(BaseModel):
    status: Literal["completed", "failed", "mock"] = "mock"
    energy: float | None = None
    converged: bool = False
    optimization_steps: int = 0
    output_files: list[str] = []
    warnings: list[str] = []


class OrcaTool(HuginnTool):
    """Run ORCA quantum chemistry calculations."""

    name = "orca_tool"
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
        "Run ORCA quantum chemistry calculations (single point, optimization, frequency). "
        "Parses .out output for energy, optimization steps, and convergence status. "
        "Supports auto-fix of common ORCA errors."
    )
    input_schema = OrcaToolInput
    _init_kwargs_map = {"orca_executable": "orca_executable"}

    def __init__(
        self,
        orca_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        super().__init__()
        self.orca_executable = orca_executable or self._find_orca()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_orca(self) -> str | None:
        env_path = os.environ.get("ORCA_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        exe = shutil.which("orca")
        return exe

    def estimate_cost(self, args: OrcaToolInput) -> dict[str, float] | None:
        return {"cpu_hours": 1.0, "walltime_hours": args.timeout / 3600.0}

    async def validate_input(
        self, args: OrcaToolInput, context: ToolContext
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
        self, args: OrcaToolInput, context: ToolContext
    ) -> ToolResult:
        work_dir = Path(args.working_dir)
        if not work_dir.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Working directory not found: {work_dir}",
            )

        inp_file = self._find_inp(work_dir, args.input_file)
        if inp_file is None:
            return ToolResult(
                data=None,
                success=False,
                error="No .inp input file found in working directory",
            )

        if args.input_overrides:
            self._apply_input_overrides(inp_file, args.input_overrides)

        if args.action == "parse":
            return self._parse_and_return(work_dir, inp_file)

        if self.orca_executable:
            return await self._run_orca(args, work_dir, inp_file)

        return self._mock_result(args, work_dir)

    # ── execution ──────────────────────────────────────────────────

    async def _run_orca(
        self, args: OrcaToolInput, work_dir: Path, inp_file: Path
    ) -> ToolResult:
        """Run ORCA with auto-fix retry loop."""
        autoheal_log: list[dict[str, Any]] = []
        out_file = inp_file.with_suffix(".out")
        rc = -1
        stderr = ""
        # 软失败原因. returncode=0 但 SCF 没收敛 / 优化没收敛都属于软失败,
        # ORCA 也会静默返回 0, 不能当成功返回.
        soft_failure_msg: str | None = None
        audit_report = None

        for attempt in range(args.max_auto_retries + 1):
            cmd = [self.orca_executable, inp_file.name]
            try:
                sb_result = self.sandbox.run(
                    cmd, cwd=str(work_dir), timeout=args.timeout
                )
            except Exception as e:
                return ToolResult(
                    data=None, success=False, error=f"ORCA execution failed: {e}"
                )

            rc = self._get_returncode(sb_result)
            stderr = self._get_stderr(sb_result)

            # 判断这次跑完到底算成功还是失败:
            # - rc != 0 → 硬失败, stderr 诊断
            # - rc == 0 但优化没收敛 / 没拿到能量 → 软失败, ORCA 也会静默返回 0
            error: str | None = None
            soft_failure_msg = None
            if rc != 0:
                error = stderr or ""
            else:
                # returncode=0 不代表收敛了, 查 .out 的优化 / SCF 状态
                parsed_now = self._parse_out(out_file) if out_file.exists() else {}
                if args.action == "opt" and not parsed_now.get("converged", False):
                    # 优化没收敛: 没出现 OPTIMIZATION HAS CONVERGED
                    error = "Optimization did not converge — OPTIMIZATION HAS NOT CONVERGED"
                    soft_failure_msg = error
                elif parsed_now.get("energy") is None:
                    # 没拿到 FINAL SINGLE POINT ENERGY, SCF 大概率没收敛
                    error = "SCF not converged — no FINAL SINGLE POINT ENERGY"
                    soft_failure_msg = error
                else:
                    # SCF 收敛且能量拿到了, 过物理审计抓 "成功但不可信"
                    try:
                        from huginn.execution.physics_auditor import (
                            PhysicsAuditor,
                        )

                        auditor = PhysicsAuditor()
                        audit_report = auditor.audit(
                            "orca_tool",
                            args.action,
                            parsed_now,
                            args.model_dump(),
                        )
                        if audit_report.has_errors:
                            errs = [
                                f.message
                                for f in audit_report.findings
                                if f.severity == "error"
                            ]
                            error = f"Physics audit found errors: {errs}"
                            soft_failure_msg = error
                    except Exception:
                        logger.debug("审计本身挂了不能阻塞结果", exc_info=True)

            if error is None:
                break  # 真正成功

            if attempt < args.max_auto_retries:
                # 硬失败时把 .out 尾部也带上, 帮 autofix 匹配错误模式
                err_text = error or ""
                if out_file.exists():
                    err_text += "\n" + out_file.read_text(
                        encoding="utf-8", errors="ignore"
                    )[-2000:]
                fixed = self._try_autofix(inp_file, err_text)
                if fixed:
                    autoheal_log.append({
                        "attempt": attempt + 1,
                        "error": err_text[:300],
                        "fixes_applied": fixed["fixes"],
                        "reasoning": fixed["reasoning"],
                    })
                    continue
                break
            break  # 重试耗尽

        parsed = self._parse_out(out_file) if out_file.exists() else {}

        # 最终成功判定: returncode==0 且没有遗留软失败 (SCF 没收敛 / 优化没收敛)
        ok = rc == 0 and soft_failure_msg is None
        output = OrcaToolOutput(
            status="completed" if ok else "failed",
            energy=parsed.get("energy"),
            converged=parsed.get("converged", False),
            optimization_steps=parsed.get("optimization_steps", 0),
            output_files=[
                f.name for f in work_dir.iterdir()
                if f.suffix in [".out", ".gbw"]
            ],
        )

        data = output.model_dump()
        data["parsed"] = parsed
        if autoheal_log:
            data["autoheal_attempts"] = autoheal_log

        # Physics audit — 循环里跑过审计就复用, 没跑过就补跑一次兜底
        if audit_report is not None:
            data["physics_audit"] = audit_report.to_dict()
        else:
            try:
                from huginn.execution.physics_auditor import PhysicsAuditor

                auditor = PhysicsAuditor()
                audit_report = auditor.audit(
                    "orca_tool", args.action, parsed, args.model_dump()
                )
                data["physics_audit"] = audit_report.to_dict()
            except Exception:
                logger.debug("audit failure can't block result delivery", exc_info=True)

        return ToolResult(
            data=data,
            success=ok,
            error=None if ok else (stderr[:500] if rc != 0 else soft_failure_msg),
        )

    def _try_autofix(self, inp_file: Path, error: str) -> dict[str, Any] | None:
        """Run AutoFixLoop, translate fixes into ORCA input keywords."""
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_input_params(inp_file)
            fixed = AutoFixLoop().apply_fix("orca_tool", error, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)

            # keep only keys we know how to write into the input
            applicable = {
                k: v for k, v in fixed.items()
                if k in _ORCA_KEYWORD_MAP and _ORCA_KEYWORD_MAP[k]
            }
            if not applicable:
                return None

            self._apply_input_overrides(inp_file, applicable)
            return {"fixes": applicable, "reasoning": reasoning}
        except Exception:
            return None

    # ── .inp handling ──────────────────────────────────────────────

    def _read_input_params(self, inp_path: Path) -> dict[str, Any]:
        """Parse the ! simple-input line into a keyword dict."""
        params: dict[str, Any] = {}
        try:
            for line in inp_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("!"):
                    for token in s.lstrip("!").split():
                        params[token.lower()] = True
        except Exception:
            logger.debug("read text failed", exc_info=True)
        return params

    def _apply_input_overrides(self, inp_path: Path, overrides: dict) -> None:
        """Append keywords to the ! simple-input line."""
        try:
            lines = inp_path.read_text(encoding="utf-8").split("\n")
            overridden: set[str] = set()

            for i, line in enumerate(lines):
                s = line.strip()
                if not s.startswith("!"):
                    continue
                existing = s.lstrip("!").split()
                new_tokens: list[str] = []
                for key, val in overrides.items():
                    token = self._format_orca_token(key, val)
                    if token is None:
                        continue
                    # replace if the keyword family is already present
                    kw = _ORCA_KEYWORD_MAP.get(key, key.upper())
                    replaced = False
                    for j, ex in enumerate(existing):
                        if kw.upper() in ex.upper():
                            existing[j] = token
                            overridden.add(key)
                            replaced = True
                            break
                    if not replaced:
                        new_tokens.append(token)
                lines[i] = "! " + " ".join(existing + new_tokens)
                break

            # any overrides that didn't match get appended to the ! line
            remaining = []
            for key, val in overrides.items():
                if key not in overridden:
                    token = self._format_orca_token(key, val)
                    if token:
                        remaining.append(token)
            if remaining:
                for i, line in enumerate(lines):
                    if line.strip().startswith("!"):
                        lines[i] = line.rstrip() + " " + " ".join(remaining)
                        break

            inp_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"Warning: failed to modify ORCA input: {e}")

    @staticmethod
    def _format_orca_token(key: str, val: Any) -> str | None:
        """Translate an autofix key+val into a valid ORCA simple-input token."""
        if key == "scf_conv":
            # 'tight' -> TightSCF, 'verytight' -> VeryTightSCF
            if isinstance(val, str):
                return f"{val.capitalize()}SCF"
            return "TightSCF"
        if key == "grid":
            # 7 -> Grid7
            return f"Grid{val}"
        if key in ("opt_maxiter", "maxiter"):
            if isinstance(val, int):
                # ORCA reads MAXITER from %scf/%geom blocks, not the ! line,
                # but we add it here as a hint for the block parser
                return f"MAXITER {val}"
            return None
        if key == "maxcore":
            if val == "double" or isinstance(val, str):
                return f"MAXCORE {val}"
            return f"MAXCORE {val}"
        kw = _ORCA_KEYWORD_MAP.get(key, key.upper())
        if val is True or val is None:
            return kw
        return f"{kw} {val}"

    # ── .out parsing ──────────────────────────────────────────────

    def _parse_out(self, out_path: Path) -> dict[str, Any]:
        """Parse an ORCA .out file for key results."""
        result: dict[str, Any] = {
            "energy": None,
            "converged": False,
            "optimization_steps": 0,
            "frequencies": [],
        }

        try:
            content = out_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            result["parse_error"] = str(e)
            return result

        # Final single point energy — last occurrence
        sp_matches = re.findall(
            r"FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)", content
        )
        if sp_matches:
            result["energy"] = float(sp_matches[-1])

        # also try "Total Energy" for older ORCA versions
        if result["energy"] is None:
            te_matches = re.findall(
                r"Total Energy\s*:\s*(-?[\d.]+)", content
            )
            if te_matches:
                result["energy"] = float(te_matches[-1])

        # optimization converged
        result["converged"] = "OPTIMIZATION HAS CONVERGED" in content.upper()

        # count optimization cycles
        result["optimization_steps"] = content.count(
            "Geometry Optimization Cycle"
        ) + content.count("GEOMETRY OPTIMIZATION CYCLE")

        # frequencies
        freq_matches = re.findall(r"VIBRATIONAL FREQUENCIES.*?(?:\n.*?)*?(-?[\d.]+)\s*cm",
                                  content, re.IGNORECASE)
        # simpler: grab lines like "  0:  0.00 cm"
        freq_lines = re.findall(r"^\s*\d+:\s+(-?[\d.]+)\s*cm", content, re.MULTILINE)
        if freq_lines:
            result["frequencies"] = [float(f) for f in freq_lines]

        # Charge and multiplicity
        charge_match = re.search(r"Charge\s*[:=]\s*(-?\d+)", content, re.IGNORECASE)
        if charge_match:
            result["charge"] = int(charge_match.group(1))
        mult_match = re.search(
            r"Multiplicity\s*[:=]\s*(\d+)", content, re.IGNORECASE
        )
        if mult_match:
            result["multiplicity"] = int(mult_match.group(1))

        return result

    # ── helpers ────────────────────────────────────────────────────

    def _find_inp(self, work_dir: Path, name: str | None) -> Path | None:
        if name:
            p = work_dir / name
            return p if p.exists() else None
        for pattern in ["*.inp"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _parse_and_return(self, work_dir: Path, inp_file: Path) -> ToolResult:
        out_file = inp_file.with_suffix(".out")
        if not out_file.exists():
            out_file = work_dir / (inp_file.stem + ".out")
        if not out_file.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"No .out file found for {inp_file.name}",
            )
        parsed = self._parse_out(out_file)
        output = OrcaToolOutput(
            status="completed" if parsed.get("energy") is not None else "failed",
            energy=parsed.get("energy"),
            converged=parsed.get("converged", False),
            optimization_steps=parsed.get("optimization_steps", 0),
        )
        data = output.model_dump()
        data["parsed"] = parsed
        return ToolResult(data=data, success=True)

    def _mock_result(
        self, args: OrcaToolInput, work_dir: Path
    ) -> ToolResult:
        """Return synthetic results when ORCA is not installed."""
        import random

        mock_energies = {"sp": -76.0, "opt": -76.1, "freq": -76.1}
        output = OrcaToolOutput(
            status="mock",
            energy=mock_energies.get(args.action, -50.0) + random.uniform(-0.5, 0.5),
            converged=True,
            optimization_steps=3 if args.action == "opt" else 0,
            warnings=[
                "ORCA executable not found. Results are MOCK data for demonstration."
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
