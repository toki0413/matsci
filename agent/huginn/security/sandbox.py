"""Command execution sandbox for Huginn.

Prevents arbitrary code execution by whitelisting executables, restricting
working directories, and enforcing timeouts/output limits.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SandboxError(Exception):
    """Raised when a sandbox policy is violated."""


@dataclass
class SandboxConfig:
    """Sandbox configuration."""

    # Allowed executable names (base names, e.g. "vasp", "lammps", "lake")
    allowed_executables: set[str] = field(
        default_factory=lambda: {
            "vasp",
            "vasp_std",
            "vasp_gam",
            "vasp_ncl",
            "lmp",
            "lammps",
            "lake",
            "lean",
            "python",
            "python3",
            "mpiexec",
            "mpirun",
            "packmol",
            "uv",
            # MD / quantum chemistry
            "gmx",
            "gmx_mpi",
            "gaussian",
            "g16",
            "orca",
            "cp2k",
            "pw.x",
            "cp.x",
            "qe",
            # FEM / CFD
            "ElmerSolver",
            "ElmerGrid",
            "ElmerSolver_mpi",
            "freefem",
            # LaTeX / docs
            "pdflatex",
            "xelatex",
            "lualatex",
            "bibtex",
            "latexmk",
            # Shell builtins / coreutils
            "echo",
            "cat",
            "ls",
            "pwd",
            "printf",
            "true",
            "false",
            "test",
            "head",
            "tail",
            "wc",
            "sort",
            "cut",
            "tr",
            "grep",
            "find",
            "mkdir",
            "cp",
            "mv",
            "rm",
            "touch",
            "diff",
            "which",
            "env",
        }
    )

    # Allowed working directory roots (default: anywhere — override for strict mode)
    allowed_work_dirs: set[Path] = field(default_factory=set)

    # Global limits
    default_timeout: float = 3600.0
    max_timeout: float = 86400.0
    max_output_bytes: int = 50 * 1024 * 1024  # 50 MiB

    # Dry-run mode: log but do not execute
    dry_run: bool = False

    # Strict mode: cwd must be under allowed_work_dirs
    strict_work_dir: bool = False


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: list[str]
    dry_run: bool
    blocked: bool = False
    block_reason: str | None = None


class SandboxExecutor:
    """Execute subprocess commands inside a security sandbox."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def _resolve_executable(self, cmd: list[str]) -> str:
        """Resolve the first element of cmd to an absolute path."""
        if not cmd:
            raise SandboxError("Empty command")
        exe = shutil.which(cmd[0])
        if exe is None:
            raise SandboxError(f"Executable not found: {cmd[0]}")
        return exe

    def _validate_command(
        self, cmd: list[str], config: SandboxConfig | None = None
    ) -> None:
        """Validate that the command complies with sandbox policy."""
        if not cmd:
            raise SandboxError("Empty command")

        # Strictly prohibit shell=True equivalents
        if isinstance(cmd, str):
            raise SandboxError("String commands are forbidden — use list only")

        cfg = config or self.config

        # Policy engine: declarative rules take priority over the
        # legacy hardcoded whitelist.  deny -> block, allow -> skip
        # whitelist, ask -> fall through to whitelist (backward compat).
        from huginn.security.policy_engine import evaluate_command_hook

        decision = evaluate_command_hook(cmd)
        if decision.action == "deny":
            raise SandboxError(
                f"Blocked by security policy '{decision.matched_rule}': "
                f"{decision.reason}"
            )
        if decision.action == "allow":
            # Still verify the executable actually exists on disk
            self._resolve_executable(cmd)
            return

        # "ask" or unmatched -> fall back to legacy whitelist
        exe_path = self._resolve_executable(cmd)
        exe_name = Path(exe_path).name.lower()

        # Remove .exe suffix for Windows normalization
        if exe_name.endswith(".exe"):
            exe_name = exe_name[:-4]

        allowed = {a.lower() for a in cfg.allowed_executables}
        if exe_name not in allowed:
            raise SandboxError(
                f"Executable '{exe_name}' not in sandbox whitelist. "
                f"Allowed: {sorted(cfg.allowed_executables)}"
            )

    def _validate_cwd(
        self, cwd: str | Path | None, config: SandboxConfig | None = None
    ) -> Path | None:
        """Validate working directory restrictions."""
        if cwd is None:
            return None
        path = Path(cwd).resolve()
        cfg = config or self.config

        if cfg.strict_work_dir and cfg.allowed_work_dirs:
            allowed = False
            for root in cfg.allowed_work_dirs:
                try:
                    path.relative_to(root.resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise SandboxError(
                    f"Working directory {path} is outside allowed roots: "
                    f"{[str(r) for r in cfg.allowed_work_dirs]}"
                )
        return path

    # Kwargs meant for remote schedulers; they must not be passed to subprocess.run.
    _REMOTE_KWARGS = {
        "queue",
        "walltime",
        "nodes",
        "ntasks_per_node",
        "modules",
        "job_name",
    }

    def run(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        text: bool = True,
        env: dict[str, str] | None = None,
        config: SandboxConfig | None = None,
        **kwargs: Any,
    ) -> SandboxResult:
        """Run a command inside the sandbox.

        Raises SandboxError if policy is violated.
        """
        cfg = config or self.config
        self._validate_command(cmd, config=cfg)
        valid_cwd = self._validate_cwd(cwd, config=cfg)

        # Clamp timeout
        if timeout is None:
            timeout = cfg.default_timeout
        timeout = min(float(timeout), cfg.max_timeout)

        if cfg.dry_run:
            return SandboxResult(
                success=True,
                returncode=0,
                stdout="[dry-run] Command would execute: " + " ".join(cmd),
                stderr="",
                command=cmd,
                dry_run=True,
            )

        # Drop scheduler-only hints so they do not reach subprocess.run.
        run_kwargs = {k: v for k, v in kwargs.items() if k not in self._REMOTE_KWARGS}

        try:
            result = subprocess.run(
                cmd,
                cwd=str(valid_cwd) if valid_cwd else None,
                capture_output=capture_output,
                text=text,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                command=cmd,
                dry_run=False,
            )

        # Truncate oversized output
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        max_bytes = self.config.max_output_bytes
        if isinstance(stdout, bytes):
            if len(stdout) > max_bytes:
                stdout = stdout[:max_bytes] + b"\n... [truncated]"
        else:
            if len(stdout.encode("utf-8", errors="replace")) > max_bytes:
                stdout = stdout[: max_bytes // 4] + "\n... [truncated]"

        if isinstance(stderr, bytes):
            if len(stderr) > max_bytes:
                stderr = stderr[:max_bytes] + b"\n... [truncated]"
        else:
            if len(stderr.encode("utf-8", errors="replace")) > max_bytes:
                stderr = stderr[: max_bytes // 4] + "\n... [truncated]"

        return SandboxResult(
            success=result.returncode == 0,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
            command=cmd,
            dry_run=False,
        )

    @staticmethod
    def hash_data(data: str | bytes) -> str:
        """Return SHA-256 hex digest of data."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]


def create_sandbox(
    config: SandboxConfig | None = None,
    prefer_docker: bool = False,
    docker_image: str = "python:3.12-slim",
) -> SandboxExecutor | "DockerSandboxExecutor":  # type: ignore[name-defined]
    """根据环境自动选择沙箱后端。

    - prefer_docker=True 且 Docker 可用 → DockerSandboxExecutor
    - 否则 → SandboxExecutor（subprocess 软沙箱）

    也可以通过环境变量 HUGINN_DOCKER_SANDBOX=1 启用 Docker 沙箱，
    效果等同 prefer_docker=True。
    """
    cfg = config or SandboxConfig()

    # 环境变量开关，方便运维侧不改代码就切后端
    env_enabled = os.environ.get("HUGINN_DOCKER_SANDBOX") == "1"
    want_docker = prefer_docker or env_enabled

    if want_docker:
        # 延迟 import，避免 docker SDK 没装时整个 sandbox 模块都加载不了
        try:
            from huginn.security.docker_sandbox import DockerSandboxExecutor
        except Exception:
            return SandboxExecutor(cfg)

        try:
            docker_executor = DockerSandboxExecutor(image=docker_image, config=cfg)
        except Exception:
            # 构造失败也别让上层挂掉
            return SandboxExecutor(cfg)

        if docker_executor.is_available():
            return docker_executor
        # Docker 不可用就静默回退
        return SandboxExecutor(cfg)

    return SandboxExecutor(cfg)
