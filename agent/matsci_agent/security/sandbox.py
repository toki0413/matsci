"""Command execution sandbox for MatSci-Agent.

Prevents arbitrary code execution by whitelisting executables, restricting
working directories, and enforcing timeouts/output limits.
"""

from __future__ import annotations

import hashlib
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

    def _validate_command(self, cmd: list[str], config: SandboxConfig | None = None) -> None:
        """Validate that the command complies with sandbox policy."""
        if not cmd:
            raise SandboxError("Empty command")

        # Strictly prohibit shell=True equivalents
        if isinstance(cmd, str):
            raise SandboxError("String commands are forbidden — use list only")

        cfg = config or self.config
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
                **kwargs,
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
