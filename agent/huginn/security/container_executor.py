"""Container-based command execution sandbox.

Runs commands inside Docker/Podman/Apptainer containers for stronger isolation
and reproducibility. Requires the chosen container runtime to be installed on
the host; no Python docker package is needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from huginn.security.sandbox import (
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
)


class ContainerExecutor:
    """Execute commands inside a container image.

        Mirrors the ``SandboxExecutor.run`` signature so it can be used as a
    drop-in replacement for local/remote execution.
    """

    _VALID_RUNTIMES = {"docker", "podman", "apptainer", "singularity"}

    def __init__(
        self,
        runtime: str,
        image: str,
        sandbox_config: SandboxConfig | None = None,
        workdir_mount: str = "/huginn_work",
    ):
        runtime = runtime.lower().strip()
        if runtime not in self._VALID_RUNTIMES:
            raise ValueError(
                f"Unsupported container runtime '{runtime}'. "
                f"Supported: {sorted(self._VALID_RUNTIMES)}"
            )
        self.runtime = runtime
        self.image = image
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.workdir_mount = workdir_mount
        self._sandbox = SandboxExecutor(self.sandbox_config)

    def run(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        text: bool = True,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SandboxResult:
        """Run ``cmd`` inside the container."""
        valid_cwd = self._sandbox._validate_cwd(cwd, config=self.sandbox_config)
        if valid_cwd is None:
            valid_cwd = Path.cwd()

        cfg = self.sandbox_config
        if timeout is None:
            timeout = cfg.default_timeout
        timeout = min(float(timeout), cfg.max_timeout)

        if cfg.dry_run:
            return SandboxResult(
                success=True,
                returncode=0,
                stdout=f"[dry-run] Would run in {self.runtime} container '{self.image}': "
                + " ".join(cmd),
                stderr="",
                command=cmd,
                dry_run=True,
            )

        runtime_exe = shutil.which(self.runtime)
        if runtime_exe is None:
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Container runtime '{self.runtime}' not found in PATH",
                command=cmd,
                dry_run=False,
            )

        container_cmd = self._build_command(runtime_exe, cmd, valid_cwd, env=env or {})

        try:
            result = subprocess.run(
                container_cmd,
                capture_output=capture_output,
                text=text,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
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

        return SandboxResult(
            success=result.returncode == 0,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            command=cmd,
            dry_run=False,
        )

    def _build_command(
        self,
        runtime_exe: str,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
    ) -> list[str]:
        if self.runtime in ("docker", "podman"):
            command = [runtime_exe, "run", "--rm"]
            command.extend(["-v", f"{cwd}:{self.workdir_mount}"])
            command.extend(["-w", self.workdir_mount])
            for k, v in env.items():
                command.extend(["-e", f"{k}={v}"])
            command.append(self.image)
            command.extend(cmd)
            return command

        # Apptainer / Singularity
        command = [runtime_exe, "exec"]
        command.extend(["--bind", f"{cwd}:{self.workdir_mount}"])
        command.extend(["--pwd", self.workdir_mount])
        for k, v in env.items():
            command.extend(["--env", f"{k}={v}"])
        command.append(self.image)
        command.extend(cmd)
        return command
