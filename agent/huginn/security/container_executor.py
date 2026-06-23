"""Container-based command execution sandbox.

Runs commands inside Docker/Podman/Apptainer containers for stronger isolation
and reproducibility. Requires the chosen container runtime to be installed on
the host; no Python docker package is needed.

Security hardening (Phase 4):
- Network isolation (``--network=none``)
- CPU and memory limits
- Image digest pinning (reject mutable tags)
- Read-only root filesystem
- Non-root user execution
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.security.sandbox import (
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
)


# ---------------------------------------------------------------------------
# Container security configuration
# ---------------------------------------------------------------------------

@dataclass
class ContainerSecurityConfig:
    """Security hardening options for container execution."""

    # Network isolation — drop all network access
    network_none: bool = True

    # Resource limits
    memory_limit: str | None = None       # e.g. "512m", "2g"
    cpu_limit: float | None = None        # e.g. 2.0 = 2 CPUs
    pids_limit: int | None = None         # max number of processes

    # Image pinning — require sha256 digest instead of mutable tags
    require_digest: bool = False
    allowed_images: set[str] = field(default_factory=set)  # empty = allow all

    # Filesystem and user
    read_only_root: bool = True
    run_as_user: str | None = "1000"      # UID to run as (None = image default)
    no_new_privileges: bool = True

    # Extra volume mounts (host:container) — use sparingly
    extra_mounts: list[str] = field(default_factory=list)

    # Drop all Linux capabilities
    drop_all_capabilities: bool = True

    # Seccomp profile path (None = Docker default)
    seccomp_profile: str | None = None


_DIGEST_RE = re.compile(r"^[a-z0-9._/-]+@sha256:[a-f0-9]{64}$")


def _is_digest_pinned(image: str) -> bool:
    """Return True if the image reference uses a sha256 digest."""
    return bool(_DIGEST_RE.match(image))


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
        security_config: ContainerSecurityConfig | None = None,
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
        self.security_config = security_config or ContainerSecurityConfig()
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

        # --- image policy checks -------------------------------------------
        sec = self.security_config
        if sec.require_digest and not _is_digest_pinned(self.image):
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=(
                    f"Image '{self.image}' is not pinned to a sha256 digest. "
                    "Use image@sha256:<hex> format or disable require_digest."
                ),
                command=cmd,
                dry_run=False,
                blocked=True,
                block_reason="image_not_pinned",
            )

        if sec.allowed_images and self.image not in sec.allowed_images:
            # Also check without digest for tag-based allowlists
            base_image = self.image.split("@")[0].split(":")[0]
            if base_image not in sec.allowed_images:
                return SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"Image '{self.image}' not in allowed list: {sorted(sec.allowed_images)}",
                    command=cmd,
                    dry_run=False,
                    blocked=True,
                    block_reason="image_not_allowed",
                )

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
        sec = self.security_config

        if self.runtime in ("docker", "podman"):
            command = [runtime_exe, "run", "--rm"]

            # Network isolation
            if sec.network_none:
                command.extend(["--network", "none"])

            # Resource limits
            if sec.memory_limit:
                command.extend(["--memory", sec.memory_limit])
            if sec.cpu_limit is not None:
                command.extend(["--cpus", str(sec.cpu_limit)])
            if sec.pids_limit is not None:
                command.extend(["--pids-limit", str(sec.pids_limit)])

            # Read-only root filesystem
            if sec.read_only_root:
                command.append("--read-only")

            # Run as non-root user
            if sec.run_as_user:
                command.extend(["--user", sec.run_as_user])

            # No new privileges
            if sec.no_new_privileges:
                command.append("--security-opt=no-new-privileges")

            # Drop all capabilities
            if sec.drop_all_capabilities:
                command.append("--cap-drop=ALL")

            # Seccomp profile
            if sec.seccomp_profile:
                command.append(f"--security-opt=seccomp={sec.seccomp_profile}")

            # Volume mounts
            command.extend(["-v", f"{cwd}:{self.workdir_mount}"])
            for mount in sec.extra_mounts:
                command.extend(["-v", mount])

            command.extend(["-w", self.workdir_mount])

            # Environment variables
            for k, v in env.items():
                command.extend(["-e", f"{k}={v}"])

            command.append(self.image)
            command.extend(cmd)
            return command

        # Apptainer / Singularity
        command = [runtime_exe, "exec"]

        # Network isolation (Apptainer uses --net --network=none or fakeroot)
        if sec.network_none:
            command.extend(["--net", "--network", "none"])

        # Bind mounts
        command.extend(["--bind", f"{cwd}:{self.workdir_mount}"])
        for mount in sec.extra_mounts:
            command.extend(["--bind", mount])

        command.extend(["--pwd", self.workdir_mount])

        # Environment variables
        for k, v in env.items():
            command.extend(["--env", f"{k}={v}"])

        command.append(self.image)
        command.extend(cmd)
        return command
