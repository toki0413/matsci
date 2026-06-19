"""Execution backend selection for sandboxed commands.

Picks between container-based and process-based execution based on the
runtime configuration, with safe defaults for production.
"""

from __future__ import annotations

import os
import shutil

from huginn.security.container_executor import ContainerExecutor
from huginn.security.sandbox import SandboxConfig, SandboxError, SandboxExecutor


def allow_local_bash() -> bool:
    """Return True if the operator explicitly enabled local bash fallback."""
    return os.environ.get("HUGINN_ALLOW_LOCAL_BASH") == "1"


def container_runtime_config() -> tuple[str, str | None]:
    """Return the configured container runtime and image."""
    runtime = os.environ.get("HUGINN_CONTAINER_RUNTIME", "none").lower().strip()
    image = os.environ.get("HUGINN_CONTAINER_IMAGE") or None
    return runtime, image


def get_executor(
    config: SandboxConfig | None = None,
) -> SandboxExecutor | ContainerExecutor:
    """Return an execution backend appropriate for the current environment.

    Priority:
    1. Container executor if ``HUGINN_CONTAINER_RUNTIME`` is set and the
       runtime binary is available on PATH.
    2. Local ``SandboxExecutor`` if ``HUGINN_ALLOW_LOCAL_BASH=1``.
    3. Otherwise raise ``SandboxError``.
    """
    runtime, image = container_runtime_config()

    if runtime != "none" and image:
        if shutil.which(runtime) is None:
            raise SandboxError(
                f"Container runtime '{runtime}' not found in PATH. "
                "Install it or set HUGINN_ALLOW_LOCAL_BASH=1 to use the local sandbox."
            )
        return ContainerExecutor(
            runtime=runtime,
            image=image,
            sandbox_config=config,
        )

    if allow_local_bash():
        return SandboxExecutor(config)

    raise SandboxError(
        "No execution backend available. "
        "Set HUGINN_CONTAINER_RUNTIME + HUGINN_CONTAINER_IMAGE for container isolation, "
        "or set HUGINN_ALLOW_LOCAL_BASH=1 to accept the local sandbox risk."
    )
