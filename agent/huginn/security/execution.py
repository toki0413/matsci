"""Execution backend selection for sandboxed commands.

Picks between container-based and process-based execution based on the
runtime configuration, with safe defaults for production.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from huginn.security.container_executor import (
    ContainerExecutor,
    ContainerSecurityConfig,
)
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


def _paranoid_security_config() -> ContainerSecurityConfig:
    """T-BCSE-09 paranoid 档的安全配置: digest 固定 + 禁提权 + 丢 caps + 断网."""
    return ContainerSecurityConfig(
        network_none=True,
        require_digest=True,
        no_new_privileges=True,
        drop_all_capabilities=True,
        read_only_root=True,
    )


def build_executor(
    tool_metadata: Any = None,
    sandbox_hint: str | None = None,
    config: SandboxConfig | None = None,
) -> SandboxExecutor | ContainerExecutor:
    """T-BCSE-09: 按工具的 ``sandbox_hint`` 选择执行后端.

    hint 优先级: 显式 ``sandbox_hint`` 参数 > ``tool_metadata.sandbox_hint``
    > 若 ``is_destructive`` 则 ``paranoid`` > 默认 ``any``.

    - ``paranoid``: 强制容器 (require_digest + no_new_privileges + drop caps +
      network_none). 缺容器 / 未配 runtime 时**拒绝** (raise SandboxError),
      绝不静默回退到软沙箱.
    - ``host``: 本地 ``SandboxExecutor`` (Landlock/seccomp 进程级隔离).
    - ``container``: 容器 ``ContainerExecutor``.
    - ``any``: 按环境自动选 (走 ``get_executor``).
    """
    hint = sandbox_hint
    if hint is None and tool_metadata is not None:
        hint = getattr(tool_metadata, "sandbox_hint", "any")
    if hint in (None, "any"):
        # is_destructive 工具默认 paranoid (B2: 破坏性工具强制隔离)
        if (
            tool_metadata is not None
            and getattr(tool_metadata, "is_destructive", False)
        ):
            hint = "paranoid"
        else:
            hint = "any"

    if hint == "host":
        return SandboxExecutor(config)

    if hint == "container":
        runtime, image = container_runtime_config()
        if not runtime or runtime == "none" or not image:
            raise SandboxError(
                "Tool requires container execution but HUGINN_CONTAINER_RUNTIME "
                "and HUGINN_CONTAINER_IMAGE are not configured."
            )
        if shutil.which(runtime) is None:
            raise SandboxError(
                f"Container runtime '{runtime}' not found in PATH."
            )
        return ContainerExecutor(
            runtime=runtime, image=image, sandbox_config=config
        )

    if hint == "paranoid":
        runtime, image = container_runtime_config()
        if not runtime or runtime == "none" or not image:
            raise SandboxError(
                "Tool requires paranoid (container) isolation but no container "
                "runtime is configured. Refusing to run in a weaker sandbox."
            )
        if shutil.which(runtime) is None:
            raise SandboxError(
                f"Paranoid isolation requires container runtime '{runtime}' "
                "but it is not installed. Refusing to run in a weaker sandbox."
            )
        return ContainerExecutor(
            runtime=runtime,
            image=image,
            sandbox_config=config,
            security_config=_paranoid_security_config(),
        )

    # hint == "any" (含未匹配): 环境自动选择.
    return get_executor(config)
