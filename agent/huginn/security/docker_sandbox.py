"""Docker 容器沙箱。

走 docker Python SDK 在隔离容器里执行命令，比 subprocess 软沙箱强一档：
网络默认 none、内存/CPU 限死、超时直接 kill 容器。
SDK 没装或者 daemon 没起来都不抛，静默退回 SandboxExecutor。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from huginn.security.sandbox import (
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
)

import logging

logger = logging.getLogger(__name__)

# docker SDK 是可选依赖，import 失败就走 subprocess 降级
try:  # pragma: no cover - 看环境
    import docker  # type: ignore[import-untyped]
    from docker.errors import APIError  # type: ignore[import-untyped]
    _DOCKER_IMPORT_OK = True
    _DOCKER_IMPORT_ERR: str | None = None
except Exception as e:  # pragma: no cover - 看环境
    docker = None  # type: ignore[assignment]

    class APIError(Exception):  # type: ignore[no-redef]
        pass

    _DOCKER_IMPORT_OK = False
    _DOCKER_IMPORT_ERR = f"{type(e).__name__}: {e}"


class DockerSandboxExecutor:
    """Docker 容器沙箱 — 在隔离容器里执行命令。

    降级策略：
    1. docker SDK 可用 + Docker daemon 运行中 → 容器内执行
    2. docker SDK 不可用 → 回退到 SandboxExecutor（subprocess 软沙箱）
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        work_dir: str = "/workspace",
        mount_root: str | None = None,
        network: str = "none",
        memory_limit: str = "2g",
        cpu_limit: float = 2.0,
        timeout: float = 3600.0,
        read_only: bool = False,
        config: SandboxConfig | None = None,
    ) -> None:
        self.image = image
        self.work_dir = work_dir
        # 宿主机挂载根目录，None 时用当前 cwd
        self.mount_root = mount_root
        self.network = network
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.default_timeout = timeout
        self.read_only = read_only
        self.config = config or SandboxConfig()

        # 没装 SDK 也要能构造，run() 时再降级
        self._client: Any = None
        self._client_err: str | None = None
        if _DOCKER_IMPORT_OK:
            try:
                self._client = docker.from_env()  # type: ignore[union-attr]
                # ping 一下 daemon，确认真的连上了
                self._client.ping()
            except Exception as e:
                self._client = None
                self._client_err = f"{type(e).__name__}: {e}"
        else:
            self._client_err = _DOCKER_IMPORT_ERR

        # 降级用的软沙箱
        self._fallback = SandboxExecutor(self.config)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检查 Docker 是否可用。"""
        if not _DOCKER_IMPORT_OK or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def fallback_reason(self) -> str | None:
        """不可用时给出原因，方便排查。"""
        if _DOCKER_IMPORT_OK and self._client is not None:
            return None
        return self._client_err or "docker unavailable"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
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
        """在 Docker 容器里执行命令。

        - workspace 目录挂载到容器的 work_dir
        - network=none 默认禁网（可配置）
        - memory/cpu 限制
        - 超时杀容器
        - 输出截断与 SandboxExecutor 一致
        """
        # 不可用就直接走软沙箱，别让上层感知到差异
        if not self.is_available():
            return self._fallback.run(
                cmd,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=text,
                env=env,
                config=self.config,
                **kwargs,
            )

        # 复用软沙箱的校验逻辑（白名单、cwd 限制）
        try:
            self._fallback._validate_command(cmd, config=self.config)
        except Exception as e:
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=str(e),
                command=cmd,
                dry_run=False,
                blocked=True,
                block_reason="policy_violation",
            )

        valid_cwd = self._fallback._validate_cwd(cwd, config=self.config)
        host_dir = str(valid_cwd) if valid_cwd else (
            self.mount_root or os.getcwd()
        )

        if timeout is None:
            timeout = self.default_timeout
        timeout = min(float(timeout), self.config.max_timeout)

        if self.config.dry_run:
            return SandboxResult(
                success=True,
                returncode=0,
                stdout=f"[dry-run] Would run in docker container '{self.image}': " + " ".join(cmd),
                stderr="",
                command=cmd,
                dry_run=True,
            )

        volumes = {
            host_dir: {"bind": self.work_dir, "mode": "rw" if not self.read_only else "ro"},
        }

        # nano_cpus 用整数纳秒表达 CPU 配额，docker SDK 接受这个参数
        nano_cpus = int(self.cpu_limit * 1e9)

        # detach=True 先拿到 container 对象，自己轮询状态 + 超时杀容器。
        # containers.run 的 detach=False 模式下 SDK 自己读 socket，超时也杀不掉容器。
        container: Any = None
        try:
            container = self._client.containers.run(
                self.image,
                command=cmd,
                volumes=volumes,
                working_dir=self.work_dir,
                environment=env or {},
                network=self.network,
                mem_limit=self.memory_limit,
                nano_cpus=nano_cpus,
                detach=True,
                stdout=True,
                stderr=True,
            )
        except APIError as e:
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"docker API error: {e}",
                command=cmd,
                dry_run=False,
            )
        except Exception:
            # 创建容器就崩了，老老实实回软沙箱
            return self._fallback.run(
                cmd,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=text,
                env=env,
                config=self.config,
                **kwargs,
            )

        # 轮询容器状态；超时直接 kill
        deadline_expired = False
        try:
            deadline = time.monotonic() + timeout
            while True:
                container.reload()
                status = container.status
                if status in ("exited", "dead", "removed"):
                    break
                if time.monotonic() >= deadline:
                    deadline_expired = True
                    try:
                        container.kill()
                    except Exception:
                        # 容器可能刚好自己退出了，忽略
                        pass
                    break
                time.sleep(0.5)
        except Exception:
            # 轮询崩了就尽量清理，再走软沙箱
            try:
                container.remove(force=True)
            except Exception:
                logger.debug("remove failed", exc_info=True)
            return self._fallback.run(
                cmd,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=text,
                env=env,
                config=self.config,
                **kwargs,
            )

        # 抓 logs，stdout/stderr 分开
        try:
            stdout_bytes = container.logs(stdout=True, stderr=False) or b""
            stderr_bytes = container.logs(stdout=False, stderr=True) or b""
        except Exception:
            stdout_bytes = b""
            stderr_bytes = b""

        # 拿退出码
        try:
            returncode = int(container.attrs["State"]["ExitCode"])
        except Exception:
            returncode = -1

        try:
            container.remove(force=True)
        except Exception:
            logger.debug("remove failed", exc_info=True)

        if deadline_expired:
            returncode = -1
            stderr_bytes = (stderr_bytes or b"") + b"\n[sandbox] container killed on timeout"

        success = (returncode == 0) and not deadline_expired

        if isinstance(stdout_bytes, (bytes, bytearray)):
            stdout = bytes(stdout_bytes).decode("utf-8", errors="replace")
        else:
            stdout = str(stdout_bytes)
        if isinstance(stderr_bytes, (bytes, bytearray)):
            stderr = bytes(stderr_bytes).decode("utf-8", errors="replace")
        else:
            stderr = str(stderr_bytes)

        # 截断，逻辑跟 SandboxExecutor.run 一致
        max_bytes = self.config.max_output_bytes
        if len(stdout.encode("utf-8", errors="replace")) > max_bytes:
            stdout = stdout[: max_bytes // 4] + "\n... [truncated]"
        if len(stderr.encode("utf-8", errors="replace")) > max_bytes:
            stderr = stderr[: max_bytes // 4] + "\n... [truncated]"

        return SandboxResult(
            success=success,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            command=cmd,
            dry_run=False,
        )
