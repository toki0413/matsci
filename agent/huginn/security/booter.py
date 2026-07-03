"""Computer Booter —— 统一的执行后端抽象。

借鉴 AstrBot 的 ComputerBooter (astrbot/core/computer/booters/base.py)。
对上层暴露统一接口, 屏蔽本地 / 容器 / SSH 三种执行后端的差异: agent 代码
只管调 booter.shell.exec(), 不用关心命令最终跑在本机还是远端 HPC 上。

架构:
  ComputerBooter (抽象基类)
    ├── LocalBooter   —— 包装现有的 SandboxExecutor
    └── SSHBooter     —— 包装 HPCClient + CredentialStore

组件 (对应 AstrBot 的 olayer 模式):
  ShellComponent.exec(cmd)    —— 跑 shell 命令, 返回 stdout/stderr/exit_code
  PythonComponent.exec(code)  —— 跑 Python 代码, 返回输出
  FileSystemComponent         —— 上传 / 下载 / 列文件
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from huginn.security.command_filter import check_command_safety
from huginn.security.sandbox import (
    SandboxConfig,
    SandboxError,
    SandboxExecutor,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """一次 shell / Python 执行的结果。"""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    latency_ms: float = 0.0
    error: str | None = None


# ── 组件抽象 (AstrBot olayer 模式) ─────────────────────────────


class ShellComponent(ABC):
    @abstractmethod
    async def exec(
        self,
        command: list[str],
        cwd: str | None = None,
        timeout: float = 300.0,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        ...


class PythonComponent(ABC):
    @abstractmethod
    async def exec(
        self,
        code: str,
        timeout: float = 30.0,
        cwd: str | None = None,
    ) -> ExecResult:
        ...


class FileSystemComponent(ABC):
    @abstractmethod
    async def upload(self, local_path: str, remote_path: str) -> bool:
        ...

    @abstractmethod
    async def download(self, remote_path: str, local_path: str) -> bool:
        ...

    @abstractmethod
    async def list_files(self, path: str) -> list[str]:
        ...


# ── Booter 抽象 ────────────────────────────────────────────────


class ComputerBooter(ABC):
    """执行后端的抽象基类。子类负责把 shell / python / fs 组件装好。"""

    shell: ShellComponent
    python: PythonComponent
    fs: FileSystemComponent | None

    @abstractmethod
    async def boot(self, session_id: str | None = None) -> None:
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...

    def capabilities(self) -> tuple[str, ...]:
        """返回该后端支持的能力, 方便上层做特性探测。"""
        caps = ("shell", "python")
        if self.fs is not None:
            caps += ("filesystem",)
        return caps


# ── 本地实现 ───────────────────────────────────────────────────


class LocalShell(ShellComponent):
    """本地 shell 执行, 内部走 SandboxExecutor。"""

    def __init__(self, sandbox: SandboxExecutor | None = None) -> None:
        self._sandbox = sandbox or SandboxExecutor()

    async def exec(
        self,
        command: list[str],
        cwd: str | None = None,
        timeout: float = 300.0,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # 第二道防线: 命令黑名单, 拦截危险参数
        filter_result = check_command_safety(command)
        if not filter_result.is_safe:
            logger.warning(
                "拦截危险命令: pattern=%s", filter_result.matched_pattern
            )
            return ExecResult(
                success=False,
                error=f"Blocked: 命中危险模式 {filter_result.matched_pattern}",
            )

        # SandboxExecutor.run 是同步阻塞调用, 丢到线程池里跑, 别卡住事件循环
        try:
            result = await asyncio.to_thread(
                self._sandbox.run,
                command,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
            return ExecResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except SandboxError as e:
            return ExecResult(success=False, error=str(e))
        except asyncio.TimeoutError:
            return ExecResult(
                success=False, error=f"命令超时 ({timeout}s)"
            )


class LocalPython(PythonComponent):
    """本地 Python 执行。代码落到临时文件再交给沙箱跑, 避开 -c 引号转义。"""

    def __init__(self, sandbox: SandboxExecutor | None = None) -> None:
        # Python 执行器只放行 python / python3, 收窄白名单
        self._sandbox = sandbox or SandboxExecutor(
            SandboxConfig(allowed_executables={"python", "python3"})
        )

    async def exec(
        self, code: str, timeout: float = 30.0, cwd: str | None = None
    ) -> ExecResult:
        fd, script_path = tempfile.mkstemp(suffix=".py", dir=cwd)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(code)
            result = await asyncio.to_thread(
                self._sandbox.run,
                ["python", script_path],
                cwd=cwd,
                timeout=timeout,
            )
            return ExecResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except SandboxError as e:
            return ExecResult(success=False, error=str(e))
        except asyncio.TimeoutError:
            return ExecResult(
                success=False, error=f"Python 执行超时 ({timeout}s)"
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass


class LocalBooter(ComputerBooter):
    """本地执行后端 —— 直接包装 SandboxExecutor。"""

    def __init__(self, sandbox: SandboxExecutor | None = None) -> None:
        sb = sandbox or SandboxExecutor()
        self.shell = LocalShell(sb)
        self.python = LocalPython(sb)
        self.fs = None  # 本地文件系统直接访问, 不需要抽象层

    async def boot(self, session_id: str | None = None) -> None:
        # 本地后端无需初始化
        pass

    async def shutdown(self) -> None:
        # 本地后端无需清理
        pass


# ── SSH 实现 ───────────────────────────────────────────────────


class SSHShell(ShellComponent):
    """远端 shell 执行, 内部走 HPCClient (paramiko SSH)。"""

    def __init__(self, hpc_config: Any) -> None:
        self._config = hpc_config
        self._client: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            from huginn.hpc.client import HPCClient

            self._client = HPCClient(self._config)
        return self._client

    async def exec(
        self,
        command: list[str],
        cwd: str | None = None,
        timeout: float = 300.0,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # 远端同样先过命令黑名单
        filter_result = check_command_safety(command)
        if not filter_result.is_safe:
            logger.warning(
                "拦截危险 SSH 命令: pattern=%s", filter_result.matched_pattern
            )
            return ExecResult(
                success=False,
                error=f"Blocked: {filter_result.matched_pattern}",
            )

        client = await self._ensure_client()
        cmd_str = " ".join(command) if isinstance(command, list) else command
        if cwd:
            cmd_str = f"cd {cwd} && {cmd_str}"

        try:
            result = await asyncio.to_thread(client._exec, cmd_str)
            # HPCClient._exec 返回 (stdout, stderr, exit_code)
            if isinstance(result, tuple):
                stdout, stderr, exit_code = result
            else:
                stdout, stderr, exit_code = str(result), "", 0
            return ExecResult(
                success=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        except Exception as e:
            return ExecResult(success=False, error=f"SSH 错误: {e}")


class SSHPython(PythonComponent):
    """远端 Python 执行: 把代码用 python3 -c 丢到 SSHShell 跑。"""

    def __init__(self, hpc_config: Any) -> None:
        self._config = hpc_config
        self._shell = SSHShell(hpc_config)

    async def exec(
        self, code: str, timeout: float = 30.0, cwd: str | None = None
    ) -> ExecResult:
        # shlex.quote 走 POSIX 转义, 远端是 Linux HPC, 用单引号包住整段代码
        escaped_code = shlex.quote(code)
        cmd = f"python3 -c {escaped_code}"
        return await self._shell.exec([cmd], cwd=cwd, timeout=timeout)


class SSHFileSystem(FileSystemComponent):
    """远端文件系统操作, 走 HPCClient 的 sftp。"""

    def __init__(self, hpc_config: Any) -> None:
        self._config = hpc_config
        self._client: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            from huginn.hpc.client import HPCClient

            self._client = HPCClient(self._config)
        return self._client

    async def upload(self, local_path: str, remote_path: str) -> bool:
        client = await self._ensure_client()
        try:
            await asyncio.to_thread(client.upload_file, local_path, remote_path)
            return True
        except Exception as e:
            logger.error("SSH 上传失败: %s", e)
            return False

    async def download(self, remote_path: str, local_path: str) -> bool:
        client = await self._ensure_client()
        try:
            await asyncio.to_thread(
                client.download_file, remote_path, local_path
            )
            return True
        except Exception as e:
            logger.error("SSH 下载失败: %s", e)
            return False

    async def list_files(self, path: str) -> list[str]:
        client = await self._ensure_client()
        try:
            result = await asyncio.to_thread(client._exec, f"ls -1 {path}")
            # HPCClient._exec 返回 (stdout, stderr, exit_code)
            if isinstance(result, tuple):
                stdout, _stderr, exit_code = result
            else:
                stdout, exit_code = str(result), 0
            if exit_code == 0:
                return [
                    line.strip() for line in stdout.splitlines() if line.strip()
                ]
            return []
        except Exception:
            return []


class SSHBooter(ComputerBooter):
    """SSH 执行后端 —— 包装 HPCClient, 并对接 CredentialStore。"""

    def __init__(self, hpc_config: Any) -> None:
        self._config = hpc_config
        self.shell = SSHShell(hpc_config)
        self.python = SSHPython(hpc_config)
        self.fs = SSHFileSystem(hpc_config)

    async def boot(self, session_id: str | None = None) -> None:
        # 触发一次连接, 顺便验证配置能不能用
        await self.shell._ensure_client()
        logger.info("SSH booter 已连接到 %s", self._config.host)

    async def shutdown(self) -> None:
        # shell / python / fs 各自懒加载了自己的 client, 都得关掉
        # HPCClient.disconnect() 内部判空, 重复调用是安全的
        for client in (
            self.shell._client,
            self.python._shell._client,
            self.fs._client if self.fs is not None else None,
        ):
            if client is not None:
                with _suppress_disconnect():
                    client.disconnect()


class _suppress_disconnect:
    """断开连接时出错别让 shutdown 抛异常, 只记日志。"""

    def __enter__(self) -> "_suppress_disconnect":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            logger.debug("断开 SSH 连接时出错, 忽略: %s", exc)
        return True  # 吞掉异常


# ── 工厂 ───────────────────────────────────────────────────────


def create_booter(
    backend: str = "local",
    credential_id: str | None = None,
    hpc_config: Any = None,
) -> ComputerBooter:
    """按后端名创建 booter。

    Args:
        backend: "local" 或 "ssh"
        credential_id: backend="ssh" 时, 从 CredentialStore 加载 SSH 配置
        hpc_config: 直接传 HPCConfig (与 credential_id 二选一)
    """
    if backend == "local":
        return LocalBooter()
    elif backend == "ssh":
        if credential_id:
            from huginn.security.credential_store import get_credential_store

            store = get_credential_store()
            hpc_config = store.to_hpc_config(credential_id)
        elif hpc_config is None:
            raise ValueError("SSH booter 需要 credential_id 或 hpc_config")
        return SSHBooter(hpc_config)
    else:
        raise ValueError(f"未知后端: {backend}")
