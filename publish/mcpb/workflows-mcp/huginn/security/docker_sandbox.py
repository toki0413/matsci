"""Docker 容器沙箱。

走 docker Python SDK 在隔离容器里执行命令，比 subprocess 软沙箱强一档：
网络默认 none、内存/CPU 限死、超时直接 kill 容器。
SDK 没装或者 daemon 没起来都不抛，静默退回 SandboxExecutor。
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from huginn.security.sandbox import (
    SandboxConfig,
    SandboxExecutor,
    SandboxResult,
)

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


# ── 三档沙箱配额 (profile) ──────────────────────────────────────
# 竞品调研后的默认值. 调用方按任务类型选 profile, 不传默认 light.
# ponytail: 三档够覆盖科研工作流, 不做连续配置 — 多一个自由度多一处出错.
# 升级路径: profile 从 env var 读, 或按 tool constraint_scope 自动选.
_PROFILES: dict[str, tuple[str, str, str | None, float]] = {
    # profile: (memory, cpu_str, disk_tmpfs, timeout_s)
    "light": ("2g", "2.0", None, 3600.0),  # 绘图/数据分析/cif 解析
    "standard": ("8g", "4.0", "20g", 21600.0),  # 中等 VASP/LAMMPS/ML 推理
    "heavy": ("16g", "8.0", "50g", 172800.0),  # 大体系 DFT/长 MD/finetune
}


def _resolve_profile(
    profile: str | None,
    explicit_mem: str | None,
    explicit_cpu: float | None,
    explicit_disk: str | None,
    explicit_timeout: float | None,
) -> tuple[str, str, str | None, float]:
    """从 profile + 显式 override 算出最终 (mem, cpu_str, disk, timeout).

    显式参数非 None 时覆盖 profile 的对应字段.
    ponytail: profile 是字符串, 不引入 enum — 减少一个 import 依赖.
    """
    p = _PROFILES.get(profile or "light", _PROFILES["light"])
    mem = explicit_mem or p[0]
    cpu_str = str(explicit_cpu) if explicit_cpu is not None else p[1]
    disk = explicit_disk if explicit_disk is not None else p[2]
    timeout = explicit_timeout if explicit_timeout is not None else p[3]
    return mem, cpu_str, disk, timeout


def _connect_docker_client() -> tuple[Any, str | None]:
    """尝试连上 Docker daemon; 失败时返回 (None, 原因)."""
    if _DOCKER_IMPORT_OK:
        try:
            client = docker.from_env()  # type: ignore[union-attr]
            # ping 一下 daemon，确认真的连上了
            client.ping()
            return client, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, _DOCKER_IMPORT_ERR


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
        memory_limit: str | None = None,
        cpu_limit: float | None = None,
        timeout: float | None = None,
        read_only: bool = False,
        config: SandboxConfig | None = None,
        profile: str = "light",
        disk_limit: str | None = None,
    ) -> None:
        # profile 优先, 显式参数覆盖 profile 对应字段.
        # ponytail: 老代码传 memory_limit="2g" 仍工作 — _resolve_profile 走 None 路径.
        _mem, _cpu_str, _disk, _timeout = _resolve_profile(
            profile,
            memory_limit,
            cpu_limit,
            disk_limit,
            timeout,
        )
        self.image = image
        self.work_dir = work_dir
        # 宿主机挂载根目录，None 时用当前 cwd
        self.mount_root = mount_root
        self.network = network
        self.memory_limit = _mem
        # cpu_limit 老字段保留 (float), 但 docker SDK 用 nano_cpus 整数纳秒
        self.cpu_limit = float(_cpu_str)
        self.default_timeout = _timeout
        self.read_only = read_only
        self.config = config or SandboxConfig()
        # P4-6: 磁盘配额 — tmpfs 挂 work_dir, 限制容器可写磁盘大小.
        # None 时不挂 tmpfs (老 light 行为, 用宿主磁盘). standard/heavy 默认挂.
        self.disk_limit = _disk
        # profile 透传, 方便观测/日志
        self.profile = profile or "light"

        # 没装 SDK 也要能构造，run() 时再降级
        self._client, self._client_err = _connect_docker_client()

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
            logger.debug("best-effort op failed", exc_info=True)
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
        unavailable = self._ds_unavailable_result(
            self, cmd, cwd, timeout, capture_output, text, env, kwargs
        )
        if unavailable is not None:
            return unavailable
        # 复用软沙箱的校验逻辑（白名单、cwd 限制）
        blocked = self._ds_check_policy(self, cmd)
        if blocked is not None:
            return blocked
        host_dir, timeout = self._ds_resolve_context(self, cwd, timeout)
        dry_run = self._ds_dry_run(self, cmd)
        if dry_run is not None:
            return dry_run
        return DockerSandboxExecutor._ds_execute(
            self, cmd, cwd, timeout, capture_output, text, env, kwargs, host_dir
        )

    @staticmethod
    def _ds_execute(
        exe: DockerSandboxExecutor,
        cmd: list[str],
        cwd: str | Path | None,
        timeout: float,
        capture_output: bool,
        text: bool,
        env: dict[str, str] | None,
        kwargs: dict[str, Any],
        host_dir: str,
    ) -> SandboxResult:
        """容器执行阶段: 装配参数 → 启动 → 轮询 → 收尾; 失败时降级软沙箱."""
        volumes, tmpfs, nano_cpus = DockerSandboxExecutor._ds_build_spec(exe, host_dir)
        container, api_error, needs_fallback = (
            DockerSandboxExecutor._ds_start_container(
                exe,
                cmd,
                volumes,
                tmpfs,
                nano_cpus,
                env,
            )
        )
        if api_error is not None:
            return api_error
        if needs_fallback:
            # 创建容器就崩了，老老实实回软沙箱
            return DockerSandboxExecutor._ds_fallback_run(
                exe, cmd, cwd, timeout, capture_output, text, env, kwargs
            )
        # 轮询容器状态；超时直接 kill
        deadline_expired, needs_fallback = DockerSandboxExecutor._ds_wait_container(
            container, timeout
        )
        if needs_fallback:
            # 轮询崩了，helper 已清理容器，回软沙箱
            return DockerSandboxExecutor._ds_fallback_run(
                exe, cmd, cwd, timeout, capture_output, text, env, kwargs
            )
        # 抓 logs / 退出码 / 清理容器，再封装结果
        stdout_bytes, stderr_bytes, returncode = (
            DockerSandboxExecutor._ds_collect_output(container)
        )
        return DockerSandboxExecutor._ds_make_result(
            exe, cmd, deadline_expired, returncode, stdout_bytes, stderr_bytes
        )

    # ------------------------------------------------------------------
    # run() 分阶段 helper — 显式收参, 只返回中间结果
    # ------------------------------------------------------------------
    @staticmethod
    def _ds_unavailable_result(
        exe: DockerSandboxExecutor,
        cmd: list[str],
        cwd: str | Path | None,
        timeout: float | None,
        capture_output: bool,
        text: bool,
        env: dict[str, str] | None,
        kwargs: dict[str, Any],
    ) -> SandboxResult | None:
        """Docker 不可用时返回软沙箱结果，可用时返回 None 继续走容器。"""
        if exe.is_available():
            return None
        return exe._fallback.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            env=env,
            config=exe.config,
            **kwargs,
        )

    @staticmethod
    def _ds_fallback_run(
        exe: DockerSandboxExecutor,
        cmd: list[str],
        cwd: str | Path | None,
        timeout: float | None,
        capture_output: bool,
        text: bool,
        env: dict[str, str] | None,
        kwargs: dict[str, Any],
    ) -> SandboxResult:
        """退回 subprocess 软沙箱执行（容器创建/轮询失败时）。"""
        return exe._fallback.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            env=env,
            config=exe.config,
            **kwargs,
        )

    @staticmethod
    def _ds_check_policy(
        exe: DockerSandboxExecutor,
        cmd: list[str],
    ) -> SandboxResult | None:
        """复用软沙箱的命令校验（白名单、deny 规则）；违规时返回 blocked 结果。"""
        try:
            exe._fallback._validate_command(cmd, config=exe.config)
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
        return None

    @staticmethod
    def _ds_resolve_context(
        exe: DockerSandboxExecutor,
        cwd: str | Path | None,
        timeout: float | None,
    ) -> tuple[str, float]:
        """算出宿主挂载目录与最终超时（含 max_timeout 截断）。"""
        valid_cwd = exe._fallback._validate_cwd(cwd, config=exe.config)
        host_dir = str(valid_cwd) if valid_cwd else (exe.mount_root or os.getcwd())
        if timeout is None:
            timeout = exe.default_timeout
        timeout = min(float(timeout), exe.config.max_timeout)
        return host_dir, timeout

    @staticmethod
    def _ds_dry_run(
        exe: DockerSandboxExecutor,
        cmd: list[str],
    ) -> SandboxResult | None:
        """dry_run 时只回显将要执行的命令，否则返回 None。"""
        if not exe.config.dry_run:
            return None
        return SandboxResult(
            success=True,
            returncode=0,
            stdout=f"[dry-run] Would run in docker container '{exe.image}': "
            + " ".join(cmd),
            stderr="",
            command=cmd,
            dry_run=True,
        )

    @staticmethod
    def _ds_build_spec(
        exe: DockerSandboxExecutor,
        host_dir: str,
    ) -> tuple[dict[str, Any], dict[str, str], int]:
        """容器参数装配: 卷挂载、CPU 纳秒配额、tmpfs 磁盘配额。"""
        volumes = {
            host_dir: {
                "bind": exe.work_dir,
                "mode": "rw" if not exe.read_only else "ro",
            },
        }

        # nano_cpus 用整数纳秒表达 CPU 配额，docker SDK 接受这个参数
        nano_cpus = int(exe.cpu_limit * 1e9)

        # P4-6: tmpfs 挂 /workspace 限制可写磁盘大小.
        # disk_limit=None 时不挂 tmpfs (light 默认行为, 兼容老配置).
        # ponytail: tmpfs 是内存支持的, 不会吃宿主磁盘, 但会占 mem_limit 额度 —
        # 写满 tmpfs 时进程收到 ENOSPC, 比吃满宿主盘安全.
        tmpfs: dict[str, str] = {}
        if exe.disk_limit:
            # docker tmpfs 接受 size=Nm/Ng 格式. /workspace 用 tmpfs,
            # 原始 host 挂载仍保留 (work_dir 同一挂载点, tmpfs 会覆盖).
            # 实际上 tmpfs 优先级高于 bind mount, 所以 work_dir 走 tmpfs.
            # ponytail: 如果需要 host 文件可见, 升级路径是 /workspace 内挂 /workspace/host.
            tmpfs = {exe.work_dir: f"size={exe.disk_limit}"}
        return volumes, tmpfs, nano_cpus

    @staticmethod
    def _ds_start_container(
        exe: DockerSandboxExecutor,
        cmd: list[str],
        volumes: dict[str, Any],
        tmpfs: dict[str, str],
        nano_cpus: int,
        env: dict[str, str] | None,
    ) -> tuple[Any, SandboxResult | None, bool]:
        """创建并启动容器；返回 (容器, API 错误结果, 是否需降级到软沙箱)。"""
        # detach=True 先拿到 container 对象，自己轮询状态 + 超时杀容器。
        # containers.run 的 detach=False 模式下 SDK 自己读 socket，超时也杀不掉容器。
        try:
            run_kwargs: dict[str, Any] = {
                "image": exe.image,
                "command": cmd,
                "volumes": volumes,
                "working_dir": exe.work_dir,
                "environment": env or {},
                "network": exe.network,
                "mem_limit": exe.memory_limit,
                "nano_cpus": nano_cpus,
                "detach": True,
                "stdout": True,
                "stderr": True,
            }
            if tmpfs:
                run_kwargs["tmpfs"] = tmpfs
            container = exe._client.containers.run(**run_kwargs)
        except APIError as e:
            return (
                None,
                SandboxResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"docker API error: {e}",
                    command=cmd,
                    dry_run=False,
                ),
                False,
            )
        except Exception:
            return None, None, True
        return container, None, False

    @staticmethod
    def _ds_wait_container(container: Any, timeout: float) -> tuple[bool, bool]:
        """轮询容器状态，超时 kill；返回 (是否超时, 是否需降级)。"""
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
                    # 容器可能刚好自己退出了，忽略
                    DockerSandboxExecutor._ds_kill_on_timeout(container)
                    break
                time.sleep(0.5)
        except Exception:
            # 轮询崩了就尽量清理，再走软沙箱
            try:
                container.remove(force=True)
            except Exception:
                logger.debug("remove failed", exc_info=True)
            return False, True
        return deadline_expired, False

    @staticmethod
    def _ds_kill_on_timeout(container: Any) -> None:
        """超时后尽力 kill 容器；容器可能刚好自己退出，异常一律忽略。"""
        with contextlib.suppress(Exception):
            container.kill()

    @staticmethod
    def _ds_collect_output(container: Any) -> tuple[Any, Any, int]:
        """抓 logs（stdout/stderr 分开）、读退出码并清理容器。"""
        # 抓 logs，stdout/stderr 分开
        try:
            stdout_bytes = container.logs(stdout=True, stderr=False) or b""
            stderr_bytes = container.logs(stdout=False, stderr=True) or b""
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
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
        return stdout_bytes, stderr_bytes, returncode

    @staticmethod
    def _ds_make_result(
        exe: DockerSandboxExecutor,
        cmd: list[str],
        deadline_expired: bool,
        returncode: int,
        stdout_bytes: Any,
        stderr_bytes: Any,
    ) -> SandboxResult:
        """超时修正 + 解码 + 截断，封装成 SandboxResult。"""
        if deadline_expired:
            returncode = -1
            stderr_bytes = (
                stderr_bytes or b""
            ) + b"\n[sandbox] container killed on timeout"

        success = (returncode == 0) and not deadline_expired

        stdout = exe._ds_decode(stdout_bytes)
        stderr = exe._ds_decode(stderr_bytes)

        # 截断，逻辑跟 SandboxExecutor.run 一致
        max_bytes = exe.config.max_output_bytes
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

    @staticmethod
    def _ds_decode(data: Any) -> str:
        """bytes/bytearray 按 utf-8 替换解码，其余直接 str()。"""
        if isinstance(data, bytes | bytearray):
            return bytes(data).decode("utf-8", errors="replace")
        return str(data)


if __name__ == "__main__":
    # P4-6 自检: 验证 profile 解析 + tmpfs 配置 + 显式 override
    # ponytail: 不真跑 docker (CI 没装), 只验证参数解析逻辑.

    # 1. 默认 profile=light → 2g/2cpu/无 tmpfs/1h
    exe = DockerSandboxExecutor()
    assert exe.memory_limit == "2g", f"light mem: {exe.memory_limit}"
    assert exe.cpu_limit == 2.0, f"light cpu: {exe.cpu_limit}"
    assert exe.disk_limit is None, f"light disk: {exe.disk_limit}"
    assert exe.default_timeout == 3600.0, f"light timeout: {exe.default_timeout}"
    assert exe.profile == "light"

    # 2. standard → 8g/4cpu/20g tmpfs/6h
    exe_s = DockerSandboxExecutor(profile="standard")
    assert exe_s.memory_limit == "8g", f"standard mem: {exe_s.memory_limit}"
    assert exe_s.cpu_limit == 4.0, f"standard cpu: {exe_s.cpu_limit}"
    assert exe_s.disk_limit == "20g", f"standard disk: {exe_s.disk_limit}"
    assert exe_s.default_timeout == 21600.0, (
        f"standard timeout: {exe_s.default_timeout}"
    )

    # 3. heavy → 16g/8cpu/50g tmpfs/48h
    exe_h = DockerSandboxExecutor(profile="heavy")
    assert exe_h.memory_limit == "16g", f"heavy mem: {exe_h.memory_limit}"
    assert exe_h.cpu_limit == 8.0, f"heavy cpu: {exe_h.cpu_limit}"
    assert exe_h.disk_limit == "50g", f"heavy disk: {exe_h.disk_limit}"
    assert exe_h.default_timeout == 172800.0, f"heavy timeout: {exe_h.default_timeout}"

    # 4. 显式参数覆盖 profile
    exe_o = DockerSandboxExecutor(profile="light", memory_limit="4g", timeout=7200.0)
    assert exe_o.memory_limit == "4g", f"override mem: {exe_o.memory_limit}"
    assert exe_o.cpu_limit == 2.0, f"keep profile cpu: {exe_o.cpu_limit}"
    assert exe_o.default_timeout == 7200.0, f"override timeout: {exe_o.default_timeout}"

    # 5. _resolve_profile 纯函数: unknown profile 走 light 默认
    m, c, d, t = _resolve_profile("unknown", None, None, None, None)
    assert m == "2g" and c == "2.0" and d is None and t == 3600.0, (
        f"unknown profile should fall back to light: {m, c, d, t}"
    )

    # 6. _resolve_profile: 显式 None 不覆盖 (None 是"未传"语义)
    m2, c2, d2, t2 = _resolve_profile("standard", None, 6.0, None, None)
    assert m2 == "8g", f"profile mem 保持: {m2}"
    assert c2 == "6.0", f"显式 cpu 覆盖: {c2}"
    assert d2 == "20g", f"profile disk 保持: {d2}"

    print("docker_sandbox profile self-check: OK")
