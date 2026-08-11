"""Command execution sandbox for Huginn.

Prevents arbitrary code execution by whitelisting executables, restricting
working directories, and enforcing timeouts/output limits.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.security.docker_sandbox import DockerSandboxExecutor

logger = logging.getLogger(__name__)


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

    # Strict mode: cwd must be under allowed_work_dirs.
    # 默认 True (安全优先); HUGINN_SANDBOX_RELAX=1 可关.
    strict_work_dir: bool = True


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

    def __init__(
        self,
        config: SandboxConfig | None = None,
        profile: str = "light",
    ) -> None:
        self.config = config or SandboxConfig()
        # Task 1.2: 软沙箱按 profile 限内存 (POSIX RLIMIT_AS), 跚 Docker 档位对齐.
        # light=2g/standard=8g/heavy=16g, 复用 docker_sandbox._PROFILES 的 mem 字段.
        self.profile = profile
        # HUGINN_SANDBOX_RELAX=1 关闭 strict_work_dir (兼容老用户)
        if os.environ.get("HUGINN_SANDBOX_RELAX") == "1":
            self.config.strict_work_dir = False
        # strict=True 但 allowed_work_dirs 空 → path scoping 实际不生效, warn
        if self.config.strict_work_dir and not self.config.allowed_work_dirs:
            import logging
            logging.getLogger(__name__).warning(
                "strict_work_dir=True but allowed_work_dirs empty — path scoping "
                "inactive. Set allowed_work_dirs or HUGINN_WORKSPACE to enable."
            )

    def _resolve_executable(self, cmd: list[str]) -> str:
        """Resolve the first element of cmd to an absolute path.

        Windows 兼容: POSIX coreutils (ls/cp/mv/rm/cat/...) 在原生 Windows
        没有 .exe. shutil.which("ls") 返回 None → 之前 agent 卡循环
        (self-modify proposal 被 reject). 这里加 Windows fallback:
          1. 先按原名 which (跨平台: 装了 git-bash/WSL 能直接用)
          2. 失败时映射到 Windows 等价命令 (ls→cmd /c dir 等)
          3. 都失败才 raise
        映射表只覆盖白名单里有的 coreutils, 不引入新可执行文件.
        ponytail: 用 cmd /c 调内置命令, 不装新依赖. 天花板: cmd /c 的
          参数语义跟 POSIX 不完全一致 (e.g. ls -la vs dir), agent 需要适配.
          升级路径: 装 git-bash 把 coreutils 加到 PATH, 或换 WSL.
        """
        if not cmd:
            raise SandboxError("Empty command")
        exe = shutil.which(cmd[0])
        if exe is not None:
            return exe

        # Windows fallback: 映射 POSIX coreutils 到 cmd /c 内置命令
        if os.name == "nt":
            _WIN_FALLBACK = {
                "ls": "dir",
                "cp": "copy",
                "mv": "move",
                "rm": "del",
                "cat": "type",
                "touch": "copy /b",
                "mkdir": "mkdir",
                "rmdir": "rmdir",
                "echo": "echo",
                "pwd": "cd",
                "which": "where",
                "find": "find",
                "sort": "sort",
                "head": "more",
                "tail": "more",
                "wc": "find /c",
                "grep": "findstr",
                "diff": "fc",
                "true": "rem",
                "false": "rem",
                "test": "if",
                "env": "set",
            }
            _win_cmd = _WIN_FALLBACK.get(cmd[0].lower())
            if _win_cmd:
                # 验证 Windows 命令可用 (cmd 内置命令 which 不到, 直接信任)
                # 返回 cmd[0] 让上层 subprocess 走 cmd /c 路径
                # ponytail: 这里返回原 cmd[0], 实际执行在 run() 里用 cmd /c 包
                return cmd[0]

        raise SandboxError(f"Executable not found: {cmd[0]}")

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

        # Policy engine: declarative rules add deny patterns on top of
        # the legacy whitelist. deny -> block, allow/ask -> whitelist.
        from huginn.security.policy_engine import evaluate_command_hook

        decision = evaluate_command_hook(cmd)
        if decision.action == "deny":
            raise SandboxError(
                f"Blocked by security policy '{decision.matched_rule}': "
                f"{decision.reason}"
            )

        # allow/ask/unmatched all fall through to the whitelist check.
        # The policy engine handles global deny patterns, but a sandbox
        # with a restrictive allowed_executables should still enforce its
        # own list — otherwise a custom whitelist is silently ignored.
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
                    logger.debug("best-effort op failed", exc_info=True)
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

        # Windows coreutils fallback: _resolve_executable 返回原 cmd[0] (e.g. "ls")
        # 但 shutil.which 找不到 → 这里用 cmd /c 把整条命令包起来, 让 cmd.exe
        # 走内置命令 (dir/copy/type/...). 白名单已含 cmd.exe 间接调用, 不开新口子.
        # ponytail: 用 cmd /c 单层包裹, 不递归. 天花板: cmd /c 参数语义跟 POSIX
        #   不完全一致 (e.g. ls -la vs dir), agent LLM 通常会自己适配 Windows 语法.
        #   升级路径: 装 git-bash 让 coreutils 在 PATH 里直接 which 到.
        _WIN_COREUTILS = {
            "ls", "cp", "mv", "rm", "cat", "touch", "mkdir", "rmdir",
            "echo", "pwd", "which", "find", "sort", "head", "tail",
            "wc", "grep", "diff", "true", "false", "test", "env",
        }
        if os.name == "nt" and cmd and cmd[0].lower() in _WIN_COREUTILS:
            # shutil.which 已经在 _resolve_executable 里试过失败, 直接走 cmd /c
            _win_map = {
                "ls": "dir", "cp": "copy", "mv": "move", "rm": "del",
                "cat": "type", "touch": "copy /b", "mkdir": "mkdir",
                "rmdir": "rmdir", "echo": "echo", "pwd": "cd",
                "which": "where", "find": "find", "sort": "sort",
                "head": "more", "tail": "more", "wc": "find /c",
                "grep": "findstr", "diff": "fc", "true": "rem",
                "false": "rem", "test": "if", "env": "set",
            }
            _mapped = _win_map.get(cmd[0].lower(), cmd[0])
            # 重组: cmd /c <mapped> <rest args>
            # ponytail: 不翻译 -la/-la 等参数, agent 自己写 Windows 语法时直接通过.
            #   这里只在 cmd[0] 是 POSIX coreutil 时翻译, 参数透传 (可能不兼容但至少不卡).
            cmd = ["cmd", "/c", _mapped, *cmd[1:]]

        # Task 1.2: POSIX 软沙箱补 RLIMIT_AS 内存上限 (按 profile, 跚 Docker 档位对齐).
        # Windows 跳过 (resource 模块不支持 RLIMIT_AS). save/restore 父进程 limit,
        # 避免污染 agent 自身内存上限. ponytail: try/except 包裹, 失败不阻塞.
        #
        # 关键: 只降 soft limit, 不动 hard limit. 非 root 用户一旦降低 hard limit
        # 就无法再升高 (POSIX 规范), 导致 setrlimit 恢复失败, RLIMIT_AS 永久卡在
        # 2GB, 后续 pytest 进程因虚拟内存不足触发 MemoryError 并挂起 (Python 3.13
        # 在 CI 上实测卡死 42 分钟直到 job timeout).
        _saved_rlimit_soft = None
        if os.name != "nt":
            try:
                import resource as _resource
                _mem = _profile_mem_bytes(self.profile)
                if _mem and _mem > 0:
                    _cur_soft, _cur_hard = _resource.getrlimit(_resource.RLIMIT_AS)
                    _saved_rlimit_soft = _cur_soft
                    _resource.setrlimit(_resource.RLIMIT_AS, (_mem, _cur_hard))
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "setrlimit(RLIMIT_AS, profile=%s) failed (non-fatal)",
                    self.profile, exc_info=True,
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
                shell=False,
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
        finally:
            # 恢复父进程 soft limit, 避免子进程内存上限反过来卡死 agent 自身.
            # 只恢复 soft limit — hard limit 从未被降低, 无需恢复.
            if _saved_rlimit_soft is not None:
                try:
                    import resource as _resource
                    _cur_soft, _cur_hard = _resource.getrlimit(_resource.RLIMIT_AS)
                    _resource.setrlimit(_resource.RLIMIT_AS, (_saved_rlimit_soft, _cur_hard))
                except Exception:
                    logger.debug("failed to restore parent rlimit", exc_info=True)

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


def _profile_mem_bytes(profile: str) -> int | None:
    """profile -> 内存上限 (bytes). 复用 docker_sandbox._PROFILES 的 mem 字段.

    light=2g/standard=8g/heavy=16g, 跚 Docker 容器档位对齐.
    ponytail: lazy import docker_sandbox 取 _PROFILES, 解析 "2g"/"512m" 字符串;
      import 失败 (docker SDK 缺) 时用本地 fallback, 值跟 _PROFILES 同步.
    """
    _FALLBACK_GB = {"light": 2, "standard": 8, "heavy": 16}
    try:
        from huginn.security.docker_sandbox import _PROFILES
        mem_str = _PROFILES.get(profile or "light", _PROFILES["light"])[0]
        mem_str = mem_str.strip().lower()
        if mem_str.endswith("g"):
            return int(float(mem_str[:-1]) * 1024 * 1024 * 1024)
        if mem_str.endswith("m"):
            return int(float(mem_str[:-1]) * 1024 * 1024)
        return int(float(mem_str))
    except Exception:
        gb = _FALLBACK_GB.get(profile or "light", 2)
        return gb * 1024 * 1024 * 1024


def create_sandbox(
    config: SandboxConfig | None = None,
    prefer_docker: bool = True,
    docker_image: str = "python:3.12-slim",
    profile: str = "light",
) -> SandboxExecutor | DockerSandboxExecutor:  # type: ignore[name-defined]
    """根据环境自动选择沙箱后端。

    - prefer_docker=True (默认) 且 Docker 可用 → DockerSandboxExecutor
    - Docker 不可用 → warn log + 自动退回 SandboxExecutor (subprocess 软沙箱), 不 fatal
    - 否则 → SandboxExecutor（subprocess 软沙箱）

    环境变量:
    - HUGINN_DOCKER_SANDBOX=1 等同 prefer_docker=True (老开关, 保留)
    - HUGINN_USE_DOCKER=0 显式回退 subprocess (兼容无 Docker 部署); =1 显式开

    profile: light/standard/heavy 三档配额, 控制 Docker 容器的 mem/cpu/disk/timeout.
    调用方按任务类型选 (绘图=light, VASP/LAMMPS=standard, 大体系 DFT=heavy).
    """
    cfg = config or SandboxConfig()
    _log = __import__("logging").getLogger(__name__)

    # 环境变量开关，方便运维侧不改代码就切后端
    # HUGINN_USE_DOCKER 显式设值优先 (0=强制 subprocess, 1=强制 docker);
    # HUGINN_DOCKER_SANDBOX=1 是老开关, 等同显式开; 都不设时跟 prefer_docker 默认走.
    use_docker_env = os.environ.get("HUGINN_USE_DOCKER")
    if use_docker_env == "0":
        want_docker = False
    elif use_docker_env == "1" or os.environ.get("HUGINN_DOCKER_SANDBOX") == "1":
        want_docker = True
    else:
        want_docker = prefer_docker

    if want_docker:
        # 延迟 import，避免 docker SDK 没装时整个 sandbox 模块都加载不了
        try:
            from huginn.security.docker_sandbox import DockerSandboxExecutor
        except Exception:
            _log.warning(
                "docker_sandbox import failed, falling back to subprocess "
                "sandbox (profile=%s)", profile, exc_info=True,
            )
            return SandboxExecutor(cfg, profile=profile)

        try:
            docker_executor = DockerSandboxExecutor(image=docker_image, config=cfg, profile=profile)
        except Exception:
            # 构造失败也别让上层挂掉
            _log.warning(
                "DockerSandboxExecutor init failed, falling back to "
                "subprocess sandbox (profile=%s)", profile, exc_info=True,
            )
            return SandboxExecutor(cfg, profile=profile)

        if docker_executor.is_available():
            return docker_executor
        # Docker daemon 不可用 → warn + 退回 subprocess (不 fatal)
        _log.warning(
            "Docker daemon unavailable, falling back to subprocess sandbox "
            "(profile=%s). Set HUGINN_USE_DOCKER=0 to silence this.",
            profile,
        )
        return SandboxExecutor(cfg, profile=profile)

    return SandboxExecutor(cfg, profile=profile)
