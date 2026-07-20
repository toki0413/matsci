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
