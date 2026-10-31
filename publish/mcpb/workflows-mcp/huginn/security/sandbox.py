"""Command execution sandbox for Huginn.

Prevents arbitrary code execution by whitelisting executables, restricting
working directories, and enforcing timeouts/output limits.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huginn.security.child_env import sanitize_child_env
from huginn.utils.common import hash_text
from huginn.utils.process import (
    kill_process_group,
    new_group_popen_kwargs,
    track_live_child,
    untrack_live_child,
)

if TYPE_CHECKING:
    from huginn.security.docker_sandbox import DockerSandboxExecutor

logger = logging.getLogger(__name__)


def _powershell_enabled_by_default() -> bool:
    """PowerShell 支持开关的默认值: posix 关、win32 开.

    显式设置 ``HUGINN_ALLOW_POWERSHELL`` (1/true/yes) 时一律按其取值
    (无论平台); 未设置时按平台默认 (win32 开, 其余关). 放在模块级以便
    在 ``SandboxConfig`` 的 dataclass ``default_factory`` 中复用.
    """
    val = os.environ.get("HUGINN_ALLOW_POWERSHELL")
    if val is not None and val.strip():
        return val.strip().lower() in ("1", "true", "yes")
    return os.name == "nt"


def _net_isolation_enabled_by_default() -> bool:
    """P1 网络隔离开关默认值: 显式 env > feature flag > 开.

    ``HUGINN_SANDBOX_ISOLATE_NETWORK`` (1/true/yes/on 开, 0/false/no/off 关)
    显式覆盖; 未设置时读 feature flag ``sandbox_net_isolation`` (默认 True,
    纵深防御). 内核不支持 Landlock 网络规则时优雅降级 (只做 FS 隔离).
    放在模块级以便在 dataclass ``default_factory`` 中复用.
    """
    val = os.environ.get("HUGINN_SANDBOX_ISOLATE_NETWORK")
    if val is not None and val.strip():
        return val.strip().lower() in ("1", "true", "yes", "on")
    try:
        from huginn.feature_flags import FeatureFlags

        return FeatureFlags.shared().is_enabled("sandbox_net_isolation")
    except Exception:  # 防御: flag 层异常不影响沙箱构造
        logger.debug("net-isolation default: flag read failed", exc_info=True)
        return False


def _env_int(name: str) -> int | None:
    """读一个可选的正整数环境变量; 未设/非法 → None (即不设该限制)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(float(raw.strip()))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return None
    return val if val > 0 else None


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

    # PowerShell 执行支持 (安全开关). 由默认白名单里**不含** pwsh/powershell,
    # 只有在此开关为真 (显式 HUGINN_ALLOW_POWERSHELL=1 或 win32 平台默认) 时,
    # 才经 __post_init__ 把它们并入 allowed_executables. 默认 posix 关、win32 开.
    allow_powershell: bool = field(
        default_factory=_powershell_enabled_by_default,
    )

    # P1: 内核级网络隔离 (Landlock ABI >= 4, Linux 6.7+). 开后在已做内核
    # confinement 的路径上把子进程的 TCP bind/connect 全部拒掉 (Landlock 网络
    # 规则是 allow 规则, 处理了这些位又不加规则 = 全拒). 只收紧网络, 不锁文件系统.
    # 默认开 (纵深防御, 不依赖外层容器兜底); HUGINN_SANDBOX_ISOLATE_NETWORK=0 或
    # feature flag sandbox_net_isolation=false 可关. 内核不支持时优雅降级 (只做 FS 隔离).
    isolate_network: bool = field(default_factory=_net_isolation_enabled_by_default)

    # P1: 本地资源限制 (POSIX setrlimit). 在 fork 出的子进程 preexec 里设, 不污染
    # 父进程 (不像 RLIMIT_AS 那样在父进程设了再恢复, 多线程下无竞态).
    # None = 不设 (零回归). 可用环境变量给默认值.
    max_cpu_seconds: int | None = field(
        default_factory=lambda: _env_int("HUGINN_SANDBOX_MAX_CPU_SECONDS")
    )
    max_file_bytes: int | None = field(
        default_factory=lambda: _env_int("HUGINN_SANDBOX_MAX_FILE_BYTES")
    )
    max_processes: int | None = field(
        default_factory=lambda: _env_int("HUGINN_SANDBOX_MAX_PROCESSES")
    )

    def __post_init__(self) -> None:
        # 开关开 → 把 PowerShell 可执行文件并入白名单; 关 → 从白名单剔除.
        # 保持默认 posix 白名单不含 pwsh (除非显式开关打开), 不破坏既有安全假设.
        self._sync_powershell()

    def _sync_powershell(self) -> None:
        """按 ``allow_powershell`` 同步白名单中的 pwsh/powershell 项."""
        if self.allow_powershell:
            self.allowed_executables.update({"pwsh", "powershell"})
        else:
            self.allowed_executables.discard("pwsh")
            self.allowed_executables.discard("powershell")


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
    timed_out: bool = False  # True 当命令因超时被终止 (returncode 仍为 -1)
    # T3: 结构化错误分类, 供下游按类决策 (重试/追踪/降级). 默认 NONE 保持兼容.
    error_kind: str = "none"


def _kill_process_tree(proc: subprocess.Popen, *, own_group: bool) -> None:
    """终止子进程**及其同组的所有后代** — 修超时孤儿泄漏.

    背景: ``subprocess.run(timeout=...)`` 超时时 CPython 只 ``kill()`` **直接子
    进程**. 若命令是 ``sh -c '...; python3 heavy.py'``, 真正跑计算的孙进程会被
    orphan 到 init (PPID→1) 后**继续全速运行** (实测 run83 遗留 8 个 python 孤儿,
    最早烧 37min CPU, 并让主进程收尾多等 300s join 线程).

    修法: 子进程自立新会话/进程组执行 (posix ``start_new_session`` / win32
    ``CREATE_NEW_PROCESS_GROUP``), 超时时整组回收. ``own_group=False`` (调用方显式
    要求共用进程组) 时退回只杀直接子进程 —— 绝不 killpg 一个可能含 agent 自身的组.
    整组回收交由 :func:`huginn.utils.process.kill_process_group` 统一实现 (sandbox /
    PersistentTerminal / 外部计算工具共用一套语义). 全部尽力而为, 失败不影响主流程
    (真正的回收兜底是随后的 ``communicate()``).
    """
    if not kill_process_group(proc.pid, own_group=own_group):
        with contextlib.suppress(Exception):
            proc.kill()


# Windows coreutils fallback 映射 (POSIX coreutil -> cmd 内置命令).
_WIN_COREUTIL_CMD_MAP = {
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


def _sb_dry_run_result(cmd: list[str]) -> SandboxResult:
    """dry-run 模式的返回值 (不实际执行命令)."""
    return SandboxResult(
        success=True,
        returncode=0,
        stdout="[dry-run] Command would execute: " + " ".join(cmd),
        stderr="",
        command=cmd,
        dry_run=True,
    )


def _sb_apply_profile_rlimit(profile: str) -> int | None:
    """Task 1.2: POSIX 软沙箱补 RLIMIT_AS 内存上限 (按 profile), 返回原 soft.

    Windows 跳过 (resource 模块不支持 RLIMIT_AS). 只降 soft limit, 不动 hard
    limit — 非 root 用户降了 hard 就再也升不回去. 失败不阻塞 (返回 None).
    """
    if os.name == "nt":
        return None
    saved: int | None = None
    try:
        import resource as _resource

        _mem = _profile_mem_bytes(profile)
        if _mem and _mem > 0:
            _cur_soft, _cur_hard = _resource.getrlimit(_resource.RLIMIT_AS)
            saved = _cur_soft
            _resource.setrlimit(_resource.RLIMIT_AS, (_mem, _cur_hard))
    except Exception:
        logger.debug(
            "setrlimit(RLIMIT_AS, profile=%s) failed (non-fatal)",
            profile,
            exc_info=True,
        )
    return saved


def _sb_restore_parent_rlimit(saved_soft: int) -> None:
    """恢复父进程 soft limit (hard limit 从未被降低, 无需恢复)."""
    try:
        import resource as _resource

        _cur_soft, _cur_hard = _resource.getrlimit(_resource.RLIMIT_AS)
        _resource.setrlimit(_resource.RLIMIT_AS, (saved_soft, _cur_hard))
    except Exception:
        logger.debug("failed to restore parent rlimit", exc_info=True)


def _sb_popen_kwargs(run_kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """组装 Popen kwargs, 并确定子进程是否自立新进程组 (供整组回收).

    调用方显式指定进程组行为时尊重之, 仅在其确实自立新组时才整组杀.
    """
    popen_kwargs = dict(run_kwargs)
    if "start_new_session" in popen_kwargs or "creationflags" in popen_kwargs:
        own_group = bool(popen_kwargs.get("start_new_session")) or bool(
            popen_kwargs.get("creationflags", 0)
            & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        return popen_kwargs, own_group
    default_kwargs, own_group = new_group_popen_kwargs()
    popen_kwargs.update(default_kwargs)
    return popen_kwargs, own_group


def _sb_timeout_result(
    proc: subprocess.Popen, cmd: list[str], own_group: bool
) -> SandboxResult:
    """超时收尾: 整组回收子进程, 回收僵尸并取回已产出的部分输出."""
    _kill_process_tree(proc, own_group=own_group)
    _out, _err = proc.communicate()
    return SandboxResult(
        success=False,
        returncode=-1,
        stdout=_out or "",
        stderr=_err or "",
        command=cmd,
        dry_run=False,
        timed_out=True,
    )


def _sb_execute(
    cmd: list[str],
    valid_cwd: Path | None,
    capture_output: bool,
    text: bool,
    env: dict[str, str] | None,
    popen_kwargs: dict[str, Any],
    own_group: bool,
    timeout: float,
    saved_rlimit_soft: int | None,
) -> tuple[subprocess.CompletedProcess | None, SandboxResult | None]:
    """阶段 5: 起子进程、超时整组回收、取回输出, 并恢复父进程 rlimit.

    返回 ``(CompletedProcess, None)`` 或 ``(None, 超时结果)``.
    """
    tracked_pid: int | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(valid_cwd) if valid_cwd else None,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            encoding="utf-8",
            errors="replace",
            env=env,
            shell=False,
            **popen_kwargs,
        )
        # 登记在册: 父进程若先退出 (autoloop CLI 无 lifespan 钩子), 退出兜底
        # ``kill_tracked_children()`` 才能按进程组整组回收, 不留 PPID→1 孤儿.
        track_live_child(proc.pid, own_group=own_group)
        tracked_pid = proc.pid
        try:
            _out, _err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, _sb_timeout_result(proc, cmd, own_group)
        return subprocess.CompletedProcess(cmd, proc.returncode, _out, _err), None
    finally:
        # 子进程已结束 (或已被整组回收) → 注销登记, 防止 pid 复用被退出兜底误杀.
        if tracked_pid is not None:
            untrack_live_child(tracked_pid)
        if saved_rlimit_soft is not None:
            _sb_restore_parent_rlimit(saved_rlimit_soft)


def _sb_truncate(value: str | bytes, max_bytes: int) -> str | bytes:
    """阶段 6: 超限输出截断 (bytes 按字节, str 按 UTF-8 字节估算后取 1/4 字符)."""
    if isinstance(value, bytes):
        if len(value) > max_bytes:
            return value[:max_bytes] + b"\n... [truncated]"
        return value
    if len(value.encode("utf-8", errors="replace")) > max_bytes:
        return value[: max_bytes // 4] + "\n... [truncated]"
    return value


def _sb_completed_result(
    completed: subprocess.CompletedProcess[str] | Any,
    cmd: list[str],
    max_bytes: int,
) -> SandboxResult:
    """把 subprocess 结果转成 SandboxResult (超限输出截断)."""
    stdout = _sb_truncate(completed.stdout or "", max_bytes)
    stderr = _sb_truncate(completed.stderr or "", max_bytes)
    return SandboxResult(
        success=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        command=cmd,
        dry_run=False,
    )


def _windows_fallback_executable(name: str) -> str | None:
    """Windows fallback: 映射 POSIX coreutils 到 cmd /c 内置命令."""
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
    return _WIN_FALLBACK.get(name.lower())


class SandboxExecutor:
    """Execute subprocess commands inside a security sandbox."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        profile: str = "light",
        allow_powershell: bool | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        # allow_powershell 参数显式覆盖 config 的开关: None=沿用 config 默认
        # (env HUGINN_ALLOW_POWERSHELL / 平台默认); 显式 True/False 强制收放,
        # 供调用方在不改全局环境的前提下按任务粒度开/关 PowerShell.
        if allow_powershell is not None:
            self.config.allow_powershell = allow_powershell
            self.config._sync_powershell()
        # T-BCSE-14: RevertibleEffect — 沙箱级可逆效应上下文 (Cordis 论文).
        # 每个经沙箱的副作用 (建文件/写 env/起进程/注册依赖) 都经它累积逆,
        # exit 时可整体回滚到沙箱建立前的状态. 见 huginn/security/revertible.py.
        from huginn.security.revertible import RevertibleContext

        self.revertible = RevertibleContext()
        # Task 1.2: 软沙箱按 profile 限内存 (POSIX RLIMIT_AS), 跚 Docker 档位对齐.
        # light=2g/standard=8g/heavy=16g, 复用 docker_sandbox._PROFILES 的 mem 字段.
        self.profile = profile
        # HUGINN_SANDBOX_RELAX=1 关闭 strict_work_dir (兼容老用户)
        if os.environ.get("HUGINN_SANDBOX_RELAX") == "1":
            self.config.strict_work_dir = False
        # strict=True 但 allowed_work_dirs 空 → path scoping 静默失效, 命令能在任意
        # cwd 落地 (ws9: 产物写进源码树). 缺省按 run workspace 建作用域; 显式给了
        # allowed_work_dirs 的调用方一律尊重, 不覆盖.
        if self.config.strict_work_dir and not self.config.allowed_work_dirs:
            from huginn.utils.common import default_workspace_dir

            self.config.allowed_work_dirs = {default_workspace_dir().resolve()}
            logger.debug(
                "allowed_work_dirs 未显式给出, 收敛到 %s",
                next(iter(self.config.allowed_work_dirs)),
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
            _win_cmd = _windows_fallback_executable(cmd[0])
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

    def _sb_validate_and_clamp(
        self,
        cmd: list[str],
        cwd: str | Path | None,
        timeout: float | None,
        config: SandboxConfig | None,
    ) -> tuple[SandboxConfig, Path | None, float]:
        """阶段 1-2: 命令/allow-list 校验 + 工作目录作用域检查 + 超时夹取."""
        cfg = config or self.config
        self._validate_command(cmd, config=cfg)
        valid_cwd = self._validate_cwd(cwd, config=cfg)

        # Clamp timeout
        if timeout is None:
            timeout = cfg.default_timeout
        timeout = min(float(timeout), cfg.max_timeout)
        return cfg, valid_cwd, timeout

    def _sb_landlock_rw_dirs(self, cfg: SandboxConfig) -> list[str]:
        """可写根: 仅 scoped 模式下给出, 否则空集 (不锁文件系统)."""
        if not (cfg.strict_work_dir and cfg.allowed_work_dirs):
            return []
        return [str(p) for p in cfg.allowed_work_dirs]

    def _sb_landlock_ro_dirs(
        self, cmd: list[str], cfg: SandboxConfig, valid_cwd: Path | None
    ) -> list[str]:
        """只读/执行根: (a) 工作目录 + (b) 白名单可执行文件自身所在目录.

        否则隔离后子进程读不到二进制本身而无法 exec (正确性缺陷).
        """
        scoped = bool(cfg.strict_work_dir and cfg.allowed_work_dirs)
        ro_dirs = [str(valid_cwd)] if (valid_cwd and scoped) else []
        if scoped:
            try:
                exe_abs = self._resolve_executable(cmd)
                exe_dir = str(Path(exe_abs).resolve().parent)
                if exe_dir not in ro_dirs:
                    ro_dirs.append(exe_dir)
            except Exception:
                logger.debug(
                    "landlock: exe dir resolution failed, confining cwd only",
                    exc_info=True,
                )
        return [d for d in ro_dirs if d]

    def _sb_inject_confinement(
        self,
        cmd: list[str],
        cfg: SandboxConfig,
        valid_cwd: Path | None,
        run_kwargs: dict[str, Any],
    ) -> None:
        """阶段 4: 内核级 confinement (Landlock FS/网络规则 + 子进程 rlimit).

        T-BCSE-07 / P1: 把 Landlock (FS 路径规则 + 可选网络隔离) 与子进程
        rlimit 合并成**单个** preexec_fn 注入, 只在 posix 且确有要收紧的东西时
        构造, 否则完全不注入 (零回归). 调用方已显式传 preexec_fn 时尊重其注入.
        """
        rlimits = _build_rlimits(cfg)
        want_confinement = bool(
            cfg.isolate_network
            or rlimits
            or (cfg.strict_work_dir and cfg.allowed_work_dirs)
        )
        if not (
            os.name == "posix" and want_confinement and "preexec_fn" not in run_kwargs
        ):
            return
        try:
            from huginn.security.landlock import make_preexec_fn

            # FS 隔离只在 scoped 模式 (strict_work_dir + allowed_work_dirs) 下
            # 生效; net-only / rlimit-only 时传空 path 集 — 不锁文件系统, 只收紧
            # 网络与资源. 这也避免"只开网络隔离"把子进程的文件访问一并封死.
            preexec = make_preexec_fn(
                self._sb_landlock_ro_dirs(cmd, cfg, valid_cwd),
                self._sb_landlock_rw_dirs(cfg),
                required=False,
                net_isolate=cfg.isolate_network,
                rlimits=rlimits,
            )
            if preexec is not None:
                run_kwargs["preexec_fn"] = preexec
        except Exception:
            logger.debug("landlock confinement skipped (non-fatal)", exc_info=True)

    @staticmethod
    def _sb_win_cmd_wrap(cmd: list[str]) -> list[str]:
        """Windows coreutils fallback: 白名单里的 POSIX coreutil 用 cmd /c 包裹.

        ``_resolve_executable`` 返回原 cmd[0] (e.g. "ls") 但 ``shutil.which``
        找不到 → 这里让 cmd.exe 走内置命令 (dir/copy/type/...). 白名单已含
        cmd.exe 间接调用, 不开新口子. 只在 cmd[0] 是 POSIX coreutil 时翻译,
        参数透传.
        ponytail: 用 cmd /c 单层包裹, 不递归. 天花板: cmd /c 参数语义跟 POSIX
          不完全一致 (e.g. ls -la vs dir), agent LLM 通常会自己适配 Windows 语法.
        """
        if os.name == "nt" and cmd and cmd[0].lower() in _WIN_COREUTIL_CMD_MAP:
            _mapped = _WIN_COREUTIL_CMD_MAP.get(cmd[0].lower(), cmd[0])
            # 重组: cmd /c <mapped> <rest args>
            return ["cmd", "/c", _mapped, *cmd[1:]]
        return cmd

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
        cfg, valid_cwd, timeout = self._sb_validate_and_clamp(cmd, cwd, timeout, config)
        if cfg.dry_run:
            return _sb_dry_run_result(cmd)
        # 密钥类变量 (API_KEY/TOKEN/...) 不进入子进程环境.
        env = sanitize_child_env(env)
        # Drop scheduler-only hints so they do not reach subprocess.run.
        run_kwargs = {k: v for k, v in kwargs.items() if k not in self._REMOTE_KWARGS}
        self._sb_inject_confinement(cmd, cfg, valid_cwd, run_kwargs)
        cmd = self._sb_win_cmd_wrap(cmd)
        # Task 1.2: POSIX 软沙箱补 RLIMIT_AS 内存上限 (按 profile, 与 Docker 档位对齐).
        _saved_rlimit_soft = _sb_apply_profile_rlimit(self.profile)

        # P0 修复 (超时孤儿泄漏): 子进程自立新会话/进程组, 超时由 _kill_process_tree 整组回收.
        _popen_kwargs, _own_group = _sb_popen_kwargs(run_kwargs)

        completed, timeout_result = _sb_execute(
            cmd,
            valid_cwd,
            capture_output,
            text,
            env,
            _popen_kwargs,
            _own_group,
            timeout,
            _saved_rlimit_soft,
        )
        if timeout_result is not None:
            return timeout_result

        # Truncate oversized output
        return _sb_completed_result(completed, cmd, self.config.max_output_bytes)

    @staticmethod
    def hash_data(data: str | bytes) -> str:
        """Return SHA-256 hex digest of data."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return hash_text(data)

    # ── RevertibleEffect (T-BCSE-14) ──────────────────────────────
    # 时间可组合: 沙箱副作用都携带逆, 由 self.revertible 累积, 可整体回滚.
    # 详见 huginn/security/revertible.py (Cordis 论文的 revertible effects).

    def transaction(self) -> Any:
        """事务边界: ``with sb.transaction():`` 块内异常自动回滚, 正常退出保留.

        用于把一组沙箱副作用包成原子单元 — 失败时沙箱状态恢复到块前.
        """
        return self.revertible.transaction()

    def create_file(self, path: str | Path, content: bytes | str | None = None) -> Path:
        """在沙箱内创建文件并登记逆 (移除时若原不存在则删除/原存在则恢复)."""
        return self.revertible.create_file(path, content)

    def create_dir(self, path: str | Path) -> Path:
        """在沙箱内创建目录并登记逆 (原本不存在且空时删除)."""
        return self.revertible.create_dir(path)

    def set_env(self, key: str, value: str, env: dict[str, str]) -> str | None:
        """设置进程环境变量并登记逆 (恢复旧值)."""
        return self.revertible.set_env(key, value, env)

    def run_revertible(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        poll_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> tuple[SandboxResult, Callable[[], None]]:
        """执行命令并返回 ``(result, dispose)``.

        与 ``run`` 相同, 但额外登记一个逆: 记录 ``poll_dir`` (默认取 ``cwd``)
        在执行前后新增的文件, ``dispose`` 把它们删除. 让调用方能在命令失败或
        放弃结果时把沙箱文件系统恢复原状 (时间可组合).
        """
        snap_dir = Path(poll_dir or cwd or Path.cwd())
        before = (
            {p for p in snap_dir.iterdir() if p.is_file()}
            if snap_dir.is_dir()
            else set()
        )

        result = self.run(cmd, cwd=cwd, **kwargs)

        after = (
            {p for p in snap_dir.iterdir() if p.is_file()}
            if snap_dir.is_dir()
            else set()
        )
        new_files = after - before

        def dispose() -> None:
            for p in new_files:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    logger.debug("revertible: failed to remove %s", p, exc_info=True)

        return result, dispose


def _build_rlimits(cfg: SandboxConfig) -> dict[int, tuple[int, int]] | None:
    """按配置组装子进程 ``{RLIMIT_*: (soft, hard)}``; 全空时返回 None.

    只在 preexec_fn 里 apply (只影响 fork 出的子进程), 不像 RLIMIT_AS 那样在
    父进程设了再恢复 — 多线程下无竞态. CPU 的 hard 取 soft+grace, 让进程收到
    SIGXCPU 后还有一点时间做收尾 (超时仍由 subprocess timeout 兜底).
    """
    if os.name == "nt":
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover - 非 POSIX
        return None

    limits: dict[int, tuple[int, int]] = {}
    if cfg.max_cpu_seconds:
        _cpu = int(cfg.max_cpu_seconds)
        limits[resource.RLIMIT_CPU] = (_cpu, _cpu + 5)
    if cfg.max_file_bytes:
        _fsize = int(cfg.max_file_bytes)
        limits[resource.RLIMIT_FSIZE] = (_fsize, _fsize)
    if cfg.max_processes:
        _nproc = int(cfg.max_processes)
        limits[resource.RLIMIT_NPROC] = (_nproc, _nproc)
    return limits or None


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


def _want_docker_backend(prefer_docker: bool) -> bool:
    """环境变量开关优先级: HUGINN_USE_DOCKER > HUGINN_DOCKER_SANDBOX > prefer_docker."""
    use_docker_env = os.environ.get("HUGINN_USE_DOCKER")
    if use_docker_env == "0":
        return False
    if use_docker_env == "1" or os.environ.get("HUGINN_DOCKER_SANDBOX") == "1":
        return True
    return prefer_docker


def _try_docker_executor(
    cfg: SandboxConfig,
    docker_image: str,
    profile: str,
    log: Any,
) -> Any | None:
    """构造 DockerSandboxExecutor; import 或构造失败时返回 None."""
    # 延迟 import，避免 docker SDK 没装时整个 sandbox 模块都加载不了
    try:
        from huginn.security.docker_sandbox import DockerSandboxExecutor
    except Exception:
        log.warning(
            "docker_sandbox import failed, falling back to subprocess "
            "sandbox (profile=%s)",
            profile,
            exc_info=True,
        )
        return None

    try:
        return DockerSandboxExecutor(image=docker_image, config=cfg, profile=profile)
    except Exception:
        # 构造失败也别让上层挂掉
        log.warning(
            "DockerSandboxExecutor init failed, falling back to "
            "subprocess sandbox (profile=%s)",
            profile,
            exc_info=True,
        )
        return None


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
    if not _want_docker_backend(prefer_docker):
        return SandboxExecutor(cfg, profile=profile)

    docker_executor = _try_docker_executor(cfg, docker_image, profile, _log)
    if docker_executor is None:
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
