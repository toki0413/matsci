"""WSL ↔ Windows 路径双向转换工具（纯函数，无平台副作用）。

背景: huginn 后端主要跑在 WSL / Linux 里, 但用户可能给出 Windows 侧路径
(``C:\\Users\\al\\...``) 或 WSL 网络共享 UNC 路径 (``\\wsl$\\Ubuntu\\home\\al\\...``),
直接扔给 ``Path.resolve()`` 既解析不出文件又丢失语义。本模块提供:

- :func:`is_wsl`: 环境/系统判定是否在 WSL
- :func:`to_wsl`: Windows 或 UNC 路径 → WSL 侧路径 (/mnt/c/...) 或发行版内路径
- :func:`to_windows`: WSL 侧 /mnt/<drive> 或发行版内路径 → Windows 路径
- :func:`detect_default_distro` / :func:`probe_wsl_distoros`: 发行版探测

设计约束:
1. 核心转换 (``to_wsl``/``to_windows``) 是纯函数, **不依赖** wsl.exe / wslpath 存在;
   wslpath / wsl 探测只在显式调用时 (包装了 try/except, 失败静默降级) 触发。
2. 不识别/相对/非 WSL 路径一律**原样返回**, 不臆造转换, 保证既有的 POSIX
   行为不受影响。

转换规则:
- `/mnt/<drive>/<rest>` → `<DRIVE>:\\<rest>`  (automount root 默认 /mnt, 可配置)
- `<DRIVE>:\\<rest>` 或 `<DRIVE>/<rest>` → `<automount>/<drive_lower>/<rest>`
- `\\wsl$\\<distro>\\<rest>` / `\\wsl.localhost\\<distro>\\<rest>` → `/<rest>`
- 其余空格路径: 若在 WSL 内且知道发行版, 映射为 `\\wsl$\\<distro>\\<rest>`;
  否则原样返回。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 环境注入开关: 允许在测试/非 WSL 环境强制模拟判定, 便于单测 is_wsl。
# 取值: "1"/"true"/"yes"→强制 WSL; "0"/"false"/"no"→强制非 WSL; 其他/未设→自动探测。
_WSL_FORCE_ENV = "HUGINN_FORCE_WSL"

# 默认 automount root (WSL 把 Windows 盘符挂到 /mnt/<drive>)。
_DEFAULT_MOUNT_ROOT = "/mnt"

# `\\wsl$\Ubuntu\rest` 或 `\\wsl.localhost\Ubuntu\rest`:
# 前缀 `\\`(2 个)+ `wsl$` + 单 `\` + distro + 单 `\` + rest
_UNC_RE = re.compile(r"^\\\\wsl(?:\$|\.localhost)\\([^\\/]+)\\(.*)$", re.IGNORECASE)
_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\\\/](.*)$")


def is_wsl() -> bool:
    """判定当前进程是否运行在 WSL 环境。

    先检查环境注入开关 ``HUGINN_FORCE_WSL`` (见模块 docstring), 未设置时自动探测:
    ``/proc/version`` 含 "microsoft" (WSL 内核名) 或存在 ``/etc/wsl.conf``。
    """
    force = os.environ.get(_WSL_FORCE_ENV)
    if force is not None:
        v = force.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return _detect_wsl_from_system()


def _detect_wsl_from_system() -> bool:
    try:
        version = Path("/proc/version").read_text(errors="replace")
        if "microsoft" in version.lower() or "wsl" in version.lower():
            return True
    except OSError:
        pass
    try:
        return Path("/etc/wsl.conf").is_file()
    except OSError:
        return False


def _automount_root() -> Path:
    """返回 WSL 的 Windows 盘符挂载根 (默认 /mnt, 读取 /etc/wsl.conf [automount] root)。

    非 WSL 环境直接返回默认 /mnt, 保证纯函数在测试里行为可预测。
    """
    if not is_wsl():
        return Path(_DEFAULT_MOUNT_ROOT)
    try:
        text = Path("/etc/wsl.conf").read_text(errors="replace")
    except OSError:
        return Path(_DEFAULT_MOUNT_ROOT)
    in_automount = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            in_automount = line.lower().startswith("[automount]")
            continue
        if in_automount and line.startswith("root"):
            key, _, value = line.partition("=")
            if key.strip().lower() == "root" and value.strip():
                return Path(value.strip())
    return Path(_DEFAULT_MOUNT_ROOT)


def detect_default_distro() -> str | None:
    """返回当前默认发行版名 (WSL_DISTRO_NAME), 无则 None。纯环境读取。"""
    name = os.environ.get("WSL_DISTRO_NAME")
    return name.strip() if name and name.strip() else None


def probe_wsl_distoros() -> list[str]:
    """尽力探测可用发行版 (调用 wsl.exe --list --quiet).

    只在 wsl.exe 存在时尝试; 失败/超时/无命令均可静默降级返回空列表。
    探测结果用于 wsl$ UNC 的重连提示, 核心转换不依赖它。
    """
    exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--list", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - 环境相关
        logger.debug("probe wsl distros failed (non-fatal): %s", exc)
        return []
    distros = []
    for line in (out or "").splitlines():
        name = line.strip().strip("\x00")
        if name and not name.lower().startswith(("wsl is", "windows", "the following")):
            distros.append(name)
    return distros


def _is_unc(path: str) -> bool:
    """判断路径是否为 `\\\\wsl$\\<distro>\\...` 或 `\\\\wsl.localhost\\<distro>\\...`。"""
    return _UNC_RE.match(path) is not None


def is_wsl_unc(path: str) -> bool:
    """公开判断: 该路径是否为 WSL 网络共享 UNC 路径 (用于 fs_open 重连提示)。"""
    return _is_unc(path)


def to_wsl(path: str) -> str:
    """Windows / UNC 路径 → WSL 侧路径。

    优先级(纯函数, 不依赖 wsl 命令):
    1. ``\\\\wsl$\\<distro>\\<rest>`` / ``\\\\wsl.localhost\\<distro>\\<rest>``
       → ``/<rest>`` (发行版文件系统根路径)
    2. ``<DRIVE>:\\<rest>`` / ``<DRIVE>/<rest>`` → ``<automount_root>/<drive_lower>/<rest>``
    3. 其余 (相对 / 非 Windows) → 原样返回, 绝不臆造转换。
    """
    if not isinstance(path, str) or not path:
        return path

    m = _UNC_RE.match(path)
    if m:
        _, rest = m.group(1), m.group(2)
        return "/" + rest.replace("\\", "/").lstrip("/")

    m = _DRIVE_RE.match(path)
    if m:
        drive, rest = m.group(1), m.group(2)
        normalized = rest.replace("\\", "/").strip("/")
        base = str(_automount_root()).rstrip("/")
        if not normalized:
            return f"{base}/{drive.lower()}"
        return f"{base}/{drive.lower()}/{normalized}"

    return path


def to_windows(path: str) -> str:
    """WSL 侧路径 → Windows 路径。

    优先级(纯函数):
    1. ``<automount_root>/<drive>/<rest>`` (例如 ``/mnt/c/Users/al``) → ``<DRIVE>:\\<rest>``
    2. 其他 WSL 绝对路径 (非挂载盘) 且处于 WSL 内且已知发行版
       → ``\\\\wsl$\\<distro>\\<rest>`` (壁路径映射)
    3. 其余 (相对 / 非 WSL 路径) → 原样返回。
    """
    if not isinstance(path, str) or not path:
        return path
    if not path.startswith("/") and not _DRIVE_RE.match(path):
        return path

    root = _automount_root()
    root_str = str(root).rstrip("/")
    if path == root_str or path.startswith(root_str + "/"):
        rest = path[len(root_str) :].strip("/")
        parts = rest.split("/")
        if parts and len(parts[0]) == 1 and parts[0].isalpha():
            drive = parts[0].upper()
            tail = "\\".join(p for p in parts[1:] if p)
            return f"{drive}:\\{tail}" if tail else f"{drive}:\\"

    # 不在挂载盘下: 若确在 WSL 内且能确定发行版, 映射为 wsl$ UNC。
    if is_wsl():
        distro = detect_default_distro()
        if distro:
            rest = path.strip("/")
            return f"\\\\wsl$\\{distro}\\{rest.replace('/', chr(92))}"
    return path
