"""本地文件系统读写端点 (ADR-0001 文件 I/O 归口后端)。

桌面/CLI 不再通过 Tauri 命令行直接做本地文件 I/O，而是统一走后端 `/v1/fs/*`，
从而把 provenance / 审计 / 安全策略（路径白名单、敏感目录拦截）收口到一处。

安全基线：迁移自 desktop/src-tauri/src/main.rs 的 `is_path_safe`，保证
「用户可写区域放行、系统敏感目录与其他用户 profile 拦截」的原有语义不因
迁移而丢失；同时把 `HUGINN_ALLOW_UNRESTRICTED_READ=1` 视为显式放行（测试/CI）。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

# WSL ↔ Windows 路径转换: fs_read/fs_list/fs_open 在收到 Windows 或 \\wsl$ UNC
# 路径时先在 WSL 侧归一, 再走既有的 resolve/安全校验。纯函数, 不依赖 wsl 命令。
from huginn.utils import wslpath

router = APIRouter(tags=["fs"])

logger = logging.getLogger(__name__)

# 系统敏感目录（迁移自 Tauri `is_in_system_sensitive`）。大小写不敏感的包含匹配。
_SYSTEM_SENSITIVE = (
    "Windows\\System32",
    "Windows\\SysWOW64",
    "ProgramData",
    "$Recycle.Bin",
    "Boot",
    "etc\\ssh",
    ".ssh",
    ".gnupg",
    "AppData\\Roaming\\Microsoft\\Credentials",
    "AppData\\Roaming\\Microsoft\\Crypto",
    "AppData\\Local\\Microsoft\\Credential Manager",
)


def _is_system_sensitive(path: Path) -> bool:
    lossy = str(path).lower()
    return any(seg.lower() in lossy for seg in _SYSTEM_SENSITIVE)


def _is_path_safe(path: Path) -> bool:
    """与 Tauri `is_path_safe` 语义一致：用户可写区域放行，系统/profile 敏感区拦截。"""
    # 显式放行开关（测试 / 本地开发取证用）
    if os.environ.get("HUGINN_ALLOW_UNRESTRICTED_READ", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return True

    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        return not _is_system_sensitive(path)

    home_path = Path(home)
    # 拦截其他用户的 profile：位于 home 的父目录下、但不在当前 home 之下
    profiles_root = home_path.parent
    if (
        str(profiles_root) != str(home_path)
        and path.is_relative_to(profiles_root)
        and not path.is_relative_to(home_path)
    ):
        return False

    return not _is_system_sensitive(path)


def _safe_resolve(raw: str) -> Path:
    # Windows / \\wsl$ UNC 路径先转成 WSL 侧 (纯函数), 不影响既有 POSIX 行为:
    # 非 Windows/相对路径 to_wsl 原样返回; 绝对 POSIX 路径原样经过。
    converted = wslpath.to_wsl(raw)
    p = Path(converted).expanduser()
    try:
        resolved = p.resolve()
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"无法解析路径: {e}") from e
    if not _is_path_safe(resolved):
        raise HTTPException(
            status_code=403, detail="Access denied: 路径在允许访问的目录之外"
        )
    return resolved


@router.get("/fs/cwd")
async def fs_cwd() -> dict[str, Any]:
    """返回后端当前工作目录（单一事实源）。"""
    return {"path": os.getcwd()}


@router.get("/fs/list")
async def fs_list(path: str = ".") -> dict[str, Any]:
    """列出目录条目（dirs 在前，同层按名称排序）。"""
    base = _safe_resolve(path)
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {path}")
    entries: list[dict[str, Any]] = []
    try:
        for entry in base.iterdir():
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            entries.append({"name": entry.name, "path": str(entry), "is_dir": is_dir})
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"读取目录失败: {e}") from e
    return {"path": str(base), "entries": entries}


# 快速打开时要跳过的目录（构建产物 / 依赖 / 版本控制 / Python 缓存）
_SKIP_DIRS = {
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "out",
    "target",
    ".idea",
    ".vscode",
}

# /fs/read 整读上限 (字节). 超过则拒绝整读并引导走 file_read_tool 懒读窗口,
# 防止超大文件被 read_text 一次性载入导致内存/上下文暴涨. 可用环境变量覆盖.
_FS_READ_MAX_BYTES = 1024 * 1024


@router.get("/fs/branch")
async def fs_branch(path: str = ".") -> dict[str, Any]:
    """返回工作目录所在 Git 仓库的当前分支；非仓库时返回 null。

    状态栏展示用，读取只读的 .git/HEAD，不产生任何写操作。
    """
    base = _safe_resolve(path)
    try:
        head = base / ".git" / "HEAD"
        if not head.is_file():
            return {"branch": None}
        ref = head.read_text(encoding="utf-8", errors="replace").strip()
        if ref.startswith("ref: refs/heads/"):
            return {"branch": ref.split("refs/heads/", 1)[1]}
        return {"branch": "detached"}
    except OSError:
        return {"branch": None}


@router.get("/fs/search")
async def fs_search(path: str = ".", max_results: int = 500) -> dict[str, Any]:
    """递归列出现有工作区内的文件，供快速打开 / 全局文件定位使用。

    跳过构建产物与依赖目录以避免巨树。返回相对路径（相对工作区根），
    便于前端分类与打开。
    """
    base = _safe_resolve(path)
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {path}")
    files: list[dict[str, str]] = []
    try:
        for root, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
            for name in names:
                files.append({"name": name, "path": str(Path(root) / name)})
                if len(files) >= max_results:
                    break
            if len(files) >= max_results:
                break
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"遍历目录失败: {e}") from e
    return {"root": str(base), "files": files}


@router.get("/fs/read")
async def fs_read(path: str) -> dict[str, Any]:
    """读取文本文件内容.

    超大文件 (超过 HUGINN_FS_READ_MAX_SIZE_BYTES, 默认 1MB) 拒绝整读并返回
    明确提示, 避免 read_text 一次性读进内存导致内存/上下文暴涨. 大文件应改走
    file_read_tool 的 window_offset/window_size 懒读窗口.
    """
    target = _safe_resolve(path)
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"不是文件: {path}")
    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    max_bytes = int(
        os.environ.get("HUGINN_FS_READ_MAX_SIZE_BYTES", str(_FS_READ_MAX_BYTES))
    )
    if size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"文件过大: {size} bytes (上限 {max_bytes} bytes). "
                "为避免整读爆内存, 请改用 file_read_tool 的 "
                "window_offset/window_size 懒读窗口分段读取."
            ),
        )
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"读取文件失败: {e}") from e
    return {"path": str(target), "content": content}


@router.put("/fs/write")
async def fs_write(params: dict[str, Any]) -> dict[str, Any]:
    """写入文本文件内容（自动创建父目录）。"""
    path = params.get("path")
    content = params.get("content", "")
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    target = _safe_resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"写入文件失败: {e}") from e
    return {"path": str(target)}


@router.post("/fs/mkdir")
async def fs_mkdir(params: dict[str, Any]) -> dict[str, Any]:
    """新建目录（自动创建缺失的父目录）。已存在时幂等返回。"""
    path = params.get("path")
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    target = _safe_resolve(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"创建目录失败: {e}") from e
    return {"path": str(target)}


@router.put("/fs/rename")
async def fs_rename(params: dict[str, Any]) -> dict[str, Any]:
    """重命名或移动文件/目录（提供同目录新名或跨界新路径）。"""
    path = params.get("path")
    new = params.get("new")
    if not isinstance(path, str) or not path or not isinstance(new, str) or not new:
        raise HTTPException(status_code=400, detail="需要 path 与 new")
    src = _safe_resolve(path)
    dst_raw = new if not new.startswith(".") else str(src.parent / new)
    dst = _safe_resolve(dst_raw) if not new.startswith(".") else _safe_resolve(dst_raw)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"源不存在: {path}")
    try:
        src.rename(dst)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"重命名失败: {e}") from e
    return {"path": str(src), "new": str(dst)}


@router.delete("/fs/delete")
async def fs_delete(path: str) -> dict[str, Any]:
    """删除文件或目录（目录需为空，避免误删整棵树）。"""
    if not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    target = _safe_resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"目标不存在: {path}")
    try:
        if target.is_dir():
            target.rmdir()  # 仅删除空目录；非空抛错，防止级联误删
        else:
            target.unlink()
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"删除失败: {e}") from e
    return {"deleted": str(target)}


@router.post("/fs/open")
async def fs_open(params: dict[str, Any]) -> dict[str, Any]:
    """在系统文件管理器（Windows 资源管理器）中打开目录并选中目标。

    对 `\\\\wsl$\\<distro>\\...` UNC 路径先转成发行版内路径；若该发行版未
    挂载（路径不存在），给出可重连提示（带发行版探测结果）；普通 POSIX 路径
    行为与原来完全一致。
    """
    path = params.get("path")
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    target = _safe_resolve(path)
    if not target.exists():
        detail = f"目标不存在: {path}"
        hint = _wsl_unc_reconnect_hint(path)
        if hint:
            detail += f"。{hint}"
        raise HTTPException(status_code=404, detail=detail)
    try:
        reveal = str(target) if target.is_dir() else str(target.parent)
        (
            os.startfile(reveal)
            if sys.platform == "win32"
            else subprocess.Popen(
                ["open" if sys.platform == "darwin" else "xdg-open", reveal]
            )
        )
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"打开失败: {e}") from e
    return {"opened": str(target)}


def _wsl_unc_reconnect_hint(path: str) -> str:
    """为 `\\\\wsl$\\...` 路径生成重连提示 (探测发行版, 失败静默)。

    只有当原始输入是 WSL 网络共享 UNC 时才提示；其余路径返回空串, 不干扰
    既有的 404 行为。探测优先读 WSL_DISTRO_NAME, 其次才尝试调用 wsl.exe (探测
    失败不影响返回, 最坏情况给通用提示)。
    """
    if not wslpath.is_wsl_unc(path):
        return ""
    distro = wslpath.detect_default_distro()
    if not distro:
        found = wslpath.probe_wsl_distoros()
        if found:
            distro = found[0]
    if distro:
        return (
            f"WSL 发行版 '{distro}' 可能未运行或该路径在该发行版内不可访问。"
            "请先在终端执行 `wsl --shutdown && wsl --distribution <发行版>` 重连后重试。"
        )
    return (
        "该 WSL 发行版可能未运行，请在 Windows 中执行 "
        "`wsl --list --online` 确认发行版并 `wsl` 启动后重连。"
    )
