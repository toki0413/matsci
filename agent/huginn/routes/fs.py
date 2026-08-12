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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

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
    if os.environ.get("HUGINN_ALLOW_UNRESTRICTED_READ", "").lower() in ("1", "true", "yes"):
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
    p = Path(raw).expanduser()
    try:
        resolved = p.resolve()
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"无法解析路径: {e}") from e
    if not _is_path_safe(resolved):
        raise HTTPException(status_code=403, detail="Access denied: 路径在允许访问的目录之外")
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


@router.get("/fs/read")
async def fs_read(path: str) -> dict[str, Any]:
    """读取文本文件内容。"""
    target = _safe_resolve(path)
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"不是文件: {path}")
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
