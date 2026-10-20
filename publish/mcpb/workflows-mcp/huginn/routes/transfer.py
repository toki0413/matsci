"""文件传输管理路由 —— 仿 MobaXterm 的 SFTP 浏览器 + rsync 风格同步。

端点一览:
    POST /transfer/upload        上传本地文件到远端 (SFTP put, 服务端本地路径)
    POST /transfer/download      下载远端文件到本地 (SFTP get, 服务端本地路径)
    GET  /transfer/browse        浏览远端目录, 带文件大小 / 修改时间 / 类型
    POST /transfer/sync          本地目录同步到远端 (按 size+mtime 判断是否需要传)
    POST /transfer/web/upload    浏览器 multipart 上传体 → SFTP put 到远端
    GET  /transfer/web/download  远端文件 → 流式 HTTP 响应体

`/transfer/upload|download` 面向「服务端已有本地路径」的调用方 (CLI / 自动化);
`/transfer/web/*` 面向浏览器 (拿不到服务端本地路径), 一个收 multipart 上传体、
一个把远端内容流式吐回。两组都复用 HPCClient 已建好的 paramiko SFTP 通道。
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
import stat as _stat
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from huginn.hpc.client import HPCClient
from huginn.routes.hpc import _resolve_hpc_config
from huginn.security.auth import require_admin_key

router = APIRouter(tags=["transfer"], dependencies=[Depends(require_admin_key)])

logger = logging.getLogger(__name__)

# 浏览器侧单次上传 / 单块读写的大小口径 (浏览器传的是整文件, 走 SFTP 批量传输)
_WEB_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024
_WEB_TRANSFER_CHUNK_BYTES = 1024 * 1024


# ── 请求模型 ─────────────────────────────────────────────────────


class TransferRequest(BaseModel):
    """上传 / 下载请求的通用结构。"""

    credential_id: str
    local_path: str
    remote_path: str
    # 内联覆盖 (可选): 没传就走 credential_id 里的配置
    host: str | None = None
    username: str | None = None
    port: int = 22

    @field_validator("local_path", "remote_path")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("路径不能为空")
        return v.strip()


class BrowseRequest(BaseModel):
    """浏览远端目录的请求。"""

    credential_id: str
    path: str = "~"
    show_hidden: bool = True


class SyncRequest(BaseModel):
    """目录同步请求, rsync 风格。"""

    credential_id: str
    local_dir: str
    remote_dir: str
    # 是否删除远端多余文件 (危险操作, 默认关)
    delete: bool = False
    # 只同步匹配这些 glob 的文件 (空 = 全部)
    include_patterns: list[str] = []
    # 排除这些 glob 的文件
    exclude_patterns: list[str] = []


# ── 辅助: 构造 HPCConfig ─────────────────────────────────────────


def _build_cfg(body: dict[str, Any]):
    """统一走 hpc 路由的配置解析, 支持 credential_id + 内联覆盖。"""
    return _resolve_hpc_config(body)


def _validate_path_safety(path: str) -> None:
    """挡掉带 shell 注入字符和路径穿越的输入。"""
    for ch in (";", "`", "$(", "${"):
        if ch in path:
            raise ValueError(f"路径包含非法字符: {ch}")
    # block path traversal — .. can escape workspace root
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError("路径包含 .. 路径穿越")


# ── 路由 ─────────────────────────────────────────────────────────


@router.post("/transfer/upload")
async def transfer_upload(req: TransferRequest) -> dict[str, Any]:
    """上传本地文件到远端服务器。

    用 SFTP put, 大文件也走单连接不中断。远端父目录会自动创建。
    """
    _validate_path_safety(req.remote_path)

    local = Path(req.local_path).expanduser()
    if not local.is_file():
        return {"success": False, "error": f"本地文件不存在: {local}"}

    body = req.model_dump()
    cfg, err = _build_cfg(body)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    file_size = local.stat().st_size
    start = time.time()

    try:
        with HPCClient(cfg) as client:
            client.upload_file(str(local), req.remote_path)
    except Exception as exc:
        logger.warning("上传失败 %s -> %s: %s", local, req.remote_path, exc)
        return {"success": False, "error": str(exc)}

    elapsed = time.time() - start
    speed = (file_size / elapsed / 1024) if elapsed > 0 else 0
    return {
        "success": True,
        "local_path": str(local),
        "remote_path": req.remote_path,
        "host": cfg.host,
        "size": file_size,
        "elapsed_sec": round(elapsed, 2),
        "speed_kbps": round(speed, 1),
    }


@router.post("/transfer/download")
async def transfer_download(req: TransferRequest) -> dict[str, Any]:
    """从远端服务器下载文件到本地。

    用 SFTP get, 保存到 local_path。本地父目录不存在会自动建。
    """
    _validate_path_safety(req.remote_path)

    local = Path(req.local_path).expanduser()
    local.parent.mkdir(parents=True, exist_ok=True)

    body = req.model_dump()
    cfg, err = _build_cfg(body)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    start = time.time()

    try:
        with HPCClient(cfg) as client:
            client.download_file(req.remote_path, str(local))
    except Exception as exc:
        logger.warning("下载失败 %s -> %s: %s", req.remote_path, local, exc)
        return {"success": False, "error": str(exc)}

    file_size = local.stat().st_size if local.exists() else 0
    elapsed = time.time() - start
    speed = (file_size / elapsed / 1024) if elapsed > 0 else 0
    return {
        "success": True,
        "remote_path": req.remote_path,
        "local_path": str(local),
        "host": cfg.host,
        "size": file_size,
        "elapsed_sec": round(elapsed, 2),
        "speed_kbps": round(speed, 1),
    }


@router.get("/transfer/browse")
async def transfer_browse(
    credential_id: str,
    path: str = "~",
    show_hidden: bool = True,
) -> dict[str, Any]:
    """浏览远端目录, 返回带元信息的文件列表。

    每个条目包含 name / size / mtime / is_dir / permissions,
    前端可以据此渲染一个文件树。
    """
    _validate_path_safety(path)

    body = {"credential_id": credential_id}
    cfg, err = _build_cfg(body)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    try:
        with HPCClient(cfg) as client:
            client._ensure_connected()
            sftp = client._sftp
            # ~ 之类的主目录符号, 先 expand 一下
            expanded = _expand_remote_path(sftp, path)

            entries: list[dict[str, Any]] = []
            for item in sftp.listdir_attr(expanded):
                name = item.filename
                if not show_hidden and name.startswith("."):
                    continue
                entries.append({
                    "name": name,
                    "size": item.st_size or 0,
                    "mtime": item.st_mtime or 0,
                    "is_dir": _is_dir_attr(item),
                    "permissions": _mode_to_str(item.st_mode or 0),
                    "path": f"{expanded.rstrip('/')}/{name}",
                })

            # 目录排前面, 文件排后面, 各自按名字排
            entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

            return {
                "success": True,
                "path": expanded,
                "host": cfg.host,
                "entries": entries,
                "count": len(entries),
            }
    except FileNotFoundError:
        return {"success": False, "error": f"远端目录不存在: {path}"}
    except Exception as exc:
        logger.warning("浏览远端目录 %s 失败: %s", path, exc)
        return {"success": False, "error": str(exc)}


@router.post("/transfer/sync")
async def transfer_sync(req: SyncRequest) -> dict[str, Any]:
    """把本地目录同步到远端, rsync 风格。

    遍历本地目录树, 按文件 size + mtime 判断是否需要重新传;
    delete=True 时会删掉远端本地已不存在的文件 (谨慎用)。
    """
    _validate_path_safety(req.remote_dir)
    _validate_path_safety(req.local_dir)

    local_root = Path(req.local_dir).expanduser()
    if not local_root.is_dir():
        return {"success": False, "error": f"本地目录不存在: {local_root}"}

    body = req.model_dump()
    cfg, err = _build_cfg(body)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    uploaded: list[dict[str, Any]] = []
    skipped = 0
    deleted = 0
    errors: list[str] = []
    total_bytes = 0
    start = time.time()

    try:
        with HPCClient(cfg) as client:
            client._ensure_connected()
            sftp = client._sftp

            # 1. 遍历本地文件, 逐个判断要不要传
            local_files: list[Path] = []
            for fp in local_root.rglob("*"):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(local_root).as_posix()
                # include/exclude 过滤
                if req.include_patterns and not any(
                    fnmatch.fnmatch(rel, pat) for pat in req.include_patterns
                ):
                    continue
                if any(fnmatch.fnmatch(rel, pat) for pat in req.exclude_patterns):
                    continue
                local_files.append(fp)

            # 2. 上传需要更新的文件
            remote_file_map: dict[str, str] = {}
            for fp in local_files:
                rel = fp.relative_to(local_root).as_posix()
                remote_path = f"{req.remote_dir.rstrip('/')}/{rel}"
                remote_file_map[rel] = remote_path
                try:
                    # 比对远端文件: 不存在 / 大小不同 / mtime 更新 -> 传
                    need_upload = True
                    try:
                        rstat = sftp.stat(remote_path)
                        local_size = fp.stat().st_size
                        local_mtime = int(fp.stat().st_mtime)
                        if (
                            rstat.st_size == local_size
                            and (rstat.st_mtime or 0) >= local_mtime
                        ):
                            need_upload = False
                    except FileNotFoundError:
                        logger.debug("best-effort op failed", exc_info=True)  # 远端没有, 必须传

                    if not need_upload:
                        skipped += 1
                        continue

                    # 确保远端父目录存在
                    remote_parent = str(Path(remote_path).parent).replace("\\", "/")
                    _ensure_remote_dir(sftp, remote_parent)

                    sftp.put(str(fp), remote_path)
                    uploaded.append({
                        "rel_path": rel,
                        "remote_path": remote_path,
                        "size": fp.stat().st_size,
                    })
                    total_bytes += fp.stat().st_size
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")

            # 3. 可选: 删掉远端多余的文件
            if req.delete:
                deleted = _delete_extra_remote_files(
                    sftp, req.remote_dir, set(remote_file_map.keys())
                )

    except Exception as exc:
        logger.warning("同步失败 %s -> %s: %s", local_root, req.remote_dir, exc)
        return {"success": False, "error": str(exc)}

    elapsed = time.time() - start
    return {
        "success": True,
        "host": cfg.host,
        "local_dir": str(local_root),
        "remote_dir": req.remote_dir,
        "uploaded": uploaded,
        "uploaded_count": len(uploaded),
        "skipped": skipped,
        "deleted": deleted,
        "total_bytes": total_bytes,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
    }


# ── 浏览器形状: multipart 上传 / 流式下载 ──────────────────────────
#
# FilesPanel 的「上传」「下载」按钮拿到的是浏览器 `File` / `Blob`, 没有服务端
# 本地路径, 套不上 `/transfer/upload|download` 的 `local_path` 契约, 故单独给
# 浏览器一套: 上传收 multipart 上传体直接写进 SFTP, 下载把远端内容流式吐回。


@router.post("/transfer/web/upload")
async def transfer_web_upload(
    credential_id: str = Form(...),
    remote_dir: str = Form("~"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """浏览器 multipart 上传 → 远端 `remote_dir/<basename>`。

    文件名只取 basename (挡路径穿越); 远端父目录按需创建; 半截失败会清掉远端
    残留文件。阻塞的 SFTP 写入放线程里跑, 不占事件循环。
    """
    name = Path(file.filename or "").name.strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="缺少合法的上传文件名")
    try:
        _validate_path_safety(remote_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cfg, err = _build_cfg({"credential_id": credential_id})
    if err or cfg is None:
        raise HTTPException(status_code=400, detail=err or "无法解析 HPC 配置")

    # Starlette 已把 multipart 落进 spooled 临时文件, 定位到尾部量一下大小
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"读取上传体失败: {e}") from e
    if size > _WEB_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {size} bytes (上限 {_WEB_UPLOAD_MAX_BYTES} bytes)",
        )

    def _put() -> tuple[str, str]:
        with HPCClient(cfg) as client:
            client._ensure_connected()
            sftp = client._sftp
            expanded = _expand_remote_path(sftp, remote_dir)
            _ensure_remote_dir(sftp, expanded)
            dest = f"{expanded.rstrip('/')}/{name}"
            try:
                with sftp.file(dest, "wb") as rf:
                    while True:
                        buf = file.file.read(_WEB_TRANSFER_CHUNK_BYTES)
                        if not buf:
                            break
                        rf.write(buf)
            except Exception:
                # 写一半失败 → 清掉远端半截文件, 不留垃圾
                with contextlib.suppress(Exception):
                    sftp.remove(dest)
                raise
            return dest, expanded

    try:
        dest, _expanded = await asyncio.to_thread(_put)
    except Exception as exc:
        logger.warning("浏览器上传 %s -> %s 失败: %s", name, remote_dir, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "success": True,
        "name": name,
        "remote_path": dest,
        "host": cfg.host,
        "size": size,
    }


@router.get("/transfer/web/download")
async def transfer_web_download(credential_id: str, path: str) -> StreamingResponse:
    """远端文件 → 浏览器流式下载 (SFTP → HTTP 响应体)。

    先探一次 stat 好把「文件不存在」变成干净的 404 (响应一旦开始就没有回头路),
    再分块读 SFTP 文件句柄 —— Starlette 在线程池里跑同步生成器, 不阻塞事件循环。
    """
    try:
        _validate_path_safety(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cfg, err = _build_cfg({"credential_id": credential_id})
    if err or cfg is None:
        raise HTTPException(status_code=400, detail=err or "无法解析 HPC 配置")

    def _open_and_stat() -> tuple[str, int]:
        with HPCClient(cfg) as client:
            client._ensure_connected()
            sftp = client._sftp
            expanded = _expand_remote_path(sftp, path)
            return expanded, (sftp.stat(expanded).st_size or 0)

    try:
        expanded, size = await asyncio.to_thread(_open_and_stat)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"远端文件不存在: {path}") from exc
    except Exception as exc:
        logger.warning("浏览器下载 %s 失败: %s", path, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _iter():
        with HPCClient(cfg) as client:
            client._ensure_connected()
            sftp = client._sftp
            with sftp.file(expanded, "rb") as rf:
                while True:
                    chunk = rf.read(_WEB_TRANSFER_CHUNK_BYTES)
                    if not chunk:
                        break
                    yield chunk

    filename = _sanitize_header_filename(Path(path).name or "download")
    return StreamingResponse(
        _iter(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(size),
        },
    )


# ── 内部工具函数 ─────────────────────────────────────────────────


def _sanitize_header_filename(name: str) -> str:
    """Content-Disposition 文件名去掉引号 / 反斜杠 / 换行, 防 header 注入。"""
    cleaned = "".join(c for c in name if c not in '"\\\r\n')
    return cleaned.strip() or "download"


def _expand_remote_path(sftp: Any, path: str) -> str:
    """展开远端路径里的 ~ 等主目录符号。

    SFTP 的 normalize 方法会把 ~ 和相对路径转成绝对路径。
    """
    if path.startswith("~"):
        # paramiko SFTP 没有 expanduser, 用 normalize 拿真实 home
        try:
            return sftp.normalize(path)
        except Exception:
            # 退而求其次: 用 $HOME 替换
            return path.replace("~", ".", 1)
    try:
        return sftp.normalize(path)
    except Exception:
        return path


def _is_dir_attr(attr: Any) -> bool:
    """判断 SFTP 目录条目是不是目录。"""
    mode = attr.st_mode or 0
    return _stat.S_ISDIR(mode)


def _mode_to_str(mode: int) -> str:
    """把 stat mode 转成 rwxr-xr-x 字符串, 前端展示用。"""
    if mode == 0:
        return "---------"
    parts = []
    for who in (_stat.S_IRUSR, _stat.S_IWUSR, _stat.S_IXUSR,
                _stat.S_IRGRP, _stat.S_IWGRP, _stat.S_IXGRP,
                _stat.S_IROTH, _stat.S_IWOTH, _stat.S_IXOTH):
        parts.append(bool(mode & who))
    perms = ["r", "w", "x"]
    return "".join(p if b else "-" for b, p in zip(parts, perms))


def _ensure_remote_dir(sftp: Any, remote_dir: str) -> None:
    """递归创建远端目录, 类似 mkdir -p。

    path 里可能是 ~ 或绝对路径, 用 normalize 统一一下。
    """
    parts = [p for p in remote_dir.split("/") if p and p != "."]
    prefix = "/" if remote_dir.startswith("/") else ""
    for i in range(1, len(parts) + 1):
        cur = prefix + "/".join(parts[:i])
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            # 并发创建时可能已经被建了, 忽略
            with contextlib.suppress(OSError):
                sftp.mkdir(cur)


def _delete_extra_remote_files(
    sftp: Any, remote_root: str, keep_rels: set[str]
) -> int:
    """删掉远端目录里本地已不存在的文件, 返回删除数量。

    只删文件不删目录, 避免误删整个目录结构。
    """
    deleted = 0
    try:
        for entry in sftp.listdir_attr(remote_root):
            remote_path = f"{remote_root.rstrip('/')}/{entry.filename}"
            if _stat.S_ISDIR(entry.st_mode or 0):
                continue
            rel = entry.filename
            if rel not in keep_rels:
                try:
                    sftp.remove(remote_path)
                    deleted += 1
                except OSError:
                    logger.debug("best-effort op failed", exc_info=True)
    except Exception as exc:
        logger.debug("清理远端多余文件失败: %s", exc)
    return deleted
