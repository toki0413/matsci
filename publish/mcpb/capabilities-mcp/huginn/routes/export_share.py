"""导出 / 导入 / 分享 — 统一 REST 端点.

提供全量导出、单组件导出、归档导入、状态查询等接口。
所有接口都在 /export 和 /import 前缀下。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from huginn.security.auth import require_api_key
from huginn.server_core import get_context

router = APIRouter(tags=["export_share"])

logger = logging.getLogger(__name__)


class ExportParams(BaseModel):
    format: str = "zip"
    include: list[str] | None = None

# 上传文件大小上限：500 MB
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024


def _get_manager():
    """从当前 ServerContext 拿到 workspace，构造 ExportShareManager."""
    from huginn.export_share import ExportShareManager

    workspace = get_context().config.workspace or "."
    return ExportShareManager(workspace)


@router.get("/export/status", dependencies=[Depends(require_api_key)])
async def export_status() -> dict[str, Any]:
    """检查当前有哪些数据可以导出."""
    try:
        mgr = _get_manager()
        return {"success": True, "status": mgr.get_export_status()}
    except Exception as e:
        logger.error("查询导出状态失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/export/all", dependencies=[Depends(require_api_key)])
async def export_all(params: ExportParams | None = None) -> Any:
    """导出全部数据，返回文件下载.

    请求体参数:
        format: "zip" / "tar.gz" / "json" (默认 zip)
        include: 要导出的组件列表，如 ["memory", "knowledge"]
                 默认全部
    """
    params = params or ExportParams()
    fmt = params.format
    include = params.include

    try:
        mgr = _get_manager()
        suffix = {"zip": ".zip", "tar.gz": ".tar.gz", "json": ".json"}.get(fmt, ".zip")
        output = Path(tempfile.gettempdir()) / f"huginn_export_all{suffix}"
        result = mgr.export_all(str(output), format=fmt, include=include)
        return FileResponse(
            result["path"],
            filename=Path(result["path"]).name,
            media_type="application/octet-stream",
        )
    except Exception as e:
        logger.error("全量导出失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/export/memory", dependencies=[Depends(require_api_key)])
async def export_memory(params: ExportParams | None = None) -> Any:
    """只导出长期记忆，返回文件下载或 JSON bytes."""
    params = params or ExportParams()
    fmt = params.format
    try:
        mgr = _get_manager()
        data = mgr.export_memory(format=fmt)
        if fmt == "json":
            return Response(
                content=data,
                media_type="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=huginn_memory.json"
                },
            )
        else:
            # zip 格式直接返回二进制
            return Response(
                content=data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=huginn_memory.zip"
                },
            )
    except Exception as e:
        logger.error("记忆导出失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/export/knowledge", dependencies=[Depends(require_api_key)])
async def export_knowledge(params: ExportParams | None = None) -> Any:
    """只导出知识库，返回文件下载或 JSON bytes."""
    params = params or ExportParams()
    fmt = params.format
    try:
        mgr = _get_manager()
        data = mgr.export_knowledge(format=fmt)
        if fmt == "json":
            return Response(
                content=data,
                media_type="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=huginn_knowledge.json"
                },
            )
        else:
            return Response(
                content=data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=huginn_knowledge.zip"
                },
            )
    except Exception as e:
        logger.error("知识库导出失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/import/all", dependencies=[Depends(require_api_key)])
async def import_all(
    file: UploadFile = File(...),
    merge: bool = True,
) -> dict[str, Any]:
    """从上传的归档文件导入数据.

    查询参数:
        merge: True=增量合并(默认)，False=覆盖
    """
    try:
        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            return {
                "success": False,
                "error": f"文件过大 ({len(content)} bytes, 上限 {_MAX_UPLOAD_BYTES})",
            }

        # 保存到临时文件
        suffix = Path(file.filename or "import.zip").suffix or ".zip"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tf:
            tf.write(content)
            tmp_path = tf.name

        try:
            mgr = _get_manager()
            result = mgr.import_all(tmp_path, merge=merge)
            return {"success": True, **result}
        finally:
            # 清理临时文件
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.error("导入失败", exc_info=True)
        return {"success": False, "error": str(e)}
