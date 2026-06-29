"""System state endpoints — expose runtime component status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_system_snapshot

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/components")
async def system_components() -> dict[str, Any]:
    """返回当前运行时各组件的初始化状态。

    用于诊断"系统是否就绪"——哪些组件已装配、哪些还缺着。
    同时把 HuginnSystem 全局单例同步为最新快照，其他模块可用
    huginn.system.get_system() 读取。
    """
    try:
        return {"success": True, **get_system_snapshot()}
    except Exception as e:
        return {"success": False, "error": str(e)}
