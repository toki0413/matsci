"""Kernel session 路由 — 有状态 Python 内核的 HTTP 接口.

所有端点都要 require_api_key 鉴权. 内核执行是阻塞 I/O, 走 asyncio.to_thread
不卡 event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from huginn.execution.kernel_session import KernelSessionManager
from huginn.security.auth import require_api_key, require_capability

logger = logging.getLogger(__name__)

# G32: kernel 跑任意 Python = 特权操作, 加 execute capability
router = APIRouter(
    prefix="/kernel",
    tags=["kernel"],
    dependencies=[Depends(require_api_key), Depends(require_capability("execute"))],
)

# 模块级单例管理器, 懒加载
_manager: KernelSessionManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> KernelSessionManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = KernelSessionManager()
    return _manager


# ── request bodies ────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    kernel_name: str = Field(default="python3")
    timeout: float = Field(default=30.0, gt=0.0)


class ExecuteRequest(BaseModel):
    code: str = Field(description="要执行的 Python 代码")
    silent: bool = Field(default=False)


# ── endpoints ──────────────────────────────────────────────────


@router.post("/session")
async def create_session(body: CreateSessionRequest) -> dict[str, Any]:
    """创建一个新的 kernel 会话."""
    mgr = _get_manager()
    # create 内部会阻塞 (启动子进程), 放线程里跑
    sess = await asyncio.to_thread(
        mgr.create,
        kernel_name=body.kernel_name,
        timeout=body.timeout,
    )
    return {
        "session_id": sess.session_id,
        "backend": sess.backend,
        "alive": sess.alive,
    }


@router.post("/{session_id}/execute")
async def execute_code(session_id: str, body: ExecuteRequest) -> dict[str, Any]:
    """在指定会话里执行代码, 返回输出."""
    mgr = _get_manager()
    sess = mgr.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    result = await asyncio.to_thread(sess.execute, body.code, body.silent)
    return {
        "session_id": session_id,
        "status": result.status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "images": result.images,
        "error": result.error,
    }


@router.get("/{session_id}/state")
async def get_state(session_id: str) -> dict[str, Any]:
    """获取会话当前的顶层变量列表."""
    mgr = _get_manager()
    sess = mgr.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    state = await asyncio.to_thread(sess.get_state)
    return {
        "session_id": session_id,
        "backend": sess.backend,
        "variables": state,
        "n_vars": len(state),
    }


@router.delete("/{session_id}")
async def close_session(session_id: str) -> dict[str, Any]:
    """关闭并移除一个会话."""
    mgr = _get_manager()
    ok = await asyncio.to_thread(mgr.close, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return {"session_id": session_id, "closed": True}


@router.get("")
async def list_sessions() -> dict[str, Any]:
    """列出所有活跃会话 (额外提供的管理端点)."""
    mgr = _get_manager()
    return {"sessions": mgr.list_sessions()}
