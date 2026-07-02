"""Side conversation HTTP routes.

暴露 ``/side`` 端点让前端在主 agent 忙的时候提交侧边问题, 之后轮询拿答案.
跟 Codex CLI 的 /side 对齐: 不打断主任务, agent 轮空时自己 drain.

端点:
- POST /side                  提交一个问题
- GET  /side                  列出全部 (pending + answered)
- GET  /side/pending          只看待答的
- GET  /side/answered         只看已答的
- GET  /side/{question_id}    查单个问题状态 / 答案
- DELETE /side                清空全部 (测试 / 重置用)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.side_conversation import get_shared_side_channel

router = APIRouter(tags=["side"])


@router.post("/side")
async def submit_side_question(params: dict[str, Any]) -> dict[str, Any]:
    """提交一个侧边问题. body: {question, metadata?}.

    返回 question id, 前端拿这个 id 去 GET /side/{id} 轮询答案.
    """
    question = str(params.get("question", "")).strip()
    if not question:
        return {"success": False, "error": "question is required"}
    metadata = params.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {"success": False, "error": "metadata must be an object"}
    channel = get_shared_side_channel()
    sq = channel.submit(question, metadata=metadata)
    return {"success": True, "question": sq.to_dict()}


@router.get("/side")
async def list_side_questions() -> dict[str, Any]:
    """列出全部侧边问题 (pending + answered)."""
    channel = get_shared_side_channel()
    items = [sq.to_dict() for sq in channel.list_all()]
    return {
        "success": True,
        "count": len(items),
        "n_pending": channel.n_pending,
        "n_answered": channel.n_answered,
        "questions": items,
    }


@router.get("/side/pending")
async def list_pending_side_questions() -> dict[str, Any]:
    """只看待答的问题."""
    channel = get_shared_side_channel()
    items = [sq.to_dict() for sq in channel.list_pending()]
    return {"success": True, "count": len(items), "questions": items}


@router.get("/side/answered")
async def list_answered_side_questions() -> dict[str, Any]:
    """只看已答的问题."""
    channel = get_shared_side_channel()
    items = [sq.to_dict() for sq in channel.list_answered()]
    return {"success": True, "count": len(items), "questions": items}


@router.get("/side/{question_id}")
async def get_side_question(question_id: str) -> dict[str, Any]:
    """查单个问题的状态 / 答案. 没找到返回 404-ish (success=False)."""
    channel = get_shared_side_channel()
    sq = channel.get(question_id)
    if sq is None:
        return {"success": False, "error": f"question '{question_id}' not found"}
    return {"success": True, "question": sq.to_dict()}


@router.delete("/side")
async def clear_side_questions() -> dict[str, Any]:
    """清空全部侧边问题. 测试和重置场景用."""
    channel = get_shared_side_channel()
    channel.clear()
    return {"success": True, "cleared": True}
