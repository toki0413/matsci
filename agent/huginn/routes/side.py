"""Side conversation HTTP routes.

两条链路并存:

1. **轻量问答队列** (``/side``) — 主 agent 轮空时 drain 的旁路问题, 不打断主任务.
2. **侧边任务** (``/side/tasks``) — 从主对话某条消息 fork 出来的独立 thread:
   继承 fork 点之前的上下文, 独立跑, 可一键转正为正式任务 (任务卡片 + 历史 +
   产物), 也可用完即弃 (thread 注册 + checkpointer 行一起删).

端点:
- POST /side                        提交一个旁路问题
- GET  /side                        列出全部旁路问题 (pending + answered)
- GET  /side/pending                只看待答的
- GET  /side/answered               只看已答的
- POST /side/tasks                  fork 出一个侧边任务
- GET  /side/tasks                  列出侧边任务 (可按 status 过滤, 顺带 TTL 清理)
- GET  /side/tasks/{task_id}        查单个侧边任务
- POST /side/tasks/{task_id}/messages   在侧边 thread 里继续对话
- POST /side/tasks/{task_id}/promote    转正为正式任务
- DELETE /side/tasks/{task_id}      丢弃 (零残留)
- GET  /side/{question_id}          查单个旁路问题状态 / 答案
- DELETE /side                      清空旁路问题 (测试 / 重置用)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from huginn.side_conversation import get_shared_side_channel
from huginn.side_tasks import (
    SideTask,
    extract_artifacts,
    get_shared_side_task_store,
    purge_thread_checkpoints,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["side"])

_LC_ROLE_MAP = {"human": "user", "ai": "assistant", "tool": "tool"}


# ── 轻量问答队列 ──────────────────────────────────────────────────


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


# ── 侧边任务: fork / 对话 / 转正 / 丢弃 ───────────────────────────
#
# 注意: 字面量路径 /side/tasks 必须声明在 /side/{question_id} 之前,
# 否则会被动态段吞掉.


async def _load_parent_messages(thread_id: str) -> list[dict[str, Any]]:
    """读父 thread 的 checkpointer 消息 (user/assistant/tool)."""
    from huginn.server_core import get_agent

    try:
        agent = await get_agent()
        graph = agent.build_graph()
        if graph is None:
            return []
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        raw = snapshot.values.get("messages", []) if snapshot else []
    except Exception:
        logger.debug("side task: load parent messages failed", exc_info=True)
        return []

    out: list[dict[str, Any]] = []
    for msg in raw:
        role = _LC_ROLE_MAP.get(getattr(msg, "type", "") or getattr(msg, "role", ""))
        if role is None:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        entry: dict[str, Any] = {"role": role, "content": content}
        mid = getattr(msg, "id", None)
        if mid:
            entry["id"] = mid
        out.append(entry)
    return out


async def _run_side_agent(thread_id: str, content: str) -> dict[str, Any]:
    """在侧边 thread 里跑一轮 agent, 返回最终回答 + 写类工具产物."""
    from huginn.server_core import get_agent_factory

    try:
        factory = get_agent_factory()
        agent = factory.create_lead(thread_id=thread_id)
    except Exception as exc:
        return {"success": False, "error": f"cannot create agent: {exc}"}

    if getattr(agent, "model", None) is None:
        return {"success": False, "error": "no LLM model configured"}

    final_text = ""
    raw_messages: list[Any] = []
    try:
        async for state in agent.chat(content, thread_id=thread_id):
            if not isinstance(state, dict):
                continue
            msgs = state.get("messages")
            if not msgs:
                continue
            raw_messages = list(msgs)
            for msg in msgs:
                if getattr(msg, "type", "") != "ai":
                    continue
                text = getattr(msg, "content", "")
                if isinstance(text, str) and text.strip():
                    final_text = text
    except Exception as exc:
        return {"success": False, "error": f"side agent failed: {exc}"}

    if not final_text.strip():
        return {"success": False, "error": "side agent returned an empty answer"}

    return {
        "success": True,
        "answer": final_text,
        "artifacts": extract_artifacts(raw_messages[-50:]),
    }


def _register_promoted_thread(task: SideTask, user_id: str | None) -> dict[str, Any]:
    """把转正后的侧边 thread 注册成正式任务 thread, 并写入任务卡片."""
    from huginn.server_core import _state_lock, _threads, get_or_create_thread

    get_or_create_thread(task.thread_id, user_id=user_id, label=task.title)
    with _state_lock:
        meta = _threads.get(task.thread_id)
        if meta is None:
            return {}
        updated = dict(meta)
        updated.update({
            "label": task.title,
            "kind": "task",
            "promoted_from": task.parent_thread_id,
            "task_card": task.task_card or {},
        })
        # 写回映射 — sqlite 后端 get 返回副本, 就地 update 不会持久化.
        _threads[task.thread_id] = updated
    return updated


def _forget_side_thread(thread_id: str) -> None:
    """零残留: 删 thread 注册 + checkpointer 行 (best-effort)."""
    from huginn.server_core import _state_lock, _threads

    with _state_lock:
        _threads.pop(thread_id, None)
    db_path: str | None = None
    try:
        from huginn.config import get_config

        db_path = get_config().checkpointer_path or None
    except Exception:
        logger.debug("side task: read checkpointer config failed", exc_info=True)
    try:
        purge_thread_checkpoints(thread_id, db_path=db_path)
    except Exception:
        logger.debug("side task: purge checkpoints failed", exc_info=True)


@router.post("/side/tasks")
async def create_side_task(params: dict[str, Any], request: Request) -> dict[str, Any]:
    """从父对话的某条消息 fork 出一个侧边任务.

    body: ``{parent_thread_id, from_message_index?, title?, inherit_context?}``

    ``from_message_index`` 是父对话 user/assistant 过滤后的下标 (含该条);
    省略则继承父对话全部上下文.
    """
    from huginn.server_core import (
        _current_user_id,
        _state_lock,
        _threads,
        get_or_create_thread,
    )

    parent = str(params.get("parent_thread_id", "")).strip()
    if not parent:
        return {"success": False, "error": "parent_thread_id is required"}

    # 父 thread 必须真实存在 — 不把"幽灵父线程"包装成空继承.
    with _state_lock:
        exists = parent in _threads
    if not exists:
        return {"success": False, "error": f"parent thread '{parent}' not found"}

    title = str(params.get("title", "")).strip()
    inherit = bool(params.get("inherit_context", True))
    idx_raw = params.get("from_message_index")
    from_idx: int | None = None
    if idx_raw is not None:
        try:
            from_idx = int(idx_raw)
        except (TypeError, ValueError):
            return {"success": False, "error": "from_message_index must be an integer"}

    messages = await _load_parent_messages(parent) if inherit else []
    try:
        task = get_shared_side_task_store().create(
            parent_thread_id=parent,
            messages=messages,
            from_message_index=from_idx,
            title=title,
            inherit_context=inherit,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    # 侧边 thread 立即注册, 让 WS / 消息读取链路能按 thread_id 工作.
    get_or_create_thread(task.thread_id, user_id=_current_user_id(request), label=task.title)
    with _state_lock:
        meta = _threads.get(task.thread_id)
        if meta is not None:
            updated = dict(meta)
            updated.update({
                "kind": "side",
                "parent_thread_id": parent,
                "fork_message_index": from_idx,
            })
            _threads[task.thread_id] = updated

    return {
        "success": True,
        "task": task.to_dict(),
        "inherited_messages": len(task.seed_messages),
    }


@router.get("/side/tasks")
async def list_side_tasks(status: str | None = None) -> dict[str, Any]:
    """列出侧边任务; 顺带清理过期侧边任务 (转正的不清理)."""
    store = get_shared_side_task_store()
    removed = store.sweep()
    for task in removed:
        _forget_side_thread(task.thread_id)
    tasks = store.list_tasks(status=status)
    return {
        "success": True,
        "count": len(tasks),
        "expired": [t.id for t in removed],
        "tasks": [t.to_dict() for t in tasks],
    }


@router.get("/side/tasks/{task_id}")
async def get_side_task(task_id: str) -> dict[str, Any]:
    task = get_shared_side_task_store().get(task_id)
    if task is None:
        return {"success": False, "error": f"side task '{task_id}' not found"}
    return {"success": True, "task": task.detail_dict()}


@router.post("/side/tasks/{task_id}/messages")
async def send_side_task_message(task_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """在侧边任务 thread 里发一条消息, 返回完整回答 (侧边任务不等流式)."""
    task = get_shared_side_task_store().get(task_id)
    if task is None:
        return {"success": False, "error": f"side task '{task_id}' not found"}
    content = str(params.get("content", "")).strip()
    if not content:
        return {"success": False, "error": "content is required"}

    result = await _run_side_agent(task.thread_id, content)
    if not result.get("success"):
        return result

    store = get_shared_side_task_store()
    store.append_turn(task_id, "user", content)
    store.append_turn(task_id, "assistant", result["answer"])
    store.add_artifacts(task_id, result.get("artifacts") or [])
    updated = store.get(task_id)
    return {
        "success": True,
        "answer": result["answer"],
        "artifacts": result.get("artifacts") or [],
        "task": updated.detail_dict() if updated else task.detail_dict(),
    }


@router.post("/side/tasks/{task_id}/promote")
async def promote_side_task(
    task_id: str, params: dict[str, Any], request: Request
) -> dict[str, Any]:
    """侧边任务转正: 生成任务卡片, thread 原地升级为正式任务."""
    from huginn.server_core import _current_user_id

    title = str(params.get("title", "")).strip()
    task = get_shared_side_task_store().promote(task_id, title=title)
    if task is None:
        return {
            "success": False,
            "error": f"side task '{task_id}' not found or already promoted",
        }
    _register_promoted_thread(task, _current_user_id(request))
    return {
        "success": True,
        "thread_id": task.thread_id,
        "task_card": task.task_card,
        "task": task.to_dict(),
    }


@router.delete("/side/tasks/{task_id}")
async def discard_side_task(task_id: str) -> dict[str, Any]:
    """丢弃侧边任务 — thread 注册与 checkpointer 行一起删 (零残留)."""
    task = get_shared_side_task_store().discard(task_id)
    if task is None:
        return {"success": False, "error": f"side task '{task_id}' not found"}
    _forget_side_thread(task.thread_id)
    return {"success": True, "discarded": task_id}


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

