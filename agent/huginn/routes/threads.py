"""Conversation thread management endpoints."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from huginn.routes.schemas import CreateThreadRequest
from huginn.server_core import (
    _current_user_id,
    _state_lock,
    _threads,
    get_agent,
    get_or_create_thread,
    touch_thread,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["threads"])


def _check_thread_owner(thread_id: str, request: Request) -> dict[str, Any] | None:
    """Return error dict if the caller doesn't own the thread, None if OK.

    Dev/shared mode bypasses the check so existing single-user setups
    keep working without a configured user store.
    """
    user_id = _current_user_id(request)
    if user_id in ("dev", "shared", None):
        return None
    with _state_lock:
        thread = _threads.get(thread_id)
        if thread is None:
            return {"success": False, "error": "thread not found"}
        owner = thread.get("user_id")
        if owner and owner != user_id:
            return {"success": False, "error": "forbidden: thread belongs to another user"}
    return None


@router.get("/threads")
async def list_threads(request: Request, include_archived: bool = False) -> dict[str, Any]:
    """List known conversation threads."""
    user_id = _current_user_id(request)
    with _state_lock:
        threads = sorted(
            _threads.values(),
            key=lambda x: x.get("last_active", ""),
            reverse=True,
        )
    # Filter to the caller's own threads in multi-tenant mode. Dev/shared
    # sessions see everything for backward compat.
    if user_id not in ("dev", "shared", None):
        threads = [t for t in threads if not t.get("user_id") or t.get("user_id") == user_id]
    if not include_archived:
        threads = [t for t in threads if not t.get("archived", False)]
    return {
        "threads": [
            {
                "id": t["id"],
                "label": t.get("label", t["id"]),
                "created_at": t.get("created_at", ""),
                "last_active": t.get("last_active", ""),
                "archived": t.get("archived", False),
            }
            for t in threads
        ]
    }


@router.post("/threads")
async def create_thread(params: dict[str, Any], request: Request) -> dict[str, Any]:
    """Create a new conversation thread."""
    # Validate the request body — title length and metadata shape.
    try:
        req = CreateThreadRequest.model_validate(params)
    except ValidationError as exc:
        return {"error": f"Invalid request: {exc.errors()}"}

    thread_id = params.get("id") or uuid.uuid4().hex[:8]
    # Prefer the validated "title" field, fall back to "label" for
    # backward compat with clients that haven't migrated yet.
    label = req.title or params.get("label") or thread_id
    # Bind the thread to the authenticated caller so multi-tenant
    # deployments can isolate session data. No-op in dev / shared-key mode.
    user_id = _current_user_id(request)
    get_or_create_thread(thread_id, user_id=user_id, label=label)
    return {"id": thread_id, "label": label}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """Return metadata for a conversation thread."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id in _threads:
            return {"thread_id": thread_id, **dict(_threads[thread_id])}
    return {"thread_id": thread_id, "exists": False}


# LangChain message types don't map 1:1 to our frontend roles
_LC_ROLE_MAP = {"human": "user", "ai": "assistant", "tool": "tool"}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str, request: Request) -> dict[str, Any]:
    """Return message history for a thread from the LangGraph checkpointer."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    try:
        agent = await get_agent()
        graph = agent.build_graph()
        if graph is None:
            return {"messages": [], "thread_id": thread_id}
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        raw_msgs = snapshot.values.get("messages", []) if snapshot else []
    except Exception:
        logger.debug("failed to fetch thread state for %s", thread_id, exc_info=True)
        return {"messages": [], "thread_id": thread_id}

    messages: list[dict[str, Any]] = []
    for msg in raw_msgs:
        lc_type = getattr(msg, "type", "") or getattr(msg, "role", "")
        role = _LC_ROLE_MAP.get(lc_type)
        if role is None:
            continue  # skip system / unknown
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": "",
        }
        if role == "tool":
            entry["tool_name"] = getattr(msg, "name", "") or ""
            entry["tool_call_id"] = getattr(msg, "tool_call_id", "") or ""
        messages.append(entry)

    return {"messages": messages, "thread_id": thread_id}


@router.get("/threads/{thread_id}/events")
async def get_thread_events(
    thread_id: str,
    request: Request,
    after: int = -1,
) -> dict[str, Any]:
    """Return the incremental UI block model for a thread (T-BCSE-06).

    Serves the ``UiProjection`` read-model from the session event log. The
    frontend can poll ``?after=<last_seq>`` to get only blocks newer than that
    cursor — the incremental-sync primitive for the block-level renderer.

    Response: ``{ thread_id, blocks: [...], next_seq, leaf_id }``. ``blocks``
    are ``UiBlock`` dicts ``{kind, text, frozen, rev, seq}``; compaction and
    branch summaries arrive as their own ``frozen`` divider blocks so history
    is never dropped.
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    try:
        from huginn.events.projection import ProjectionEngine, UiProjection
        from huginn.events.session_log import SessionEventLog

        log = SessionEventLog.open(thread_id, load=True)
        engine = ProjectionEngine()
        engine.register(UiProjection())
        blocks = engine.build(log, "ui")
        if after >= 0:
            blocks = [b for b in blocks if b["seq"] > after]
        return {
            "thread_id": thread_id,
            "blocks": blocks,
            "next_seq": log.seq,
            "leaf_id": log.leaf_id,
        }
    except Exception:
        logger.debug("failed to serve session events for %s", thread_id, exc_info=True)
        return {"thread_id": thread_id, "blocks": [], "next_seq": 0, "leaf_id": None}


@router.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str, request: Request) -> dict[str, Any]:
    """返回 thread 的任务状态 (goal/mode/iteration/key_findings).

    前端 switchThread 时调, 恢复 mode 显示 + 显示研究进度.
    之前 switchThread 只拉 messages, 切回 research thread 后 mode/进度全丢.
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    try:
        from huginn.memory.task_state import get_tracker
        state = get_tracker().get(thread_id)
        return {
            "thread_id": thread_id,
            "goal": state.goal,
            "mode": state.mode,
            "iteration": state.iteration,
            "key_findings": list(state.key_findings[-10:]),
            "open_questions": list(state.open_questions[-5:]),
            "steps_done": sum(1 for s in state.steps if s.status == "done"),
            "steps_total": len(state.steps),
        }
    except Exception:
        logger.debug("failed to fetch task state for %s", thread_id, exc_info=True)
        return {"thread_id": thread_id, "goal": "", "mode": "chat", "iteration": 0}


@router.patch("/threads/{thread_id}")
async def rename_thread(thread_id: str, params: dict[str, Any], request: Request) -> dict[str, Any]:
    """Rename a thread."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}
        _threads[thread_id]["label"] = params.get("label", thread_id)
        return {"success": True, "label": _threads[thread_id]["label"]}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """Remove a thread from the registry."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id in _threads:
            del _threads[thread_id]
    return {"success": True}


@router.post("/threads/{thread_id}/archive")
async def archive_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """Archive a thread — hides it from the active list."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}
        _threads[thread_id]["archived"] = True
    return {"success": True, "archived": True}


@router.post("/threads/{thread_id}/unarchive")
async def unarchive_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """Unarchive a thread — restores it to the active list."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}
        _threads[thread_id]["archived"] = False
    return {"success": True, "archived": False}


@router.post("/threads/{thread_id}/fork")
async def fork_thread(thread_id: str, request: Request, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fork the conversation tree from the current position (or a given node)."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    agent = await get_agent()
    from_node_id = (params or {}).get("from_node_id")
    result = agent.fork_conversation(from_node_id=from_node_id)

    if result.get("success"):
        # touch_thread refreshes both last_active and last_accessed_ts so
        # the TTL sweeper treats this thread as recently used.
        touch_thread(thread_id)
    return {"thread_id": thread_id, **result}


@router.get("/threads/{thread_id}/branches")
async def list_branches(thread_id: str, request: Request) -> dict[str, Any]:
    """List all branches in the conversation tree for this thread."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    agent = await get_agent()
    branches = agent.conversation_branches()
    return {"thread_id": thread_id, **branches}


@router.post("/threads/{thread_id}/switch-branch")
async def switch_branch(thread_id: str, params: dict[str, Any], request: Request) -> dict[str, Any]:
    """Switch the active conversation path to end at the given node."""
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    node_id = params.get("node_id")
    if not node_id:
        return {"success": False, "error": "node_id is required"}

    agent = await get_agent()
    result = agent.switch_branch(node_id)

    if result.get("success"):
        touch_thread(thread_id)
    return {"thread_id": thread_id, **result}


@router.post("/threads/{thread_id}/event-branch")
async def event_branch(
    thread_id: str,
    params: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """事件级分支/回溯 (H2): 移动 SessionEventLog 的叶指针到 ``seq``.

    与 ``switch-branch`` (走内存 ConversationTree) 互补: 这里直接在事件日志
    上做 O(1) 叶指针移动 — 历史事件永不删, 后续 append 从新叶继续, 形成
    可回放/可合并的事件分支. 支持 ``seq`` (整数事件序号) 或事件 ``id``.

    Response: ``{ thread_id, leaf_id, next_seq, seq }``.
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    target = params.get("seq") or params.get("event_id")
    if target is None:
        return {"success": False, "error": "seq (int) or event_id (str) is required"}
    try:
        from huginn.events.session_log import SessionEventLog

        log = SessionEventLog.open(thread_id, load=True)
        leaf_id = log.branch(target)
        return {
            "thread_id": thread_id,
            "success": True,
            "leaf_id": leaf_id,
            "next_seq": log.seq,
            "seq": target,
        }
    except KeyError as exc:
        return {"success": False, "error": str(exc)}
    except Exception:
        logger.debug("event-branch failed for %s", thread_id, exc_info=True)
        return {"success": False, "error": "event branch failed"}


@router.get("/threads/{thread_id}/event-path")
async def event_path(thread_id: str, request: Request) -> dict[str, Any]:
    """事件级读取 (H2): 返回当前叶指针的事件路径 (root→leaf).

    路径上的 ``seq`` 序列可直接用于 ``event-branch`` 回溯/建分支.
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    try:
        from huginn.events.session_log import SessionEventLog

        log = SessionEventLog.open(thread_id, load=True)
        path = log.events_on_path()
        return {
            "thread_id": thread_id,
            "leaf_id": log.leaf_id,
            "next_seq": log.seq,
            "path": [
                {
                    "seq": ev.seq,
                    "kind": ev.kind,
                    "id": ev.id,
                    "parent_id": ev.parent_id,
                }
                for ev in path
            ],
        }
    except Exception:
        logger.debug("event-path failed for %s", thread_id, exc_info=True)
        return {"thread_id": thread_id, "leaf_id": None, "path": []}


@router.post("/threads/{thread_id}/compact")
async def compact_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """手动触发一次长会话智能压缩 (对标 Codex /compact).

    读取 checkpointer 里的会话历史, 把超出预算的旧消息让 LLM 折叠成一条摘要,
    再从 checkpointer 真删除旧消息, 摘要写入 agent 的会话摘要侧 (下轮注入
    system prompt, 跟运行时自动压缩同一条链路, 消息顺序因此不会被破坏).

    Response: ``{ success, thread_id, before, after, summarized, summary }``.
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err

    import asyncio

    from langchain_core.messages import RemoveMessage

    from huginn.utils.context import summarize_compact_messages

    try:
        agent = await get_agent()
        graph = agent.build_graph()
        if graph is None:
            return {"success": False, "error": "graph unavailable"}
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        raw_msgs = snapshot.values.get("messages", []) if snapshot else []
    except Exception:
        logger.debug("compact failed to load thread state for %s", thread_id, exc_info=True)
        return {"success": False, "error": "failed to load thread state"}

    # 太短没有压缩价值, 直接跳过 (与 _sliding_window_compact 同阈值).
    if len(raw_msgs) < 10:
        return {"success": True, "thread_id": thread_id, "summarized": 0}

    # 预算复用运行时配置; 摘要模型走与运行时相同的便宜模型分诊.
    budget = getattr(agent, "context_budget_tokens", 0) or 0
    summarizer = agent._make_summarizer()
    existing = agent._build_compact_summary() if hasattr(agent, "_build_compact_summary") else ""

    try:
        compacted, summary_text = await summarize_compact_messages(
            raw_msgs,
            budget,
            summarizer=summarizer,
            existing_summary=existing,
        )
    except Exception:
        logger.debug("summarize_compact_messages failed", exc_info=True)
        return {"success": False, "error": "compaction failed"}

    # 计算被折叠掉的旧消息 id, 用 LangGraph 官方 RemoveMessage + update_state
    # 真删 checkpointer 历史 (与运行时 G34 同一套, 避免 checkpoint 无限膨胀).
    keep_ids = {
        m.id for m in compacted if getattr(m, "id", None)
    }
    drop_ids = [
        m.id for m in raw_msgs
        if getattr(m, "id", None) and m.id not in keep_ids
    ]
    removed = 0
    if drop_ids and graph is not None:
        try:
            removals = [RemoveMessage(id=mid) for mid in drop_ids]
            await asyncio.to_thread(graph.update_state, config, {"messages": removals})
            removed = len(drop_ids)
        except Exception:
            logger.warning("checkpointer remove failed (compact endpoint)", exc_info=True)

    # 摘要进会话摘要侧, 下轮自动注入 system prompt.
    if summary_text:
        try:
            base = getattr(agent, "_conversation_summary", "") or ""
            agent._conversation_summary = (
                f"{base}\n{summary_text}".strip() if base else summary_text
            )
        except Exception:
            logger.debug("store conversation summary failed", exc_info=True)

    # 事件源记录一条 compaction divider, 让前端块模型渲染成 ── compacted ──.
    try:
        from huginn.events.session_writer import record_compaction
        record_compaction(thread_id, summary_text or "")
    except Exception:
        logger.debug("record_compaction skipped", exc_info=True)

    touch_thread(thread_id)
    return {
        "success": True,
        "thread_id": thread_id,
        "summarized": len(raw_msgs) - len(compacted),
        "removed_from_checkpointer": removed,
        "before": len(raw_msgs),
        "after": len(compacted),
        "summary": summary_text or "",
    }
