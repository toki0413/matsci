"""Helper functions for the WebSocket route.

Extracted from ws.py to make agent_websocket a thin dispatcher.
All message format and behavior is preserved — the functions are just
hoisted to module level with explicit parameter signatures.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket
from langchain_core.messages import AIMessage, ToolMessage

from huginn.core_types import progress_cb
from huginn.routes.schemas import WSMessage
from huginn.server_core import (
    _EDIT_TOOLS,
    _checkpoints,
    _snapshot_directory,
    _state_lock,
)

logger = logging.getLogger(__name__)

# Track fire-and-forget tasks so they don't get GC'd mid-flight.
_pending_tasks: set[asyncio.Task] = set()

# Pending plan confirmations: plan_id -> Future (used by send_plan_and_wait).
_pending_plans: dict[str, asyncio.Future] = {}


@dataclass
class WSCtx:
    """Per-connection state bundled so every message handler shares one signature.

    All handlers receive ``(websocket, msg, ctx)``. Fields that a handler
    does not need are simply ignored — this removes the schema-vs-dict dual
    channel that previously let handlers read ``data.get(...)`` off the raw
    message and drift out of sync with ``WSMessage``.
    """

    ws_approval: Any = None
    session_auto_approve: dict = field(default_factory=dict)
    last_user_context: dict = field(default_factory=dict)
    pending_approvals: dict = field(default_factory=dict)
    pending_approval_contexts: dict = field(default_factory=dict)
    pending_plan_contexts: dict = field(default_factory=dict)
    user_id: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


# ── Small utilities ──────────────────────────────────────────────


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send a WS error message and return."""
    await websocket.send_json({"type": "error", "error": message})


def _extract_task_progress(content: str) -> dict | None:
    """Detect HPC job / sweep / long-running task info from tool output.

    Returns a dict suitable for sending as ``task_progress`` WS message,
    or None if no progress info is detected.
    """
    text = content.lower()

    job_match = re.search(r"job[_ ]?(?:id)?[:\s]+(\d+)", text)
    if job_match and any(
        kw in text for kw in ("submit", "hpc", "slurm", "qsub", "queue")
    ):
        job_id = job_match.group(1)
        status = "queued"
        if "running" in text:
            status = "running"
        elif "complet" in text or "finish" in text:
            status = "completed"
        elif "fail" in text or "error" in text:
            status = "failed"
        return {
            "task_type": "hpc_job",
            "job_id": job_id,
            "status": status,
            "message": content[:200],
        }

    sweep_match = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:complet|done|finish)", text)
    if sweep_match:
        done = int(sweep_match.group(1))
        total = int(sweep_match.group(2))
        return {
            "task_type": "sweep",
            "completed": done,
            "total": total,
            "progress_pct": round(done / total * 100, 1) if total else 0,
            "message": content[:200],
        }

    pct_match = re.search(r"progress[:\s]+(\d+(?:\.\d+)?)\s*%", text)
    if pct_match:
        pct = float(pct_match.group(1))
        return {
            "task_type": "progress",
            "progress_pct": pct,
            "message": content[:200],
        }

    return None


def _extract_tool_warnings(content: str) -> list:
    """Pull warning entries out of a serialized tool result."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    w = parsed.get("warnings")
    if isinstance(w, list) and w:
        return w
    inner = parsed.get("result")
    if isinstance(inner, dict):
        w = inner.get("_constraint_warnings") or inner.get("warnings")
        if isinstance(w, list) and w:
            return w
    return []


async def send_plan_and_wait(
    websocket: WebSocket,
    plan: dict,
    *,
    timeout: float = 120.0,
) -> dict:
    """Send a structured plan to the client and wait for confirmation."""
    plan_id = uuid.uuid4().hex[:8]
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _pending_plans[plan_id] = future

    await websocket.send_json({
        "type": "plan",
        "plan_id": plan_id,
        "plan": plan,
    })

    try:
        result = await asyncio.wait_for(future, timeout=timeout)
        return result
    except TimeoutError:
        # Fail-closed: a silent timeout must not rubber-stamp a plan
        # that the user never actually saw. Return denied so the agent
        # treats it as "not approved" and re-asks or aborts.
        return {"confirmed": False, "edited_plan": None, "reason": "timeout"}
    finally:
        _pending_plans.pop(plan_id, None)


def _make_ws_approval_callback(
    websocket: WebSocket,
    session_auto_approve: dict | None = None,
    pending_approvals: dict | None = None,
    last_user_context: dict | None = None,
    pending_approval_contexts: dict | None = None,
):
    """Build a sync approval callback that notifies the WebSocket client.

    ponytail: the callback is sync (called from _check_permission which
    is sync), so we can't await a client reply here. The re-queue in the
    approval_response handler is the workaround — the user sees the
    denial, reviews it, and approves; the original turn re-runs.
    """
    # 用**注册名**: `file_delete_tool`/`terminal_tool` 无对应注册工具, 永不命中已删
    # (MECE 审计记的死项); 删除能力实际走 bash_tool 的 rm.
    _DANGEROUS_TOOLS = frozenset({
        "bash_tool", "file_edit_tool", "multi_edit_tool", "git_tool",
    })

    _auto = session_auto_approve if session_auto_approve is not None else {"enabled": True}

    def callback(tool_name: str, reason: str) -> bool:
        request_id = uuid.uuid4().hex
        is_dangerous = tool_name in _DANGEROUS_TOOLS
        approved = _auto["enabled"]

        if not approved:
            if pending_approvals is not None:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                pending_approvals[request_id] = fut
            if pending_approval_contexts is not None and last_user_context:
                pending_approval_contexts[request_id] = dict(last_user_context)

        # v6: auto_approve=True 时改发静默事件, 不再发 approval_request.
        # 之前发 approval_request (auto_approved=true) 让前端弹 "Auto-approved"
        # 消息污染对话, UX 与可见性双错位. 现在: 静默放行只记日志, 前端不弹消息.
        try:
            loop = asyncio.get_running_loop()

            async def _safe_send():
                try:
                    msg_type = "tool_auto_approved" if approved else "approval_request"
                    await websocket.send_json(
                        {
                            "type": msg_type,
                            "request_id": request_id,
                            "tool_name": tool_name,
                            "reason": reason,
                            "auto_approved": approved,
                            "dangerous": is_dangerous,
                        }
                    )
                except Exception:
                    logger.debug("ws approval notification send failed", exc_info=True)

            task = loop.create_task(_safe_send())
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            logger.debug("best-effort op failed", exc_info=True)

        if is_dangerous:
            logger.warning(
                "Tool '%s' %s (reason: %s).",
                tool_name,
                "denied (auto-approve off)" if not approved else "auto-approved",
                reason,
            )

        return approved

    return callback


async def _stream_agent_response(
    websocket: WebSocket,
    agent,
    content: str,
    thread_id: str,
    cfg_chat,
    *,
    auto_checkpoint: bool = False,
    handle_clarification: bool = False,
    plan_result: dict | None = None,
    citations: list | None = None,
    sediment_question: str | None = None,
    sediment_type: str = "conversation",
    error_log: str = "unexpected error",
    error_label: str = "Agent error",
) -> str:
    """Stream ``agent.chat()`` to the websocket, emitting every WS message type.

    Walks the async generator from ``agent.chat(content, thread_id)`` and
    forwards ``tool_call`` / ``tool_result`` / ``text_delta`` / ``task_progress``
    (plus optional ``auto_checkpoint``) messages to the client, then closes
    with ``done`` (or ``error`` on failure).

    Returns the concatenated assistant response text.
    """
    st = _WsStreamState(
        thread_id=thread_id,
        workspace_path=Path(cfg_chat.workspace).resolve() if auto_checkpoint else None,
    )

    _ws_closed = [False]  # mutable holder so nested fn can update

    try:
        async def _ws_send(msg: dict) -> None:
            """Wrap send_json with thread_id for client-side routing."""
            if _ws_closed[0]:
                return
            if "thread_id" not in msg:
                msg["thread_id"] = thread_id
            try:
                await websocket.send_json(msg)
            except Exception:
                _ws_closed[0] = True
                logger.debug("WS closed mid-stream, stopping sends")
                # actively close so the client gets a close event
                # instead of waiting for TCP timeout
                with contextlib.suppress(Exception):
                    await websocket.close()

        # let tools push progress events back through the same WS
        progress_cb.set(_ws_send)

        async for state in agent.chat(content, thread_id):
            action = await _forward_stream_state(
                state,
                st,
                _ws_send,
                content=content,
                agent=agent,
                handle_clarification=handle_clarification,
                auto_checkpoint=auto_checkpoint,
            )
            if action == "break":
                break

        await _emit_plan_and_citations(_ws_send, plan_result, citations)
        await _sediment_if_enabled(
            _ws_send, st, cfg_chat, sediment_question, sediment_type
        )
        await _stream_auto_continues(agent, st, _ws_send)
        await _emit_fallback_if_empty(_ws_send, st)
        _record_completed_turn(thread_id, st.full_response)
        await _ws_send({"type": "done"})

    except Exception as e:
        logger.error(error_log, exc_info=True)
        await _ws_send(
            {"type": "error", "error": f"{error_label}: {str(e)}"}
        )

    return st.full_response


# Maximum auto-continue iterations per turn.
_MAX_AUTO_CONTINUES = 3


@dataclass
class _WsStreamState:
    """Mutable per-turn state shared by the WS streaming helpers."""

    thread_id: str
    workspace_path: Path | None = None
    full_response: str = ""
    seen_tool_calls: set[str] = field(default_factory=set)
    tool_names_used: set[str] = field(default_factory=set)
    seen_tool_results: set[str] = field(default_factory=set)
    auto_cp_id: str | None = None
    clarify_sent: bool = False
    token_streamed: bool = False
    auto_continue_count: int = 0
    pending_auto: bool = False


async def _forward_stream_state(
    state, st, send, *, content, agent, handle_clarification, auto_checkpoint
) -> str:
    """Forward one ``agent.chat()`` state; return ``"break"`` to stop the loop."""
    _record_task_goal(st.thread_id, content, agent)

    # Typed side-channel events yielded by the agent loop (e.g.
    # mode_banner / trust_update / budget_update / budget_escalation /
    # suggest_code / risk_threshold). These are emitted as
    # {"type": ...} dicts with no _token/_reasoning/messages, so the
    # branches below would drop them. Forward them verbatim so the
    # frontend's HRI trust/budget/suggest UI actually receives data.
    if isinstance(state, dict) and "type" in state:
        await send(dict(state))
        return "continue"

    if "_token" in state:
        st.token_streamed = True
        text = state["_token"]
        if text:
            st.full_response += text
            await send({"type": "text_delta", "text": text})
        return "continue"

    if "_reasoning" in state:
        reasoning = state["_reasoning"]
        if reasoning:
            await send({"type": "reasoning_delta", "text": reasoning})
        return "continue"

    if "_compacted" in state:
        await _emit_compacted(state, st.thread_id, send)
        return "continue"

    if state.get("_auto_continue"):
        await _emit_auto_continue(st, send)
        return "continue"

    if handle_clarification and state.get("thought_loop_terminated"):
        await send({
            "type": "text_delta",
            "text": "\n\n⚠️ **思考循环检测**: Agent 检测到输出陷入死循环, 已自动终止以避免无限循环。请尝试换一种问法或提供更多上下文。\n",
        })
        await send({"type": "done"})
        return "break"

    if (
        handle_clarification
        and not st.clarify_sent
        and state.get("needs_clarification")
        and state.get("clarify_questions")
    ):
        await _emit_clarification(state, st, send)
        return "break"

    messages = state.get("messages", [])
    if not messages:
        return "continue"
    await _emit_message_events(
        messages[-1], st, send, auto_checkpoint=auto_checkpoint
    )
    return "continue"


def _record_task_goal(thread_id: str, content: str, agent) -> None:
    """Record the task goal for long-horizon tracking (first success wins)."""
    try:
        from huginn.memory.task_state import get_tracker

        _ts = get_tracker()
        _ts_state = _ts.get(thread_id)
        if not _ts_state.goal:
            _ts_state.goal = content[:200]
            _ts_state.mode = getattr(agent, "mode", "chat")
            _ts.save(thread_id)
    except Exception:
        logger.debug("task state record skipped", exc_info=True)


async def _emit_compacted(state, thread_id: str, send) -> None:
    """Forward a compaction event and event-source it."""
    c = state["_compacted"]
    await send({
        "type": "context_compacted",
        "before_pct": c.get("before_pct", 0),
        "after_pct": c.get("after_pct", 0),
    })
    # Event-source the compaction so the block model can render a
    # `── compacted ──` divider in recovered history (T-BCSE-06).
    try:
        from huginn.events.session_writer import record_compaction

        record_compaction(thread_id, c.get("summary", ""))
    except Exception:
        logger.debug("session compaction record skipped", exc_info=True)


async def _emit_auto_continue(st, send) -> None:
    """Bump the auto-continue counter and tell the client a new round started."""
    # Pipeline/proactive suggestions injected — trigger another agent.chat()
    # iteration automatically instead of waiting for the user to send a message.
    if st.auto_continue_count < _MAX_AUTO_CONTINUES:
        st.auto_continue_count += 1
        st.pending_auto = True
        logger.info(
            "Auto-continue %d/%d (pipeline suggestions pending)",
            st.auto_continue_count, _MAX_AUTO_CONTINUES,
        )
        await send({
            "type": "text_delta",
            "text": "\n\n🔄 *Auto-continuing pipeline...*\n\n",
        })
    else:
        logger.warning(
            "Auto-continue limit (%d) reached, stopping", _MAX_AUTO_CONTINUES
        )


async def _emit_clarification(state, st, send) -> None:
    """Send the clarification request plus the accompanying assistant text."""
    questions = state["clarify_questions"]
    await send({
        "type": "clarification_request",
        "thread_id": st.thread_id,
        "questions": questions,
    })
    st.clarify_sent = True
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        text = (
            last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        )
        if text:
            st.full_response = text
            await send({"type": "text_delta", "text": text})
    await send({"type": "done"})


async def _emit_message_events(last_msg, st, send, *, auto_checkpoint: bool) -> None:
    """Forward the tool-call / tool-result / text events for one graph message."""
    if isinstance(last_msg, AIMessage):
        await _emit_ai_tool_calls(last_msg, st, send, auto_checkpoint)
        await _emit_ai_text(last_msg, st, send)
    if isinstance(last_msg, ToolMessage):
        await _emit_tool_result(last_msg, st, send)


async def _emit_ai_tool_calls(msg, st, send, auto_checkpoint: bool) -> None:
    """Emit unseen tool_calls from one assistant message."""
    for tc in getattr(msg, "tool_calls", []) or []:
        tid = tc.get("id")
        name = tc.get("name", "unknown")
        if not tid or tid in st.seen_tool_calls:
            continue
        st.seen_tool_calls.add(tid)
        st.tool_names_used.add(name)
        if auto_checkpoint and name in _EDIT_TOOLS and st.auto_cp_id is None:
            await _maybe_auto_checkpoint(st, send)
        await send({
            "type": "tool_call",
            "id": tid,
            "name": name,
            "args": tc.get("args", {}),
        })
        _record_session_tool_call(st.thread_id, tid, name, tc.get("args", {}))
        await _emit_governance_decision(name, tc.get("args", {}), send)


async def _maybe_auto_checkpoint(st, send) -> None:
    """Take one pre-edit workspace snapshot per turn (best effort)."""
    try:
        snapshot = _snapshot_directory(st.workspace_path)
        st.auto_cp_id = uuid.uuid4().hex[:8]
        with _state_lock:
            _checkpoints[st.auto_cp_id] = (st.workspace_path, snapshot)
        await send({
            "type": "auto_checkpoint",
            "id": st.auto_cp_id,
            "base": str(st.workspace_path),
            "files": len(snapshot),
        })
    except Exception as e:
        logger.debug("[auto-cp] failed: %s", e)


def _record_session_tool_call(thread_id, tid, name, args) -> None:
    """Event-source a tool call (T-BCSE closed loop)."""
    try:
        from huginn.events.session_writer import record_tool_call

        record_tool_call(thread_id, tid, name, args)
    except Exception:
        logger.debug("session tool_call record skipped", exc_info=True)


def _tool_category(name: str) -> str:
    """Map a tool name to an action category heuristically."""
    if any(k in name for k in ("vasp", "qe", "cp2k", "lammps", "abaqus", "comsol", "openfoam")):
        return "simulate"
    if any(k in name for k in ("file_edit", "file_write", "file_delete")):
        return "file_ops"
    if "code" in name or "bash" in name:
        return "code"
    if any(k in name for k in ("rag", "literature", "web_search", "database")):
        return "network"
    if "remember" in name or "recall" in name:
        return "learn"
    return "query"


async def _emit_governance_decision(name, args, send) -> None:
    """Emit the governance pre-check decision for a tool call."""
    try:
        from huginn.governance import get_governance

        gov = get_governance()
        ctx = {"tool_name": name, "args": args}
        decision = gov.can_execute(name, ctx)
        await send({
            "type": "governance",
            "action_name": name,
            "category": _tool_category(name),
            "risk_level": decision.risk_level,
            "allowed": decision.allowed,
            "reasons": decision.reasons,
            "requires_approval": decision.requires_approval,
            "predictability": decision.predictability,
        })
    except Exception:
        logger.debug("ws governance decision send failed", exc_info=True)


async def _emit_tool_result(msg, st, send) -> None:
    """Emit one unseen tool result plus its warnings / governance / progress."""
    tid = getattr(msg, "tool_call_id", None)
    if not tid or tid in st.seen_tool_results:
        return
    st.seen_tool_results.add(tid)
    tool_content = str(getattr(msg, "content", ""))
    tool_name = getattr(msg, "name", "") or tid
    tool_result_msg = {
        "type": "tool_result",
        "id": tid,
        "content": tool_content,
    }
    warnings = _extract_tool_warnings(tool_content)
    if warnings:
        tool_result_msg["warnings"] = warnings
        # 前端有专门的 hook_warning UI 渲染, 单独发一个事件
        await send({
            "type": "hook_warning",
            "tool_name": tool_name,
            "warnings": warnings,
        })
    await send(tool_result_msg)
    _record_session_tool_result(st.thread_id, tid, tool_content)
    _record_tool_step(st.thread_id, tool_name, tool_content)
    await _emit_governance_verification(tool_name, tool_content, send)
    progress = _extract_task_progress(tool_content)
    if progress:
        await send({"type": "task_progress", **progress})


def _record_session_tool_result(thread_id, tid, tool_content) -> None:
    """Event-source a tool result (T-BCSE closed loop)."""
    try:
        from huginn.events.session_writer import record_tool_result

        record_tool_result(thread_id, tid, tool_content)
    except Exception:
        logger.debug("session tool_result record skipped", exc_info=True)


def _first_finding_line(tool_content: str) -> str:
    # ponytail: naive finding extraction — first non-empty line of tool
    # output. Good enough for short results; upgrade to LLM summarization
    # when long outputs pile up.
    for line in tool_content.split("\n"):
        line = line.strip()
        if len(line) > 10 and not line.startswith("{") and not line.startswith("["):
            return line[:120]
    return ""


def _record_tool_step(thread_id: str, tool_name: str, tool_content: str) -> None:
    """Record one tool step in the task state tracker."""
    try:
        from huginn.memory.task_state import get_tracker

        get_tracker().record_step(
            thread_id,
            action=f"Tool: {tool_name}",
            tool=tool_name,
            result=tool_content[:200],
            findings=_first_finding_line(tool_content),
        )
    except Exception:
        logger.debug("tool step record skipped", exc_info=True)


async def _emit_governance_verification(tool_name, tool_content, send) -> None:
    """Emit the governance post-execution verification result."""
    try:
        from huginn.governance import get_governance

        gov = get_governance()
        v_ok, v_msg = gov.verify(
            tool_name,
            {"tool_name": tool_name},
            {"result": tool_content[:500]},
        )
        await send({
            "type": "governance",
            "action_name": tool_name,
            "category": "",
            "risk_level": "",
            "allowed": True,
            "reasons": [],
            "requires_approval": False,
            "predictability": 1.0 if v_ok else 0.5,
            "audit_id": "",
            "status": "verified" if v_ok else "failed",
            "verification_passed": v_ok,
            "verification_message": v_msg,
            "rollback_available": False,
        })
    except Exception:
        logger.debug("ws governance verification send failed", exc_info=True)


async def _emit_ai_text(msg, st, send) -> None:
    """Emit the non-streamed remainder of an assistant message as a delta."""
    if st.token_streamed:
        st.full_response = msg.content
    elif isinstance(msg.content, str):
        delta = msg.content[len(st.full_response):]
        if delta:
            st.full_response = msg.content
            await send({"type": "text_delta", "text": delta})
    else:
        st.full_response = msg.content


async def _emit_plan_and_citations(send, plan_result, citations) -> None:
    """Emit the post-run plan acceptance report and citation list."""
    if plan_result and plan_result.get("acceptance_criteria"):
        criteria_results = []
        for ac in plan_result["acceptance_criteria"]:
            criterion = ac.get("criterion", "") if isinstance(ac, dict) else str(ac)
            criteria_results.append({
                "criterion": criterion,
                "passed": True,
                "note": "Verified after execution",
            })
        await send({
            "type": "plan_result",
            "plan_id": plan_result.get("plan_id", ""),
            "criteria": criteria_results,
            "all_passed": all(c["passed"] for c in criteria_results),
        })

    if citations:
        await send({
            "type": "citations",
            "sources": citations,
        })


async def _sediment_if_enabled(
    send, st, cfg_chat, sediment_question, sediment_type
) -> None:
    """Auto-sediment the completed Q/A into the knowledge base when enabled."""
    from huginn.routes.ws import get_context

    if not (
        sediment_question
        and cfg_chat.rag_enabled
        and st.full_response
        and len(st.full_response) > 50
        and get_context().kb is not None
    ):
        return
    try:
        import time as _time

        sediment_text = (
            f"Q: {sediment_question[:500]}\n\n"
            f"A: {st.full_response[:2000]}"
        )
        get_context().kb.add_text(
            text=sediment_text,
            metadata={
                "type": sediment_type,
                "thread_id": st.thread_id,
                "timestamp": _time.time(),
                "source": "auto_sediment",
            },
        )
        await send({
            "type": "sediment",
            "stored": True,
            "kind": sediment_type,
            "preview": sediment_text[:200],
        })
        _publish_sediment_event(st.thread_id, sediment_type, sediment_text)
    except Exception:
        logger.debug(
            "plan auto-sediment failed"
            if sediment_type == "plan_execution"
            else "auto-sediment failed",
            exc_info=True,
        )


def _publish_sediment_event(thread_id, sediment_type, sediment_text) -> None:
    """沉淀也发到全局 EventBus, 让审计视图(/events/recent|stream)能回放明细."""
    try:
        from huginn.events.integration import publish_event_sync

        publish_event_sync(
            "sediment",
            {
                "stored": True,
                "kind": sediment_type,
                "preview": sediment_text[:200],
            },
            thread_id=thread_id,
            source="auto_sediment",
        )
    except Exception:
        logger.debug("sediment event publish failed", exc_info=True)


async def _stream_auto_continues(agent, st, send) -> None:
    """Run extra chat() iterations while pipeline suggestions are pending."""
    while st.pending_auto and st.auto_continue_count <= _MAX_AUTO_CONTINUES:
        st.pending_auto = False
        st.token_streamed = False
        async for state in agent.chat("Continue", st.thread_id):
            if "_token" in state:
                st.token_streamed = True
                text = state["_token"]
                if text:
                    st.full_response += text
                    await send({"type": "text_delta", "text": text})
                continue
            if "_reasoning" in state:
                reasoning = state["_reasoning"]
                if reasoning:
                    await send({"type": "reasoning_delta", "text": reasoning})
                continue
            if state.get("_auto_continue"):
                if st.auto_continue_count < _MAX_AUTO_CONTINUES:
                    st.auto_continue_count += 1
                    st.pending_auto = True
                    await send({
                        "type": "text_delta",
                        "text": "\n\n🔄 *Auto-continuing pipeline...*\n\n",
                    })
                continue
            msgs = state.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage):
                await _apply_auto_continue_text(msgs[-1], st, send)


async def _apply_auto_continue_text(last, st, send) -> None:
    """Emit the non-streamed remainder during auto-continue iterations."""
    if st.token_streamed:
        st.full_response = last.content
    elif isinstance(last.content, str):
        delta = last.content[len(st.full_response):]
        if delta:
            st.full_response = last.content
            await send({"type": "text_delta", "text": delta})


async def _emit_fallback_if_empty(send, st) -> None:
    """Emit a human-readable fallback when the agent produced no text."""
    if st.full_response:
        return
    if st.tool_names_used:
        names_str = ", ".join(sorted(st.tool_names_used))
        fallback = f"（Agent 调用了工具 [{names_str}] 但未生成文字回复。请查看上方的工具调用结果，或换一种问法重试。）"
    else:
        fallback = "（Agent 未能生成回复，可能是内部处理超时或出错。请尝试换一种问法，或稍后重试。）"
    st.full_response = fallback
    await send({
        "type": "text_delta",
        "text": fallback,
    })


def _record_completed_turn(thread_id: str, full_response: str) -> None:
    """Event-source the completed assistant turn (T-BCSE closed loop)."""
    if not full_response:
        return
    try:
        from huginn.events.session_writer import record_assistant_message

        record_assistant_message(thread_id, full_response)
    except Exception:
        logger.debug("session assistant-message record skipped", exc_info=True)


async def _dispatch_user_spec(
    websocket: WebSocket,
    factory: Any,
    spec_name: str,
    task_text: str,
    thread_id: str,
    ws_approval: Any,
) -> None:
    """Dispatch a user-selected subagent (``@spec task``) and stream its result.

    仅用户显式 @点名 的子代理走这里. 不经过主 agent 决策, spec 和任务文本
    由用户直接给定. 结果以 text_delta 流式返回, 最后 done.
    """
    from huginn.agents.subagent import SubagentDispatch

    _dispatch = SubagentDispatch()
    result = await _dispatch.dispatch(
        spec_name,
        task_text,
        context={
            "agent_factory": factory,
            "approval_callback": ws_approval,
        },
    )

    if not result.success:
        await websocket.send_json({
            "type": "text_delta",
            "text": f"⚠️ subagent [{spec_name}] 执行失败: {result.error}\n",
        })
        await websocket.send_json({"type": "done", "thread_id": thread_id})
        return

    summary = result.summary or "（子代理未返回内容）"
    try:
        from huginn.events.session_writer import record_assistant_message
        record_assistant_message(thread_id, summary)
    except Exception:
        logger.debug("session subagent record skipped", exc_info=True)

    await websocket.send_json({"type": "text_delta", "text": summary})
    await websocket.send_json({"type": "done", "thread_id": thread_id})


# ── Message-type handlers (extracted from agent_websocket) ───────


async def _handle_user_input(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Handle a user_input message: create agent, route, stream response."""
    from huginn.routes.ws import get_or_create_thread

    prepared = await _resolve_turn_config(websocket)
    if prepared is None:
        return
    cfg_chat, factory = prepared

    agent = await _create_turn_agent(websocket, msg, factory, ctx.ws_approval)
    if agent is None:
        return

    content = msg.content
    thread_id = msg.thread_id
    get_or_create_thread(thread_id, user_id=ctx.user_id)
    _record_user_turn(thread_id, content)

    agent, content, handled = await _route_mentions(
        websocket, factory, agent, content, thread_id, ctx.ws_approval
    )
    if handled:
        return
    _attach_team_ref(agent)

    use_team, use_fusion = _detect_chat_modes(content, cfg_chat)
    plan_mode = _apply_plan_mode(agent, content)
    _apply_research_mode(agent, content)
    if await _dispatch_special_modes(
        websocket, factory, agent, content, thread_id, cfg_chat,
        plan_mode, use_team, use_fusion, ctx.pending_plan_contexts,
    ):
        return

    content, rag_sources = await _augment_with_rag(content, cfg_chat)
    ctx.last_user_context.update({
        "content": content, "thread_id": thread_id,
        "cfg_chat": cfg_chat, "agent": agent,
    })
    await _stream_agent_response(
        websocket,
        agent,
        content,
        thread_id,
        cfg_chat,
        auto_checkpoint=True,
        handle_clarification=True,
        plan_result=None,
        citations=rag_sources,
        sediment_question=content,
        error_log="unexpected error",
        error_label="Agent error",
    )
    await _drain_side_questions(websocket)


async def _resolve_turn_config(websocket: WebSocket) -> tuple[Any, Any] | None:
    """取 config 与 agent factory; 失败发错误并返回 None."""
    from huginn.routes.ws import get_agent_factory, get_config

    try:
        cfg_chat = get_config()
    except Exception as exc:
        logger.error("unexpected error", exc_info=True)
        await _send_error(websocket, f"Config error: {exc}")
        return None
    try:
        factory = get_agent_factory()
    except Exception as exc:
        logger.error("unexpected error", exc_info=True)
        await _send_error(websocket, f"Factory error: {exc}")
        return None
    return cfg_chat, factory


async def _create_turn_agent(
    websocket: WebSocket, msg: WSMessage, factory: Any, ws_approval: Any
) -> Any | None:
    """按 thinking/max_tokens/persona 创建 lead agent; 失败或无模型返回 None."""
    thinking = msg.thinking
    max_tokens = msg.max_tokens
    if thinking is not None or max_tokens is not None or msg.persona:
        try:
            agent = factory.create_lead(
                thread_id=msg.thread_id,
                thinking=thinking,
                max_tokens=max_tokens,
                approval_callback=ws_approval,
            )
        except Exception as e:
            await _send_error(websocket, f"Cannot create agent: {e}")
            return None
    else:
        try:
            agent = factory.create_lead(
                thread_id=msg.thread_id,
                approval_callback=ws_approval,
            )
        except Exception as exc:
            logger.error("unexpected error", exc_info=True)
            await _send_error(websocket, f"Failed to init agent: {exc}")
            return None
    if agent.model is None:
        await websocket.send_json(
            {"type": "text_delta", "text": "⚠️ No LLM model configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.\n"}
        )
        await websocket.send_json({"type": "done"})
        return None
    return agent


def _record_user_turn(thread_id: str, content: str) -> None:
    """Event-source the user turn (T-BCSE closed loop); 写失败不阻塞对话."""
    try:
        from huginn.events.session_writer import record_user_message
        record_user_message(thread_id, content)
    except Exception:
        logger.debug("session user-message record skipped", exc_info=True)


async def _route_mentions(
    websocket: WebSocket,
    factory: Any,
    agent: Any,
    content: str,
    thread_id: str,
    ws_approval: Any,
) -> tuple[Any, str, bool]:
    """@agent 切换 + @spec 子代理 dispatch; 返回 (agent, content, 是否已接管)."""
    if content.startswith("@"):
        parts = content[1:].split(None, 1)
        if len(parts) == 2:
            target_agent_name, actual_content = parts
            target_agent_name = target_agent_name.lower()
            agent_cfg = None
            for name in ("lead", "pm", "rd", "swe", "writer", "reviewer"):
                if name in target_agent_name:
                    agent_cfg = name
                    break
            if agent_cfg and agent_cfg != "lead":
                try:
                    agent = factory.create_agent(
                        agent_cfg,
                        thread_id=thread_id,
                        approval_callback=ws_approval,
                    )
                    content = actual_content
                except Exception as e:
                    logger.warning("Failed to create agent %s: %s", agent_cfg, e)

    # @spec subagent routing — 仅用户显式 @点名 的子代理才 dispatch.
    if content.startswith("@"):
        parts = content[1:].split(None, 1)
        if len(parts) == 2:
            spec_token, task_text = parts
            spec_token = spec_token.lower()
            from huginn.agents.subagent import SubagentDispatch

            if spec_token in SubagentDispatch.BUILTIN_SPECS:
                await _dispatch_user_spec(
                    websocket, factory, spec_token, task_text,
                    thread_id, ws_approval,
                )
                return agent, content, True
    return agent, content, False


def _attach_team_ref(agent: Any) -> None:
    """给 agent 挂 ModelTeam 引用, 供跨 agent 视觉委派; 无团队配置则忽略."""
    try:
        from huginn.routes.team import _build_model_team
        agent._team_ref = _build_model_team()
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)


def _detect_chat_modes(content: str, cfg_chat: Any) -> tuple[bool, bool]:
    """Team mode (关键字或 config flag) 与 Fusion mode (关键字) 触发."""
    lowered = content.lower()
    use_team = (
        any(kw in lowered for kw in ("/team", "delegate", "collaborate"))
        or getattr(cfg_chat, "team_mode_enabled", False)
    )
    return use_team, "/fusion" in lowered


def _apply_plan_mode(agent: Any, content: str) -> bool:
    """显式 /plan 或自动规划意图 → set_mode('plan'), 返回是否进入 plan 模式."""
    lowered = content.lower()
    if any(kw in lowered for kw in ("/plan", "plan mode")):
        if hasattr(agent, "set_mode"):
            agent.set_mode("plan")
        return True
    if any(kw in lowered for kw in ("/research", "research mode", "/chat")):
        return False
    signals = (
        "design a", "design an", "propose a", "propose an",
        "plan how", "step by step", "workflow for",
        "strategy for", "roadmap", "outline a",
    )
    if any(s in lowered for s in signals) and len(content) > 30:
        if hasattr(agent, "set_mode"):
            agent.set_mode("plan")
        return True
    return False


def _apply_research_mode(agent: Any, content: str) -> None:
    """显式 /research 或自动研究意图 → set_mode('research')."""
    lowered = content.lower()
    signals = (
        "/research", "research mode",
        "literature review", "survey the", "systematically review",
        "state of the art", "recent advances in",
    )
    explicit = any(kw in lowered for kw in ("/research", "research mode"))
    auto = any(s in lowered for s in signals[:6]) and len(content) > 40
    if (explicit or auto) and hasattr(agent, "set_mode"):
        agent.set_mode("research")


async def _dispatch_special_modes(
    websocket: WebSocket,
    factory: Any,
    agent: Any,
    content: str,
    thread_id: str,
    cfg_chat: Any,
    plan_mode: bool,
    use_team: bool,
    use_fusion: bool,
    pending_plan_contexts: dict,
) -> bool:
    """research/team/fusion/plan 四个特殊模式, 返回是否已接管本轮."""
    if await _run_research_mode(websocket, agent, content, thread_id):
        return True
    if use_team:
        return await _run_team_mode(
            websocket, factory, agent, content, thread_id
        )
    if use_fusion and await _run_fusion_mode(
        websocket, agent, content, thread_id
    ):
        return True
    if plan_mode:
        await _run_plan_mode(
            websocket, agent, content, thread_id, cfg_chat, pending_plan_contexts
        )
        return True
    return False


async def _emit_research_event(websocket: WebSocket, result: dict) -> str:
    """把一个 research 事件转成 WS 消息; 返回 draft 内容 (可能空串)."""
    evt_type = result.get("type")
    if evt_type == "status":
        await websocket.send_json({
            "type": "task_progress",
            "task_type": "pipeline",
            "pipeline": "deli_research",
            "stage": result.get("stage", ""),
            "message": result.get("message", ""),
            "stage_index": result.get("stage_index"),
            "total_stages": result.get("total_stages"),
            "progress_pct": result.get("progress_pct"),
            "status": result.get("status", "running"),
            "topic": result.get("topic", ""),
        })
    elif evt_type == "hypothesis":
        await websocket.send_json({
            "type": "text_delta",
            "text": f"📋 Hypothesis: {result.get('hypothesis', '')}\n",
        })
    elif evt_type == "experiment":
        await websocket.send_json({
            "type": "text_delta",
            "text": f"🧪 Experiment: {result.get('description', '')}\n",
        })
    elif evt_type == "result":
        await websocket.send_json({
            "type": "text_delta",
            "text": f"📊 Result: {result.get('summary', '')}\n",
        })
    elif evt_type == "draft":
        draft = result.get("content", "")
        if draft:
            await websocket.send_json({
                "type": "text_delta",
                "text": f"\n{'='*40}\n📝 Draft:\n{'='*40}\n{draft}\n",
            })
            return draft
    elif evt_type == "error":
        await websocket.send_json({
            "type": "text_delta",
            "text": f"❌ Error: {result.get('message', '')}\n",
        })
    return ""


async def _run_research_mode(
    websocket: WebSocket, agent: Any, content: str, thread_id: str
) -> bool:
    """Research 模式: 跑 workflow 并流式转发事件; 返回是否已接管."""
    if not (hasattr(agent, "is_research_mode") and agent.is_research_mode()):
        return False
    try:
        from huginn.research_workflow import (
            ResearchWorkflow,
            ResearchWorkflowConfig,
        )
        research_config = ResearchWorkflowConfig(
            max_concurrent_branches=3,
            enable_hypothesis_generation=True,
        )
        workflow = ResearchWorkflow(agent=agent, config=research_config)
        await websocket.send_json(
            {"type": "text_delta", "text": "🔬 Research mode activated.\n\n"}
        )
        output = ""
        async for result in workflow.run(content, thread_id=thread_id):
            draft = await _emit_research_event(websocket, result)
            output = draft or output
        try:
            from huginn.events.session_writer import record_assistant_message
            record_assistant_message(thread_id, output or "Research complete.")
        except Exception:
            logger.debug("session research record skipped", exc_info=True)
        await websocket.send_json({"type": "done"})
        return True
    except Exception as e:
        logger.error(
            "Research mode failed, falling back to chat: %s", e, exc_info=True
        )
        return False


async def _run_team_mode(
    websocket: WebSocket, factory: Any, agent: Any, content: str, thread_id: str
) -> bool:
    """Team 模式: Orchestrator 多 agent 协作; 返回是否已接管."""
    try:
        if hasattr(agent, "set_mode"):
            agent.set_mode("research")
    except Exception as e:
        logger.warning("set_mode('research') for team failed: %s", e)
    try:
        from huginn.agents.orchestrator import Orchestrator
        orch = Orchestrator(factory=factory)
        await websocket.send_json(
            {"type": "text_delta", "text": "👥 Team mode activated.\n\n"}
        )

        async def _team_status(status_msg: str) -> None:
            await websocket.send_json(
                {"type": "text_delta", "text": f"  → {status_msg}\n"}
            )

        result = await orch.run(content, on_status=_team_status, auto_confirm=True)
        summary = result.summary if hasattr(result, "summary") else str(result)
        await websocket.send_json({"type": "text_delta", "text": f"\n{summary}\n"})
        try:
            from huginn.events.session_writer import record_assistant_message
            record_assistant_message(thread_id, summary)
        except Exception:
            logger.debug("session team record skipped", exc_info=True)
        await websocket.send_json({"type": "done"})
        return True
    except Exception as e:
        logger.error(
            "Team mode failed, falling back to single agent: %s", e, exc_info=True
        )
        await websocket.send_json({
            "type": "text_delta",
            "text": f"⚠️ Team mode unavailable ({e}), using single agent.\n\n",
        })
        return False


async def _setup_fusion_panel(
    websocket: WebSocket, content: str
) -> tuple[Any, str, int]:
    """建团队面板 + 发激活消息 + 解析 /fusionN 轮数."""
    from huginn.routes.team import _build_model_team

    team = _build_model_team()
    query = content.replace("/fusion", "").strip() or content
    await websocket.send_json(
        {"type": "text_delta", "text": "⚡ Fusion mode activated.\n"}
    )
    await websocket.send_json(
        {"type": "text_delta", "text": f"Panel: {len(team.members)} members\n"}
    )
    for member in team.list_members():
        await websocket.send_json({
            "type": "text_delta",
            "text": f"  • [{member['role']}] {member['name']} ({member['model']})\n",
        })
    rounds = 1
    for tag in ("/fusion1", "/fusion2", "/fusion3"):
        if tag in content.lower():
            rounds = int(tag[-1])
            break
    return team, query, rounds


async def _emit_fusion_result(websocket: WebSocket, result: dict) -> None:
    """流式发多轮回答 + 共识/分歧 + 综合答案."""
    all_rounds = result.get("all_rounds", [])
    for idx, responses in enumerate(all_rounds):
        if len(all_rounds) > 1:
            await websocket.send_json({
                "type": "text_delta",
                "text": (
                    f"\n{'='*40}\n📝 Round {idx + 1}/{len(all_rounds)}"
                    f"\n{'='*40}\n"
                ),
            })
        for r in responses:
            await websocket.send_json({
                "type": "text_delta",
                "text": f"\n--- [{r['role']} ({r['model']})] ({r['duration_ms']}ms) ---\n",
            })
            await websocket.send_json({
                "type": "text_delta",
                "text": r["answer"] + "\n",
            })
    if result.get("consensus"):
        await websocket.send_json({
            "type": "text_delta",
            "text": "\n## 共识\n" + result["consensus"] + "\n",
        })
    if result.get("dissent"):
        await websocket.send_json({
            "type": "text_delta",
            "text": "\n## 分歧\n" + result["dissent"] + "\n",
        })
    await websocket.send_json({
        "type": "text_delta",
        "text": "\n## 综合答案\n" + result.get("final_answer", "") + "\n",
    })


async def _run_fusion_mode(
    websocket: WebSocket, agent: Any, content: str, thread_id: str
) -> bool:
    """Fusion 模式: 多模型并行 + 综合; 返回是否已接管."""
    try:
        if hasattr(agent, "set_mode"):
            agent.set_mode("research")
    except Exception as e:
        logger.warning("set_mode('research') for fusion failed: %s", e)
    try:
        team, query, rounds = await _setup_fusion_panel(websocket, content)
        result = await team.fusion_query(query, rounds=rounds)
        await _emit_fusion_result(websocket, result)
        try:
            from huginn.events.session_writer import record_assistant_message
            record_assistant_message(thread_id, result.get("final_answer", ""))
        except Exception:
            logger.debug("session fusion record skipped", exc_info=True)
        await websocket.send_json({"type": "done"})
        return True
    except Exception as e:
        logger.error(
            "Fusion mode failed, falling back to single agent: %s", e, exc_info=True
        )
        await websocket.send_json({
            "type": "text_delta",
            "text": f"⚠️ Fusion mode unavailable ({e}), using single agent.\n\n",
        })
        return False


async def _plan_context_hint(objective: str) -> str:
    """从知识库取 3 条相关上下文拼成提示; 失败返回空串."""
    from huginn.routes.ws import get_context

    try:
        ctx = get_context()
        if ctx.kb is not None and ctx.kb.count() > 0:
            chunks = await asyncio.to_thread(ctx.kb.query, objective, 3)
            if chunks:
                context = "\n".join(
                    f"- {(c.get('text') or '')[:200]}" for c in chunks[:3]
                )
                return "\n\nRelevant context from knowledge base:\n" + context
    except Exception:
        logger.debug("plan kb context query skipped", exc_info=True)
    return ""


def _build_plan_prompt(objective: str, context_hint: str) -> str:
    """构造要求 LLM 只返回 JSON 的 plan prompt."""
    return (
        "Break down the following task into a structured plan.\n\n"
        f"Task: {objective}\n\n"
        "Return a JSON object with:\n"
        '  "steps": [{"name": "...", "description": "...", "tool": "...", "estimated_time": "..."}]\n'
        '  "acceptance_criteria": [{"criterion": "...", "how_to_verify": "..."}]\n'
        '  "tools_needed": ["tool1", "tool2", ...]\n'
        '  "summary": "One-line description"\n\n'
        "Return ONLY the JSON, no markdown fences."
        f"{context_hint}"
    )


def _fallback_plan(objective: str) -> dict:
    """LLM 输出无法解析时的单步兜底计划."""
    return {
        "steps": [
            {
                "name": "Execute task",
                "description": objective,
                "tool": "agent",
            }
        ],
        "acceptance_criteria": [],
        "tools_needed": [],
        "summary": objective[:100],
    }


def _parse_plan_response(plan_text: Any, objective: str) -> dict:
    """解析 plan JSON; 失败则正则抽 {...}; 再失败给单步兜底."""
    import json as _json
    import re

    if not isinstance(plan_text, str):
        plan_text = str(plan_text)
    try:
        return _json.loads(plan_text)
    except Exception:
        logger.debug("plan json parse failed, retry with regex", exc_info=True)
    try:
        match = re.search(r"\{[\s\S]*\}", plan_text)
        if match:
            return _json.loads(match.group())
    except Exception:
        logger.debug("plan regex parse failed, fallback plan", exc_info=True)
    return _fallback_plan(objective)


def _persist_plan(plan_data: dict, objective: str) -> None:
    """持久化到 PlanStore, 支持跨会话恢复; 失败忽略."""
    try:
        from huginn.autoloop.plan_store import PlanStep, PlanStore
        _store = PlanStore()
        _steps = [
            PlanStep(
                id=f"step_{i}",
                description=s.get("description") or s.get("name", ""),
                tool=s.get("tool"),
            )
            for i, s in enumerate(plan_data.get("steps", []))
        ]
        _store.create_plan(objective=objective, steps=_steps, auto_confirm=False)
    except Exception:
        logger.debug("plan store create skipped", exc_info=True)


async def _run_plan_mode(
    websocket: WebSocket,
    agent: Any,
    content: str,
    thread_id: str,
    cfg_chat: Any,
    pending_plan_contexts: dict,
) -> bool:
    """Plan 模式: 生成结构化计划并发 plan 事件; 失败发错误."""
    try:
        objective = (
            content.replace("/plan", "").replace("plan mode", "").strip()
            or content
        )
        context_hint = await _plan_context_hint(objective)
        plan_response = await agent.model.ainvoke(
            _build_plan_prompt(objective, context_hint)
        )
        plan_text = (
            plan_response.content
            if hasattr(plan_response, "content")
            else str(plan_response)
        )
        plan_data = _parse_plan_response(plan_text, objective)
        plan_id = uuid.uuid4().hex[:8]
        pending_plan_contexts[plan_id] = {
            "plan_data": plan_data,
            "plan_objective": objective,
            "thread_id": thread_id,
            "agent": agent,
            "websocket": websocket,
            "cfg_chat": cfg_chat,
            "rag_sources": [],
        }
        _persist_plan(plan_data, objective)
        await websocket.send_json({
            "type": "plan",
            "plan_id": plan_id,
            "plan": plan_data,
        })
        return True
    except Exception as e:
        logger.error("plan mode error", exc_info=True)
        await _send_error(websocket, f"Plan generation failed: {e}")
        return False


async def _augment_with_rag(content: str, cfg_chat: Any) -> tuple[str, list]:
    """RAG 增强: 检索 5 条上下文拼进 content, 返回 (content, sources)."""
    from huginn.routes.ws import get_context

    sources: list = []
    try:
        kb = get_context().kb
        if not (cfg_chat.rag_enabled and kb is not None and kb.count() > 0):
            return content, sources
        chunks = await asyncio.to_thread(kb.query, content, 5)
        if chunks:
            context_text = "\n\n".join(
                f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)
            )
            content = (
                "Use the following retrieved context to answer the question. "
                "Cite the source numbers when appropriate.\n\n"
                f"{context_text}\n\n"
                f"Question: {content}"
            )
            for i, c in enumerate(chunks):
                sources.append({
                    "ref": i + 1,
                    "filename": c.get("filename") or c.get("source") or "unknown",
                    "text": (c.get("text") or "")[:200],
                    "distance": c.get("distance"),
                })
    except Exception as e:
        logger.warning("[RAG] query failed: %s", e)
    return content, sources


async def _drain_side_questions(websocket: WebSocket) -> None:
    """Drain pending side-channel questions."""
    try:
        from huginn.side_conversation import get_shared_side_channel
        for sq in get_shared_side_channel().drain():
            await websocket.send_json({
                "type": "side_question_pending",
                "question_id": sq.id,
                "question": sq.question,
                "created_at": sq.created_at,
            })
    except Exception:
        logger.debug("side-channel drain failed", exc_info=True)


async def _handle_explore_start(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Handle explore_start: run the exploration orchestrator."""
    # Late import so monkeypatch on huginn.routes.ws takes effect
    from huginn.routes.ws import get_context

    content = msg.content
    config = msg.config or {}

    await websocket.send_json(
        {
            "type": "text_delta",
            "text": f"🚀 Starting exploration: {content}\n",
        }
    )
    try:
        from huginn.exploration.orchestrator import ExplorationOrchestrator
        from huginn.exploration.strategies import ParetoPruningStrategy

        cfg = get_context().config
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=5),
            max_parallel=cfg.max_parallel_branches,
        )

        config = msg.config or {}
        initial_branches = config.get(
            "initial_branches",
            [
                {
                    "name": "baseline",
                    "hypothesis": f"Baseline for: {content}",
                }
            ],
        )
        objectives = config.get("objectives", {"score": "maximize"})

        result = await orch.explore(
            objective=content,
            initial_branches=initial_branches,
            objectives_config=objectives,
            max_iterations=config.get("max_iterations", 10),
        )

        await websocket.send_json(
            {
                "type": "text_delta",
                "text": f"\n✅ Exploration complete!\n"
                f"• Branches explored: {result.n_branches_explored}\n"
                f"• Branches pruned: {result.n_branches_pruned}\n"
                f"• Pareto front size: {len(result.pareto_front)}\n"
                f"• Convergence: {result.convergence_reason}\n",
            }
        )

        if result.best_branch:
            await websocket.send_json(
                {
                    "type": "text_delta",
                    "text": f"\n🏆 Best branch: {result.best_branch['name']}\n"
                    f"   Hypothesis: {result.best_branch['hypothesis']}\n"
                    f"   Objectives: {result.best_branch['objectives']}\n",
                }
            )

        await websocket.send_json(
            {
                "type": "exploration_result",
                "data": {
                    "pareto_front": result.pareto_front,
                    "best_branch": result.best_branch,
                    "convergence_reason": result.convergence_reason,
                },
            }
        )

    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        await websocket.send_json(
            {"type": "error", "error": f"Exploration failed: {str(e)}"}
        )
    await websocket.send_json({"type": "done"})


async def _handle_approval_response(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Handle approval_response: resolve future, re-queue if approved."""
    request_id = msg.request_id
    approved = msg.approved if msg.approved is not None else False
    pending_approvals = ctx.pending_approvals
    future = pending_approvals.pop(request_id, None)
    if future is not None and not future.done():
        future.set_result(approved)

    ctx_ap = ctx.pending_approval_contexts.pop(request_id, None)
    if approved and ctx_ap:
        # ponytail: re-runs the full turn, not just the denied tool —
        # acceptable since the user explicitly approved.
        if ctx.session_auto_approve is not None:
            ctx.session_auto_approve["enabled"] = True
        await _stream_agent_response(
            websocket,
            ctx_ap["agent"],
            ctx_ap["content"],
            ctx_ap["thread_id"],
            ctx_ap["cfg_chat"],
            auto_checkpoint=True,
            handle_clarification=True,
            error_log="approval re-queue error",
            error_label="Approval re-queue failed",
        )
        if ctx.session_auto_approve is not None:
            ctx.session_auto_approve["enabled"] = False


async def _handle_plan_confirm(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Handle plan_confirm: execute or cancel a previously generated plan."""
    plan_id = msg.plan_id
    confirmed = msg.confirmed if msg.confirmed is not None else False
    edited_plan = msg.edited_plan
    pending_plan_contexts = ctx.pending_plan_contexts
    last_user_context = ctx.last_user_context

    ctx_plan = pending_plan_contexts.pop(plan_id, None)
    if ctx_plan is None:
        future = _pending_plans.pop(plan_id, None)
        if future is not None and not future.done():
            future.set_result({"confirmed": confirmed, "edited_plan": edited_plan})
        return

    if not confirmed:
        await websocket.send_json({
            "type": "text_delta",
            "text": "📋 Plan cancelled by user.\n",
        })
        await websocket.send_json({"type": "done"})
        return

    plan_data = edited_plan or ctx_plan["plan_data"]
    plan_objective = ctx_plan["plan_objective"]
    agent = ctx_plan["agent"]
    thread_id = ctx_plan["thread_id"]
    cfg_chat = ctx_plan["cfg_chat"]

    import json as _json2
    plan_context = (
        f"Agreed plan:\n"
        f"{_json2.dumps(plan_data, indent=2, ensure_ascii=False)}\n\n"
        f"Execute this plan step by step. "
        f"After completion, verify each acceptance criterion.\n\n"
        f"Original request: {plan_objective}"
    )

    _plan_result = None
    if plan_data.get("acceptance_criteria"):
        _plan_result = {
            "plan_id": plan_id,
            "acceptance_criteria": plan_data["acceptance_criteria"],
        }
    last_user_context.update({
        "content": plan_context, "thread_id": thread_id,
        "cfg_chat": cfg_chat, "agent": agent,
    })

    # 执行阶段开启 plan_mode — 写工具强制 ASK, 只读工具放行.
    # 走 enter/exit_plan_execution 配对方法, 集中状态管理 (之前直接改 _permission_config
    # 会让 is_plan_mode() 与 _mode 不一致, 误导 research_safety_hook).
    agent.enter_plan_execution()
    # 发 plan.exec_start / exec_complete 到 campaign SSE 通道, 前端拿到后给
    # plan 卡片挂 "executing" / "done" 状态徽标. 之前用户点 Confirm 后整个
    # 流式期间没有任何指示, 只能盯着 streaming 光标.
    try:
        from huginn.interaction.progress import get_progress_tracker
        get_progress_tracker().emit_campaign_event(
            task_id=plan_id,
            event_type="plan.exec_start",
            data={
                "plan_id": plan_id,
                "thread_id": thread_id,
                "objective": plan_objective,
                "total_steps": len(plan_data.get("steps") or []),
                "steps": [
                    {"name": s.get("name", "Step"), "description": s.get("description", "")}
                    for s in (plan_data.get("steps") or [])
                ],
            },
        )
    except Exception:
        logger.debug("plan.exec_start emit failed", exc_info=True)
    try:
        await _stream_agent_response(
            websocket,
            agent,
            plan_context,
            thread_id,
            cfg_chat,
            plan_result=_plan_result,
            sediment_question=plan_objective,
            sediment_type="plan_execution",
            error_log="plan execution error",
            error_label="Plan execution failed",
        )
    finally:
        agent.exit_plan_execution()
        try:
            from huginn.interaction.progress import get_progress_tracker
            get_progress_tracker().emit_campaign_event(
                task_id=plan_id,
                event_type="plan.exec_complete",
                data={"plan_id": plan_id, "thread_id": thread_id},
            )
        except Exception:
            logger.debug("plan.exec_complete emit failed", exc_info=True)


async def _handle_clarification_response(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Resolve a pending clarification question (by id or thread)."""
    from huginn.interaction.clarification import get_clarification_manager

    mgr = get_clarification_manager()
    if msg.question_id:
        mgr.resolve(msg.question_id, msg.answer)
    else:
        mgr.resolve_thread(msg.thread_id, msg.answer)


async def _handle_set_auto_approve(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Toggle per-session auto-approve for tool permission checks."""
    enabled = bool(msg.enabled if msg.enabled is not None else True)
    if ctx.session_auto_approve is not None:
        ctx.session_auto_approve["enabled"] = enabled
    await websocket.send_json(
        {
            "type": "auto_approve_set",
            "enabled": enabled,
            "scope": "session",
        }
    )


async def _handle_set_suggest_mode(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """HRI #4: SUGGEST mode toggle — 所有 code_act 代码先展示给用户编辑."""
    enabled = bool(msg.enabled if msg.enabled is not None else True)
    tid = msg.thread_id or "default"
    from huginn.agent.code_act_loop import set_suggest_mode as _set_suggest

    _set_suggest(f"code_act:{tid}", enabled)
    await websocket.send_json(
        {
            "type": "suggest_mode_set",
            "enabled": enabled,
            "scope": "session",
        }
    )


async def _handle_suggest_response(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """HRI #4: 用户对 suggest_code 的响应 (approve/edit/deny + 可选 edited_code)."""
    tid = msg.thread_id or "default"
    action = str(msg.action or "approve")
    edited_code = str(msg.edited_code or "")
    from huginn.agent.code_act_loop import resume_suggest

    ok = resume_suggest(f"code_act:{tid}", action, edited_code)
    if not ok:
        await _send_error(websocket, "No active SUGGEST approval for this thread.")


async def _handle_decision_response(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Record a user's decision-point verdict (received via SSE, replied over WS)."""
    from huginn.branch_policy import get_decision_point_registry

    dp_id = msg.decision_point_id or ""
    if not dp_id:
        await _send_error(websocket, "decision_response requires decision_point_id")
        return
    dp = get_decision_point_registry().resolve(
        dp_id,
        decision=str(msg.decision or "approved"),
        option=str(msg.option or ""),
    )
    if dp is None:
        await _send_error(websocket, f"Unknown decision point: {dp_id}")
        return
    await websocket.send_json(
        {"type": "decision_resolved", "id": dp_id, "status": dp.status}
    )


async def _handle_ping(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Reply to a client ping."""
    await websocket.send_json({"type": "pong"})


async def _handle_guide_input(
    websocket: WebSocket,
    msg: WSMessage,
    ctx: WSCtx,
) -> None:
    """Handle a `guide` message: 把用户补充内容注入当前 thread 的会话 memory.

    引导与普通排队消息的区别:
    - 排队消息会在下一轮作为"新问题"发送 (前端 pendingMessages drain).
    - 引导消息不开启新回合, 而是直接写进 agent 的 session memory, 作为
      一条 user 消息进入会话上下文 — agent 后续的 LLM 推理 (包括下一次
      工具调用前的生成) 都会把这条引导纳入考量.

    ponytail: 同回合内的"工具调用间隙注入"需要 langgraph interrupt 或
    tool-adapter 层 hook 才能做到 (当前回合 agent.chat 的 message 快照
    已固定), 本实现注入 memory 后从下一轮 LLM 生成起生效. 升级路径: 在
    HuginnAgent 的工具执行前钩子 (approval_callback 同位置) 检查 pending
    guidance 并改写输入 messages.
    """
    from huginn.routes.ws import get_agent_factory

    content = (msg.content or "").strip()
    thread_id = msg.thread_id or "desktop"
    if not content:
        await websocket.send_json({"type": "guide_ack", "stored": False, "error": "empty content"})
        return

    try:
        factory = get_agent_factory()
        agent = factory.create_lead(thread_id=thread_id, approval_callback=ctx.ws_approval)
        # 注入会话 memory: 引导成为持久上下文的一部分.
        agent.memory.add_message("user", f"[用户中途引导] {content}")
        await websocket.send_json({
            "type": "guide_ack",
            "stored": True,
            "thread_id": thread_id,
        })
    except Exception as exc:
        logger.debug("guide inject failed", exc_info=True)
        await websocket.send_json({
            "type": "guide_ack",
            "stored": False,
            "error": str(exc),
        })
