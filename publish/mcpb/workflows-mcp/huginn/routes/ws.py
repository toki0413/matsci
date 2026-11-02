"""WebSocket endpoint for real-time Agent chat.

Thin dispatcher — all heavy logic lives in ws_helpers.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from huginn.config import get_config
from huginn.routes.schemas import WSMessage
from huginn.routes.ws_helpers import (
    WSCtx,
    _handle_approval_response,
    _handle_clarification_response,
    _handle_decision_response,
    _handle_explore_start,
    _handle_guide_input,
    _handle_ping,
    _handle_plan_confirm,
    _handle_set_auto_approve,
    _handle_set_suggest_mode,
    _handle_suggest_response,
    _handle_user_input,
    _make_ws_approval_callback,
    _pending_plans,
    _pending_tasks,
    _send_error,
    _stream_agent_response,
    send_plan_and_wait,
)
from huginn.server_core import (
    get_agent,
    get_agent_factory,
    get_context,
    get_memory_manager,
    get_or_create_thread,
)

# Re-export for backward compat (tests import / monkeypatch from here)
__all__ = [
    "router",
    "_stream_agent_response",
    "send_plan_and_wait",
    "_pending_plans",
    "_pending_tasks",
    "get_config",
    "get_agent",
    "get_agent_factory",
    "get_context",
    "get_memory_manager",
    "get_or_create_thread",
]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

# Message-type → handler registry. Every handler shares the signature
# ``(websocket, msg: WSMessage, ctx: WSCtx)``, so adding a new message type
# is a one-line mapping instead of a new elif branch. Handlers read their
# arguments off the validated ``WSMessage`` (msg), never the raw dict.
_MESSAGE_HANDLERS: dict[str, Any] = {
    "user_input": _handle_user_input,
    "explore_start": _handle_explore_start,
    "guide": _handle_guide_input,
    "approval_response": _handle_approval_response,
    "plan_confirm": _handle_plan_confirm,
    "clarification_response": _handle_clarification_response,
    "set_auto_approve": _handle_set_auto_approve,
    "set_suggest_mode": _handle_set_suggest_mode,
    "suggest_response": _handle_suggest_response,
    "decision_response": _handle_decision_response,
    "ping": _handle_ping,
}

_WS_HEARTBEAT_INTERVAL = 30.0


def _ws_build_ctx(websocket: WebSocket, identity: Any) -> WSCtx:
    """Build the per-connection state bundle + approval callback."""
    pending_approvals: dict[str, asyncio.Future[bool]] = {}
    session_auto_approve: dict[str, bool] = {"enabled": True}
    last_user_context: dict = {}
    pending_approval_contexts: dict[str, dict] = {}
    ws_approval = _make_ws_approval_callback(
        websocket,
        session_auto_approve=session_auto_approve,
        pending_approvals=pending_approvals,
        last_user_context=last_user_context,
        pending_approval_contexts=pending_approval_contexts,
    )
    return WSCtx(
        ws_approval=ws_approval,
        session_auto_approve=session_auto_approve,
        last_user_context=last_user_context,
        pending_approvals=pending_approvals,
        pending_approval_contexts=pending_approval_contexts,
        pending_plan_contexts={},
        user_id=identity,
    )


def _ws_start_task(coro: Any) -> asyncio.Task:
    """Start a background task and track it for shutdown."""
    task = asyncio.create_task(coro)
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return task


async def _ws_heartbeat_loop(
    websocket: WebSocket, last_recv: dict[str, float]
) -> None:
    while True:
        await asyncio.sleep(_WS_HEARTBEAT_INTERVAL)
        now = asyncio.get_running_loop().time()
        if now - last_recv["t"] <= _WS_HEARTBEAT_INTERVAL:
            continue
        try:
            await websocket.send_json({"type": "ping", "ts": now})
        except Exception as hb_exc:
            logger.debug("heartbeat send failed: %s", hb_exc)
            return


async def _ws_forward_pet_events(websocket: WebSocket) -> None:
    from huginn.pet import get_pet_bus

    bus = get_pet_bus()
    q, unsub = await bus.queue()
    try:
        while True:
            event = await q.get()
            st = bus.state
            await websocket.send_json({
                "type": "pet_update",
                "mood": st.mood.value,
                "message": event.message,
                "xp": st.experience,
                "level": st.level,
                "hunger": st.hunger,
                "happiness": st.happiness,
                "active_tasks": st.active_tasks,
            })
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("pet event forward error", exc_info=True)
    finally:
        unsub()


def _ws_parse_envelope(message: str) -> tuple[WSMessage | None, str | None]:
    """Parse raw JSON into a validated WSMessage; return (msg, error)."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None, "Malformed JSON, expected a valid JSON object."
    try:
        return WSMessage(**data), None
    except ValidationError as exc:
        return None, f"Invalid message: {exc.errors()}"


async def _ws_dispatch(websocket: WebSocket, msg: WSMessage, ctx: WSCtx) -> None:
    """Look up the handler for ``msg.type`` and run it, reporting errors."""
    handler = _MESSAGE_HANDLERS.get(msg.type)
    if handler is None:
        await _send_error(websocket, f"Unknown message type: {msg.type}")
        return
    try:
        await handler(websocket, msg, ctx)
    except Exception as dispatch_exc:
        logger.error("dispatch error: %s", dispatch_exc, exc_info=True)
        await _send_error(websocket, f"Handler error: {dispatch_exc}")


async def _ws_receive_loop(
    websocket: WebSocket, ctx: WSCtx, last_recv: dict[str, float]
) -> None:
    from huginn.middleware.ws_governance import WSMessageRateLimiter

    rate_limiter = WSMessageRateLimiter()
    while True:
        message = await websocket.receive_text()
        if not rate_limiter.check():
            await _send_error(websocket, "Too many messages, slow down.")
            continue
        last_recv["t"] = asyncio.get_running_loop().time()

        # Handle client-side pong
        try:
            if json.loads(message).get("type") == "pong":
                continue
        except (json.JSONDecodeError, TypeError):
            logger.debug("best-effort op failed", exc_info=True)

        msg, error = _ws_parse_envelope(message)
        if error is not None:
            await _send_error(websocket, error)
            continue
        await _ws_dispatch(websocket, msg, ctx)


async def _ws_cancel_tasks(*tasks: asyncio.Task | None) -> None:
    for task in tasks:
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@router.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat.

    Authenticates, sets up heartbeat + pet event forwarding, then dispatches
    incoming messages to handler functions in ws_helpers.py.
    """
    from huginn.middleware.ws_governance import get_tracker, ws_auth_and_track

    identity = await ws_auth_and_track(websocket)
    if identity is None:
        return

    # 握手已由 ws_auth_and_track 统一完成 (含 first-message auth 与断开保护).
    ctx = _ws_build_ctx(websocket, identity)
    last_recv: dict[str, float] = {"t": asyncio.get_running_loop().time()}
    heartbeat_task = _ws_start_task(_ws_heartbeat_loop(websocket, last_recv))
    pet_task = _ws_start_task(_ws_forward_pet_events(websocket))

    try:
        await _ws_receive_loop(websocket, ctx, last_recv)
    except WebSocketDisconnect:
        logger.debug("best-effort op failed", exc_info=True)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        await _ws_cancel_tasks(heartbeat_task, pet_task)
        get_tracker().release(identity)
