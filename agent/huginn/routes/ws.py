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
    _handle_explore_start,
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
    "approval_response": _handle_approval_response,
    "plan_confirm": _handle_plan_confirm,
    "clarification_response": _handle_clarification_response,
    "set_auto_approve": _handle_set_auto_approve,
    "set_suggest_mode": _handle_set_suggest_mode,
    "suggest_response": _handle_suggest_response,
    "ping": _handle_ping,
}

_WS_HEARTBEAT_INTERVAL = 30.0


@router.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat.

    Authenticates, sets up heartbeat + pet event forwarding, then dispatches
    incoming messages to handler functions in ws_helpers.py.
    """
    from huginn.middleware.ws_governance import (
        WSMessageRateLimiter,
        get_tracker,
        ws_auth_and_track,
    )

    identity = await ws_auth_and_track(websocket)
    if identity is None:
        return

    # ws_auth_and_track may have already accepted (first-message auth)
    if not websocket.scope.get("_ws_pre_accepted"):
        await websocket.accept()

    # Per-connection state
    _pending_approvals: dict[str, asyncio.Future[bool]] = {}
    session_auto_approve: dict[str, bool] = {"enabled": True}
    _last_user_context: dict = {}
    _pending_approval_contexts: dict[str, dict] = {}
    _ws_approval = _make_ws_approval_callback(
        websocket,
        session_auto_approve=session_auto_approve,
        pending_approvals=_pending_approvals,
        last_user_context=_last_user_context,
        pending_approval_contexts=_pending_approval_contexts,
    )
    _pending_plan_contexts: dict[str, dict] = {}

    # ── Heartbeat ────────────────────────────────────────────────
    last_recv: dict[str, float] = {"t": asyncio.get_running_loop().time()}
    heartbeat_task: asyncio.Task | None = None

    async def _heartbeat():
        while True:
            await asyncio.sleep(_WS_HEARTBEAT_INTERVAL)
            now = asyncio.get_running_loop().time()
            idle = now - last_recv["t"]
            if idle > _WS_HEARTBEAT_INTERVAL:
                try:
                    await websocket.send_json({"type": "ping", "ts": now})
                except Exception as hb_exc:
                    logger.debug("heartbeat send failed: %s", hb_exc)
                    return

    heartbeat_task = asyncio.create_task(_heartbeat())
    _pending_tasks.add(heartbeat_task)
    heartbeat_task.add_done_callback(_pending_tasks.discard)

    # ── Pet event forwarding ──────────────────────────────────────
    async def _forward_pet_events():
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

    pet_task = asyncio.create_task(_forward_pet_events())
    _pending_tasks.add(pet_task)
    pet_task.add_done_callback(_pending_tasks.discard)

    rate_limiter = WSMessageRateLimiter()
    try:
        while True:
            message = await websocket.receive_text()
            if not rate_limiter.check():
                await _send_error(websocket, "Too many messages, slow down.")
                continue
            last_recv["t"] = asyncio.get_running_loop().time()

            # Handle client-side pong
            try:
                data = json.loads(message)
                if data.get("type") == "pong":
                    continue
            except (json.JSONDecodeError, TypeError):
                logger.debug("best-effort op failed", exc_info=True)

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await _send_error(
                    websocket, "Malformed JSON, expected a valid JSON object."
                )
                continue

            # Validate envelope
            try:
                msg = WSMessage(**data)
            except ValidationError as exc:
                await _send_error(websocket, f"Invalid message: {exc.errors()}")
                continue

            msg_type = msg.type

            # Look up handler from the registry (one-line per message type)
            # instead of a long if/elif chain. Every handler receives the
            # same (websocket, msg, ctx) signature.
            handler = _MESSAGE_HANDLERS.get(msg_type)
            if handler is None:
                await _send_error(
                    websocket, f"Unknown message type: {msg_type}"
                )
                continue

            ctx = WSCtx(
                ws_approval=_ws_approval,
                session_auto_approve=session_auto_approve,
                last_user_context=_last_user_context,
                pending_approvals=_pending_approvals,
                pending_approval_contexts=_pending_approval_contexts,
                pending_plan_contexts=_pending_plan_contexts,
                user_id=identity,
            )

            # Wrap handlers so errors always reach the client instead of
            # silently killing the connection and hanging on receive_json.
            try:
                await handler(websocket, msg, ctx)

            except Exception as dispatch_exc:
                logger.error("dispatch error: %s", dispatch_exc, exc_info=True)
                await _send_error(websocket, f"Handler error: {dispatch_exc}")

    except WebSocketDisconnect:
        logger.debug("best-effort op failed", exc_info=True)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        for _t in (heartbeat_task, pet_task):
            if _t and not _t.done():
                _t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await _t
        get_tracker().release(identity)
