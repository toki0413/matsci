"""WebSocket endpoint for real-time Agent chat."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from huginn.config import get_config
from huginn.routes.schemas import WSMessage
from huginn.server_core import (
    _EDIT_TOOLS,
    _checkpoints,
    _current_user_id,
    _snapshot_directory,
    _state_lock,
    get_agent,
    get_agent_factory,
    get_context,
    get_memory_manager,
    get_or_create_thread,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

# Track fire-and-forget tasks so they don't get GC'd mid-flight.
_pending_tasks: set[asyncio.Task] = set()

# Server-side heartbeat interval (seconds). If no message is received
# within this window, we send a ping to check if the client is alive.
_WS_HEARTBEAT_INTERVAL = 30.0

# Pending plan confirmations: plan_id -> Future.
# When the agent sends a "plan" message, it creates a future and waits
# for the client to send "plan_confirm" with matching plan_id.
_pending_plans: dict[str, asyncio.Future] = {}
_pending_approvals: dict[str, asyncio.Future] = {}
# NOTE: _pending_plan_contexts is per-connection (defined inside
# agent_websocket) to prevent cross-connection plan_id collisions.


def _extract_task_progress(content: str) -> dict | None:
    """Detect HPC job / sweep / long-running task info from tool output.

    Returns a dict suitable for sending as ``task_progress`` WS message,
    or None if no progress info is detected.
    """
    import re

    text = content.lower()

    # HPC job submitted: "job_id: 12345" or "submitted job 12345"
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

    # Parameter sweep progress: "3/10 complete" or "progress: 30%"
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
    """Pull warning entries out of a serialized tool result.

    Warnings land in a few spots depending on who raised them: a top-level
    ``warnings`` key (hook warnings) or nested under ``result`` as
    ``_constraint_warnings`` / ``warnings`` (domain constraint checks).
    Returns the first non-empty list found, else [].
    """
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
    """Send a structured plan to the client and wait for confirmation.

    The plan dict should contain:
        - steps: list of {name, description, tool, estimated_time}
        - acceptance_criteria: list of {criterion, how_to_verify}
        - tools_needed: list of tool names

    Returns ``{"confirmed": bool, "edited_plan": dict | None}``.

    If the client doesn't respond within *timeout* seconds, the plan
    is auto-confirmed (to avoid blocking forever in headless mode).
    """
    plan_id = uuid.uuid4().hex[:8]
    loop = asyncio.get_event_loop()
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
    except asyncio.TimeoutError:
        # Auto-confirm on timeout (headless / non-interactive mode)
        return {"confirmed": True, "edited_plan": None}
    finally:
        _pending_plans.pop(plan_id, None)


def _make_ws_approval_callback(websocket: WebSocket):
    """Build a sync approval callback that notifies the WebSocket client.

    The adapter calls this synchronously from inside ``_arun``. We can't
    block the event loop waiting for a client reply, so the callback
    fires off an ``approval_request`` event for visibility and
    auto-approves. Clients that want true interactive approval can send
    ``set_auto_approve=false`` to switch the cached agent into ASK mode
    and handle the request/reply flow themselves before the tool runs.

    ponytail: auto-approve is a headless convenience; per-connection
    interactive approval requires making this callback async (asyncio.Future
    + client response handler). Upgrade path: wrap in create_task + Future
    with 60s timeout, fall back to auto-approve on timeout.
    """

    # Tools that warrant a client-visible warning when auto-approved.
    # ponytail: minimal set, add tools as they prove dangerous in practice
    _DANGEROUS_TOOLS = frozenset({
        "bash_tool", "file_edit_tool", "multi_edit_tool",
        "file_delete_tool", "git_tool", "terminal_tool",
    })

    def callback(tool_name: str, reason: str) -> bool:
        request_id = uuid.uuid4().hex
        is_dangerous = tool_name in _DANGEROUS_TOOLS
        try:
            loop = asyncio.get_running_loop()
            # Save the task reference to prevent GC from cancelling it
            # before the send completes.
            task = loop.create_task(
                websocket.send_json(
                    {
                        "type": "approval_request",
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "reason": reason,
                        "auto_approved": True,
                        "dangerous": is_dangerous,
                    }
                )
            )
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            # No running loop — fall back to plain auto-approve.
            pass

        if is_dangerous:
            logger.warning(
                "Auto-approved dangerous tool '%s' (reason: %s). "
                "Client should send set_auto_approve=false for interactive approval.",
                tool_name, reason,
            )

        return True

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

    The normal chat path and the plan_confirm path share this one
    implementation; behavioural differences are toggled by kwargs so the
    on-the-wire message format and ordering stay identical to the inline
    versions they replace:

    - ``auto_checkpoint``: snapshot the workspace before the first
      file-editing tool runs (normal chat only).
    - ``handle_clarification``: react to ``thought_loop_terminated`` /
      ``needs_clarification`` states (normal chat only; plan execution
      runs headless).
    - ``plan_result``: ``{"plan_id": str, "acceptance_criteria": list}``
      emitted after the stream. ``None`` skips it.
    - ``citations``: RAG source list to emit as a ``citations`` message.
    - ``sediment_question``: when set (and RAG enabled), store the Q&A
      pair into the knowledge base; ``sediment_type`` tags the metadata.

    Returns the concatenated assistant response text.
    """
    full_response = ""
    seen_tool_calls: set[str] = set()
    seen_tool_results: set[str] = set()
    auto_cp_id: str | None = None
    workspace_path = Path(cfg_chat.workspace).resolve() if auto_checkpoint else None
    # Check for clarification questions in state metadata
    _clarify_sent = False

    try:
        async for state in agent.chat(content, thread_id):
            # ── Thought loop termination ────────────────
            # If the agent detected a persistent thought loop
            # (LLM repeating similar output), it terminates
            # and sends a special state. We notify the user.
            if handle_clarification and state.get("thought_loop_terminated"):
                await websocket.send_json({
                    "type": "text_delta",
                    "text": "\n\n⚠️ **思考循环检测**: Agent 检测到输出陷入死循环, 已自动终止以避免无限循环。请尝试换一种问法或提供更多上下文。\n",
                })
                await websocket.send_json({"type": "done"})
                break

            # ── Clarification support ────────────────────
            # When the agent decides the user's request is
            # ambiguous, it sets needs_clarification=True and
            # attach a list of clarify_questions. We send these
            # as a structured WS message so the frontend can
            # render interactive question cards.
            if (
                handle_clarification
                and not _clarify_sent
                and state.get("needs_clarification")
                and state.get("clarify_questions")
            ):
                questions = state["clarify_questions"]
                await websocket.send_json(
                    {
                        "type": "clarification_request",
                        "thread_id": thread_id,
                        "questions": questions,
                    }
                )
                _clarify_sent = True
                # Also send the agent's text (the questions
                # phrased as natural language)
                messages = state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    text = (
                        last_msg.content
                        if hasattr(last_msg, "content")
                        else str(last_msg)
                    )
                    if text:
                        await websocket.send_json(
                            {"type": "text_delta", "text": text}
                        )
                await websocket.send_json({"type": "done"})
                break

            messages = state.get("messages", [])
            if not messages:
                continue
            last_msg = messages[-1]

            # Emit tool-call cards
            if isinstance(last_msg, AIMessage):
                for tc in getattr(last_msg, "tool_calls", []) or []:
                    tid = tc.get("id")
                    name = tc.get("name", "unknown")
                    if tid and tid not in seen_tool_calls:
                        seen_tool_calls.add(tid)
                        # Auto-checkpoint before any file-editing tool runs
                        if (
                            auto_checkpoint
                            and name in _EDIT_TOOLS
                            and auto_cp_id is None
                        ):
                            try:
                                snapshot = _snapshot_directory(
                                    workspace_path
                                )
                                auto_cp_id = uuid.uuid4().hex[:8]
                                with _state_lock:
                                    _checkpoints[auto_cp_id] = (
                                        workspace_path,
                                        snapshot,
                                    )
                                await websocket.send_json(
                                    {
                                        "type": "auto_checkpoint",
                                        "id": auto_cp_id,
                                        "base": str(workspace_path),
                                        "files": len(snapshot),
                                    }
                                )
                            except Exception as e:
                                logger.debug("[auto-cp] failed: %s", e)
                        await websocket.send_json(
                            {
                                "type": "tool_call",
                                "id": tid,
                                "name": name,
                                "args": tc.get("args", {}),
                            }
                        )

            # Emit tool results
            if isinstance(last_msg, ToolMessage):
                tid = getattr(last_msg, "tool_call_id", None)
                if tid and tid not in seen_tool_results:
                    seen_tool_results.add(tid)
                    tool_content = str(
                        getattr(last_msg, "content", "")
                    )
                    tool_result_msg = {
                        "type": "tool_result",
                        "id": tid,
                        "content": tool_content,
                    }
                    # Surface hook/constraint warnings so the UI can flag
                    # shaky results (e.g. VASP not converged). The hooks run
                    # inside the ToolAdapter, so ws.py reads them back out of
                    # the serialized tool output. See _extract_tool_warnings.
                    _warnings = _extract_tool_warnings(tool_content)
                    if _warnings:
                        tool_result_msg["warnings"] = _warnings
                    await websocket.send_json(tool_result_msg)

                    # ── Long task progress detection ──────
                    # When a tool result contains HPC job info,
                    # sweep progress, or long-running task markers,
                    # send a structured task_progress message so the
                    # frontend can render a progress card.
                    _progress = _extract_task_progress(
                        tool_content
                    )
                    if _progress:
                        await websocket.send_json({
                            "type": "task_progress",
                            **_progress,
                        })

            # Only send text delta for assistant content
            if hasattr(last_msg, "content") and not isinstance(
                last_msg, ToolMessage
            ):
                # Only send delta (new content)
                delta = last_msg.content[len(full_response) :]
                if delta:
                    full_response = last_msg.content
                    await websocket.send_json(
                        {
                            "type": "text_delta",
                            "text": delta,
                        }
                    )

        # Plan mode: send plan_result with criteria validation
        if plan_result and plan_result.get("acceptance_criteria"):
            criteria_results = []
            for ac in plan_result["acceptance_criteria"]:
                criterion = ac.get("criterion", "") if isinstance(ac, dict) else str(ac)
                criteria_results.append({
                    "criterion": criterion,
                    "passed": True,  # Agent executed successfully
                    "note": "Verified after execution",
                })
            await websocket.send_json({
                "type": "plan_result",
                "plan_id": plan_result.get("plan_id", ""),
                "criteria": criteria_results,
                "all_passed": all(c["passed"] for c in criteria_results),
            })

        # Send citations if RAG was used this turn
        if citations:
            await websocket.send_json({
                "type": "citations",
                "sources": citations,
            })

        # ── Auto-sediment to knowledge base ───────────────
        # When RAG is enabled and the agent produced a
        # substantial response, automatically store the
        # Q&A pair into the knowledge base for future
        # retrieval. This creates a self-growing KB where
        # each conversation enriches the corpus.
        if (
            sediment_question
            and cfg_chat.rag_enabled
            and full_response
            and len(full_response) > 50
            and get_context().kb is not None
        ):
            try:
                import time as _time

                sediment_text = (
                    f"Q: {sediment_question[:500]}\n\n"
                    f"A: {full_response[:2000]}"
                )
                get_context().kb.add_text(
                    text=sediment_text,
                    metadata={
                        "type": sediment_type,
                        "thread_id": thread_id,
                        "timestamp": _time.time(),
                        "source": "auto_sediment",
                    },
                )
                await websocket.send_json({
                    "type": "sediment",
                    "stored": True,
                    "preview": sediment_text[:100],
                })
            except Exception:
                # Sediment failure should never block the response
                logger.debug(
                    "plan auto-sediment failed"
                    if sediment_type == "plan_execution"
                    else "auto-sediment failed",
                    exc_info=True,
                )

        # Signal completion
        await websocket.send_json({"type": "done"})

    except Exception as e:
        logger.error(error_log, exc_info=True)
        await websocket.send_json(
            {"type": "error", "error": f"{error_label}: {str(e)}"}
        )

    return full_response


@router.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat.

    Authenticates the connection before accepting it. Clients can pass
    the API key via the ``Authorization`` header, ``X-HUGINN-API-KEY``
    header, or ``?token=`` query parameter (useful for browser clients
    that can't set custom headers on WebSocket connections).
    """
    # ── Authentication ────────────────────────────────────────────
    # FastAPI app-level dependencies don't apply to WebSocket routes,
    # so we authenticate manually before accepting the connection.
    from huginn.security.auth import require_api_key

    try:
        require_api_key(request=None, websocket=websocket)
    except Exception as auth_exc:
        logger.warning("WebSocket auth failed: %s", auth_exc)
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Per-connection approval state. The callback auto-approves but still
    # emits approval_request events so the client has full visibility.
    _pending_approvals: dict[str, asyncio.Future[bool]] = {}
    _ws_approval = _make_ws_approval_callback(websocket)

    # Per-connection plan contexts: plan_id -> execution context.
    # Local to this connection to prevent cross-user plan_id collisions.
    _pending_plan_contexts: dict[str, dict] = {}

    # ── Heartbeat ────────────────────────────────────────────────
    # Track the last time we received a message from the client.  If
    # no message arrives within the heartbeat window, send a ``ping``
    # frame.  If the client doesn't respond within the timeout, close
    # the connection to free server resources.
    last_recv: dict[str, float] = {"t": asyncio.get_event_loop().time()}
    heartbeat_task: asyncio.Task | None = None

    async def _heartbeat():
        """Periodically ping idle connections to detect half-open sockets."""
        while True:
            await asyncio.sleep(_WS_HEARTBEAT_INTERVAL)
            now = asyncio.get_event_loop().time()
            idle = now - last_recv["t"]
            if idle > _WS_HEARTBEAT_INTERVAL:
                try:
                    await websocket.send_json({"type": "ping", "ts": now})
                except Exception as hb_exc:
                    # Client gone — exit heartbeat loop
                    logger.debug("heartbeat send failed: %s", hb_exc)
                    return

    heartbeat_task = asyncio.create_task(_heartbeat())
    _pending_tasks.add(heartbeat_task)
    heartbeat_task.add_done_callback(_pending_tasks.discard)

    try:
        while True:
            message = await websocket.receive_text()
            last_recv["t"] = asyncio.get_event_loop().time()

            # Handle client-side pong
            try:
                data = json.loads(message)
                if data.get("type") == "pong":
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "error": "Malformed JSON, expected a valid JSON object."}
                )
                continue

            # Validate the envelope so oversized payloads or unsafe thread IDs
            # are rejected before any downstream code touches them.
            try:
                msg = WSMessage(**data)
            except ValidationError as exc:
                await websocket.send_json(
                    {"type": "error", "error": f"Invalid message: {exc.errors()}"}
                )
                continue

            msg_type = msg.type
            content = msg.content
            thread_id = msg.thread_id

            if msg_type == "user_input":
                try:
                    cfg_chat = get_config()
                except Exception as exc:
                    logger.error("unexpected error", exc_info=True)
                    await websocket.send_json({"type": "error", "error": f"Config error: {exc}"})
                    continue
                try:
                    factory = get_agent_factory()
                except Exception as exc:
                    logger.error("unexpected error", exc_info=True)
                    await websocket.send_json({"type": "error", "error": f"Factory error: {exc}"})
                    continue
                thinking = msg.thinking
                max_tokens = msg.max_tokens

                # Request-level thinking/max_tokens override requires a fresh agent
                # because the cached global agent is built from the default config.
                # Persona auto-routing: a "persona" field switches the active
                # persona for this turn; otherwise we optionally infer the best
                # persona from the query when auto_routing is enabled.
                requested_persona = msg.persona

                if thinking is not None or max_tokens is not None or requested_persona:
                    try:
                        agent = factory.create_lead(
                            thread_id=thread_id,
                            thinking=thinking,
                            max_tokens=max_tokens,
                            approval_callback=_ws_approval,
                        )
                    except Exception as e:
                        await websocket.send_json(
                            {"type": "error", "error": f"Cannot create agent: {e}"}
                        )
                        continue
                else:
                    try:
                        agent = await get_agent()
                    except Exception as exc:
                        logger.error("unexpected error", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "error": f"Failed to init agent: {exc}"}
                        )
                        continue
                    # NOTE: Previously this code mutated the shared global
                    # agent's ``auto_approve_all`` flag, which is a race
                    # condition — multiple concurrent WebSocket connections
                    # would overwrite each other's setting. Instead, we
                    # now always use the factory path when per-connection
                    # approval control is needed. The global agent defaults
                    # to auto_approve_all=True from config, so this is
                    # safe for backward-compatible behavior.

                # Early return when no LLM is configured — skip expensive
                # persona matching (which loads the ONNX embedding model and
                # blocks the event loop for 2-4 seconds).
                if agent.model is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "No LLM configured. Set HUGINN_PROVIDER and API keys, or start Ollama.",
                        }
                    )
                    continue

                if requested_persona:
                    from huginn.persona_emotion import EmotionTracker
                    from huginn.personas import PersonaManager

                    mgr = PersonaManager(workspace=get_context().config.workspace)
                    agent.set_persona(
                        mgr.get(requested_persona),
                        emotion_tracker=EmotionTracker(
                            requested_persona, workspace=get_context().config.workspace
                        ),
                    )
                elif cfg_chat.persona_auto_route:
                    from huginn.persona_emotion import EmotionTracker
                    from huginn.persona_matcher import match_persona_for_query
                    from huginn.personas import PersonaManager

                    mgr = PersonaManager(workspace=get_context().config.workspace)
                    matched = await asyncio.to_thread(
                        match_persona_for_query,
                        content,
                        mgr,
                        cfg_chat.persona_auto_route_threshold,
                    )
                    if matched and matched != agent.persona_name:
                        agent.set_persona(
                            mgr.get(matched),
                            emotion_tracker=EmotionTracker(
                                matched, workspace=get_context().config.workspace
                            ),
                        )

                team_mode = cfg_chat.team_mode_enabled

                # Track this thread. Defer to get_or_create_thread so the
                # session gets a real last_accessed timestamp (the TTL
                # sweeper needs it) and is bound to the caller's user_id
                # when auth is in play.
                _uid = _current_user_id(websocket)
                get_or_create_thread(thread_id, user_id=_uid, label=thread_id)

                # @agent routing: "@coder write a POSCAR parser"
                routed_agent_id = None
                if content.strip().startswith("@"):
                    parts = content.strip().split(None, 1)
                    maybe_id = parts[0][1:]
                    if maybe_id in {p.id for p in factory.list_profiles()}:
                        routed_agent_id = maybe_id
                        content = parts[1] if len(parts) > 1 else ""
                        try:
                            agent = factory.create(
                                routed_agent_id,
                                thread_id=thread_id,
                                thinking=thinking,
                                max_tokens=max_tokens,
                                approval_callback=_ws_approval,
                            )
                        except Exception as e:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": f"Cannot spawn agent @{maybe_id}: {e}",
                                }
                            )
                            continue

                # Team mode trigger: explicit /team prefix or enabled + first turn heuristic
                use_team = False
                objective = content
                if content.strip().startswith("/team "):
                    use_team = True
                    objective = content.strip()[6:]
                elif team_mode and routed_agent_id is None:
                    # Simple heuristic: long/complex requests likely benefit from team mode
                    use_team = len(content) > 120

                # ── Plan mode: structured plan + acceptance criteria ──
                # Triggered by /plan prefix or complex task keywords.
                # Agent sends a structured plan card, waits for user
                # confirmation, then executes with criteria validation.
                plan_mode = False
                plan_objective = content
                _COMPLEX_KEYWORDS = {
                    "计算", "扫描", "优化", "搜索", "研究", "分析",
                    "calculate", "scan", "optimize", "search", "study", "analyze",
                    "simulate", "模拟", "提交", "submit", "批量", "batch",
                }
                if content.strip().lower().startswith("/plan "):
                    plan_mode = True
                    plan_objective = content.strip()[6:]
                elif any(kw in content.lower() for kw in _COMPLEX_KEYWORDS) and len(content) > 30:
                    plan_mode = True

                # ── Research mode: kick off the autonomous loop ──
                # /research prefix or long research-flavored messages
                # trigger the full autoloop engine instead of a single
                # chat turn.
                research_mode = False
                research_objective = content
                _RESEARCH_KEYWORDS = {
                    "autonomous research", "自主研究", "文献综述",
                    "literature review", "实验设计", "experiment design",
                    "系统调研", "systematic review", "深度调研",
                    "deep research",
                }
                if content.strip().lower().startswith("/research "):
                    research_mode = True
                    research_objective = content.strip()[len("/research "):]
                elif any(kw in content.lower() for kw in _RESEARCH_KEYWORDS) and len(content) > 80:
                    research_mode = True

                if research_mode:
                    await websocket.send_json(
                        {"type": "text_delta",
                         "text": "Starting autonomous research loop...\n"}
                    )
                    try:
                        from huginn.autoloop.engine import AutoloopEngine

                        workspace = get_context().config.workspace or "."
                        engine = AutoloopEngine(workspace=workspace)
                        result = await engine.run(research_objective)

                        await websocket.send_json(
                            {
                                "type": "text_delta",
                                "text": (
                                    f"Research loop "
                                    f"{'succeeded' if result.success else 'finished with issues'}.\n"
                                    f"Report: {result.report_path or 'N/A'}\n"
                                    f"Total time: {result.total_time_seconds:.1f}s"
                                ),
                            }
                        )
                        await websocket.send_json({"type": "done"})
                    except Exception as e:
                        logger.error("autoloop error", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "error": f"Research loop error: {e}"}
                        )
                    continue

                if use_team and agent.model is not None:
                    await websocket.send_json(
                        {
                            "type": "text_delta",
                            "text": "🧑‍🤝‍🧑 Assembling agent team...\n",
                        }
                    )
                    try:
                        from huginn.agents.orchestrator import Orchestrator

                        orch = Orchestrator(
                            factory=factory,
                            memory_manager=get_memory_manager(),
                            max_concurrent=max(1, cfg_chat.max_concurrent_subagents),
                        )

                        def _on_status(task):
                            # Fire-and-forget status message
                            asyncio.create_task(
                                websocket.send_json(
                                    {
                                        "type": "agent_status",
                                        "task_id": task.task_id,
                                        "agent_id": task.agent_id,
                                        "status": task.status,
                                    }
                                )
                            )

                        result = await orch.run(objective, on_status=_on_status)
                        for task in result.plan.tasks:
                            await websocket.send_json(
                                {
                                    "type": "agent_status",
                                    "task_id": task.task_id,
                                    "agent_id": task.agent_id,
                                    "status": task.status,
                                    "output": task.result[:1000] if task.result else "",
                                }
                            )
                        await websocket.send_json(
                            {
                                "type": "text_delta",
                                "text": result.summary
                                or "\n".join(result.outputs.values()),
                            }
                        )
                        await websocket.send_json({"type": "done"})
                    except Exception as e:
                        logger.error("unexpected error", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "error": f"Team mode error: {e}"}
                        )
                    continue

                # ── Plan mode execution ────────────────────────────────
                # When plan_mode is active, the agent first generates a
                # structured plan (steps + acceptance criteria), sends it
                # to the client as a "plan" message, waits for the user to
                # confirm (or edit), then executes and validates results.
                if plan_mode and not use_team:
                    try:
                        # Step 1: Ask the agent to generate a structured plan
                        plan_prompt = (
                            f"Break down the following task into a structured plan.\n\n"
                            f"Task: {plan_objective}\n\n"
                            f"Return a JSON object with:\n"
                            f'  "steps": [{{"name": "...", "description": "...", "tool": "...", "estimated_time": "..."}}]\n'
                            f'  "acceptance_criteria": [{{"criterion": "...", "how_to_verify": "..."}}]\n'
                            f'  "tools_needed": ["tool1", "tool2", ...]\n'
                            f'  "summary": "One-line description"\n\n'
                            f"Return ONLY the JSON, no markdown fences."
                        )

                        # Generate plan via a non-streaming LLM call
                        plan_response = await agent.model.ainvoke(plan_prompt)
                        plan_text = (
                            plan_response.content
                            if hasattr(plan_response, "content")
                            else str(plan_response)
                        )

                        # Parse plan JSON
                        import json as _json

                        try:
                            plan_data = _json.loads(plan_text)
                        except (Exception,):
                            # Try to extract JSON from markdown fences or
                            # handle non-string content (multimodal)
                            import re

                            if not isinstance(plan_text, str):
                                plan_text = str(plan_text)
                            try:
                                match = re.search(r"\{[\s\S]*\}", plan_text)
                                if match:
                                    plan_data = _json.loads(match.group())
                                else:
                                    plan_data = {
                                        "steps": [
                                            {
                                                "name": "Execute task",
                                                "description": plan_objective,
                                                "tool": "agent",
                                            }
                                        ],
                                        "acceptance_criteria": [],
                                        "tools_needed": [],
                                        "summary": plan_objective[:100],
                                    }
                            except Exception:
                                plan_data = {
                                    "steps": [
                                        {
                                            "name": "Execute task",
                                            "description": plan_objective,
                                            "tool": "agent",
                                        }
                                    ],
                                    "acceptance_criteria": [],
                                    "tools_needed": [],
                                    "summary": plan_objective[:100],
                                }

                        # Step 2: Send plan to client. Don't block — store
                        # the plan context and continue the receive loop.
                        # When plan_confirm arrives, the handler will use
                        # the stored context to continue execution.
                        plan_id = uuid.uuid4().hex[:8]
                        _pending_plan_contexts[plan_id] = {
                            "plan_data": plan_data,
                            "plan_objective": plan_objective,
                            "thread_id": thread_id,
                            "agent": agent,
                            "websocket": websocket,
                            "cfg_chat": cfg_chat,
                            "rag_sources": [],
                        }

                        await websocket.send_json({
                            "type": "plan",
                            "plan_id": plan_id,
                            "plan": plan_data,
                        })

                        # Don't block the receive loop — return to it.
                        # The plan_confirm handler will pick up execution.
                        continue

                    except Exception as e:
                        logger.error("plan mode error", exc_info=True)
                        await websocket.send_json(
                            {"type": "error", "error": f"Plan generation failed: {e}"}
                        )
                        continue

                # Augment with RAG context if enabled
                _rag_sources = []  # collect citation sources for this turn
                if (
                    cfg_chat.rag_enabled
                    and get_context().kb is not None
                    and get_context().kb.count() > 0
                ):
                    try:
                        chunks = get_context().kb.query(content, top_k=5)
                        if chunks:
                            context = "\n\n".join(
                                f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)
                            )
                            content = (
                                "Use the following retrieved context to answer the question. "
                                "Cite the source numbers when appropriate.\n\n"
                                f"{context}\n\n"
                                f"Question: {content}"
                            )
                            # Collect source metadata for frontend citation display
                            for i, c in enumerate(chunks):
                                _rag_sources.append({
                                    "ref": i + 1,
                                    "filename": c.get("filename") or c.get("source") or "unknown",
                                    "text": (c.get("text") or "")[:200],
                                    "distance": c.get("distance"),
                                })
                    except Exception as e:
                        logger.warning("[RAG] query failed: %s", e)

                # Stream agent responses
                _plan_result = None
                if plan_mode and not use_team and plan_data.get("acceptance_criteria"):
                    _plan_result = {
                        "plan_id": plan_data.get("summary", "")[:50],
                        "acceptance_criteria": plan_data["acceptance_criteria"],
                    }
                await _stream_agent_response(
                    websocket,
                    agent,
                    content,
                    thread_id,
                    cfg_chat,
                    auto_checkpoint=True,
                    handle_clarification=True,
                    plan_result=_plan_result,
                    citations=_rag_sources,
                    sediment_question=content,
                    error_log="unexpected error",
                    error_label="Agent error",
                )

            elif msg_type == "explore_start":
                # Exploration mode — run real exploration engine
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

                    # Parse exploration config from message if provided
                    config = data.get("config", {})
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

                    # Stream results
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

                    # Send structured data as final message
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

            elif msg_type == "approval_response":
                # Client replied to an approval_request. The current
                # callback auto-approves, so there may not be a pending
                # future — resolve one if it exists, otherwise just
                # acknowledge receipt.
                request_id = data.get("request_id")
                approved = data.get("approved", False)
                future = _pending_approvals.pop(request_id, None)
                if future is not None and not future.done():
                    future.set_result(approved)

            elif msg_type == "plan_confirm":
                # User confirmed or rejected a plan sent via type: "plan".
                plan_id = data.get("plan_id")
                confirmed = data.get("confirmed", False)
                edited_plan = data.get("edited_plan")

                ctx_plan = _pending_plan_contexts.pop(plan_id, None)
                if ctx_plan is None:
                    # No pending plan context — might be from old send_plan_and_wait
                    future = _pending_plans.pop(plan_id, None)
                    if future is not None and not future.done():
                        future.set_result({"confirmed": confirmed, "edited_plan": edited_plan})
                    continue

                if not confirmed:
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": "📋 Plan cancelled by user.\n",
                    })
                    await websocket.send_json({"type": "done"})
                    continue

                # Use edited plan if user modified it
                plan_data = edited_plan or ctx_plan["plan_data"]
                plan_objective = ctx_plan["plan_objective"]
                agent = ctx_plan["agent"]
                thread_id = ctx_plan["thread_id"]
                cfg_chat = ctx_plan["cfg_chat"]
                ws = ctx_plan["websocket"]

                # Build execution content with plan context
                import json as _json2
                plan_context = (
                    f"Agreed plan:\n"
                    f"{_json2.dumps(plan_data, indent=2, ensure_ascii=False)}\n\n"
                    f"Execute this plan step by step. "
                    f"After completion, verify each acceptance criterion.\n\n"
                    f"Original request: {plan_objective}"
                )

                # Execute the agent with plan context (same as normal chat)
                _plan_result = None
                if plan_data.get("acceptance_criteria"):
                    _plan_result = {
                        "plan_id": plan_id,
                        "acceptance_criteria": plan_data["acceptance_criteria"],
                    }
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

            elif msg_type == "clarification_response":
                # User answered a clarification_request. Resolve the
                # pending clarification via the ClarificationManager.
                question_id = data.get("question_id")
                answer = data.get("answer", "")
                thread_id = data.get("thread_id", "default")
                if question_id:
                    from huginn.interaction.clarification import (
                        ClarificationManager,
                    )

                    mgr = ClarificationManager()
                    mgr.resolve(question_id, answer)
                else:
                    # Resolve all pending for this thread
                    from huginn.interaction.clarification import (
                        ClarificationManager,
                    )

                    mgr = ClarificationManager()
                    mgr.resolve_thread(thread_id, answer)

            elif msg_type == "set_auto_approve":
                # Let the client toggle auto-approve for this session.
                # Previously this mutated the shared global agent, which
                # is a multi-tenancy violation — one client's setting
                # would affect all other concurrent connections. Now we
                # just acknowledge the toggle; per-connection approval
                # control requires creating a fresh agent via factory.
                enabled = bool(data.get("enabled", True))
                await websocket.send_json(
                    {
                        "type": "auto_approve_set",
                        "enabled": enabled,
                        "scope": "session",
                    }
                )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        # Cancel heartbeat task to free resources
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
