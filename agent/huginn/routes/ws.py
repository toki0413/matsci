"""WebSocket endpoint for real-time Agent chat."""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from huginn.config import HuginnConfig
from huginn.routes.schemas import WSMessage
from huginn.server_core import (
    _EDIT_TOOLS,
    _checkpoints,
    _snapshot_directory,
    _state_lock,
    _threads,
    get_agent,
    get_agent_factory,
    get_context,
    get_memory_manager,
)

router = APIRouter(tags=["ws"])


def _make_ws_approval_callback(websocket: WebSocket):
    """Build a sync approval callback that notifies the WebSocket client.

    The adapter calls this synchronously from inside ``_arun``. We can't
    block the event loop waiting for a client reply, so the callback
    fires off an ``approval_request`` event for visibility and
    auto-approves. Clients that want true interactive approval can send
    ``set_auto_approve=false`` to switch the cached agent into ASK mode
    and handle the request/reply flow themselves before the tool runs.
    """

    def callback(tool_name: str, reason: str) -> bool:
        request_id = uuid.uuid4().hex
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                websocket.send_json(
                    {
                        "type": "approval_request",
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "reason": reason,
                        "auto_approved": True,
                    }
                )
            )
        except RuntimeError:
            # No running loop — fall back to plain auto-approve.
            pass
        return True

    return callback


@router.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat."""
    await websocket.accept()

    # Per-connection approval state. The callback auto-approves but still
    # emits approval_request events so the client has full visibility.
    _pending_approvals: dict[str, asyncio.Future[bool]] = {}
    _ws_approval = _make_ws_approval_callback(websocket)

    try:
        while True:
            message = await websocket.receive_text()
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
                cfg_chat = get_config()
                factory = get_agent_factory()
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
                    agent = await get_agent()
                    # The cached global agent was built without an approval
                    # callback, so ASK-mode tools would be silently denied.
                    # Flip on auto-approve to keep them working; the
                    # tool_call events already give the client visibility.
                    if not agent._permission_config.auto_approve_all:
                        agent._permission_config.auto_approve_all = True

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

                # Track this thread
                with _state_lock:
                    if thread_id not in _threads:
                        _threads[thread_id] = {
                            "id": thread_id,
                            "label": thread_id,
                            "created_at": uuid.uuid4().hex,
                            "last_active": uuid.uuid4().hex,
                        }
                    _threads[thread_id]["last_active"] = uuid.uuid4().hex

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
                        traceback.print_exc()
                        await websocket.send_json(
                            {"type": "error", "error": f"Team mode error: {e}"}
                        )
                    continue

                # Augment with RAG context if enabled
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
                    except Exception as e:
                        print(f"[RAG] query failed: {e}")

                # Stream agent responses
                try:
                    full_response = ""
                    seen_tool_calls: set[str] = set()
                    seen_tool_results: set[str] = set()
                    auto_cp_id: str | None = None
                    workspace_path = Path(cfg_chat.workspace).resolve()
                    async for state in agent.chat(content, thread_id):
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
                                    if name in _EDIT_TOOLS and auto_cp_id is None:
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
                                            print(f"[auto-cp] failed: {e}")
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
                                await websocket.send_json(
                                    {
                                        "type": "tool_result",
                                        "id": tid,
                                        "content": str(
                                            getattr(last_msg, "content", "")
                                        ),
                                    }
                                )

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

                    # Signal completion
                    await websocket.send_json({"type": "done"})

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json(
                        {"type": "error", "error": f"Agent error: {str(e)}"}
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
                    traceback.print_exc()
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

            elif msg_type == "set_auto_approve":
                # Let the client toggle auto-approve on the cached agent.
                # When disabled, ASK-mode tools will emit approval_request
                # events (via the callback on fresh agents) or be denied
                # (on the cached agent, which has no callback wired in).
                enabled = bool(data.get("enabled", True))
                try:
                    agent = await get_agent()
                    agent._permission_config.auto_approve_all = enabled
                    await websocket.send_json(
                        {
                            "type": "auto_approve_set",
                            "enabled": enabled,
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "error": f"Cannot set auto_approve: {e}"}
                    )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
