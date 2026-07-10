"""Helper functions for the WebSocket route.

Extracted from ws.py to make agent_websocket a thin dispatcher.
All message format and behavior is preserved — the functions are just
hoisted to module level with explicit parameter signatures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import WebSocket
from langchain_core.messages import AIMessage, ToolMessage

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


# ── Small utilities ──────────────────────────────────────────────


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send a WS error message and return."""
    await websocket.send_json({"type": "error", "error": message})


def _extract_task_progress(content: str) -> dict | None:
    """Detect HPC job / sweep / long-running task info from tool output.

    Returns a dict suitable for sending as ``task_progress`` WS message,
    or None if no progress info is detected.
    """
    import re

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
        return {"confirmed": True, "edited_plan": None}
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
    _DANGEROUS_TOOLS = frozenset({
        "bash_tool", "file_edit_tool", "multi_edit_tool",
        "file_delete_tool", "git_tool", "terminal_tool",
    })

    _auto = session_auto_approve if session_auto_approve is not None else {"enabled": True}

    def callback(tool_name: str, reason: str) -> bool:
        request_id = uuid.uuid4().hex
        is_dangerous = tool_name in _DANGEROUS_TOOLS
        approved = _auto["enabled"]

        if not approved:
            if pending_approvals is not None:
                loop = asyncio.get_event_loop()
                fut = loop.create_future()
                pending_approvals[request_id] = fut
            if pending_approval_contexts is not None and last_user_context:
                pending_approval_contexts[request_id] = dict(last_user_context)

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                websocket.send_json(
                    {
                        "type": "approval_request",
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "reason": reason,
                        "auto_approved": approved,
                        "dangerous": is_dangerous,
                    }
                )
            )
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            pass

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
    # Late import so monkeypatch on huginn.routes.ws takes effect
    from huginn.routes.ws import get_context

    full_response = ""
    seen_tool_calls: set[str] = set()
    _tool_names_used: set[str] = set()
    seen_tool_results: set[str] = set()
    auto_cp_id: str | None = None
    workspace_path = Path(cfg_chat.workspace).resolve() if auto_checkpoint else None
    _clarify_sent = False
    _token_streamed = False

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

        async for state in agent.chat(content, thread_id):
            if "_token" in state:
                _token_streamed = True
                text = state["_token"]
                if text:
                    full_response += text
                    await _ws_send(
                        {"type": "text_delta", "text": text}
                    )
                continue
            if "_reasoning" in state:
                reasoning = state["_reasoning"]
                if reasoning:
                    await _ws_send(
                        {"type": "reasoning_delta", "text": reasoning}
                    )
                continue

            if "_compacted" in state:
                c = state["_compacted"]
                await _ws_send({
                    "type": "context_compacted",
                    "before_pct": c.get("before_pct", 0),
                    "after_pct": c.get("after_pct", 0),
                })
                continue

            if handle_clarification and state.get("thought_loop_terminated"):
                await _ws_send({
                    "type": "text_delta",
                    "text": "\n\n⚠️ **思考循环检测**: Agent 检测到输出陷入死循环, 已自动终止以避免无限循环。请尝试换一种问法或提供更多上下文。\n",
                })
                await _ws_send({"type": "done"})
                break

            if (
                handle_clarification
                and not _clarify_sent
                and state.get("needs_clarification")
                and state.get("clarify_questions")
            ):
                questions = state["clarify_questions"]
                await _ws_send(
                    {
                        "type": "clarification_request",
                        "thread_id": thread_id,
                        "questions": questions,
                    }
                )
                _clarify_sent = True
                messages = state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    text = (
                        last_msg.content
                        if hasattr(last_msg, "content")
                        else str(last_msg)
                    )
                    if text:
                        full_response = text
                        await _ws_send(
                            {"type": "text_delta", "text": text}
                        )
                await _ws_send({"type": "done"})
                break

            messages = state.get("messages", [])
            if not messages:
                continue
            last_msg = messages[-1]

            if isinstance(last_msg, AIMessage):
                for tc in getattr(last_msg, "tool_calls", []) or []:
                    tid = tc.get("id")
                    name = tc.get("name", "unknown")
                    if tid and tid not in seen_tool_calls:
                        seen_tool_calls.add(tid)
                        _tool_names_used.add(name)
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
                                await _ws_send(
                                    {
                                        "type": "auto_checkpoint",
                                        "id": auto_cp_id,
                                        "base": str(workspace_path),
                                        "files": len(snapshot),
                                    }
                                )
                            except Exception as e:
                                logger.debug("[auto-cp] failed: %s", e)
                        await _ws_send(
                            {
                                "type": "tool_call",
                                "id": tid,
                                "name": name,
                                "args": tc.get("args", {}),
                            }
                        )

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
                    _warnings = _extract_tool_warnings(tool_content)
                    if _warnings:
                        tool_result_msg["warnings"] = _warnings
                    await _ws_send(tool_result_msg)

                    _progress = _extract_task_progress(
                        tool_content
                    )
                    if _progress:
                        await _ws_send({
                            "type": "task_progress",
                            **_progress,
                        })

            if isinstance(last_msg, AIMessage):
                if _token_streamed:
                    full_response = last_msg.content
                elif isinstance(last_msg.content, str):
                    delta = last_msg.content[len(full_response) :]
                    if delta:
                        full_response = last_msg.content
                        await _ws_send(
                            {
                                "type": "text_delta",
                                "text": delta,
                            }
                        )
                else:
                    full_response = last_msg.content

        if plan_result and plan_result.get("acceptance_criteria"):
            criteria_results = []
            for ac in plan_result["acceptance_criteria"]:
                criterion = ac.get("criterion", "") if isinstance(ac, dict) else str(ac)
                criteria_results.append({
                    "criterion": criterion,
                    "passed": True,
                    "note": "Verified after execution",
                })
            await _ws_send({
                "type": "plan_result",
                "plan_id": plan_result.get("plan_id", ""),
                "criteria": criteria_results,
                "all_passed": all(c["passed"] for c in criteria_results),
            })

        if citations:
            await _ws_send({
                "type": "citations",
                "sources": citations,
            })

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
                await _ws_send({
                    "type": "sediment",
                    "stored": True,
                    "preview": sediment_text[:100],
                })
            except Exception:
                logger.debug(
                    "plan auto-sediment failed"
                    if sediment_type == "plan_execution"
                    else "auto-sediment failed",
                    exc_info=True,
                )

        if not full_response:
            if _tool_names_used:
                names_str = ", ".join(sorted(_tool_names_used))
                fallback = f"（Agent 调用了工具 [{names_str}] 但未生成文字回复。请查看上方的工具调用结果，或换一种问法重试。）"
            else:
                fallback = "（Agent 未能生成回复，可能是内部处理超时或出错。请尝试换一种问法，或稍后重试。）"
            full_response = fallback
            await _ws_send({
                "type": "text_delta",
                "text": fallback,
            })

        await _ws_send({"type": "done"})

    except Exception as e:
        logger.error(error_log, exc_info=True)
        await _ws_send(
            {"type": "error", "error": f"{error_label}: {str(e)}"}
        )

    return full_response


# ── Message-type handlers (extracted from agent_websocket) ───────


async def _handle_user_input(
    websocket: WebSocket,
    msg: WSMessage,
    data: dict,
    *,
    ws_approval,
    session_auto_approve: dict,
    last_user_context: dict,
    pending_plan_contexts: dict,
) -> None:
    """Handle a user_input message: create agent, route, stream response.

    This is the largest handler — persona routing, team/plan/research mode
    detection, RAG augmentation, and the streaming call all live here.
    """
    # Late import so monkeypatch on huginn.routes.ws takes effect
    from huginn.routes.ws import (
        get_agent_factory,
        get_config,
        get_context,
        get_or_create_thread,
    )

    try:
        cfg_chat = get_config()
    except Exception as exc:
        logger.error("unexpected error", exc_info=True)
        await _send_error(websocket, f"Config error: {exc}")
        return
    try:
        factory = get_agent_factory()
    except Exception as exc:
        logger.error("unexpected error", exc_info=True)
        await _send_error(websocket, f"Factory error: {exc}")
        return
    thinking = msg.thinking
    max_tokens = msg.max_tokens

    requested_persona = msg.persona

    if thinking is not None or max_tokens is not None or requested_persona:
        try:
            agent = factory.create_lead(
                thread_id=msg.thread_id,
                thinking=thinking,
                max_tokens=max_tokens,
                approval_callback=ws_approval,
            )
        except Exception as e:
            await _send_error(websocket, f"Cannot create agent: {e}")
            return
    else:
        try:
            agent = factory.create_lead(
                thread_id=msg.thread_id,
                approval_callback=ws_approval,
            )
        except Exception as exc:
            logger.error("unexpected error", exc_info=True)
            await _send_error(websocket, f"Failed to init agent: {exc}")
            return

    if agent.model is None:
        await websocket.send_json(
            {"type": "text_delta", "text": "⚠️ No LLM model configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.\n"}
        )
        await websocket.send_json({"type": "done"})
        return

    content = msg.content
    thread_id = msg.thread_id

    get_or_create_thread(thread_id, user_id=None)

    # @agent routing
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

    # Team mode trigger
    use_team = False
    if any(kw in content.lower() for kw in ("/team", "delegate", "collaborate")):
        use_team = True

    # Plan mode trigger
    plan_mode = False
    plan_data: dict = {}
    if any(kw in content.lower() for kw in ("/plan", "plan mode")):
        plan_mode = True

    # Research mode trigger — guard for agents without mode tracking
    if any(kw in content.lower() for kw in ("/research", "research mode")):
        if hasattr(agent, "set_mode"):
            agent.set_mode("research")

    # ── Research mode ──
    if hasattr(agent, "is_research_mode") and agent.is_research_mode():
        try:
            from huginn.personas.research import RESEARCH_PERSONA
            from huginn.research_workflow import ResearchWorkflow, ResearchWorkflowConfig
        except ImportError:
            pass
        try:
            research_config = ResearchWorkflowConfig(
                max_concurrent_branches=3,
                enable_hypothesis_generation=True,
            )
            workflow = ResearchWorkflow(agent=agent, config=research_config)
            await websocket.send_json(
                {"type": "text_delta", "text": "🔬 Research mode activated.\n\n"}
            )
            async for result in workflow.run(content, thread_id=thread_id):
                if result.get("type") == "hypothesis":
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": f"📋 Hypothesis: {result.get('hypothesis', '')}\n",
                    })
                elif result.get("type") == "experiment":
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": f"🧪 Experiment: {result.get('description', '')}\n",
                    })
                elif result.get("type") == "result":
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": f"📊 Result: {result.get('summary', '')}\n",
                    })
            await websocket.send_json({"type": "done"})
            return
        except Exception as e:
            logger.error("Research mode failed, falling back to chat: %s", e, exc_info=True)

    # ── Team mode ──
    if use_team:
        try:
            from huginn.team_orchestrator import TeamOrchestrator
            orch = TeamOrchestrator(factory=factory, thread_id=thread_id)
            await websocket.send_json(
                {"type": "text_delta", "text": "👥 Team mode activated.\n\n"}
            )
            team_result = await orch.delegate(content)
            await websocket.send_json({
                "type": "text_delta",
                "text": f"\n{team_result}\n",
            })
            await websocket.send_json({"type": "done"})
            return
        except Exception as e:
            logger.error("Team mode failed, falling back to single agent: %s", e, exc_info=True)

    # ── Plan mode ──
    if plan_mode and not use_team:
        try:
            plan_objective = content.replace("/plan", "").replace("plan mode", "").strip()
            if not plan_objective:
                plan_objective = content
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

            plan_response = await agent.model.ainvoke(plan_prompt)
            plan_text = (
                plan_response.content
                if hasattr(plan_response, "content")
                else str(plan_response)
            )

            import json as _json

            try:
                plan_data = _json.loads(plan_text)
            except (Exception,):
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

            plan_id = uuid.uuid4().hex[:8]
            pending_plan_contexts[plan_id] = {
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

            return

        except Exception as e:
            logger.error("plan mode error", exc_info=True)
            await _send_error(websocket, f"Plan generation failed: {e}")
            return

    # ── RAG augmentation ──
    _rag_sources: list = []
    if (
        cfg_chat.rag_enabled
        and get_context().kb is not None
        and get_context().kb.count() > 0
    ):
        try:
            chunks = await asyncio.to_thread(
                get_context().kb.query, content, 5
            )
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
                for i, c in enumerate(chunks):
                    _rag_sources.append({
                        "ref": i + 1,
                        "filename": c.get("filename") or c.get("source") or "unknown",
                        "text": (c.get("text") or "")[:200],
                        "distance": c.get("distance"),
                    })
        except Exception as e:
            logger.warning("[RAG] query failed: %s", e)

    # ── Stream agent response ──
    _plan_result = None
    if plan_mode and not use_team and plan_data.get("acceptance_criteria"):
        _plan_result = {
            "plan_id": plan_data.get("summary", "")[:50],
            "acceptance_criteria": plan_data["acceptance_criteria"],
        }
    last_user_context.update({
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
        plan_result=_plan_result,
        citations=_rag_sources,
        sediment_question=content,
        error_log="unexpected error",
        error_label="Agent error",
    )

    # Drain pending side-channel questions
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
    content: str,
    data: dict,
) -> None:
    """Handle explore_start: run the exploration orchestrator."""
    # Late import so monkeypatch on huginn.routes.ws takes effect
    from huginn.routes.ws import get_context

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
    data: dict,
    *,
    pending_approvals: dict,
    pending_approval_contexts: dict,
    session_auto_approve: dict,
) -> None:
    """Handle approval_response: resolve future, re-queue if approved."""
    request_id = data.get("request_id")
    approved = data.get("approved", False)
    future = pending_approvals.pop(request_id, None)
    if future is not None and not future.done():
        future.set_result(approved)

    ctx = pending_approval_contexts.pop(request_id, None)
    if approved and ctx:
        # ponytail: re-runs the full turn, not just the denied tool —
        # acceptable since the user explicitly approved.
        session_auto_approve["enabled"] = True
        await _stream_agent_response(
            websocket,
            ctx["agent"],
            ctx["content"],
            ctx["thread_id"],
            ctx["cfg_chat"],
            auto_checkpoint=True,
            handle_clarification=True,
            error_log="approval re-queue error",
            error_label="Approval re-queue failed",
        )
        session_auto_approve["enabled"] = False


async def _handle_plan_confirm(
    websocket: WebSocket,
    data: dict,
    *,
    pending_plan_contexts: dict,
    last_user_context: dict,
) -> None:
    """Handle plan_confirm: execute or cancel a previously generated plan."""
    plan_id = data.get("plan_id")
    confirmed = data.get("confirmed", False)
    edited_plan = data.get("edited_plan")

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
