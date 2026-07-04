"""Plan-build mode endpoint."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter
from langchain_core.messages import ToolMessage

from huginn.server_core import get_context, get_planner_agent

router = APIRouter(tags=["planner"])

logger = logging.getLogger(__name__)


@router.post("/plan")
async def generate_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a step-by-step plan without executing any tools."""
    agent = get_planner_agent()
    if agent.model is None:
        return {
            "error": "No LLM configured. Set provider and API key to generate plans."
        }

    content = params.get("content", "")
    thread_id = params.get("thread_id", "plan")
    if not content.strip():
        return {"error": "content is required"}

    # Optionally ground the plan with codebase search results
    if get_context().codebase is not None:
        try:
            results = await asyncio.to_thread(
                get_context().codebase.search, content, top_k=3
            )
            if results:
                ctx = "\n\n".join(
                    f"[{i+1}] {r['path']}\n{r['text']}" for i, r in enumerate(results)
                )
                content = (
                    "Use the following relevant codebase snippets to inform your plan. "
                    "Do not execute any actions; just plan.\n\n"
                    f"{ctx}\n\n"
                    f"Request: {content}"
                )
        except Exception as e:
            print(f"[plan] codebase search warning: {e}")

    try:
        full_response = ""
        async for state in agent.chat(content, thread_id):
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "content") and not isinstance(last, ToolMessage):
                    full_response = last.content
        return {"plan": full_response}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"error": f"Planner error: {str(e)}"}
