"""Multi-provider, multi-agent, personas, orchestration, telemetry, and swarm endpoints."""

from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import APIRouter

from huginn.config import HuginnConfig
from huginn.models.registry import ModelRegistry
from huginn.server_core import (
    get_agent,
    get_agent_factory,
    get_context,
    get_memory_manager,
)

router = APIRouter(tags=["agents"])


# ── Models & Agents ──────────────────────────────────────────────


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List configured model aliases."""
    try:
        cfg = HuginnConfig.from_env()
        registry = ModelRegistry.from_config(cfg)
        return {"models": [m.__dict__ for m in registry.list()]}
    except Exception as e:
        return {"error": str(e)}


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List configured agent profiles."""
    try:
        factory = get_agent_factory()
        profiles = factory.list_profiles()
        return {
            "agents": [
                {
                    "id": p.id,
                    "name": p.name or p.id,
                    "model_alias": p.model_alias,
                    "persona": p.persona,
                    "tools": p.tools,
                    "enabled": p.enabled,
                    "max_steps": p.max_steps,
                }
                for p in profiles
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a single-turn message to a specific agent profile."""
    try:
        factory = get_agent_factory()
        agent = factory.create(
            agent_id,
            thread_id=params.get("thread_id", "default"),
            thinking=params.get("thinking"),
            max_tokens=params.get("max_tokens"),
        )
        state = agent.invoke(params.get("message", ""))
        messages = state.get("messages", [])
        content = ""
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
        return {"agent_id": agent_id, "content": content}
    except Exception as e:
        return {"error": str(e)}


# ── Personas ─────────────────────────────────────────────────────


@router.get("/personas")
async def list_personas() -> dict[str, Any]:
    """List available personas and the current default."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        return {
            "default": mgr.get_default_name(),
            "personas": [
                {
                    "name": name,
                    "system_prompt": mgr.get(name).system_prompt[:200],
                    "begin_dialogs": mgr.get(name).begin_dialogs,
                    "avatar": mgr.get(name).avatar,
                }
                for name in mgr.list()
            ],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/personas/{name}")
async def get_persona(name: str) -> dict[str, Any]:
    """Get a single persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.get(name)
        return {
            "success": True,
            "name": p.name,
            "system_prompt": p.system_prompt,
            "begin_dialogs": p.begin_dialogs,
            "mood_dialogs": p.mood_dialogs,
            "variables": p.variables,
            "avatar": p.avatar,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas")
async def create_persona(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.create(
            name=params["name"],
            system_prompt=params.get("system_prompt", ""),
            begin_dialogs=params.get("begin_dialogs", []),
            mood_dialogs=params.get("mood_dialogs", []),
            variables=params.get("variables", {}),
            avatar=params.get("avatar"),
        )
        return {"success": True, "persona": p.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas/match")
async def match_persona(params: dict[str, Any]) -> dict[str, Any]:
    """Match a query to the most suitable persona."""
    from huginn.persona_matcher import PersonaMatcher
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        matcher = PersonaMatcher(manager=mgr)
        results = matcher.match(
            params.get("query", ""),
            top_k=int(params.get("top_k", 3)),
            score_threshold=float(params.get("threshold", 0.3)),
        )
        return {
            "success": True,
            "matches": [
                {
                    "name": p.name,
                    "score": float(score),
                    "description": p.description,
                    "when_to_use": p.when_to_use,
                }
                for p, score in results
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.patch("/personas/{name}/default")
async def set_default_persona(name: str) -> dict[str, Any]:
    """Set the default persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        mgr.set_default(name)
        return {"success": True, "default": mgr.get_default_name()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/personas/{name}")
async def delete_persona(name: str) -> dict[str, Any]:
    """Delete a user-defined persona."""
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        mgr.delete(name)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/personas/{name}/switch")
async def switch_persona(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Switch the active persona for the current chat session."""
    from huginn.persona_emotion import EmotionTracker
    from huginn.personas import PersonaManager

    try:
        mgr = PersonaManager(workspace=get_context().config.workspace)
        p = mgr.get(name)
        os.environ["HUGINN_PERSONA"] = name
        get_context().agent = None  # force re-init with new default persona
        tracker = EmotionTracker(name, workspace=get_context().config.workspace)
        return {
            "success": True,
            "persona": p.name,
            "system_prompt": p.system_prompt,
            "begin_dialogs": p.begin_dialogs,
            "emotion": tracker.current_state().to_dict(),
            "context_prompt": tracker.context_prompt(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/personas/{name}/emotion")
async def get_persona_emotion(name: str) -> dict[str, Any]:
    """Return the current emotional trajectory for a persona."""
    from huginn.persona_emotion import EmotionTracker

    try:
        tracker = EmotionTracker(name, workspace=get_context().config.workspace)
        state = tracker.current_state()
        return {
            "success": True,
            "persona": name,
            "state": state.to_dict(),
            "context_prompt": tracker.context_prompt(),
            "trajectory": tracker.trajectory(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Orchestration ────────────────────────────────────────────────


@router.post("/orchestrate")
async def orchestrate(params: dict[str, Any]) -> dict[str, Any]:
    """Run the multi-agent orchestrator on an objective."""
    try:
        factory = get_agent_factory()
        from huginn.agents.orchestrator import Orchestrator

        orch = Orchestrator(
            factory=factory,
            memory_manager=get_memory_manager(),
            max_concurrent=params.get(
                "max_concurrent", factory.config.max_concurrent_subagents
            ),
        )
        result = await orch.run(params.get("objective", ""))
        return {
            "success": result.success,
            "objective": result.objective,
            "plan": [
                {
                    "task_id": t.task_id,
                    "agent_id": t.agent_id,
                    "status": t.status,
                    "prompt": t.prompt,
                }
                for t in result.plan.tasks
            ],
            "outputs": result.outputs,
            "summary": result.summary,
            "error": result.error,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ── Telemetry ────────────────────────────────────────────────────


@router.get("/telemetry/summary")
async def telemetry_summary() -> dict[str, Any]:
    """Return coarse telemetry summary for the global agent."""
    try:
        agent = await get_agent()
        return {"summary": agent.telemetry_summary()}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@router.get("/telemetry/spans")
async def telemetry_spans() -> dict[str, Any]:
    """Return all recorded telemetry spans for the global agent."""
    try:
        agent = await get_agent()
        return {"spans": agent.telemetry_spans()}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ── Swarm ────────────────────────────────────────────────────────


@router.post("/swarm/run")
async def swarm_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run a task through the multi-agent swarm."""
    from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent

    try:
        agent = await get_agent()
        task = params.get("task", "")
        if not task:
            return {"error": "task is required"}

        workers = [
            SwarmAgent(
                "planner", AgentRole.PLANNER, agent, "Break the task into steps."
            ),
            SwarmAgent(
                "scientist", AgentRole.SCIENTIST, agent, "Choose physical models."
            ),
            SwarmAgent("coder", AgentRole.CODER, agent, "Write code or tool calls."),
            SwarmAgent("executor", AgentRole.EXECUTOR, agent, "Run the solution."),
            SwarmAgent("critic", AgentRole.CRITIC, agent, "Review correctness."),
        ]
        result = await HuginnSwarm(workers).run(task)
        return {"success": True, **result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
