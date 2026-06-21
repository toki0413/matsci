"""Shared server state, factory functions, and common helpers.

All route modules import from here instead of from ``server.py`` to
avoid circular imports.  This module has **no** FastAPI dependency.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from huginn.agent import HuginnAgent
from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator
from huginn.config import HuginnConfig
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.models.registry import ModelRegistry
from huginn.permissions import PermissionMode
from huginn.personas import PersonaManager
from huginn.project_context import load_project_context
from huginn.server_context import ServerContext, get_server_context
from huginn.tools.registry import ToolRegistry

# ── optional imports ────────────────────────────────────────────────

try:
    from huginn.knowledge import get_knowledge_base  # noqa: F401

    _KB_AVAILABLE = True
except Exception:
    _KB_AVAILABLE = False
    get_knowledge_base = None  # type: ignore

try:
    from huginn.codebase import get_codebase_index  # noqa: F401

    _CODEBASE_AVAILABLE = True
except Exception:
    _CODEBASE_AVAILABLE = False
    get_codebase_index = None  # type: ignore


# ── module-level shared state ───────────────────────────────────────

_context: ServerContext | None = None

_checkpoints: dict[str, tuple[Path, dict[str, str]]] = {}

_threads: dict[str, dict[str, Any]] = {}

_EDIT_TOOLS: set[str] = {"file_write_tool", "file_edit_tool"}


# ── context accessor ────────────────────────────────────────────────


def get_context() -> ServerContext:
    """Return the server context, initializing it lazily if needed."""
    global _context
    if _context is None:
        _context = get_server_context()
    return _context


# ── Ollama helpers ──────────────────────────────────────────────────


def _check_ollama_available_sync(base_url: str, timeout: float = 2.0) -> bool:
    """Synchronous check if Ollama is responding (runs in thread pool)."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _check_ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    """Async wrapper — delegates to thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(_check_ollama_available_sync, base_url, timeout)


# ── factory functions ───────────────────────────────────────────────


async def get_agent() -> HuginnAgent:
    """Get or create the HuginnAgent instance."""
    if get_context().agent is not None:
        return get_context().agent

    cfg = HuginnConfig.from_env()
    memory_manager = get_memory_manager()
    factory = get_agent_factory()

    # Ollama: check availability first (async — no event-loop blocking)
    if cfg.provider == "ollama" and cfg.models:
        ollama_models = [m for m in cfg.models if m.provider == "ollama" and m.enabled]
        if ollama_models and not await _check_ollama_available(cfg.ollama_host):
            print(f"Warning: Ollama not responding at {cfg.ollama_host}")
            print("Falling back to mock mode (no LLM)")
            get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
            get_context().agent.register_tools_from_registry()
            return get_context().agent

    try:
        get_context().agent = factory.create_lead()
    except ImportError as e:
        print(f"Warning: Missing dependency for configured model: {e}")
        print("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()
    except Exception as e:
        print(f"Warning: Failed to initialize agent: {e}")
        print("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()

    return get_context().agent


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager."""
    if get_context().memory_manager is not None:
        return get_context().memory_manager
    cfg = HuginnConfig.from_env()
    memory_md = Path(cfg.workspace) / "MEMORY.md" if cfg.workspace else None
    get_context().memory_manager = MemoryManager(
        config=MemoryConfig(memory_md_path=memory_md),
    )
    return get_context().memory_manager


def get_agent_factory() -> AgentFactory:
    """Get or create the global AgentFactory from current config."""
    if get_context().agent_factory is not None:
        return get_context().agent_factory
    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    get_context().agent_factory = AgentFactory(
        config=cfg,
        model_registry=registry,
        memory_manager=get_memory_manager(),
    )
    return get_context().agent_factory


def get_orchestrator() -> Orchestrator:
    """Get or create the global multi-agent Orchestrator."""
    if get_context().orchestrator is not None:
        return get_context().orchestrator
    cfg = HuginnConfig.from_env()
    get_context().orchestrator = Orchestrator(
        factory=get_agent_factory(),
        memory_manager=get_memory_manager(),
        max_concurrent=cfg.max_concurrent_subagents,
    )
    return get_context().orchestrator


# ── planner ─────────────────────────────────────────────────────────

PLANNER_SUFFIX = """
# Planning Mode

You are in PLAN mode. Do NOT execute any file edits, shell commands, or external tools.
Your job is to produce a clear, step-by-step plan for how to satisfy the user's request.
For each step, briefly state:
1. What will be done
2. Which files or tools are likely involved
3. What success looks like

If the request is simple enough that no plan is needed, respond normally but still avoid taking actions.
"""


def get_planner_agent() -> HuginnAgent:
    """Get or create a read-only planning agent (no tools registered)."""
    if get_context().planner_agent is not None:
        return get_context().planner_agent

    cfg = HuginnConfig.from_env()
    persona_manager = PersonaManager(workspace=get_context().config.workspace)
    base_prompt = persona_manager.get(cfg.persona).system_prompt

    try:
        project_ctx = load_project_context(cfg.workspace)
        if project_ctx.strip():
            base_prompt = f"{base_prompt}\n\n# Project Context\n\n{project_ctx}"
    except Exception as e:
        print(f"[planner] project context warning: {e}")

    system_prompt = base_prompt + PLANNER_SUFFIX

    if cfg.provider == "default" and not cfg.models:
        get_context().planner_agent = HuginnAgent(
            model=None, system_prompt=system_prompt
        )
        return get_context().planner_agent

    try:
        factory = get_agent_factory()
        get_context().planner_agent = factory.create_lead(
            system_prompt_override=system_prompt
        )
    except Exception as e:
        print(f"Warning: Failed to initialize planner model: {e}")
        get_context().planner_agent = HuginnAgent(
            model=None, system_prompt=system_prompt
        )

    return get_context().planner_agent


# ── shared helpers ──────────────────────────────────────────────────


def _snapshot_directory(base: Path) -> dict[str, str]:
    """Snapshot text files under base into a dict keyed by relative path."""
    snapshot: dict[str, str] = {}
    if not base.exists():
        return snapshot
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                continue
            snapshot[str(path.relative_to(base))] = data.decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            continue
    return snapshot


def _server_allows_tool(tool_name: str, input_data: Any) -> tuple[bool, str | None]:
    """Check server-side permission policy for a tool call."""
    mode = get_context().permission_config.get_mode(tool_name)

    if get_context().permission_config.auto_approve_all or mode == PermissionMode.AUTO:
        return True, None

    if mode == PermissionMode.DENY:
        return False, f"Tool '{tool_name}' is blocked by permission policy"

    reasons: list[str] = []
    try:
        if tool_name in _EDIT_TOOLS or getattr(input_data, "destructive", False):
            reasons.append("this operation is destructive")
    except Exception:
        pass

    reason = f"Tool '{tool_name}' requires approval"
    if reasons:
        reason += f" ({', '.join(reasons)})"

    if os.environ.get("HUGINN_AUTO_APPROVE") == "1":
        return True, None

    return False, reason
