"""Shared server state, factory functions, and common helpers.

All route modules import from here instead of from ``server.py`` to
avoid circular imports.  This module has **no** FastAPI dependency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from huginn.agent import HuginnAgent
from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator
from huginn.autoloop.plan_store import PlanStore
from huginn.config import HuginnConfig, get_config
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.models.registry import ModelRegistry
from huginn.permissions import PermissionMode
from huginn.personas import PersonaManager
from huginn.project_context import load_project_context
from huginn.server_context import ServerContext, get_server_context
from huginn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

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

# Protects _checkpoints and _threads against concurrent access from
# multiple async handlers or thread-pool workers.
_state_lock = threading.RLock()

_EDIT_TOOLS: set[str] = {"file_write_tool", "file_edit_tool"}

# Shared visual-encoder / image-index singletons (lazily built on first use).
# Kept here rather than in ServerContext so they don't pull the heavy ML
# stack into every context construction.
_visual_encoder = None
_image_index = None

# ── session / thread lifecycle ─────────────────────────────────────
# Threads held in _threads are ephemeral conversation handles. Without a
# TTL the dict grows without bound as new threads are created, so we reap
# any thread that hasn't been touched within the configured window.

_log = logging.getLogger("huginn.server_core")

# How long an idle thread stays alive before cleanup reaps it.
_SESSION_TTL_HOURS = float(os.environ.get("HUGINN_SESSION_TTL_HOURS", "24"))
# Run the sweep every N get_or_create_thread() calls instead of every call,
# otherwise the lock churn dominates under burst traffic.
_CLEANUP_INTERVAL = 50
_cleanup_counter = 0
_cleanup_lock = threading.Lock()


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

    cfg = get_config()
    memory_manager = get_memory_manager()
    factory = get_agent_factory()

    # Ollama: check availability first (async — no event-loop blocking)
    if cfg.provider == "ollama" and cfg.models:
        ollama_models = [m for m in cfg.models if m.provider == "ollama" and m.enabled]
        if ollama_models and not await _check_ollama_available(cfg.ollama_host):
            logger.warning("Ollama not responding at {cfg.ollama_host}")
            logger.info("Falling back to mock mode (no LLM)")
            get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
            get_context().agent.register_tools_from_registry()
            return get_context().agent

    try:
        get_context().agent = factory.create_lead()
    except ImportError as e:
        logger.warning("Missing dependency for configured model: {e}")
        logger.info("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()
    except ValueError as e:
        logger.warning("{e}")
        logger.info("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()
    except Exception as e:
        logger.warning("Failed to initialize agent: {e}")
        logger.info("Falling back to mock mode (no LLM)")
        get_context().agent = HuginnAgent(model=None, memory_manager=memory_manager)
        get_context().agent.register_tools_from_registry()

    # Wire the agent's LLM into the memory manager for insight extraction
    try:
        agent = get_context().agent
        if agent is not None and getattr(agent, "memory", None) is not None:
            agent.memory.set_llm(getattr(agent, "model", None))
    except Exception:
        logger.debug("把 agent LLM 接到 memory manager 失败", exc_info=True)

    return get_context().agent


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager."""
    if get_context().memory_manager is not None:
        return get_context().memory_manager
    cfg = get_config()
    memory_md = Path(cfg.workspace) / "MEMORY.md" if cfg.workspace else None

    # Try to wire up semantic search via VectorStore + chromadb.
    # Falls back to FTS5-only long-term memory if chromadb is missing.
    vector_store = None
    longterm = None
    try:
        from huginn.rag.vector_store import VectorStore

        vector_store = VectorStore()
        from huginn.memory.longterm import LongTermMemory

        longterm = LongTermMemory(vector_store=vector_store, enable_semantic=True)
    except Exception:
        # chromadb not installed or VectorStore init failed — FTS5 only
        longterm = None

    get_context().memory_manager = MemoryManager(
        config=MemoryConfig(memory_md_path=memory_md),
        longterm=longterm,
    )
    return get_context().memory_manager


def get_agent_factory() -> AgentFactory:
    """Get or create the global AgentFactory from current config."""
    if get_context().agent_factory is not None:
        return get_context().agent_factory
    # Use the same config source as the REST config route —
    # huginn.toml if it exists, otherwise env vars.
    from huginn.routes.config import _load_runtime_config
    cfg = _load_runtime_config()
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
    cfg = get_config()
    get_context().orchestrator = Orchestrator(
        factory=get_agent_factory(),
        memory_manager=get_memory_manager(),
        max_concurrent=cfg.max_concurrent_subagents,
        plan_store=get_plan_store(),
        auto_confirm=cfg.plan_auto_confirm,
    )
    return get_context().orchestrator


def get_plan_store() -> PlanStore:
    """Get or create the global PlanStore."""
    if get_context().plan_store is not None:
        return get_context().plan_store
    get_context().plan_store = PlanStore()
    return get_context().plan_store


# ── visual perception (I-JEPA image encoder + index) ────────────────


def get_visual_encoder():
    """Return the shared frozen-image encoder (I-JEPA / CLIP / ResNet50).

    Built lazily and cached at module level so the model is loaded at most
    once per process. May return an encoder whose ``available`` flag is
    False when no ML backend could be constructed.
    """
    global _visual_encoder
    if _visual_encoder is None:
        from huginn.perception.visual_encoder import get_encoder

        _visual_encoder = get_encoder()
    return _visual_encoder


def get_image_index():
    """Return the shared image vector index, persisting under the workspace.

    The JSON store lives at ``<workspace>/.huginn/visual_index.json`` so the
    indexed images survive across server restarts.
    """
    global _image_index
    if _image_index is None:
        from huginn.perception.image_index import ImageIndex

        workspace = "."
        try:
            workspace = get_context().config.workspace or "."
        except Exception:
            logger.debug("读取 workspace 失败, 用默认 '.'", exc_info=True)
        store_path = Path(workspace) / ".huginn" / "visual_index.json"
        _image_index = ImageIndex(store_path=store_path)
    return _image_index


# ── thread lifecycle (TTL + user isolation) ────────────────────────


def _current_user_id(conn: Any) -> str | None:
    """Best-effort extraction of the caller's user_id from a request/ws.

    The auth dependency stashes a ``RequestContext`` on ``request.state.auth``
    (HTTP) — for WebSockets we fall back to decoding the bearer token if the
    connection carries one. Returns None when no user can be identified,
    which preserves the legacy shared behaviour.
    """
    # HTTP path: request.state.auth.user.user_id
    try:
        state = getattr(conn, "state", None)
        auth = getattr(state, "auth", None) if state is not None else None
        user = getattr(auth, "user", None) if auth is not None else None
        if user is not None:
            uid = getattr(user, "user_id", None)
            if uid:
                return uid
    except Exception:
        logger.debug("从 request.state 提取 user_id 失败", exc_info=True)

    # WebSocket / fallback: pull the bearer token from headers and decode it.
    try:
        headers = getattr(conn, "headers", {})
        authz = headers.get("authorization", "") if headers else ""
        if isinstance(authz, bytes):
            authz = authz.decode("utf-8", "replace")
        if authz.lower().startswith("bearer "):
            token = authz[7:]
            from huginn.security.auth import _decode_token

            claims = _decode_token(token)
            return claims.get("sub")
    except Exception:
        logger.debug("从 bearer token 提取 user_id 失败", exc_info=True)
    return None


def _cleanup_expired_threads() -> int:
    """Drop threads not accessed within the TTL window.

    Uses the ``last_accessed_ts`` epoch stamp set by ``get_or_create_thread``.
    Threads created before that field existed are left alone on the first
    pass — they pick it up the next time they're touched. Returns the number
    of reaped threads.
    """
    cutoff = time.time() - _SESSION_TTL_HOURS * 3600.0
    removed = 0
    with _state_lock:
        for tid in list(_threads):
            meta = _threads[tid]
            ts = meta.get("last_accessed_ts")
            if ts is not None and ts < cutoff:
                del _threads[tid]
                removed += 1
    if removed:
        _log.info(
            "Reaped %d expired thread(s) older than %.1fh", removed, _SESSION_TTL_HOURS
        )
    return removed


def touch_thread(thread_id: str) -> dict[str, Any] | None:
    """Mark a thread as freshly accessed. Returns a copy of its metadata.

    No-ops (returns None) when the thread doesn't exist — callers that need
    create-on-access semantics should use ``get_or_create_thread`` instead.
    """
    now_iso = datetime.now().isoformat()
    now_ts = time.time()
    with _state_lock:
        meta = _threads.get(thread_id)
        if meta is None:
            return None
        meta["last_active"] = now_iso
        meta["last_accessed_ts"] = now_ts
        return dict(meta)


def get_or_create_thread(
    thread_id: str,
    *,
    user_id: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Return thread metadata, creating the entry if it's new.

    Associates the thread with *user_id* (when supplied) so multi-tenant
    deployments can tell sessions apart, and refreshes ``last_accessed_ts``
    so the TTL sweeper can reap idle threads. The periodic cleanup runs
    inline every ``_CLEANUP_INTERVAL`` calls to keep ``_threads`` bounded.
    """
    global _cleanup_counter

    # Throttle the sweep — a lock check per call is cheaper than a full scan.
    with _cleanup_lock:
        _cleanup_counter += 1
        run_cleanup = _cleanup_counter >= _CLEANUP_INTERVAL
        if run_cleanup:
            _cleanup_counter = 0
    if run_cleanup:
        _cleanup_expired_threads()

    now_iso = datetime.now().isoformat()
    now_ts = time.time()
    with _state_lock:
        meta = _threads.get(thread_id)
        if meta is None:
            meta = {
                "id": thread_id,
                "label": label or thread_id,
                "created_at": now_iso,
                "last_active": now_iso,
                "last_accessed_ts": now_ts,
            }
            if user_id is not None:
                meta["user_id"] = user_id
            _threads[thread_id] = meta
        else:
            meta["last_active"] = now_iso
            meta["last_accessed_ts"] = now_ts
            # Backfill user_id on threads that predate per-user isolation.
            if user_id is not None and "user_id" not in meta:
                meta["user_id"] = user_id
        return dict(meta)


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

    cfg = get_config()
    persona_manager = PersonaManager(workspace=get_context().config.workspace)
    base_prompt = persona_manager.get(cfg.persona).system_prompt

    try:
        project_ctx = load_project_context(cfg.workspace)
        if project_ctx.strip():
            base_prompt = f"{base_prompt}\n\n# Project Context\n\n{project_ctx}"
    except Exception as e:
        logger.info("[planner] project context warning: {e}")

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
        logger.warning("Failed to initialize planner model: {e}")
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
        logger.debug("destructive 标记检查失败", exc_info=True)

    reason = f"Tool '{tool_name}' requires approval"
    if reasons:
        reason += f" ({', '.join(reasons)})"

    if os.environ.get("HUGINN_AUTO_APPROVE") == "1":
        return True, None

    return False, reason


# ── system snapshot ─────────────────────────────────────────────────


def get_system_snapshot() -> dict[str, Any]:
    """用 HuginnSystem 封装当前 ServerContext 的状态快照。

    HuginnSystem 设计为 ServerContext 的统一替代容器，这里把它作为
    只读快照暴露给 /system/components 等运维端点，避免每加一个诊断
    端点都要直接翻 ServerContext 的字段。同时同步到全局单例，让其他
    模块可以通过 huginn.system.get_system() 拿到最新状态。
    """
    from huginn.system import HuginnSystem, set_system

    ctx = get_context()

    # _threads 在 _state_lock 保护下，这里拷一份快照避免后续 mutation
    with _state_lock:
        threads_snapshot = dict(_threads)

    system = HuginnSystem(
        config=ctx.config,
        tool_registry=ctx.tool_registry,
        skill_registry=ctx.skill_registry,
        audit_logger=ctx.audit_logger,
        memory_backend=ctx.memory_backend,
        checkpointer_backend=ctx.checkpointer_backend,
        remote_job_backend=ctx.remote_job_backend,
        agent_factory=ctx.agent_factory,
        orchestrator=ctx.orchestrator,
        memory_manager=ctx.memory_manager,
        kb=ctx.kb,
        codebase=ctx.codebase,
        agent=ctx.agent,
        planner_agent=ctx.planner_agent,
        mcp_manager=ctx.mcp_manager,
        plan_store=ctx.plan_store,
        permission_config=ctx.permission_config,
        active_threads=threads_snapshot,
        edit_tools=set(_EDIT_TOOLS),
    )
    # 同步到全局单例，其他模块可用 get_system() 读取
    set_system(system)

    components = system.list_components()
    initialized = sum(1 for v in components.values() if v)
    return {
        "configured": system.is_configured,
        "components": components,
        "initialized_count": initialized,
        "total_count": len(components),
        "active_threads": len(threads_snapshot),
    }
