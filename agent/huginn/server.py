"""FastAPI + WebSocket server for Huginn.

Serves the desktop frontend with:
- HTTP API for tools, workflows, and health checks
- WebSocket endpoint for real-time Agent chat
- Compatibility stubs for math-anything frontend APIs
"""

from __future__ import annotations

import json
import os
import asyncio
import difflib
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from langchain_core.messages import AIMessage, ToolMessage

try:
    from huginn.knowledge import get_knowledge_base
    _KB_AVAILABLE = True
except Exception:
    _KB_AVAILABLE = False
    get_knowledge_base = None  # type: ignore

try:
    from huginn.codebase import get_codebase_index
    _CODEBASE_AVAILABLE = True
except Exception:
    _CODEBASE_AVAILABLE = False
    get_codebase_index = None  # type: ignore

from huginn import __version__
from huginn.agent import HuginnAgent
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry
from huginn.config import HuginnConfig
from huginn.personas import PERSONAS
from huginn.project_context import (
    load_project_context,
    save_project_context,
    context_source,
    project_context_path,
)
from huginn.types import ToolContext
from huginn.permissions import PermissionConfig, PermissionMode
from huginn.security.audit import AuditLogger
from huginn.pet import get_pet_bus, PetMood, configure_pet
from huginn.memory.manager import MemoryManager, MemoryConfig
from huginn.models.registry import ModelRegistry
from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator, SubTask
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import get_template
from huginn.hpc.client import HPCClient, HPCConfig
from huginn.skills.base import DeclarativeSkillExecutor
from huginn.skills.registry import SkillRegistry


# Register all tools
register_all_tools()

# Global agent and MCP manager
_agent: HuginnAgent | None = None
_planner_agent: HuginnAgent | None = None
_mcp_manager = None
_kb: Any | None = None
_codebase: Any | None = None
_memory_manager: Any | None = None
_agent_factory: AgentFactory | None = None
_orchestrator: Orchestrator | None = None

# Server-wide security policy
_permission_config = PermissionConfig()
_audit_logger = AuditLogger(Path.home() / ".huginn" / "audit.jsonl")

# In-memory checkpoints for diff review (snapshot path -> content)
_checkpoints: dict[str, tuple[Path, dict[str, str]]] = {}

# Thread registry (thread_id -> metadata)
_threads: dict[str, dict[str, Any]] = {}

# Tools that modify files and should trigger an auto-checkpoint
_EDIT_TOOLS = {"file_write_tool", "file_edit_tool"}


async def _init_mcp_tools():
    """Connect to local MCP servers and register their tools."""
    global _mcp_manager
    try:
        from huginn.mcp_client import MCPClientManager, MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools
        from pathlib import Path

        _mcp_manager = MCPClientManager()
        base = Path(__file__).parent.parent.parent  # repo root

        # Try mat-db-mcp
        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="mat-db",
                command="python",
                args=[str(mat_db_path)],
            ))

        # Try math-anything-mcp
        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="math-anything",
                command="python",
                args=[str(math_path)],
            ))

        registered = register_mcp_tools(_mcp_manager)
        print(f"[MCP] Registered {len(registered)} tools from MCP servers")
    except Exception as e:
        print(f"[MCP] Warning: Could not initialize MCP tools: {e}")


async def _shutdown_mcp():
    """Disconnect all MCP servers."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.disconnect_all()
        _mcp_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kb, _codebase
    await _init_mcp_tools()
    if _KB_AVAILABLE and _kb is None:
        try:
            cfg = HuginnConfig.from_env()
            _kb = get_knowledge_base(cfg.workspace)
        except Exception as e:
            print(f"[KB] Warning: could not initialize knowledge base: {e}")
    if _CODEBASE_AVAILABLE and _codebase is None:
        try:
            cfg = HuginnConfig.from_env()
            _codebase = get_codebase_index(cfg.workspace)
        except Exception as e:
            print(f"[Codebase] Warning: could not initialize codebase index: {e}")
    try:
        cfg = HuginnConfig.from_env()
        configure_pet(cfg.pet_name, cfg.pet_personality)
    except Exception as e:
        print(f"[Pet] Warning: could not configure pet: {e}")
    yield
    await _shutdown_mcp()


def _get_cors_origins() -> list[str]:
    """Return allowed CORS origins.

    Defaults to local development / Tauri origins. Set ``HUGINN_CORS_ORIGINS``
    to a comma-separated list to override. Use ``*`` only if you understand the
    credential implications.
    """
    raw = os.environ.get("HUGINN_CORS_ORIGINS", "")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:1420",
        "http://localhost:8000",
        "tauri://localhost",
    ]


app = FastAPI(title="Huginn Server", version=__version__, lifespan=lifespan)

_cors_origins = _get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def _check_ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    """Quick check if Ollama is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_agent() -> HuginnAgent:
    """Get or create the HuginnAgent instance."""
    global _agent
    if _agent is not None:
        return _agent
    
    cfg = HuginnConfig.from_env()
    memory_manager = get_memory_manager()
    factory = get_agent_factory()

    # Ollama: check availability first
    if cfg.provider == "ollama" and cfg.models:
        ollama_models = [m for m in cfg.models if m.provider == "ollama" and m.enabled]
        if ollama_models and not _check_ollama_available(cfg.ollama_host):
            print(f"Warning: Ollama not responding at {cfg.ollama_host}")
            print("Falling back to mock mode (no LLM)")
            _agent = HuginnAgent(model=None, memory_manager=memory_manager)
            _agent.register_tools_from_registry()
            return _agent

    try:
        _agent = factory.create_lead()
    except ImportError as e:
        print(f"Warning: Missing dependency for configured model: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = HuginnAgent(model=None, memory_manager=memory_manager)
        _agent.register_tools_from_registry()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = HuginnAgent(model=None, memory_manager=memory_manager)
        _agent.register_tools_from_registry()
    except Exception as e:
        print(f"Warning: Failed to initialize agent: {e}")
        print("Falling back to mock mode (no LLM)")
        _agent = HuginnAgent(model=None, memory_manager=memory_manager)
        _agent.register_tools_from_registry()

    return _agent


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager."""
    global _memory_manager
    if _memory_manager is not None:
        return _memory_manager
    cfg = HuginnConfig.from_env()
    memory_md = Path(cfg.workspace) / "MEMORY.md" if cfg.workspace else None
    _memory_manager = MemoryManager(
        config=MemoryConfig(memory_md_path=memory_md),
    )
    return _memory_manager


def get_agent_factory() -> AgentFactory:
    """Get or create the global AgentFactory from current config."""
    global _agent_factory
    if _agent_factory is not None:
        return _agent_factory
    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    _agent_factory = AgentFactory(
        config=cfg,
        model_registry=registry,
        memory_manager=get_memory_manager(),
    )
    return _agent_factory


def get_orchestrator() -> Orchestrator:
    """Get or create the global multi-agent Orchestrator."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
    cfg = HuginnConfig.from_env()
    _orchestrator = Orchestrator(
        factory=get_agent_factory(),
        memory_manager=get_memory_manager(),
        max_concurrent=cfg.max_concurrent_subagents,
    )
    return _orchestrator


# ── Health & Info ──────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = HuginnConfig.from_env()
    return {
        "status": "ok",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": cfg.provider != "default" and bool(cfg.resolved_api_key),
    }


@app.get("/health/rust")
async def health_rust() -> dict[str, Any]:
    """Report whether the Rust acceleration extension is available."""
    try:
        import huginn_ext

        functions = [name for name in dir(huginn_ext) if not name.startswith("_")]
        return {
            "available": True,
            "module": "huginn_ext",
            "functions": functions,
        }
    except Exception as e:
        return {
            "available": False,
            "module": "huginn_ext",
            "error": str(e),
            "functions": [],
        }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Return current server-side configuration (API key masked)."""
    return HuginnConfig.from_env().to_dict(mask_key=True)


@app.post("/config")
async def update_config(params: dict[str, Any]) -> dict[str, Any]:
    """Update server-side configuration and reset the agent so changes take effect."""
    global _agent, _agent_factory, _planner_agent

    if "provider" in params:
        os.environ["HUGINN_PROVIDER"] = str(params["provider"])
    if "model" in params:
        os.environ["HUGINN_MODEL"] = str(params["model"])
    if "api_key" in params:
        if params["api_key"]:
            os.environ["HUGINN_API_KEY"] = str(params["api_key"])
        else:
            os.environ.pop("HUGINN_API_KEY", None)
    if "base_url" in params:
        if params["base_url"]:
            os.environ["HUGINN_BASE_URL"] = str(params["base_url"])
        else:
            os.environ.pop("HUGINN_BASE_URL", None)
    if "ollama_host" in params:
        os.environ["OLLAMA_HOST"] = str(params["ollama_host"])
    if "persona" in params:
        os.environ["HUGINN_PERSONA"] = str(params["persona"])
    if "rag_enabled" in params:
        os.environ["HUGINN_RAG_ENABLED"] = "true" if params["rag_enabled"] else "false"
    if "team_mode_enabled" in params:
        os.environ["HUGINN_TEAM_MODE"] = "true" if params["team_mode_enabled"] else "false"
    if "max_concurrent_subagents" in params:
        os.environ["HUGINN_MAX_CONCURRENT_SUBAGENTS"] = str(params["max_concurrent_subagents"])
    if "models" in params:
        os.environ["HUGINN_MODELS"] = json.dumps(params["models"])
    if "agents" in params:
        os.environ["HUGINN_AGENTS"] = json.dumps(params["agents"])
    if "pet_name" in params:
        os.environ["HUGINN_PET_NAME"] = str(params["pet_name"])
    if "pet_personality" in params:
        os.environ["HUGINN_PET_PERSONALITY"] = str(params["pet_personality"])

    _agent = None
    _agent_factory = None
    _planner_agent = None
    global _orchestrator
    _orchestrator = None
    cfg = HuginnConfig.from_env()
    configure_pet(cfg.pet_name, cfg.pet_personality)
    return {"success": True, "config": cfg.to_dict(mask_key=True)}


# ── Project Context ─────────────────────────────────────────────

@app.get("/project-context")
async def get_project_context() -> dict[str, Any]:
    """Return the current project context file content and source."""
    cfg = HuginnConfig.from_env()
    return {
        "source": context_source(cfg.workspace),
        "path": str(project_context_path(cfg.workspace)),
        "content": load_project_context(cfg.workspace),
    }


@app.post("/project-context")
async def update_project_context(params: dict[str, Any]) -> dict[str, Any]:
    """Update the `.huginn.md` project context file."""
    cfg = HuginnConfig.from_env()
    content = params.get("content", "")
    try:
        result = save_project_context(cfg.workspace, content)
        global _agent
        _agent = None  # force re-init so new context is loaded
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Planner / Plan-Build Mode ───────────────────────────────────

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
    global _planner_agent
    if _planner_agent is not None:
        return _planner_agent

    cfg = HuginnConfig.from_env()
    base_prompt = PERSONAS.get(cfg.persona, PERSONAS["default"])

    try:
        project_ctx = load_project_context(cfg.workspace)
        if project_ctx.strip():
            base_prompt = f"{base_prompt}\n\n# Project Context\n\n{project_ctx}"
    except Exception as e:
        print(f"[planner] project context warning: {e}")

    system_prompt = base_prompt + PLANNER_SUFFIX

    if cfg.provider == "default" and not cfg.models:
        _planner_agent = HuginnAgent(model=None, system_prompt=system_prompt)
        return _planner_agent

    try:
        factory = get_agent_factory()
        _planner_agent = factory.create_lead(system_prompt_override=system_prompt)
    except Exception as e:
        print(f"Warning: Failed to initialize planner model: {e}")
        _planner_agent = HuginnAgent(model=None, system_prompt=system_prompt)

    # Planner is read-only: strip all tools
    _planner_agent.langchain_tools.clear()
    return _planner_agent


@app.post("/plan")
async def generate_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a step-by-step plan without executing any tools."""
    agent = get_planner_agent()
    if agent.model is None:
        return {"error": "No LLM configured. Set provider and API key to generate plans."}

    content = params.get("content", "")
    thread_id = params.get("thread_id", "plan")
    if not content.strip():
        return {"error": "content is required"}

    # Optionally ground the plan with codebase search results
    if _codebase is not None:
        try:
            results = await asyncio.to_thread(_codebase.search, content, top_k=3)
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
        traceback.print_exc()
        return {"error": f"Planner error: {str(e)}"}


# ── Codebase Semantic Search ────────────────────────────────────

@app.get("/codebase")
async def codebase_status() -> dict[str, Any]:
    """Return codebase index status."""
    if _codebase is None:
        return {"available": False, "error": "Codebase index not initialized"}
    try:
        return _codebase.status()
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/codebase/index")
async def codebase_index() -> dict[str, Any]:
    """Re-index the workspace codebase."""
    if _codebase is None:
        return {"success": False, "error": "Codebase index not initialized"}
    try:
        return {"success": True, **await asyncio.to_thread(_codebase.index_workspace)}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/codebase/search")
async def codebase_search(params: dict[str, Any]) -> dict[str, Any]:
    """Search the codebase index."""
    if _codebase is None:
        return {"results": [], "error": "Codebase index not initialized"}
    try:
        query = params.get("query", "")
        top_k = int(params.get("top_k", 5))
        results = await asyncio.to_thread(_codebase.search, query, top_k)
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        return {"results": [], "error": str(e)}


# ── Knowledge Base / RAG ───────────────────────────────────────

@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a document to the private knowledge base."""
    if _kb is None:
        return {"error": "Knowledge base is not available"}
    try:
        content = await file.read()
        result = _kb.add_document(file.filename or "unnamed", content)
        return {"success": True, "document": result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/knowledge")
async def list_knowledge() -> dict[str, Any]:
    """List documents in the knowledge base."""
    if _kb is None:
        return {"documents": [], "available": False}
    try:
        return {"documents": _kb.list_documents(), "count": _kb.count(), "available": True}
    except Exception as e:
        return {"documents": [], "error": str(e), "available": False}


@app.delete("/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str) -> dict[str, Any]:
    """Remove a document from the knowledge base."""
    if _kb is None:
        return {"success": False, "error": "Knowledge base is not available"}
    try:
        deleted = _kb.delete_document(doc_id)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/knowledge/query")
async def query_knowledge(params: dict[str, Any]) -> dict[str, Any]:
    """Query the knowledge base and return relevant chunks."""
    if _kb is None:
        return {"chunks": [], "error": "Knowledge base is not available"}
    try:
        text = params.get("query", "")
        top_k = int(params.get("top_k", 5))
        chunks = _kb.query(text, top_k=top_k)
        return {"chunks": chunks}
    except Exception as e:
        return {"chunks": [], "error": str(e)}


@app.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their schemas."""
    return ToolRegistry.get_all_schemas()


def _server_allows_tool(tool_name: str, input_data: Any) -> tuple[bool, str | None]:
    """Check server-side permission policy for a tool call."""
    mode = _permission_config.get_mode(tool_name)

    if _permission_config.auto_approve_all or mode == PermissionMode.AUTO:
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

    # Server is non-interactive: allow ASK only when HUGINN_AUTO_APPROVE is set.
    if os.environ.get("HUGINN_AUTO_APPROVE") == "1":
        return True, None

    return False, reason


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool directly via HTTP."""
    from huginn.types import ToolContext

    tool = ToolRegistry.get(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    if not tool.input_schema:
        return {"error": f"Tool '{tool_name}' has no input schema"}

    try:
        input_data = tool.input_schema(**args)
    except Exception as e:
        return {"error": f"Invalid input: {e}"}

    allowed, reason = _server_allows_tool(tool_name, input_data)
    if not allowed:
        _audit_logger.log(
            event_type="tool_call",
            actor="http",
            action=tool_name,
            details={"approved": False, "reason": reason},
            input_data=json.dumps(args, sort_keys=True, default=str),
        )
        return {"error": reason}

    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
        audit_logger=_audit_logger,
    )
    result = await tool.call(input_data, context)

    _audit_logger.log(
        event_type="tool_call",
        actor="http",
        action=tool_name,
        details={"approved": True, "success": result.success},
        input_data=json.dumps(args, sort_keys=True, default=str),
        output_data=json.dumps(result.data, sort_keys=True, default=str) if result.data else None,
    )

    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
    }


@app.get("/events")
async def events_stream() -> StreamingResponse:
    """Server-sent events stream for the desktop pet and activity indicators."""
    bus = get_pet_bus()
    queue = await bus.queue()

    async def generator() -> AsyncIterator[str]:
        # Send current state immediately.
        yield f"data: {json.dumps({'type': 'state', 'state': bus.state.to_dict()})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                payload = {
                    "type": "event",
                    "mood": event.mood.value,
                    "message": event.message,
                    "details": event.details,
                    "timestamp": event.timestamp,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.TimeoutError:
                # Keep connection alive with the latest state.
                yield f"data: {json.dumps({'type': 'heartbeat', 'state': bus.state.to_dict()})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Memory Management ────────────────────────────────────────────

@app.get("/memory")
async def list_memories(category: str | None = None, tier: str | None = None, limit: int = 100) -> dict[str, Any]:
    """List long-term memories, optionally filtered by category or tier."""
    try:
        mgr = get_memory_manager()
        if category:
            entries = mgr.longterm.list_by_category(category, limit=limit, alive_only=True)
        else:
            entries = mgr.longterm.list_all(limit=limit, alive_only=True)
        if tier:
            entries = [e for e in entries if e.get("tier") == tier]
        return {"entries": entries}
    except Exception as e:
        return {"error": str(e)}


@app.post("/memory/search")
async def search_memories(params: dict[str, Any]) -> dict[str, Any]:
    """Search long-term memory by query."""
    try:
        mgr = get_memory_manager()
        results = mgr.recall(
            query=params.get("query", ""),
            category=params.get("category"),
            tier=params.get("tier"),
            top_k=params.get("top_k", 10),
        )
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@app.post("/memory")
async def create_memory(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new memory entry."""
    try:
        mgr = get_memory_manager()
        mid = mgr.remember(
            content=params["content"],
            category=params.get("category", "fact"),
            tags=params.get("tags", []),
            importance=params.get("importance", 0.5),
            tier=params.get("tier", "mid"),
        )
        return {"memory_id": mid, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.patch("/memory/{memory_id}")
async def update_memory(memory_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Update a memory entry (content/importance/tags/tier)."""
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.update(
            memory_id,
            content=params.get("content"),
            importance=params.get("importance"),
            tags=params.get("tags"),
            tier=params.get("tier"),
        )
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a memory entry."""
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.delete(memory_id)
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/memory/promote/{memory_id}")
async def promote_memory(memory_id: str, params: dict[str, Any] = {}) -> dict[str, Any]:
    """Promote a memory to a higher tier (default long)."""
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.promote(memory_id, target_tier=params.get("tier", "long"))
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/memory/prune")
async def prune_memories(params: dict[str, Any] = {}) -> dict[str, Any]:
    """Prune expired and low-importance memories."""
    try:
        mgr = get_memory_manager()
        expired = mgr.longterm.prune_expired()
        low = mgr.longterm.prune_low_importance(
            threshold=params.get("threshold", 0.2),
            older_than_days=params.get("older_than_days", 30),
        )
        return {"expired": expired, "low_importance": low}
    except Exception as e:
        return {"error": str(e)}


@app.post("/memory/sync-md")
async def sync_memory_md() -> dict[str, Any]:
    """Sync curated long-tier memories to MEMORY.md."""
    try:
        mgr = get_memory_manager()
        path = mgr.sync_memory_md()
        return {"path": str(path) if path else None}
    except Exception as e:
        return {"error": str(e)}


@app.get("/memory/stats")
async def memory_stats() -> dict[str, Any]:
    """Return memory system statistics."""
    try:
        return get_memory_manager().stats()
    except Exception as e:
        return {"error": str(e)}


# ── Multi-Provider / Multi-Agent ─────────────────────────────────

@app.get("/models")
async def list_models() -> dict[str, Any]:
    """List configured model aliases."""
    try:
        cfg = HuginnConfig.from_env()
        registry = ModelRegistry.from_config(cfg)
        return {"models": [m.__dict__ for m in registry.list()]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/agents")
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


@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a single-turn message to a specific agent profile."""
    try:
        factory = get_agent_factory()
        agent = factory.create(agent_id, thread_id=params.get("thread_id", "default"))
        state = agent.invoke(params.get("message", ""))
        messages = state.get("messages", [])
        content = ""
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
        return {"agent_id": agent_id, "content": content}
    except Exception as e:
        return {"error": str(e)}


@app.post("/orchestrate")
async def orchestrate(params: dict[str, Any]) -> dict[str, Any]:
    """Run the multi-agent orchestrator on an objective."""
    try:
        factory = get_agent_factory()
        from huginn.agents.orchestrator import Orchestrator
        orch = Orchestrator(
            factory=factory,
            memory_manager=get_memory_manager(),
            max_concurrent=params.get("max_concurrent", factory.config.max_concurrent_subagents),
        )
        result = await orch.run(params.get("objective", ""))
        return {
            "success": result.success,
            "objective": result.objective,
            "plan": [
                {"task_id": t.task_id, "agent_id": t.agent_id, "status": t.status, "prompt": t.prompt}
                for t in result.plan.tasks
            ],
            "outputs": result.outputs,
            "summary": result.summary,
            "error": result.error,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/workflows")
async def list_workflows() -> list[str]:
    from huginn.workflows.templates import list_templates
    return list_templates()


@app.post("/workflows/execute")
async def execute_workflow(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow template.
    
    Args:
        template: Template name (e.g., "standard_dft", "aimd")
        args: Arguments passed to the template function
    """
    template_name = params.get("template")
    template_args = params.get("args", {})
    
    template_fn = get_template(template_name)
    if not template_fn:
        return {"error": f"Template '{template_name}' not found"}
    
    try:
        stages = template_fn(**template_args)
    except Exception as e:
        return {"error": f"Failed to build workflow: {e}"}
    
    engine = WorkflowEngine(ToolRegistry)
    context = ToolContext(session_id="http", workspace=".", memory_manager=get_memory_manager(), agent_factory=get_agent_factory())

    result = await engine.execute(stages, context)
    
    return {
        "success": result.success,
        "total_walltime": result.total_walltime,
        "stages": {
            sid: {
                "name": s.name,
                "status": s.status,
                "attempts": s.attempts,
                "error": s.result.error if s.result else None,
            }
            for sid, s in result.stages.items()
        },
        "outputs": result.outputs,
        "error": result.error,
    }


# ── Skills ─────────────────────────────────────────────────────

@app.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    """List all registered skills."""
    # Ensure presets are loaded and registered
    from huginn.skills import presets  # noqa: F401

    return [
        {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in skill.parameters
            ],
            "tags": skill.tags,
        }
        for skill in SkillRegistry.get_all_definitions()
    ]


@app.post("/skills/execute")
async def execute_skill(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a skill by name with the provided arguments."""
    from huginn.skills import presets  # noqa: F401

    skill_name = params.get("skill")
    skill_args = params.get("args", {})

    skill = SkillRegistry.get(skill_name)
    if not skill:
        return {"error": f"Skill '{skill_name}' not found"}

    executor = DeclarativeSkillExecutor(ToolRegistry)
    context = ToolContext(session_id="http", workspace=".", memory_manager=get_memory_manager(), agent_factory=get_agent_factory())
    result = await executor.execute(skill, skill_args, {})
    return result


# ── MCP / Plugin Management ────────────────────────────────────

@app.get("/mcp/servers")
async def list_mcp_servers() -> dict[str, Any]:
    """List connected MCP servers and their discovered tools."""
    if _mcp_manager is None:
        return {"servers": [], "connected": []}
    servers = []
    for name in _mcp_manager._sessions.keys():
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in _mcp_manager._tools if t.server_name == name
        ]
        servers.append({"name": name, "connected": True, "tools": tools})
    return {"servers": servers}


@app.get("/mcp/servers/discover")
async def discover_mcp_servers() -> dict[str, Any]:
    """Discover local MCP server directories under servers/*."""
    base = Path(__file__).parent.parent.parent / "servers"
    found = []
    if base.exists():
        for entry in base.iterdir():
            server_py = entry / "server.py"
            if entry.is_dir() and server_py.exists():
                found.append({
                    "name": entry.name,
                    "path": str(server_py),
                    "command": "python",
                    "args": [str(server_py)],
                })
    return {"servers": found}


@app.post("/mcp/servers/connect")
async def connect_mcp_server(params: dict[str, Any]) -> dict[str, Any]:
    """Connect to an MCP server and register its tools."""
    global _mcp_manager
    if _mcp_manager is None:
        from huginn.mcp_client import MCPClientManager
        _mcp_manager = MCPClientManager()

    name = params.get("name", "")
    command = params.get("command", "python")
    args = params.get("args", [])
    env = params.get("env")
    if not name:
        return {"success": False, "error": "name is required"}

    try:
        from huginn.mcp_client import MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        await _mcp_manager.connect(MCPServerConfig(
            name=name,
            command=command,
            args=args,
            env=env,
        ))
        registered = register_mcp_tools(_mcp_manager)
        return {
            "success": True,
            "server": name,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in registered if t.server_name == name
            ],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/servers/{name}/disconnect")
async def disconnect_mcp_server(name: str) -> dict[str, Any]:
    """Disconnect an MCP server and unregister its tools."""
    global _mcp_manager
    if _mcp_manager is None:
        return {"success": False, "error": "MCP manager not initialized"}

    try:
        tools_to_remove = [
            t.name for t in _mcp_manager._tools if t.server_name == name
        ]
        await _mcp_manager.disconnect(name)
        for tool_name in tools_to_remove:
            ToolRegistry.unregister(tool_name)
        return {"success": True, "unregistered": tools_to_remove}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Thread Management ────────────────────────────────────────────

@app.get("/threads")
async def list_threads() -> dict[str, Any]:
    """List known conversation threads."""
    return {
        "threads": [
            {
                "id": t["id"],
                "label": t.get("label", t["id"]),
                "created_at": t.get("created_at", ""),
                "last_active": t.get("last_active", ""),
            }
            for t in sorted(_threads.values(), key=lambda x: x.get("last_active", ""), reverse=True)
        ]
    }


@app.post("/threads")
async def create_thread(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new conversation thread."""
    thread_id = params.get("id") or uuid.uuid4().hex[:8]
    label = params.get("label") or thread_id
    _threads[thread_id] = {
        "id": thread_id,
        "label": label,
        "created_at": uuid.uuid4().hex,
        "last_active": uuid.uuid4().hex,
    }
    return {"id": thread_id, "label": label}


@app.patch("/threads/{thread_id}")
async def rename_thread(thread_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rename a thread."""
    if thread_id not in _threads:
        return {"success": False, "error": "thread not found"}
    _threads[thread_id]["label"] = params.get("label", thread_id)
    return {"success": True, "label": _threads[thread_id]["label"]}


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Remove a thread from the registry."""
    if thread_id in _threads:
        del _threads[thread_id]
    return {"success": True}


# ── WebSocket ──────────────────────────────────────────────────

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time Agent chat."""
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            msg_type = data.get("type", "user_input")
            content = data.get("content", "")
            thread_id = data.get("thread_id", "default")
            
            if msg_type == "user_input":
                cfg_chat = HuginnConfig.from_env()
                factory = get_agent_factory()
                agent = get_agent()
                team_mode = cfg_chat.team_mode_enabled

                # Track this thread
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
                            agent = factory.create(routed_agent_id, thread_id=thread_id)
                        except Exception as e:
                            await websocket.send_json({"type": "error", "error": f"Cannot spawn agent @{maybe_id}: {e}"})
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
                    await websocket.send_json({"type": "text_delta", "text": "🧑‍🤝‍🧑 Assembling agent team...\n"})
                    try:
                        from huginn.agents.orchestrator import Orchestrator
                        orch = Orchestrator(
                            factory=factory,
                            memory_manager=get_memory_manager(),
                            max_concurrent=max(1, cfg_chat.max_concurrent_subagents),
                        )

                        def _on_status(task):
                            # Fire-and-forget status message
                            asyncio.create_task(websocket.send_json({
                                "type": "agent_status",
                                "task_id": task.task_id,
                                "agent_id": task.agent_id,
                                "status": task.status,
                            }))

                        result = await orch.run(objective, on_status=_on_status)
                        for task in result.plan.tasks:
                            await websocket.send_json({
                                "type": "agent_status",
                                "task_id": task.task_id,
                                "agent_id": task.agent_id,
                                "status": task.status,
                                "output": task.result[:1000] if task.result else "",
                            })
                        await websocket.send_json({"type": "text_delta", "text": result.summary or "\n".join(result.outputs.values())})
                        await websocket.send_json({"type": "done"})
                    except Exception as e:
                        traceback.print_exc()
                        await websocket.send_json({"type": "error", "error": f"Team mode error: {e}"})
                    continue

                if agent.model is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": "No LLM configured. Set HUGINN_PROVIDER and API keys, or start Ollama."
                    })
                    continue

                # Augment with RAG context if enabled
                if cfg_chat.rag_enabled and _kb is not None and _kb.count() > 0:
                    try:
                        chunks = _kb.query(content, top_k=5)
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
                                            snapshot = _snapshot_directory(workspace_path)
                                            auto_cp_id = uuid.uuid4().hex[:8]
                                            _checkpoints[auto_cp_id] = (workspace_path, snapshot)
                                            await websocket.send_json({
                                                "type": "auto_checkpoint",
                                                "id": auto_cp_id,
                                                "base": str(workspace_path),
                                                "files": len(snapshot),
                                            })
                                        except Exception as e:
                                            print(f"[auto-cp] failed: {e}")
                                    await websocket.send_json({
                                        "type": "tool_call",
                                        "id": tid,
                                        "name": name,
                                        "args": tc.get("args", {}),
                                    })

                        # Emit tool results
                        if isinstance(last_msg, ToolMessage):
                            tid = getattr(last_msg, "tool_call_id", None)
                            if tid and tid not in seen_tool_results:
                                seen_tool_results.add(tid)
                                await websocket.send_json({
                                    "type": "tool_result",
                                    "id": tid,
                                    "content": str(getattr(last_msg, "content", "")),
                                })

                        # Only send text delta for assistant content
                        if hasattr(last_msg, "content") and not isinstance(last_msg, ToolMessage):
                            # Only send delta (new content)
                            delta = last_msg.content[len(full_response):]
                            if delta:
                                full_response = last_msg.content
                                await websocket.send_json({
                                    "type": "text_delta",
                                    "text": delta,
                                })
                    
                    # Signal completion
                    await websocket.send_json({"type": "done"})
                    
                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Agent error: {str(e)}"
                    })
            
            elif msg_type == "explore_start":
                # Exploration mode — run real exploration engine
                await websocket.send_json({
                    "type": "text_delta",
                    "text": f"🚀 Starting exploration: {content}\n"
                })
                try:
                    from huginn.exploration.orchestrator import ExplorationOrchestrator
                    from huginn.exploration.strategies import ParetoPruningStrategy

                    orch = ExplorationOrchestrator(
                        strategy=ParetoPruningStrategy(max_active=5),
                        max_parallel=3,
                    )

                    # Parse exploration config from message if provided
                    config = data.get("config", {})
                    initial_branches = config.get("initial_branches", [
                        {"name": "baseline", "hypothesis": f"Baseline for: {content}"}
                    ])
                    objectives = config.get("objectives", {"score": "maximize"})

                    result = await orch.explore(
                        objective=content,
                        initial_branches=initial_branches,
                        objectives_config=objectives,
                        max_iterations=config.get("max_iterations", 10),
                    )

                    # Stream results
                    await websocket.send_json({
                        "type": "text_delta",
                        "text": f"\n✅ Exploration complete!\n"
                                f"• Branches explored: {result.n_branches_explored}\n"
                                f"• Branches pruned: {result.n_branches_pruned}\n"
                                f"• Pareto front size: {len(result.pareto_front)}\n"
                                f"• Convergence: {result.convergence_reason}\n",
                    })

                    if result.best_branch:
                        await websocket.send_json({
                            "type": "text_delta",
                            "text": f"\n🏆 Best branch: {result.best_branch['name']}\n"
                                    f"   Hypothesis: {result.best_branch['hypothesis']}\n"
                                    f"   Objectives: {result.best_branch['objectives']}\n",
                        })

                    # Send structured data as final message
                    await websocket.send_json({
                        "type": "exploration_result",
                        "data": {
                            "pareto_front": result.pareto_front,
                            "best_branch": result.best_branch,
                            "convergence_reason": result.convergence_reason,
                        }
                    })

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Exploration failed: {str(e)}"
                    })
                await websocket.send_json({"type": "done"})
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# ── math-anything compatibility stubs ──────────────────────────

@app.get("/firewall/status")
async def firewall_status() -> dict[str, Any]:
    return {"enabled": False}


@app.post("/sandbox/execute")
async def sandbox_execute(params: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a sandbox."""
    code = params.get("code", "")
    timeout = params.get("timeout_seconds", 10)

    # Pre-validate code against restricted execution policy
    try:
        from huginn.security import validate_code, RestrictedPythonError
        validate_code(code)
    except RestrictedPythonError as e:
        return {"success": False, "error": f"Policy violation: {e}"}

    try:
        import subprocess
        from huginn.security import SandboxExecutor, SandboxConfig
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        sandbox = SandboxExecutor(
            SandboxConfig(
                allowed_executables={"python", "python3"},
                default_timeout=min(float(timeout), 300.0),
                max_timeout=300.0,
                max_output_bytes=10 * 1024 * 1024,
            )
        )
        sb_result = sandbox.run(
            ["python", tmp_path],
            timeout=min(float(timeout), 300.0),
        )
        result = sb_result

        os.unlink(tmp_path)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/analyze/symmetry")
async def analyze_symmetry(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented. Install spglib for symmetry analysis."}


@app.post("/analyze/spectral")
async def analyze_spectral(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/dynamics")
async def analyze_dynamics(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/tda")
async def analyze_tda(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/analyze/sindy")
async def analyze_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@app.post("/viz/dos")
async def viz_dos(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>DOS visualization stub</div>"}


@app.post("/viz/phase")
async def viz_phase(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Phase portrait stub</div>"}


@app.post("/viz/persistence")
async def viz_persistence(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Persistence diagram stub</div>"}


@app.post("/viz/sindy")
async def viz_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>SINDy visualization stub</div>"}


# ── HPC endpoints ──────────────────────────────────────────────

@app.post("/hpc/test")
async def hpc_test_connection(params: dict[str, Any]) -> dict[str, Any]:
    """Test SSH connection to an HPC cluster."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        port=params.get("port", 22),
    )
    
    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}
    
    try:
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            if rc == 0:
                return {
                    "success": True,
                    "hostname": stdout,
                    "scheduler": cfg.scheduler,
                }
            else:
                return {"success": False, "error": stderr or "Connection failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hpc/submit")
async def hpc_submit(params: dict[str, Any]) -> dict[str, Any]:
    """Submit a job to remote HPC."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        remote_work_dir=params.get("remote_work_dir", "~/huginn_jobs"),
    )
    
    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}
    
    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=params.get("command", "echo 'Hello HPC'"),
                job_name=params.get("job_name", "huginn_job"),
                walltime=params.get("walltime", "01:00:00"),
                nodes=params.get("nodes", 1),
                ntasks_per_node=params.get("ntasks_per_node", 4),
                queue=params.get("queue"),
                modules=params.get("modules", []),
                env_vars=params.get("env_vars", {}),
            )
            job_id = client.submit_job(script, job_name=params.get("job_name", "huginn_job"))
            return {"success": True, "job_id": job_id, "host": cfg.host}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hpc/status")
async def hpc_status(params: dict[str, Any]) -> dict[str, Any]:
    """Poll status of a remote HPC job."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
    )
    
    job_id = params.get("job_id")
    if not job_id:
        return {"success": False, "error": "job_id is required"}
    
    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(job_id)
            return {
                "success": True,
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "runtime": status.runtime,
                "message": status.message,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Multi-Agent Team ───────────────────────────────────────────

@app.get("/team/profiles")
async def team_profiles() -> dict[str, Any]:
    """List enabled agent profiles available for team tasks."""
    try:
        factory = get_agent_factory()
        profiles = [
            {
                "id": p.id,
                "name": p.name or p.id,
                "model_alias": p.model_alias,
                "persona": p.persona,
                "tools": p.tools,
                "enabled": p.enabled,
            }
            for p in factory.list_profiles()
        ]
        return {"profiles": profiles}
    except Exception as e:
        return {"error": str(e)}


@app.post("/team/plan")
async def team_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Ask the lead agent to break an objective into subtasks."""
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}
    try:
        orchestrator = get_orchestrator()
        plan = await orchestrator.plan(objective)
        return {
            "success": True,
            "objective": plan.objective,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "agent_id": t.agent_id,
                    "prompt": t.prompt,
                    "depends_on": t.depends_on,
                }
                for t in plan.tasks
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/team/run")
async def team_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run a multi-agent plan and return the synthesized result."""
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}

    async def on_status(task: SubTask) -> None:
        mood = (
            PetMood.WORKING
            if task.status == "running"
            else PetMood.SUCCESS
            if task.status == "done"
            else PetMood.ERROR
        )
        get_pet_bus().publish(
            mood=mood,
            message=f"{task.task_id} ({task.agent_id}): {task.status}",
            details={"task_id": task.task_id, "agent_id": task.agent_id, "status": task.status},
        )

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.run(objective, on_status=on_status)
        return {
            "success": result.success,
            "objective": result.objective,
            "summary": result.summary,
            "outputs": result.outputs,
            "error": result.error,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Checkpoint / Diff Review ───────────────────────────────────

def _snapshot_directory(base: Path) -> dict[str, str]:
    """Snapshot text files under base into a dict keyed by relative path."""
    snapshot: dict[str, str] = {}
    if not base.exists():
        return snapshot
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        # Skip obvious binary / large files
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                continue
            snapshot[str(path.relative_to(base))] = data.decode("utf-8", errors="ignore")
        except Exception:
            continue
    return snapshot


@app.post("/checkpoints")
async def create_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Create a checkpoint of the given directory for later diff review."""
    base = Path(params.get("path", ".")).resolve()
    snapshot = _snapshot_directory(base)
    cp_id = uuid.uuid4().hex[:8]
    _checkpoints[cp_id] = (base, snapshot)
    return {"id": cp_id, "base": str(base), "files": len(snapshot)}


@app.get("/checkpoints/{cp_id}")
async def get_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    return {"id": cp_id, "base": str(base), "files": list(snapshot.keys())}


@app.get("/checkpoints/{cp_id}/diff")
async def checkpoint_diff(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    current = _snapshot_directory(base)
    diffs = []
    all_files = set(snapshot.keys()) | set(current.keys())
    for rel in sorted(all_files):
        old = snapshot.get(rel, "")
        new = current.get(rel, "")
        if old == new:
            continue
        status = "added" if rel not in snapshot else "deleted" if rel not in current else "modified"
        diff_text = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        diffs.append({
            "path": rel,
            "status": status,
            "diff": diff_text,
            "old": old,
            "new": new,
        })
    return {"id": cp_id, "base": str(base), "diffs": diffs}


@app.post("/checkpoints/{cp_id}/accept")
async def accept_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    del _checkpoints[cp_id]
    return {"success": True}


@app.post("/checkpoints/{cp_id}/reject")
async def reject_checkpoint(cp_id: str) -> dict[str, Any]:
    if cp_id not in _checkpoints:
        return {"error": "checkpoint not found"}
    base, snapshot = _checkpoints[cp_id]
    current = _snapshot_directory(base)
    for rel, content in snapshot.items():
        if rel in current:
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    # Remove files that were added after the checkpoint
    for rel in current:
        if rel not in snapshot:
            path = base / rel
            try:
                path.unlink()
            except Exception:
                pass
    del _checkpoints[cp_id]
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
