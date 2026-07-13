"""Server-scoped context for Huginn.

Holds all long-lived objects (config, registries, agent factory, memory,
knowledge base, audit logger) so that the FastAPI server is not tied to a
handful of module-level global variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator
from huginn.autoloop.plan_store import PlanStore
from huginn.config import HuginnConfig
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.models.registry import ModelRegistry
from huginn.permissions import PermissionConfig
from huginn.persistence import (
    CheckpointerBackend,
    JSONRemoteJobBackend,
    MemoryBackend,
    SQLiteCheckpointerBackend,
    SQLiteMemoryBackend,
)
from huginn.security.audit import AuditLogger
from huginn.skills.registry import SkillRegistry
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


def _default_audit_logger() -> AuditLogger:
    base = os.environ.get("HUGINN_CACHE_DIR")
    log_path = Path(base) / "audit.jsonl" if base else Path.home() / ".huginn" / "audit.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return AuditLogger(log_path)


def _default_memory_backend() -> MemoryBackend:
    return SQLiteMemoryBackend()


@dataclass
class ServerContext:
    """Container for all server-wide Huginn state."""

    config: HuginnConfig
    tool_registry: type[ToolRegistry] = field(default=ToolRegistry)
    skill_registry: type[SkillRegistry] = field(default=SkillRegistry)
    permission_config: PermissionConfig = field(default_factory=PermissionConfig)
    audit_logger: AuditLogger = field(default_factory=_default_audit_logger)
    checkpointer_backend: CheckpointerBackend = field(
        default_factory=lambda: SQLiteCheckpointerBackend()
    )
    memory_backend: MemoryBackend = field(default_factory=_default_memory_backend)
    remote_job_backend: JSONRemoteJobBackend = field(
        default_factory=lambda: JSONRemoteJobBackend()
    )
    agent_factory: AgentFactory | None = None
    orchestrator: Orchestrator | None = None
    memory_manager: MemoryManager | None = None
    kb: Any | None = None
    codebase: Any | None = None
    agent: Any | None = None
    planner_agent: Any | None = None
    plan_store: PlanStore | None = None
    mcp_manager: Any | None = None
    # 加密 RAG 管理器 (可选): encryption_enabled=True 时由 lifespan 初始化,
    # 提供文档级 + DB 级加密. 与 kb (ChromaDB 明文) 独立存储.
    encrypted_rag: Any | None = None


def create_server_context(config: HuginnConfig | None = None) -> ServerContext:
    """Create and initialize a server context.

    Core tools are registered synchronously; optional tools are registered
    in the background by lifespan. This keeps context creation fast.
    """
    from huginn.tools import register_core_tools
    register_core_tools()
    # Use _load_runtime_config so huginn.toml is picked up the same way
    # the REST /config route does it — from_env() alone misses the file.
    if config is not None:
        cfg = config
    else:
        from huginn.routes.config import _load_runtime_config
        cfg = _load_runtime_config()

    permission_config = PermissionConfig()
    audit_logger = _default_audit_logger()

    memory_md = Path(cfg.workspace) / "MEMORY.md" if cfg.workspace else None
    memory_backend = SQLiteMemoryBackend()
    memory_manager = MemoryManager(
        config=MemoryConfig(memory_md_path=memory_md),
        longterm=memory_backend._impl,
    )

    agent_factory = AgentFactory(
        config=cfg,
        model_registry=ModelRegistry.from_config(cfg),
        memory_manager=memory_manager,
    )

    plan_store = PlanStore()
    orchestrator = Orchestrator(
        factory=agent_factory,
        memory_manager=memory_manager,
        max_concurrent=cfg.max_concurrent_subagents,
        plan_store=plan_store,
        auto_confirm=cfg.plan_auto_confirm,
    )

    return ServerContext(
        config=cfg,
        permission_config=permission_config,
        audit_logger=audit_logger,
        memory_manager=memory_manager,
        agent_factory=agent_factory,
        orchestrator=orchestrator,
        plan_store=plan_store,
    )


_server_context: ServerContext | None = None


def get_server_context() -> ServerContext:
    """Return the global server context, initializing it if necessary."""
    global _server_context
    if _server_context is None:
        _server_context = create_server_context()
    return _server_context


def set_server_context(ctx: ServerContext) -> None:
    """Replace the global server context (useful for tests and multi-tenant)."""
    global _server_context
    _server_context = ctx
