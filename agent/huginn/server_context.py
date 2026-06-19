"""Server-scoped context for Huginn.

Holds all long-lived objects (config, registries, agent factory, memory,
knowledge base, audit logger) so that the FastAPI server is not tied to a
handful of module-level global variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator
from huginn.config import HuginnConfig
from huginn.memory.manager import MemoryConfig, MemoryManager
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
    log_path = Path.home() / ".huginn" / "audit.jsonl"
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
    mcp_manager: Any | None = None


def create_server_context(config: HuginnConfig | None = None) -> ServerContext:
    """Create and initialize a server context.

    Tool registration happens once at startup.
    """
    register_all_tools()
    cfg = config or HuginnConfig.from_env()

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
        memory_manager=memory_manager,
    )

    orchestrator = Orchestrator(
        factory=agent_factory,
        memory_manager=memory_manager,
        max_concurrent=cfg.max_concurrent_subagents,
    )

    return ServerContext(
        config=cfg,
        permission_config=permission_config,
        audit_logger=audit_logger,
        memory_manager=memory_manager,
        agent_factory=agent_factory,
        orchestrator=orchestrator,
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
