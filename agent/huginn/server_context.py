"""Server-scoped context for Huginn.

Holds all long-lived objects (config, registries, agent factory, memory,
knowledge base, audit logger) so that the FastAPI server is not tied to a
handful of module-level global variables.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator
from huginn.agents.tool_dedupe import ToolDeduper
from huginn.autoloop.plan_store import PlanStore
from huginn.config import HuginnConfig
from huginn.events.transcript import TranscriptStore
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.models.registry import ModelRegistry
from huginn.permissions import PermissionConfig
from huginn.persistence import (
    CheckpointerBackend,
    JSONRemoteJobBackend,
    MemoryBackend,
    SQLiteCheckpointerBackend,
    SQLiteMemoryBackend,
    StateRegistry,
)
from huginn.security.audit import AuditLogger
from huginn.skills.registry import SkillRegistry
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
    """Container for all server-wide Huginn state.

    v23 状态树合并: ServerContext 是唯一的状态树根节点. HuginnSystem 已
    折叠为 ServerContext 的子类 (deprecated alias), 不再独立维护字段.
    active_threads / edit_tools 从 server_core 模块全局提升为字段.
    """

    # config 设默认 None, 让 ServerContext() 可无参构造 (匹配 HuginnSystem 旧契约).
    # 生产路径 create_server_context() 总是显式传 cfg, 不会留 None.
    config: HuginnConfig | None = None
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
    # 字段级状态注册表: 跨 step / 跨进程恢复. 对标 Kimi defineState.
    state_registry: StateRegistry = field(default_factory=StateRegistry.shared)
    # 对话转录: 订阅 EventBus 落 JSONL, benchmark 评审回放用.
    transcript_store: TranscriptStore = field(default_factory=TranscriptStore.shared)
    # 工具调用去重: 同一 step 内同 (tool, args) 命中即返回上次结果.
    tool_deduper: ToolDeduper = field(default_factory=ToolDeduper)
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
    # v23: 从 server_core._threads / _EDIT_TOOLS 提升为字段, 统一状态树.
    # 生产路径仍由 server_core.get_system_snapshot() 在快照时填入.
    active_threads: dict[str, Any] = field(default_factory=dict)
    edit_tools: set[str] = field(default_factory=set)

    # ── 诊断 / 运维接口 (从 HuginnSystem 移植, 避免重复实现) ───────

    @property
    def is_configured(self) -> bool:
        """Check if the system has minimum required configuration."""
        return self.config is not None

    def get_component(self, name: str) -> Any:
        """Get a system component by name, returning None if not set."""
        return getattr(self, name, None)

    def list_components(self) -> dict[str, bool]:
        """List all components and whether they are initialized.

        Returns a dict of {field_name: bool_is_set}. Includes the canonical
        set of long-lived components (state_registry / transcript_store /
        tool_deduper 等基础设施不在此列 — 它们总是初始化的).
        """
        components: dict[str, bool] = {}
        for attr in [
            "config",
            "tool_registry",
            "skill_registry",
            "audit_logger",
            "memory_backend",
            "checkpointer_backend",
            "remote_job_backend",
            "agent_factory",
            "orchestrator",
            "memory_manager",
            "kb",
            "codebase",
            "agent",
            "planner_agent",
            "mcp_manager",
            "plan_store",
        ]:
            components[attr] = getattr(self, attr, None) is not None
        return components


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

    # 启动对话转录订阅 (best-effort, 失败不影响服务启动)
    transcript_store = TranscriptStore.shared()
    with contextlib.suppress(Exception):
        transcript_store.start()

    return ServerContext(
        config=cfg,
        permission_config=permission_config,
        audit_logger=audit_logger,
        memory_manager=memory_manager,
        agent_factory=agent_factory,
        orchestrator=orchestrator,
        plan_store=plan_store,
        transcript_store=transcript_store,
        state_registry=StateRegistry.shared(),
        tool_deduper=ToolDeduper(),
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
