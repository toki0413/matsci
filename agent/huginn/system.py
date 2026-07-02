"""System object — single source of truth for Huginn runtime state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HuginnSystem:
    """Consolidates all Huginn runtime state into a single object.

    Replaces scattered module-level globals from server_core.py.
    """

    config: Any | None = None
    tool_registry: Any | None = None
    skill_registry: Any | None = None
    audit_logger: Any | None = None
    memory_backend: Any | None = None
    checkpointer_backend: Any | None = None
    remote_job_backend: Any | None = None
    agent_factory: Any | None = None
    orchestrator: Any | None = None
    memory_manager: Any | None = None
    kb: Any | None = None
    codebase: Any | None = None
    agent: Any | None = None
    planner_agent: Any | None = None
    mcp_manager: Any | None = None
    plan_store: Any | None = None

    # Thread/execution tracking
    active_threads: dict[str, Any] = field(default_factory=dict)
    edit_tools: set[str] = field(default_factory=set)

    # Permission config
    permission_config: Any | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the system has minimum required configuration."""
        return self.config is not None

    def get_component(self, name: str) -> Any:
        """Get a system component by name, returning None if not set."""
        return getattr(self, name, None)

    def list_components(self) -> dict[str, bool]:
        """List all components and whether they are initialized."""
        components = {}
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


_system: HuginnSystem | None = None


def get_system() -> HuginnSystem:
    """Return the global HuginnSystem, creating one lazily if needed."""
    global _system
    if _system is None:
        _system = HuginnSystem()
    return _system


def set_system(system: HuginnSystem) -> None:
    """Replace the global HuginnSystem instance."""
    global _system
    _system = system
