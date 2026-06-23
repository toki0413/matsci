"""Core type definitions for Huginn.

Inspired by Claude Code's Tool.ts — every type is explicit and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class PermissionMode(Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


class BudgetDecision(Enum):
    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


class HandleType(str, Enum):
    """Types of opaque handles used by tools to reference external resources."""

    FILE_PATH = "file_path"
    JOB_ID = "job_id"
    MATERIAL_ID = "material_id"
    FORMULA = "formula"


@dataclass
class PermissionResult:
    mode: PermissionMode
    reason: str | None = None


@dataclass
class ToolResult:
    """Result of a tool execution, mirroring Claude Code's ToolResult<T>."""

    data: Any
    success: bool = True
    error: str | None = None
    new_messages: list[dict] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    result: bool
    message: str = ""
    error_code: int = 0


@dataclass
class AgentMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """Runtime context passed to every tool call."""

    session_id: str
    workspace: str
    abort_controller: Any | None = None
    permissions: dict[str, PermissionMode] = field(default_factory=dict)
    memory_manager: Any | None = None
    agent_factory: Any | None = None
    audit_logger: Any | None = None
    boundary_state: Any | None = None
    config: Any | None = None


@dataclass
class CostEstimate:
    cpu_hours: float
    gpu_hours: float
    memory_gb: float
    storage_gb: float
    walltime_hours: float


@dataclass
class BudgetPolicy:
    max_cpu_hours: float = float("inf")
    max_gpu_hours: float = float("inf")
    max_storage_gb: float = float("inf")
    max_parallel_jobs: int = 5
    max_walltime_hours: float = 168.0

    def check(self, estimate: CostEstimate) -> tuple[BudgetDecision, str]:
        if estimate.cpu_hours > self.max_cpu_hours:
            return (
                BudgetDecision.DENY,
                f"CPU hours {estimate.cpu_hours:.1f} exceed budget {self.max_cpu_hours:.1f}",
            )
        if estimate.gpu_hours > self.max_gpu_hours:
            return (
                BudgetDecision.DENY,
                f"GPU hours {estimate.gpu_hours:.1f} exceed budget {self.max_gpu_hours:.1f}",
            )
        if estimate.storage_gb > self.max_storage_gb:
            return (
                BudgetDecision.DENY,
                f"Storage {estimate.storage_gb:.1f}GB exceed budget {self.max_storage_gb:.1f}GB",
            )
        return BudgetDecision.ALLOW, ""
