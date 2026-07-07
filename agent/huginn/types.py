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
    PLAN = "plan"  # 只读模式, 所有写工具强制 ASK


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
    """Result of a tool execution, mirroring Claude Code's ToolResult<T>.

    CLI-Anything 契约: 任何工具的输出都必须可序列化为 JSON.
    to_dict() / to_json() 处理常见不可序列化类型.
    """

    data: Any
    success: bool = True
    error: str | None = None
    new_messages: list[dict] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (CLI-Anything --json contract)."""
        d = {
            "data": _jsonify(self.data),
            "success": self.success,
            "error": self.error,
            "side_effects": list(self.side_effects),
        }
        if self.metadata:
            d["metadata"] = _jsonify(self.metadata)
        return d

    def to_json(self, **kwargs: Any) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


def _jsonify(obj: Any) -> Any:
    """Recursively convert non-serializable types to JSON-safe equivalents."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1 fallback
    if hasattr(obj, "dict") and not isinstance(obj, dict):
        return obj.dict()
    # numpy
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, dict):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return [ _jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    # datetime / other isoformat-able objects
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


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
