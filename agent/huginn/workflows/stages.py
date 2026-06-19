"""Workflow dataclasses shared by the engine, templates, and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from huginn.types import ToolResult


@dataclass
class ValidationRule:
    """Rule for validating stage output."""

    check: Literal["convergence", "energy_sign", "force_threshold", "custom"]
    threshold: float | None = None
    custom_fn: str | None = None  # Name of validation function


@dataclass
class RetryPolicy:
    """Retry policy for failed stages."""

    max_retries: int = 2
    backoff_factor: float = 2.0
    retry_on: list[
        Literal["convergence_fail", "timeout", "oom", "remote_failure", "any"]
    ] = field(default_factory=lambda: ["convergence_fail", "timeout", "remote_failure"])
    auto_diagnose: bool = True  # Whether to call diagnose_tool before retry
    apply_auto_fix: bool = True  # Whether to apply suggested fixes from diagnosis


@dataclass
class ComputationalStage:
    """Single stage in a computational workflow."""

    id: str
    name: str
    tool: str  # Tool name to invoke
    tool_input: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    validation: ValidationRule | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    # Execution state
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result: ToolResult | None = None
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""

    success: bool
    stages: dict[str, ComputationalStage]
    outputs: dict[str, Any]
    error: str | None = None
    total_walltime: float = 0.0  # seconds
