"""Tool base class — inspired by Claude Code's Tool<T> interface.

Every tool is self-contained with:
- name, description, input/output schemas
- permission checking
- input validation
- execution logic
- result mapping
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, get_type_hints
from pydantic import BaseModel

from huginn.types import ToolResult, ToolContext, PermissionResult, ValidationResult, PermissionMode

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class HuginnTool(ABC, Generic[InputT, OutputT]):
    """Base class for all Huginn tools.
    
    Mirrors Claude Code's Tool<Input, Output, Progress> interface.
    """
    
    name: str = ""
    description: str = ""

    # Static hints for UI / permission systems
    destructive: bool = False
    read_only: bool = False

    # Schema definitions (Pydantic v2, replacing Zod)
    input_schema: type[InputT] | None = None
    output_schema: type[OutputT] | None = None
    
    @property
    def input_json_schema(self) -> dict[str, Any] | None:
        if self.input_schema:
            return self.input_schema.model_json_schema()
        return None
    
    @abstractmethod
    async def call(self, args: InputT, context: ToolContext) -> ToolResult:
        """Execute the tool. Must be implemented by subclasses."""
        ...
    
    async def check_permissions(self, args: InputT, context: ToolContext) -> PermissionResult:
        """Check if the tool can be executed under current permissions.
        
        Default: allow. Override for tools that need explicit approval
        (e.g., job submission, file deletion).
        """
        return PermissionResult(mode=PermissionMode.AUTO)
    
    async def validate_input(self, args: InputT, context: ToolContext) -> ValidationResult:
        """Validate input before execution. Pydantic already handles schema validation,
        but this allows additional semantic checks (e.g., file existence, path validity).
        """
        return ValidationResult(result=True)
    
    def is_read_only(self, args: InputT) -> bool:
        """Return True if this tool call is read-only (no side effects).
        Read-only tools can be auto-executed without user confirmation.
        """
        return self.read_only

    def is_destructive(self, args: InputT) -> bool:
        """Return True if this tool call is destructive (deletes/overwrites data).
        Destructive tools ALWAYS require explicit user confirmation.
        """
        return self.destructive
    
    def estimate_cost(self, args: InputT) -> dict[str, float] | None:
        """Estimate computational cost for this tool call.
        Returns dict with keys like cpu_hours, gpu_hours, walltime_hours.
        Return None if cost is negligible.
        """
        return None
