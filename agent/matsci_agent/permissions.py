"""Permission system — inspired by Claude Code's utils/permissions/.

Three-level permission model:
- AUTO: read-only / safe tools execute without confirmation
- ASK: potentially expensive / destructive tools require confirmation
- DENY: explicitly blocked tools cannot be executed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from matsci_agent.types import PermissionMode, PermissionResult


# Default permission rules for material science tools
DEFAULT_PERMISSION_RULES: dict[str, PermissionMode] = {
    # Read-only / safe tools
    "structure_tool": PermissionMode.AUTO,
    "extract_tool": PermissionMode.AUTO,
    "diff_tool": PermissionMode.AUTO,
    "database_tool": PermissionMode.AUTO,
    "validate_tool": PermissionMode.AUTO,
    "visualize_tool": PermissionMode.AUTO,
    "file_read_tool": PermissionMode.AUTO,
    "web_search_tool": PermissionMode.AUTO,
    
    # Medium risk — ask for confirmation
    "vasp_tool": PermissionMode.ASK,
    "lammps_tool": PermissionMode.ASK,
    "comsol_tool": PermissionMode.ASK,
    "qe_tool": PermissionMode.ASK,
    "cp2k_tool": PermissionMode.ASK,
    "openfoam_tool": PermissionMode.ASK,
    "packing_tool": PermissionMode.ASK,
    "abaqus_tool": PermissionMode.ASK,
    "code_tool": PermissionMode.ASK,
    "gromacs_tool": PermissionMode.ASK,
    "job_tool": PermissionMode.ASK,
    "potential_tool": PermissionMode.ASK,
    "batch_tool": PermissionMode.ASK,
    "container_tool": PermissionMode.ASK,
    
    # Destructive — always ask
    "file_write_tool": PermissionMode.ASK,
    "file_edit_tool": PermissionMode.ASK,
    "notebook_edit_tool": PermissionMode.ASK,
    "git_commit_tool": PermissionMode.ASK,
    
    # Dangerous — deny by default
    "file_delete_tool": PermissionMode.DENY,
    "system_shell_tool": PermissionMode.DENY,
    # Coder tools
    "file_read_tool": PermissionMode.AUTO,
    "git_tool": PermissionMode.AUTO,
    "file_write_tool": PermissionMode.ASK,
    "file_edit_tool": PermissionMode.ASK,
    "bash_tool": PermissionMode.ASK,
}


@dataclass
class PermissionConfig:
    """User-configurable permission settings."""
    rules: dict[str, PermissionMode] = field(default_factory=lambda: DEFAULT_PERMISSION_RULES.copy())
    auto_approve_all: bool = False  # For CI/automation mode
    
    def get_mode(self, tool_name: str) -> PermissionMode:
        if self.auto_approve_all:
            return PermissionMode.AUTO
        return self.rules.get(tool_name, PermissionMode.ASK)
    
    def set_mode(self, tool_name: str, mode: PermissionMode) -> None:
        self.rules[tool_name] = mode


class PermissionChecker:
    """Checks permissions before tool execution."""
    
    def __init__(self, config: PermissionConfig | None = None):
        self.config = config or PermissionConfig()
    
    async def check(
        self,
        tool_name: str,
        is_read_only: bool = False,
        is_destructive: bool = False,
        cost_estimate: dict[str, float] | None = None
    ) -> PermissionResult:
        mode = self.config.get_mode(tool_name)
        
        if mode == PermissionMode.DENY:
            return PermissionResult(
                mode=PermissionMode.DENY,
                reason=f"Tool '{tool_name}' is explicitly blocked by permission policy"
            )
        
        if mode == PermissionMode.AUTO:
            return PermissionResult(mode=PermissionMode.AUTO)
        
        # ASK mode — build a reason string
        reasons = []
        if is_destructive:
            reasons.append("this operation is destructive")
        if cost_estimate:
            cpu = cost_estimate.get("cpu_hours", 0)
            if cpu > 1:
                reasons.append(f"estimated cost: {cpu:.1f} CPU hours")
        
        reason = f"Tool '{tool_name}' requires approval"
        if reasons:
            reason += f" ({', '.join(reasons)})"
        
        return PermissionResult(mode=PermissionMode.ASK, reason=reason)
