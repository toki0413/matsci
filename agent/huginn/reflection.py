"""Reflection — lightweight post-task assessment after each tool execution.

Runs after each tool call in the chat() loop, before the response is sent.
Checks:
  1. Physics plausibility (delegates to PhysicsAuditor if applicable)
  2. Plan advancement (did we complete a step?)
  3. Evolution trigger (should we learn from this?)
  4. Cognitive mode assessment (should we switch discover↔construct?)

Pure rule-based, no LLM calls, <1ms. Zero token consumption.

This is the "reflection" phase in the loop engineering cycle:
  EXPLORE → PLAN → EXECUTE → [REFLECT] → REPORT → EXPLORE → ...
                              ↑ you are here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Outcome of reflecting on a tool execution."""
    # Did the tool succeed?
    tool_succeeded: bool = True
    # Physics audit findings (if applicable)
    has_physics_errors: bool = False
    has_physics_warnings: bool = False
    # Plan advancement
    plan_step_completed: bool = False
    # Should we trigger evolution?
    should_evolve: bool = False
    evolve_signal: str = ""  # "failure" | "success" | ""
    # Should we switch cognitive mode?
    should_switch_mode: bool = False
    suggested_mode: str = ""  # "discover" | "construct"
    # Human-facing message
    message: str = ""
    # Whether to ask user for confirmation before proceeding
    needs_user_input: bool = False
    confirm_type: str = ""  # "continue" | "replan" | "mode_switch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_succeeded": self.tool_succeeded,
            "has_physics_errors": self.has_physics_errors,
            "has_physics_warnings": self.has_physics_warnings,
            "plan_step_completed": self.plan_step_completed,
            "should_evolve": self.should_evolve,
            "evolve_signal": self.evolve_signal,
            "should_switch_mode": self.should_switch_mode,
            "suggested_mode": self.suggested_mode,
            "message": self.message,
            "needs_user_input": self.needs_user_input,
            "confirm_type": self.confirm_type,
        }


class TaskReflector:
    """Reflects on tool execution results after each tool call.

    Usage in chat() loop:
        reflector = TaskReflector()
        result = reflector.reflect(
            tool_name="vasp_tool",
            tool_result=tool_output,
            session_state=session_state,
        )
        if result.should_evolve:
            evolution_engine.evolve_from_failures(...)
        if result.needs_user_input:
            yield confirmation_request
    """

    def reflect(
        self,
        tool_name: str,
        tool_result: dict[str, Any],
        session_state: Any = None,
        input_params: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        """Assess a tool execution result.

        Args:
            tool_name: name of the tool that was called
            tool_result: the tool's output dict (may contain "physics_audit", "parsed", etc.)
            session_state: UnifiedSessionState (optional, for plan context)
            input_params: the tool's input parameters (for evolution context)
        """
        result = ReflectionResult()
        input_params = input_params or {}

        # 1. Check tool success
        success = tool_result.get("success", True)
        error = tool_result.get("error", "")
        result.tool_succeeded = success

        # 2. Check physics audit (if present)
        audit = tool_result.get("physics_audit")
        if audit:
            result.has_physics_errors = audit.get("has_errors", False)
            result.has_physics_warnings = audit.get("has_warnings", False)

        # 3. Determine if this was a failure worth learning from
        if not success or result.has_physics_errors:
            result.should_evolve = True
            result.evolve_signal = "failure"
            result.message = self._build_failure_message(
                tool_name, error, audit
            )
            # On failure, suggest switching to discovery mode to explore alternatives
            result.should_switch_mode = True
            result.suggested_mode = "discover"
            result.needs_user_input = True
            result.confirm_type = "replan"
        elif result.has_physics_warnings:
            # Warnings don't block, but inform the user
            findings = audit.get("findings", [])
            warning_msgs = [
                f["message"] for f in findings
                if f.get("severity") == "warning"
            ]
            result.message = "Tool completed with warnings:\n" + "\n".join(warning_msgs)
            result.needs_user_input = True
            result.confirm_type = "continue"
        else:
            # Success — check if plan step is complete
            result.message = self._build_success_message(tool_name, tool_result)

        # 4. Check plan advancement
        if session_state and session_state.active_plan_id:
            # If the tool succeeded and we're executing a plan,
            # consider this step potentially complete
            if success and not result.has_physics_errors:
                result.plan_step_completed = True
                # Check if this was the last step
                # (the caller will verify and update the plan)
                result.should_evolve = True
                result.evolve_signal = "success"

        # 5. Cognitive mode assessment
        # If we just completed a plan step, suggest staying in construct mode
        if result.plan_step_completed and not result.should_switch_mode:
            result.suggested_mode = "construct"

        # If tool failed and we're in construct mode, suggest switching to discover
        if not success and session_state:
            current_mode = getattr(session_state, "cognitive_mode", None)
            if current_mode and current_mode.value == "construct":
                result.should_switch_mode = True
                result.suggested_mode = "discover"

        return result

    def _build_failure_message(
        self,
        tool_name: str,
        error: str,
        audit: dict[str, Any] | None,
    ) -> str:
        """Build a human-readable failure message."""
        parts = [f"Tool '{tool_name}' encountered an issue."]
        if error:
            parts.append(f"Error: {error}")
        if audit and audit.get("has_errors"):
            error_findings = [
                f["message"] for f in audit.get("findings", [])
                if f.get("severity") == "error"
            ]
            if error_findings:
                parts.append("Physics concerns:")
                for msg in error_findings:
                    parts.append(f"  - {msg}")
        parts.append("Would you like to adjust the approach or try alternatives?")
        return "\n".join(parts)

    def _build_success_message(
        self,
        tool_name: str,
        tool_result: dict[str, Any],
    ) -> str:
        """Build a human-readable success message."""
        parts = [f"Tool '{tool_name}' completed successfully."]
        # Try to extract a summary from the result
        parsed = tool_result.get("parsed", {})
        if isinstance(parsed, dict):
            # Highlight key results
            highlights = []
            if "energy" in parsed and parsed["energy"] is not None:
                highlights.append(f"energy={parsed['energy']:.4f} eV")
            if "band_gap" in parsed and parsed["band_gap"] is not None:
                highlights.append(f"band_gap={parsed['band_gap']:.2f} eV")
            if "converged" in parsed:
                highlights.append(f"converged={parsed['converged']}")
            if highlights:
                parts.append("Key results: " + ", ".join(highlights))
        return "\n".join(parts)
