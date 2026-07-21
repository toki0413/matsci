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
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# P2-6 belief upgrade: 单次失败切 mode 太激进, 跟用户 "prefer warnings + force
# proceed" 偏好冲突. Beta(α, β) 共轭: α += success, β += fail. 当 β/(α+β) >
# threshold 时才切 mode. 跨 turn 状态在 TaskReflector 实例上累积.
#
# ceiling: alpha/beta 跑久了会膨胀, 切 mode 决策变迟钝. 用滑动窗口 (最近 N 次)
# 而非全历史. ponytail: 不引入新依赖, Beta 共轭闭合解.
_MODE_BELIEF_WINDOW = 20  # 最近 N 次 tool result
_MODE_SWITCH_THRESHOLD = 0.5  # 失败率超过 50% 才切 discover


def _belief_switch_enabled() -> bool:
    """toggle: env HUGINN_BELIEF_MODE_SWITCH (默认 on). off 时回退原硬规则."""
    return os.environ.get("HUGINN_BELIEF_MODE_SWITCH", "1") != "0"


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

    def to_transition_signal(self) -> str:
        """Map reflection result to a cognitive state machine signal type.

        Returns the signal_type string for TransitionSignal, or "" if no signal.
        """
        if not self.tool_succeeded or self.has_physics_errors:
            return "physics_error" if self.has_physics_errors else "tool_failure"
        if self.plan_step_completed:
            return "tool_success"
        if self.has_physics_warnings:
            return "tool_success"  # warnings don't block
        return "tool_success"

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

    def __init__(self) -> None:
        # P2-6 Beta belief: 滑动窗口记 success/fail, 累积 belief 后才切 mode.
        # 单次失败只降 confidence, 不立即切. 满足用户 "prefer warnings + force proceed".
        self._mode_window: list[bool] = []  # True=success, False=fail

    def _record_and_decide_switch(
        self, success: bool, session_state: Any,
    ) -> tuple[bool, str]:
        """P2-6 belief-driven mode switch. 返回 (should_switch, suggested_mode).

        滑动窗口 Beta 共轭: α = 窗口内 success 数, β = 窗口内 fail 数.
        失败率 = β/(α+β) > threshold 才切 discover. 单次失败只降 confidence (alpha 减),
        不切. 成功多时建议 construct.
        """
        self._mode_window.append(success)
        if len(self._mode_window) > _MODE_BELIEF_WINDOW:
            self._mode_window = self._mode_window[-_MODE_BELIEF_WINDOW:]

        # toggle off → 回退原硬规则: 单次失败立即切
        if not _belief_switch_enabled():
            if not success and session_state:
                current_mode = getattr(session_state, "cognitive_mode", None)
                if current_mode and current_mode.value == "construct":
                    return True, "discover"
            return False, ""

        # 数据不足时不切 (至少 3 次结果)
        if len(self._mode_window) < 3:
            return False, ""

        alpha = sum(1 for s in self._mode_window if s)
        beta = len(self._mode_window) - alpha
        fail_rate = beta / (alpha + beta)

        # metrics
        try:
            from huginn.routes.metrics import track_belief_update
            track_belief_update("beta")
        except Exception:
            pass

        current_mode = getattr(session_state, "cognitive_mode", None) if session_state else None
        in_construct = bool(current_mode and current_mode.value == "construct")

        # 失败率高 + 在 construct → 切 discover
        if fail_rate > _MODE_SWITCH_THRESHOLD and in_construct:
            return True, "discover"
        # 失败率低 + 在 discover → 切 construct (信号: 探索成功, 可以建了)
        if fail_rate < (1 - _MODE_SWITCH_THRESHOLD) and current_mode and current_mode.value == "discover":
            return True, "construct"
        return False, ""

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
            # P2-6 belief: 不再单次失败立即切. _record_and_decide_switch 算窗口 belief.
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

        # 5. Cognitive mode assessment — P2-6 belief-driven
        # 单次失败不再立即切; 走滑动窗口 Beta 共轭.
        should_switch, suggested = self._record_and_decide_switch(success, session_state)
        if should_switch:
            result.should_switch_mode = True
            result.suggested_mode = suggested
        elif result.plan_step_completed:
            # 稳定成功时建议留在 construct
            result.suggested_mode = "construct"

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


def _selfcheck() -> int:
    """assert-based demo for P2-6 belief-driven mode switch."""
    import os as _os
    from enum import Enum

    class _Mode(Enum):
        DISCOVER = "discover"
        CONSTRUCT = "construct"

    class _FakeState:
        def __init__(self, mode: str) -> None:
            self.cognitive_mode = _Mode(mode)
            self.active_plan_id = None

    # save env, 强制 belief 模式
    _saved = _os.environ.get("HUGINN_BELIEF_MODE_SWITCH")
    _os.environ["HUGINN_BELIEF_MODE_SWITCH"] = "1"

    # 48. 单次失败不切 (窗口 < 3 不决策)
    r = TaskReflector()
    res = r.reflect("vasp_tool", {"success": False}, _FakeState("construct"))
    assert not res.should_switch_mode, "单次失败不应立即切"
    assert res.confirm_type == "replan", "但仍应提示 replan"

    # 49. 窗口 >= 3, 失败率 > 50% 在 construct → 切 discover
    r2 = TaskReflector()
    for _ in range(2):
        r2.reflect("t", {"success": False}, _FakeState("construct"))
    res = r2.reflect("t", {"success": False}, _FakeState("construct"))
    assert res.should_switch_mode and res.suggested_mode == "discover", \
        f"3 次失败应切 discover, got switch={res.should_switch_mode} mode={res.suggested_mode}"

    # 50. 窗口有混合, 失败率 <= 50% 不切
    r3 = TaskReflector()
    r3.reflect("t", {"success": True}, _FakeState("construct"))
    r3.reflect("t", {"success": True}, _FakeState("construct"))
    res = r3.reflect("t", {"success": False}, _FakeState("construct"))
    assert not res.should_switch_mode, "2 成功 1 失败不应切"

    # 51. 在 discover 高成功率 → 切 construct
    r4 = TaskReflector()
    for _ in range(4):
        r4.reflect("t", {"success": True}, _FakeState("discover"))
    res = r4.reflect("t", {"success": True}, _FakeState("discover"))
    assert res.should_switch_mode and res.suggested_mode == "construct", \
        f"5 成功在 discover 应切 construct, got switch={res.should_switch_mode} mode={res.suggested_mode}"

    # 52. toggle off → 回退原硬规则 (单次失败立即切)
    _os.environ["HUGINN_BELIEF_MODE_SWITCH"] = "0"
    r5 = TaskReflector()
    res = r5.reflect("t", {"success": False}, _FakeState("construct"))
    assert res.should_switch_mode and res.suggested_mode == "discover", \
        "toggle off 时单次失败应立即切 (原硬规则)"

    # restore env
    if _saved is None:
        _os.environ.pop("HUGINN_BELIEF_MODE_SWITCH", None)
    else:
        _os.environ["HUGINN_BELIEF_MODE_SWITCH"] = _saved

    print("task_reflector selfcheck OK (48-52 P2-6 belief-driven mode switch)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
