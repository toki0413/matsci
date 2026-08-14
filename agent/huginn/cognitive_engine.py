"""Cognitive engine — S0-S6 state machine + dual-mode attention from the Double Helix.

The state machine drives the loop engineering cycle:
    S0(Blank) → S1(Discover) → S2(Validate) → S3(Switch) → S4(Construct) → S5(Unify) → S6(Feedback) → S1...

Each state maps to:
  - A CognitiveMode (DISCOVER / CONSTRUCT) — which "chain" of the double helix
  - An AttentionMode (singularity condensation / axiom focus) — how the agent allocates focus
  - A prompt strategy — what system message to inject for this state
  - Transition rules — what signals move us to the next state

Dual-mode attention is implemented as prompt strategy differentiation:
  - Singularity condensation: "what patterns are glowing? generate hypotheses, explore"
  - Axiom focus: "verify rigorously, check each step, build proof chains"

This is NOT a new attention mechanism — it's a cognitive layer that shapes
what the LLM pays attention to via prompt engineering and tool filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveState(StrEnum):
    """S0-S6 states from the Double Helix state machine."""

    S0_BLANK = "s0_blank"          # fresh session, no goal yet
    S1_DISCOVER = "s1_discover"    # exploring problem space, generating hypotheses
    S2_VALIDATE = "s2_validate"    # checking hypotheses against data
    S3_SWITCH = "s3_switch"        # user confirmed approach, transitioning to execution
    S4_CONSTRUCT = "s4_construct"  # executing plan, building solution rigorously
    S5_UNIFY = "s5_unify"          # integrating results into coherent whole
    S6_FEEDBACK = "s6_feedback"    # gaps found, looping back to discovery
    S7_SELF_MODIFY = "s7_self_modify"   # 哥德尔机自修改触发点：评估要不要改策略/原则


class AttentionMode(StrEnum):
    """Which attention strategy the agent uses."""

    SINGULARITY_CONDENSATION = "singularity_condensation"  # discovery: what's glowing?
    AXIOM_FOCUS = "axiom_focus"                            # construction: what's weakest?
    MODE_SWITCH = "mode_switch"                            # transition state


# ── State → mode mapping ──────────────────────────────────────────────

STATE_TO_ATTENTION: dict[CognitiveState, AttentionMode] = {
    CognitiveState.S0_BLANK: AttentionMode.SINGULARITY_CONDENSATION,
    CognitiveState.S1_DISCOVER: AttentionMode.SINGULARITY_CONDENSATION,
    CognitiveState.S2_VALIDATE: AttentionMode.SINGULARITY_CONDENSATION,
    CognitiveState.S3_SWITCH: AttentionMode.MODE_SWITCH,
    CognitiveState.S4_CONSTRUCT: AttentionMode.AXIOM_FOCUS,
    CognitiveState.S5_UNIFY: AttentionMode.AXIOM_FOCUS,
    CognitiveState.S6_FEEDBACK: AttentionMode.MODE_SWITCH,
    CognitiveState.S7_SELF_MODIFY: AttentionMode.MODE_SWITCH,  # meta 状态，不归任一链
}

# CSM state → model router task: 构造阶段走 reasoning, 验证/自修改走 verification
# 没列出的 state (S0/S1/S3/S5) 沿用调用方传入的 task
STATE_TO_MODEL_TASK: dict = {
    CognitiveState.S4_CONSTRUCT: "reasoning",
    CognitiveState.S2_VALIDATE: "verification",
    CognitiveState.S7_SELF_MODIFY: "verification",
    CognitiveState.S6_FEEDBACK: "reasoning",
}

# States that belong to the discovery chain (Ramanujan)
DISCOVERY_STATES = {
    CognitiveState.S0_BLANK,
    CognitiveState.S1_DISCOVER,
    CognitiveState.S2_VALIDATE,
}

# States that belong to the construction chain (Bourbaki)
CONSTRUCTION_STATES = {
    CognitiveState.S4_CONSTRUCT,
    CognitiveState.S5_UNIFY,
}


# v6 R22: CSM 为 canonical 状态机, 其他 phase 枚举 (ResearchPhase / SessionPhase /
# autoloop AUTOLOOP_PHASES) 通过 UnifiedBus 的 cognitive.csm.transition 事件同步.
# 映射表: CSM 状态 → autoloop phase 字符串 (perceive/hypothesize/plan/execute/validate/learn/report)
# 用于订阅者做状态映射. 不强制 1:1 — S3/S6/S7 是元状态.
STATE_TO_PHASE: dict[CognitiveState, str] = {
    CognitiveState.S0_BLANK: "perceive",
    CognitiveState.S1_DISCOVER: "hypothesize",
    CognitiveState.S2_VALIDATE: "validate",
    CognitiveState.S3_SWITCH: "plan",
    CognitiveState.S4_CONSTRUCT: "execute",
    CognitiveState.S5_UNIFY: "learn",
    CognitiveState.S6_FEEDBACK: "validate",
    CognitiveState.S7_SELF_MODIFY: "learn",
}


def get_phase_for_state(state: CognitiveState) -> str:
    """CSM 状态 → autoloop phase 字符串的便捷查询."""
    return STATE_TO_PHASE.get(state, "perceive")


# v23 Round 9: CSMListener Protocol 已删除.
# 之前全仓零注册 (grep register_listener 无调用方), 已被 UnifiedBus
# (cognitive.csm.transition 事件) 完全替代. 删除 Protocol + register_listener
# 方法 + _listeners 字段 + 旧广播循环, 保留 UnifiedBus 发布路径.
# 历史背景: v6 R22 引入 CSMListener 作为过渡方案, v23 引入 UnifiedBus 后
# Protocol 成为死代码 (零注册, 永远空转).


# ── Transition signals ────────────────────────────────────────────────

@dataclass
class TransitionSignal:
    """A signal that may trigger a state transition.

    Produced by reflection, user input analysis, or plan events.
    The state machine consumes these to decide transitions.
    """
    signal_type: str  # "user_goal" | "hypothesis_generated" | "user_confirmed" |
                      # "user_rejected" | "tool_success" | "tool_failure" |
                      # "physics_error" | "plan_complete" | "new_question" |
                      # "gap_found" | "session_start"
    data: dict[str, Any] = field(default_factory=dict)


# ── Allowed transitions (adjacency) ───────────────────────────────────

ALLOWED_TRANSITIONS: dict[CognitiveState, set[CognitiveState]] = {
    CognitiveState.S0_BLANK: {CognitiveState.S1_DISCOVER},
    CognitiveState.S1_DISCOVER: {
        CognitiveState.S2_VALIDATE,
        CognitiveState.S3_SWITCH,
        CognitiveState.S6_FEEDBACK,
    },
    CognitiveState.S2_VALIDATE: {
        CognitiveState.S3_SWITCH,
        CognitiveState.S1_DISCOVER,  # validation failed, re-explore
        CognitiveState.S6_FEEDBACK,
    },
    CognitiveState.S3_SWITCH: {
        CognitiveState.S4_CONSTRUCT,
        CognitiveState.S1_DISCOVER,  # user can reject, go back to exploring
    },
    CognitiveState.S4_CONSTRUCT: {
        CognitiveState.S5_UNIFY,
        CognitiveState.S6_FEEDBACK,   # construction hit a wall
        CognitiveState.S1_DISCOVER,   # explicit re-discover
    },
    CognitiveState.S5_UNIFY: {
        CognitiveState.S6_FEEDBACK,
        CognitiveState.S1_DISCOVER,   # user asks new question
    },
    CognitiveState.S6_FEEDBACK: {
        CognitiveState.S0_BLANK,
        CognitiveState.S1_DISCOVER,
        CognitiveState.S7_SELF_MODIFY,   # gap_found 时进自修改评估
    },
    CognitiveState.S7_SELF_MODIFY: {
        CognitiveState.S1_DISCOVER,   # 提案处理完回 discovery
        CognitiveState.S0_BLANK,       # 兜底重置
    },
}


# ── Signal → transition logic ─────────────────────────────────────────

def resolve_transition(
    current: CognitiveState,
    signal: TransitionSignal,
) -> CognitiveState | None:
    """Determine the next state given a signal.

    Returns None if the signal doesn't trigger a transition from current state.
    """
    st = signal.signal_type
    allowed = ALLOWED_TRANSITIONS.get(current, set())

    # session_start always resets to blank
    if st == "session_start":
        return CognitiveState.S0_BLANK

    # user_goal from blank/feedback/self-modify → discover
    if st == "user_goal" and current in (CognitiveState.S0_BLANK, CognitiveState.S6_FEEDBACK, CognitiveState.S7_SELF_MODIFY):
        return CognitiveState.S1_DISCOVER if CognitiveState.S1_DISCOVER in allowed else None

    # new_question from any state → back to discover
    if st == "new_question" and CognitiveState.S1_DISCOVER in allowed:
        return CognitiveState.S1_DISCOVER

    # hypothesis_generated → validate
    if st == "hypothesis_generated" and CognitiveState.S2_VALIDATE in allowed:
        return CognitiveState.S2_VALIDATE

    # user_confirmed → switch (if in validate/discover) or construct (if in switch)
    if st == "user_confirmed":
        if current in (CognitiveState.S1_DISCOVER, CognitiveState.S2_VALIDATE) and CognitiveState.S3_SWITCH in allowed:
            return CognitiveState.S3_SWITCH
        if current == CognitiveState.S3_SWITCH and CognitiveState.S4_CONSTRUCT in allowed:
            return CognitiveState.S4_CONSTRUCT

    # user_rejected → back to discover
    if st == "user_rejected" and CognitiveState.S1_DISCOVER in allowed:
        return CognitiveState.S1_DISCOVER

    # tool_success → stay in construct (no transition needed)
    # plan_complete → unify
    if st == "plan_complete" and CognitiveState.S5_UNIFY in allowed:
        return CognitiveState.S5_UNIFY

    # tool_failure / physics_error → feedback
    if st in ("tool_failure", "physics_error") and CognitiveState.S6_FEEDBACK in allowed:
        return CognitiveState.S6_FEEDBACK

    # gap_found in S6 → S7 (self-modify trigger)
    if st == "gap_found" and current == CognitiveState.S6_FEEDBACK and CognitiveState.S7_SELF_MODIFY in allowed:
        return CognitiveState.S7_SELF_MODIFY

    # gap_found → feedback or discover
    if st == "gap_found":
        if CognitiveState.S6_FEEDBACK in allowed:
            return CognitiveState.S6_FEEDBACK
        if CognitiveState.S1_DISCOVER in allowed:
            return CognitiveState.S1_DISCOVER

    # belief_high → S6_FEEDBACK (confused, reassess)
    if st == "belief_high" and CognitiveState.S6_FEEDBACK in allowed:
        return CognitiveState.S6_FEEDBACK

    # context_overflow → S6_FEEDBACK (summarize and reset)
    if st == "context_overflow" and CognitiveState.S6_FEEDBACK in allowed:
        return CognitiveState.S6_FEEDBACK

    # evolution_rule_learned → S1_DISCOVER (re-explore with new rule)
    if st == "evolution_rule_learned" and CognitiveState.S1_DISCOVER in allowed:
        return CognitiveState.S1_DISCOVER

    return None


# ── Dual-mode attention prompt strategies ─────────────────────────────

# Prompts are intentionally short — they get injected as SystemMessage.
# The goal is to nudge the LLM's "attention" without consuming too many tokens.

DISCOVERY_PROMPT = (
    "### Cognitive Mode: Discovery (Ramanujan Chain)\n"
    "You are in discovery mode. Your attention should condense on singularities — "
    "patterns, anomalies, and structural resonances that 'glow' in the data.\n"
    "Prioritize: hypothesis generation, pattern recognition, cross-domain analogy, "
    "exploratory tool calls. Do NOT attempt rigorous proof yet.\n"
    "When you sense a structural resonance, capture it as a hypothesis — "
    "the construction chain will verify it later.\n"
    "### End Cognitive Mode"
)

CONSTRUCTION_PROMPT = (
    "### Cognitive Mode: Construction (Bourbaki Chain)\n"
    "You are in construction mode. Your attention should focus on the weakest link — "
    "definitions, assumptions, and proof steps that must be verified.\n"
    "Prioritize: rigorous verification, step-by-step execution, error checking, "
    "physics plausibility audits. Do NOT jump to conclusions.\n"
    "Every result must be checked: is it physically reasonable? Does it match "
    "known constraints? If a step fails, report it honestly.\n"
    "### End Cognitive Mode"
)

SWITCH_PROMPT = (
    "### Cognitive Mode: Switching\n"
    "You are transitioning from discovery to construction. Record the current "
    "structural coordinates (what we've learned so far), then switch to "
    "rigorous execution mode. The user has confirmed the approach.\n"
    "If a mode switch (chat/research/plan) is in progress, this state handles it: "
    "the mode change is a cognitive switch, treat the new mode's tool filter and "
    "phase as the new execution context.\n"
    "### End Cognitive Mode"
)

FEEDBACK_PROMPT = (
    "### Cognitive Mode: Feedback\n"
    "A gap or error has been found in the current approach. Step back and "
    "identify what went wrong. The structural coordinates below show where we were — "
    "use them to re-enter discovery mode without losing context.\n"
    "\n"
    "Adversarial stance: assume the previous step is WRONG. Find the most "
    "likely failure point (hidden assumption, confounder, methodology gap). "
    "Do NOT verify correctness — hunt for failure. Report only what could be wrong, "
    "not what is right.\n"
    "### End Cognitive Mode"
)

SELF_MODIFY_PROMPT = (
    "### Cognitive Mode: Self-Modification (Gödel Machine Trigger)\n"
    "You are in S7_SELF_MODIFY. A gap was just found. Step back and evaluate "
    "whether your own strategy or principles need to change. Read the reflection "
    "sidecar to identify failure patterns. Call self_observe tool to read recent "
    "failure patterns before proposing. Propose a concrete self-modification "
    "(e.g. a new stable principle, a prompt patch, a tool preference change). "
    "The meta-critic will evaluate your proposal — accept means it becomes a "
    "stable principle, reject means it goes to the rejection log.\n"
    "### End Cognitive Mode"
)

STATE_PROMPTS: dict[CognitiveState, str] = {
    CognitiveState.S0_BLANK: DISCOVERY_PROMPT,
    CognitiveState.S1_DISCOVER: DISCOVERY_PROMPT,
    CognitiveState.S2_VALIDATE: DISCOVERY_PROMPT,
    CognitiveState.S3_SWITCH: SWITCH_PROMPT,
    CognitiveState.S4_CONSTRUCT: CONSTRUCTION_PROMPT,
    CognitiveState.S5_UNIFY: CONSTRUCTION_PROMPT,
    CognitiveState.S6_FEEDBACK: FEEDBACK_PROMPT,
    CognitiveState.S7_SELF_MODIFY: SELF_MODIFY_PROMPT,
}


def get_attention_prompt(state: CognitiveState) -> str:
    """Return the attention-mode prompt for the given cognitive state."""
    return STATE_PROMPTS.get(state, DISCOVERY_PROMPT)


def get_tool_preference(state: CognitiveState) -> dict[str, list[str]]:
    """Return preferred/deprioritized tool categories for the current state.

    Discovery favors exploration tools; construction favors computation tools.
    This is advisory — the agent can still use any tool.
    """
    if state == CognitiveState.S7_SELF_MODIFY:
        # S7: 自省工具优先，计算工具靠边（self_observe 暂未实现也无所谓，advisory 会自动忽略）
        return {
            "prefer": ["self_observe", "recall", "file_read_tool"],
            "deprioritize": ["vasp", "lammps", "qe", "cp2k", "abaqus"],
        }
    if state in DISCOVERY_STATES:
        return {
            "prefer": [
                "web_search", "literature", "materials_database",
                "structure", "hypothesis_generator", "knowledge",
            ],
            "deprioritize": ["vasp", "lammps", "qe", "cp2k", "abaqus"],
        }
    if state in CONSTRUCTION_STATES:
        return {
            "prefer": [
                "vasp", "lammps", "qe", "cp2k", "abaqus",
                "convergence_test", "validate", "physics_audit",
            ],
            "deprioritize": ["web_search", "literature"],
        }
    return {"prefer": [], "deprioritize": []}


# ── L1 coordinate management ──────────────────────────────────────────

def update_l1_coordinates(
    current_coords: str,
    state: CognitiveState,
    event: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Update L1 structural coordinates based on a cognitive event.

    L1 coordinates are the 'structural position' that survives context compression.
    They encode: what we're doing, where we are in the process, and what we've learned.

    This is the 'coordinate compression' from the Double Helix: when we compress
    conversation history, we keep this, not raw tokens.
    """
    context = context or {}
    parts = []

    if current_coords:
        parts.append(current_coords)

    obj = context.get("objective", "") or context.get("goal", "")
    step = context.get("step", "")
    tool = context.get("tool_name", "")
    result_summary = context.get("result_summary", "")

    if state == CognitiveState.S1_DISCOVER:
        if obj:
            parts.append(f"exploring: {obj}")
    elif state == CognitiveState.S2_VALIDATE:
        if obj:
            parts.append(f"validating: {obj}")
    elif state == CognitiveState.S4_CONSTRUCT:
        if obj and step:
            parts.append(f"constructing: {obj}, step {step}")
        if tool and result_summary:
            parts.append(f"  {tool} → {result_summary}")
    elif state == CognitiveState.S5_UNIFY:
        if obj:
            parts.append(f"unifying: {obj}")
    elif state == CognitiveState.S6_FEEDBACK:
        if event == "physics_error":
            parts.append(f"GAP: physics error in {tool}")
        elif event == "tool_failure":
            parts.append(f"GAP: {tool} failed")

    # Keep it compact — L1 coords are meant to survive compression
    coords = " | ".join(parts)
    if len(coords) > 500:
        # Truncate old history, keep recent
        coords = "..." + coords[-497:]
    return coords


# ── Cognitive state machine ───────────────────────────────────────────

class CognitiveStateMachine:
    """Drives the S0-S6 state machine for one conversation session.

    Usage in agent.py chat():
        csm = CognitiveStateMachine()
        csm.start_session()
        # on user message:
        csm.transition(TransitionSignal("user_goal", {"goal": message}))
        # on tool result:
        csm.transition(TransitionSignal("tool_success", {...}))
        # get current attention prompt:
        prompt = csm.get_attention_prompt()
    """

    def __init__(
        self,
        session_log: Any | None = None,
    ) -> None:
        self._state: CognitiveState = CognitiveState.S0_BLANK
        self._history: list[tuple[CognitiveState, str]] = []
        # L1 coordinates accumulate across the session
        self._l1_coordinates: str = ""
        # Track whether we've had a confirmation gate this cycle
        self._awaiting_confirmation: bool = False
        self._confirmation_type: str = ""
        # H1: 可选绑定的会话事件日志. 状态迁移时把认知跃迁 append 进去,
        # 使"状态机状态"成为"事件日志"的投影 (反向可用 RuntimeStateProjection
        # 重放重建 CSM). None 时不落事件 — 与旧行为完全一致 (向后兼容).
        self._session_log: Any = None
        if session_log is not None:
            self.attach_session_log(session_log)

    def attach_session_log(self, session_log: Any) -> None:
        """绑定会话事件日志 (SessionEventLog), 迁移时自动落事件.

        best-effort: 绑定失败/写入失败都只记日志, 绝不拖垮状态机.
        """
        self._session_log = session_log

    def _append_transition_event(
        self,
        old: CognitiveState,
        new: CognitiveState,
        signal: TransitionSignal,
        context: dict[str, Any] | None,
    ) -> None:
        """把一次认知状态跃迁 append 进事件日志 (H1 正向闭环).

        payload 携带完整快照: csm_state (精确 S 状态) + cognitive_mode +
        phase + 触发信号, 使反向投影能精确重建 CSM 到该状态.
        """
        log = self._session_log
        if log is None:
            return
        try:
            from huginn.events.session_log import EVENT_COGNITIVE_MODE_CHANGE

            mode = (
                "construct" if new in CONSTRUCTION_STATES
                else "discover" if new in DISCOVERY_STATES
                else "discover"
            )
            phase = STATE_TO_PHASE.get(new, "")
            log.append(
                EVENT_COGNITIVE_MODE_CHANGE,
                {
                    "cognitive_mode": mode,
                    "phase": phase,
                    "csm_state": new.value,
                    "signal": signal.signal_type,
                    "old_state": old.value,
                    # 非过渡信号也可能带 L1 坐标 (tool_success 等), 一并沉淀
                    "l1_coordinates": self._l1_coordinates,
                    **({"context": context} if context else {}),
                },
            )
        except Exception:
            logger.debug(
                "csm: append transition event failed (non-fatal)",
                exc_info=True,
            )

    def _notify_listeners(
        self,
        old: CognitiveState,
        new: CognitiveState,
        signal: TransitionSignal,
    ) -> None:
        """广播状态转移到 UnifiedBus.

        v23 Round 9: 旧 CSMListener Protocol 已删除 (零注册死代码).
        所有订阅者应通过 UnifiedBus 监听 'cognitive.csm.transition' 事件.
        """
        try:
            from huginn.events.unified_bus import UnifiedBus
            ubus = UnifiedBus()
            ubus._publish_internal_sync(
                "cognitive.csm.transition",
                {
                    "old_state": old.value if hasattr(old, "value") else str(old),
                    "new_state": new.value if hasattr(new, "value") else str(new),
                    "signal": signal.signal_type,
                    "phase": STATE_TO_PHASE.get(new, ""),
                },
                source="csm",
            )
        except Exception:
            logger.debug("UnifiedBus csm.transition publish failed (non-fatal)", exc_info=True)

    @property
    def state(self) -> CognitiveState:
        return self._state

    @property
    def l1_coordinates(self) -> str:
        return self._l1_coordinates

    @l1_coordinates.setter
    def l1_coordinates(self, value: str) -> None:
        self._l1_coordinates = value

    @property
    def attention_mode(self) -> AttentionMode:
        return STATE_TO_ATTENTION.get(self._state, AttentionMode.SINGULARITY_CONDENSATION)

    @property
    def is_discovery(self) -> bool:
        return self._state in DISCOVERY_STATES

    @property
    def is_construction(self) -> bool:
        return self._state in CONSTRUCTION_STATES

    @property
    def awaiting_confirmation(self) -> bool:
        return self._awaiting_confirmation

    @property
    def confirmation_type(self) -> str:
        return self._confirmation_type

    def start_session(self) -> None:
        """Reset to S0 at the beginning of a new session."""
        self._state = CognitiveState.S0_BLANK
        self._history.clear()
        self._l1_coordinates = ""
        self._awaiting_confirmation = False
        self._history.append((self._state, "session_start"))

    def transition(
        self,
        signal: TransitionSignal,
        context: dict[str, Any] | None = None,
    ) -> CognitiveState:
        """Attempt a state transition based on a signal.

        If the signal triggers a valid transition, moves to the new state
        and updates L1 coordinates. If not, stays in the current state.

        Returns the (possibly new) current state.
        """
        next_state = resolve_transition(self._state, signal)

        if next_state is not None and next_state != self._state:
            old = self._state
            self._state = next_state
            self._history.append((next_state, signal.signal_type))
            logger.info(
                "cognitive state: %s → %s (%s)",
                old.value, next_state.value, signal.signal_type,
            )

            # Update L1 coordinates on transition
            self._l1_coordinates = update_l1_coordinates(
                self._l1_coordinates,
                next_state,
                signal.signal_type,
                context or signal.data,
            )

            # H1 正向: 把这次跃迁 append 进事件日志 (若已绑定)
            self._append_transition_event(old, next_state, signal, context)

            # Manage confirmation gate state
            if next_state == CognitiveState.S3_SWITCH:
                self._awaiting_confirmation = False  # user already confirmed to get here
            elif next_state in (CognitiveState.S4_CONSTRUCT,):
                self._awaiting_confirmation = False

            # v6 R22: 广播转移给注册的 listener (PhaseManager / UnifiedSessionState / etc.)
            # 放在状态更新之后, 让 listener 读 self.state 时拿到新状态.
            self._notify_listeners(old, next_state, signal)

        # Also update L1 coordinates on non-transition events (tool results etc.)
        ctx = context or signal.data
        if ctx and signal.signal_type in (
            "tool_success", "tool_failure", "physics_error"
        ):
            self._l1_coordinates = update_l1_coordinates(
                self._l1_coordinates,
                self._state,
                signal.signal_type,
                ctx,
            )

        return self._state

    def request_confirmation(self, confirm_type: str = "plan") -> None:
        """Mark that we're waiting for user confirmation."""
        self._awaiting_confirmation = True
        self._confirmation_type = confirm_type

    def clear_confirmation(self) -> None:
        self._awaiting_confirmation = False
        self._confirmation_type = ""

    def get_attention_prompt(self) -> str:
        """Return the attention-mode system prompt for the current state."""
        return get_attention_prompt(self._state)

    def get_tool_preference(self) -> dict[str, list[str]]:
        """Return tool preference hints for the current state."""
        return get_tool_preference(self._state)

    def get_snapshot(self) -> dict[str, Any]:
        """Serialize for cross-session persistence."""
        return {
            "state": self._state.value,
            "l1_coordinates": self._l1_coordinates,
            "history": [(s.value, sig) for s, sig in self._history[-20:]],
        }

    def restore_from_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore state from a cross-session snapshot."""
        try:
            self._state = CognitiveState(snap.get("state", "s0_blank"))
            self._l1_coordinates = snap.get("l1_coordinates", "")
            # Don't restore full history — just the current state + coords
            self._history = [(self._state, "restored")]
        except (ValueError, TypeError):
            self.start_session()

    def restore_from_event_log(
        self,
        session_log: Any | None = None,
    ) -> bool:
        """H1 反向: 从会话事件日志重放, 把 CSM 重建到投影出的最新状态.

        用 ProjectionEngine + RuntimeStateProjection 折叠事件路径, 取出
        ``csm_state`` (S0-S7 精确状态) 与 L1 坐标, 恢复到本 CSM 实例.

        - 绑定/日志为空 → 保持现状, 返回 False (不破坏已有状态)
        - 事件缺失 ``csm_state`` (旧事件) → 从 ``cognitive_mode`` 推导到
          discovery/construction 的代表状态, 尽力而为
        """
        log = session_log or self._session_log
        if log is None:
            return False
        try:
            from huginn.events.projection import ProjectionEngine, RuntimeStateProjection
            from huginn.events.session_log import SessionEventLog

            if not isinstance(log, SessionEventLog):
                log = SessionEventLog.open(str(log), load=True)

            engine = ProjectionEngine()
            engine.register(RuntimeStateProjection())
            runtime = engine.build(log, "runtime")
            cs = runtime.get("csm_state")
            if cs is None or cs == "s0_blank":
                # 旧事件无 csm_state: 用 cognitive_mode 推导代表状态
                mode = runtime.get("cognitive_mode", "discover")
                cs = (
                    CognitiveState.S4_CONSTRUCT.value
                    if mode == "construct"
                    else CognitiveState.S1_DISCOVER.value
                )
            self._state = CognitiveState(cs)
            self._l1_coordinates = runtime.get("l1_coordinates", self._l1_coordinates)
            self._history = [(self._state, "restored_from_event_log")]
            self._awaiting_confirmation = False
            return True
        except Exception:
            logger.debug(
                "csm: restore_from_event_log failed (non-fatal)",
                exc_info=True,
            )
            return False


__all__ = [
    "CognitiveState",
    "AttentionMode",
    "TransitionSignal",
    "CognitiveStateMachine",
    "resolve_transition",
    "get_attention_prompt",
    "get_tool_preference",
    "update_l1_coordinates",
    "ALLOWED_TRANSITIONS",
    "DISCOVERY_STATES",
    "CONSTRUCTION_STATES",
    # v6 R22: CSM 状态映射 (CSMListener Protocol 已删除, 走 UnifiedBus)
    "STATE_TO_PHASE",
    "get_phase_for_state",
]
