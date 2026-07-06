"""UnifiedSessionState — single coherent state object for one conversation session.

Replaces the fragmented SessionContext (which only had messages + tool_calls).
This object travels through every turn of the chat() loop, carrying:
  - persona: who the agent is right now
  - plan: what the user and agent agreed to do
  - cognitive_mode: which "double helix" chain we're on (discover vs construct)
  - l1_coordinates: structural coordinates that survive context compression

Design principle: human-in-the-loop. The user drives the loop — the agent
proposes, the user confirms, the agent executes, the agent reflects, the user
decides next step. This is NOT full auto-pilot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveMode(str, Enum):
    """Which chain of the double helix we're on.

    DISCOVER: Ramanujan chain — scan structure manifold, generate hypotheses,
    sense patterns. Attention mode: singularity condensation (what's glowing?).

    CONSTRUCT: Bourbaki chain — build axiomatic system, prove theorems,
    verify results. Attention mode: axiom focus (what's weakest?).

    The switch between modes happens at human confirmation gates — the user
    decides when to move from exploration to execution.
    """

    DISCOVER = "discover"
    CONSTRUCT = "construct"


class SessionPhase(str, Enum):
    """Where we are in the loop engineering cycle.

    EXPLORE: user just arrived, we're figuring out what they want
    PLAN: we've understood the goal, creating a structured plan
    EXECUTE: plan is confirmed, we're running tools
    REFLECT: a tool just finished, we're checking if results make sense
    REPORT: sharing results with user, waiting for next direction
    """

    EXPLORE = "explore"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    REPORT = "report"


@dataclass
class UnifiedSessionState:
    """The single state object that travels through the entire chat loop.

    Created at session start, updated every turn, saved at session end.
    Every component (persona, memory, plan, context_builder) reads from this.
    """

    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # ── Persona ────────────────────────────────────────────────────
    persona_name: str = "default"
    persona_system_prompt: str = ""

    # ── Plan ───────────────────────────────────────────────────────
    active_plan_id: str | None = None
    active_plan_objective: str = ""
    active_plan_step_index: int = 0  # which step we're on

    # ── Double Helix cognitive state ───────────────────────────────
    cognitive_mode: CognitiveMode = CognitiveMode.DISCOVER
    phase: SessionPhase = SessionPhase.EXPLORE
    # L1 coordinates — structural position that survives context compression.
    # When we compress conversation history, we keep this, not raw tokens.
    # Format: free text, e.g. "GaN band structure calculation, step 2/3 (SCF), converged"
    l1_coordinates: str = ""

    # ── Cross-session continuity ───────────────────────────────────
    last_session_summary: str = ""
    last_session_id: str = ""

    # ── Working state (transient, not persisted) ───────────────────
    pending_confirmation: dict[str, Any] | None = None
    # When not None, we're waiting for user to confirm something.
    # Keys: type ("new_plan"|"plan_update"|"high_cost_tool"|"mode_switch"),
    #       message (what we're asking), data (plan dict, tool name, etc.)

    # Cognitive prompt from the state machine — context_builder reads this
    # to inject the dual-mode attention prompt into the LLM messages.
    _cognitive_prompt: str = ""

    tool_results_this_turn: list[dict[str, Any]] = field(default_factory=list)
    # Results from the current turn's tool calls, for reflection.

    # ── Metadata ───────────────────────────────────────────────────
    turns_count: int = 0
    user_goals_history: list[str] = field(default_factory=list)
    # Every time the user states a new goal, we append it here.
    # This helps the agent understand the user's evolving intent.

    def advance_phase(self, new_phase: SessionPhase) -> None:
        """Move to a new phase, logging the transition."""
        old = self.phase
        self.phase = new_phase
        logger.debug("session %s: %s → %s", self.session_id[:8], old.value, new_phase.value)

    def switch_cognitive_mode(self, mode: CognitiveMode, reason: str = "") -> None:
        """Switch between discovery and construction chains.

        Per the double helix switching protocol:
        - Record current L1 coordinates
        - Clear path details (working memory is transient anyway)
        - Enter new mode
        """
        old = self.cognitive_mode
        self.cognitive_mode = mode
        logger.info(
            "cognitive mode switch: %s → %s (%s)",
            old.value, mode.value, reason or "no reason given",
        )

    def set_plan(self, plan_id: str, objective: str, n_steps: int = 0) -> None:
        """Record the active plan."""
        self.active_plan_id = plan_id
        self.active_plan_objective = objective
        self.active_plan_step_index = 0
        # When a plan is set, we're in construction mode
        self.switch_cognitive_mode(CognitiveMode.CONSTRUCT, "plan confirmed")
        self.advance_phase(SessionPhase.EXECUTE)

    def clear_plan(self) -> None:
        """Clear the active plan (completed or abandoned)."""
        self.active_plan_id = None
        self.active_plan_objective = ""
        self.active_plan_step_index = 0
        # Back to discovery mode
        self.switch_cognitive_mode(CognitiveMode.DISCOVER, "plan cleared")

    def advance_step(self) -> None:
        """Move to the next step in the plan."""
        self.active_plan_step_index += 1
        # Update L1 coordinates to reflect new position
        self.l1_coordinates = (
            f"{self.active_plan_objective}, "
            f"step {self.active_plan_step_index + 1}"
        )

    def request_confirmation(
        self,
        confirm_type: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Set a pending confirmation request. The chat loop will yield this
        to the user and wait for their response."""
        self.pending_confirmation = {
            "type": confirm_type,
            "message": message,
            "data": data or {},
        }

    def clear_confirmation(self) -> None:
        """Clear a pending confirmation after user responds."""
        self.pending_confirmation = None

    def add_tool_result(self, result: dict[str, Any]) -> None:
        """Record a tool result for the reflection phase."""
        self.tool_results_this_turn.append(result)

    def clear_turn_results(self) -> None:
        """Clear tool results after reflection is done."""
        self.tool_results_this_turn.clear()

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize the persistent parts for cross-session storage.
        Working state (pending_confirmation, tool_results) is NOT included.
        """
        return {
            "session_id": self.session_id,
            "persona_name": self.persona_name,
            "cognitive_mode": self.cognitive_mode.value,
            "phase": self.phase.value,
            "l1_coordinates": self.l1_coordinates,
            "active_plan_id": self.active_plan_id,
            "active_plan_objective": self.active_plan_objective,
            "active_plan_step_index": self.active_plan_step_index,
            "turns_count": self.turns_count,
            "user_goals_history": self.user_goals_history[-10:],  # last 10 goals
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "UnifiedSessionState":
        """Restore from a snapshot (for cross-session continuity)."""
        state = cls()
        state.session_id = snap.get("session_id", "")
        state.persona_name = snap.get("persona_name", "default")
        state.cognitive_mode = CognitiveMode(
            snap.get("cognitive_mode", "discover")
        )
        state.phase = SessionPhase(snap.get("phase", "explore"))
        state.l1_coordinates = snap.get("l1_coordinates", "")
        state.active_plan_id = snap.get("active_plan_id")
        state.active_plan_objective = snap.get("active_plan_objective", "")
        state.active_plan_step_index = snap.get("active_plan_step_index", 0)
        state.turns_count = snap.get("turns_count", 0)
        state.user_goals_history = snap.get("user_goals_history", [])
        return state
