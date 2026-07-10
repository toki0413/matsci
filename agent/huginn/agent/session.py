"""Session state properties and cross-session continuity."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SessionMixin:
    """Unified session state accessors and session-continuity loading."""

    @property
    def session_state(self) -> Any:
        """Expose the unified session state for external access (e.g. routes)."""
        return self._session_state

    @property
    def cognitive_state(self) -> str:
        """Current S0-S6 cognitive state (read-only)."""
        return self._csm.state.value if hasattr(self, "_csm") else "s0_blank"

    @property
    def l1_coordinates(self) -> str:
        """Current L1 structural coordinates (read-only)."""
        return self._csm.l1_coordinates if hasattr(self, "_csm") else ""

    def _build_compact_summary(self) -> str:
        """Build the existing_summary for the summarizer, prepending L1 coords.

        When the conversation gets compressed, the L1 structural coordinates
        must survive — they're the agent's 'position on the structure manifold'.
        """
        base = self._conversation_summary or ""
        l1 = self._csm.l1_coordinates if hasattr(self, "_csm") else ""
        if l1:
            return f"[Structural Position: {l1}]\n{base}"
        return base

    def _init_session_continuity(self) -> None:
        """Load previous session context at the start of a new session.

        Cross-session continuity: pick up the last summary and any
        in-flight plan so the agent remembers what was discussed last
        time.  Both lookups are best-effort.
        """
        self._csm.start_session()

        try:
            ctx = self.memory.load_last_session_context()
            if ctx and ctx.get("summary"):
                self._session_state.last_session_summary = ctx["summary"]
                self._session_state.last_session_id = ctx.get("session_id", "")
                l1 = ctx.get("l1_coordinates", "")
                if l1:
                    self._csm.l1_coordinates = l1
                    self._session_state.l1_coordinates = l1
                logger.info(
                    "loaded last session context: %s...",
                    ctx["summary"][:100],
                )
        except Exception:
            logger.debug("session continuity load failed", exc_info=True)

        try:
            plan = self.memory.load_active_plan()
            if plan and plan.get("status") not in ("completed", "abandoned"):
                plan_id = plan.get("plan_id", "")
                objective = plan.get("objective", "")
                step_index = plan.get("step_index", 0)
                if plan_id and objective:
                    self._session_state.set_plan(plan_id, objective)
                    self._session_state.active_plan_step_index = step_index
                    from huginn.cognitive_engine import CognitiveState
                    self._csm._state = CognitiveState.S4_CONSTRUCT
                    self._csm.l1_coordinates = plan.get("l1_coordinates", "")
                logger.info(
                    "resumed active plan: %s (step %d)",
                    objective[:80],
                    step_index,
                )
        except Exception:
            logger.debug("active plan load failed", exc_info=True)
