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

        # 结构化 snapshot 恢复: 上面只恢复 summary / plan, 这里补 _mode / _csm / _phase /
        # turns_count. 之前 session resume 这几个字段全丢, agent 默默回 chat mode + S0 + OPEN.
        # ponytail: 只读最新一条, 不覆盖已恢复的 plan. 升级: 按 session_id 精确读 + 版本化.
        try:
            snap = self.memory.load_session_snapshot()
            if isinstance(snap, dict):
                mode = snap.get("mode")
                if mode in ("chat", "research", "plan"):
                    # 不覆盖 plan 恢复触发的 S4_CONSTRUCT, 只在 csm 还是 S0 时恢复
                    self._mode = mode
                csm_state = snap.get("csm_state")
                if isinstance(csm_state, str) and csm_state:
                    from huginn.cognitive_engine import CognitiveState
                    try:
                        target = CognitiveState(csm_state)
                        # plan 恢复已经设了 S4_CONSTRUCT, 别覆盖; 否则恢复 csm
                        if self._csm._state == CognitiveState.S0_BLANK:
                            self._csm._state = target
                    except ValueError:
                        pass
                l1 = snap.get("l1_coordinates", "")
                if l1 and not self._csm.l1_coordinates:
                    self._csm.l1_coordinates = l1
                phase = snap.get("research_phase")
                if phase:
                    try:
                        from huginn.phases import ResearchPhase
                        # plan 恢复路径没动 _phase_manager, 这里恢复
                        self._phase_manager.reset(ResearchPhase(phase))
                    except (ValueError, KeyError):
                        pass
                turns = snap.get("turns_count")
                if isinstance(turns, int) and turns > 0:
                    self._turn_count = turns
                logger.info(
                    "restored session snapshot: mode=%s csm=%s phase=%s turns=%d",
                    self._mode, csm_state, phase, self._turn_count,
                )
        except Exception:
            logger.debug("session snapshot restore failed", exc_info=True)

    def _save_session_snapshot(self) -> None:
        """保存当前 session 状态快照. reflection 末尾节流调用.

        之前 session resume 只恢复消息历史, _mode/_csm/_phase_manager/_session_state 丢.
        ponytail: 走 memory.save_session_snapshot + JSON. 升级: 增量 diff + 独立 store.
        """
        try:
            from huginn.cognitive_engine import CognitiveState
            csm_state = ""
            if hasattr(self, "_csm") and hasattr(self._csm, "_state"):
                csm_state = self._csm._state.value if isinstance(
                    self._csm._state, CognitiveState
                ) else str(self._csm._state)
            snap = {
                "session_id": self._session_state.session_id,
                "mode": self._mode,
                "turns_count": self._turn_count,
                "csm_state": csm_state,
                "l1_coordinates": self._csm.l1_coordinates if hasattr(self, "_csm") else "",
                "research_phase": (
                    self._phase_manager.phase.value
                    if hasattr(self, "_phase_manager") else ""
                ),
                "session_state": self._session_state.to_snapshot(),
            }
            self.memory.save_session_snapshot(snap)
        except Exception:
            logger.debug("save_session_snapshot failed", exc_info=True)
