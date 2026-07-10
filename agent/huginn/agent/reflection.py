"""Evolution engine, summarizer factory, and post-turn reflection."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Prompt for the conversation summarizer — preserves research context.
_SUMMARY_SYSTEM_PROMPT = (
    "You are a research conversation summarizer. Condense the following "
    "conversation excerpt into a concise summary that preserves:\n"
    "1. Key decisions and their rationale\n"
    "2. Important numerical results (energies, parameters, convergence criteria)\n"
    "3. Failed approaches and why they failed\n"
    "4. Pending tasks and next steps\n"
    "5. Any file paths, structure IDs, or job IDs referenced\n"
    "Be terse — use bullet points. Do not include greetings or filler."
)


class ReflectionMixin:
    """Evolution engine lifecycle, summarizer creation, and reflection."""

    def _get_evolution_engine(self):
        """Lazy-init and reuse a single EvolutionEngine.

        The engine wraps an ExecutionLogger that persists to disk, so the
        first call loads any existing history.
        """
        if self._evolution_engine is None:
            from huginn.evolution.engine import EvolutionEngine
            from huginn.evolution.logger import ExecutionLogger

            self._evolution_engine = EvolutionEngine(logger=ExecutionLogger())
        return self._evolution_engine

    def _make_summarizer(self):
        """Create an async callable for conversation summarization.

        Prefers the model router's cheap/summarize model to avoid burning
        expensive main-model tokens on compaction.
        """
        model = None
        if self.model_router is not None:
            try:
                model = self.model_router.select("summarize", prefer_cheap=True)
            except Exception:
                logger.warning("model_router.select failed for summarize model", exc_info=True)
        if model is None:
            model = self.model
        if model is None:
            return None

        async def _summarize(transcript: str):
            from langchain_core.messages import HumanMessage, SystemMessage

            from huginn.llm_retry import FallbackTriggeredError, call_with_fallback

            messages = [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript),
            ]
            if hasattr(model, "ainvoke"):

                async def _call():
                    return await model.ainvoke(messages)

                try:
                    return await with_retry(_call, source="summarize")
                except FallbackTriggeredError:
                    logger.warning("summarize: primary model overloaded, trying fallback")

                    async def _fallback_call(prompt, mdl):
                        if hasattr(mdl, "ainvoke"):
                            return await mdl.ainvoke(messages)
                        return mdl.invoke(messages)

                    return await call_with_fallback(
                        prompt=transcript,
                        primary_model=getattr(model, "model", ""),
                        llm_call_fn=_fallback_call,
                    )
            return model.invoke(messages)

        return _summarize

    def _sync_plan_from_store(self) -> None:
        """Sync an executing plan from PlanStore to session_state.

        Best-effort: if the autoloop engine created a plan during a tool
        call, pick it up here so the cognitive state machine and context
        builder know we're in execution mode.
        """
        if self._session_state.active_plan_id:
            return
        try:
            from huginn.autoloop.plan_store import PlanStore
            ps = PlanStore()
            executing = ps.list_plans(status="executing")
            if executing:
                p = executing[-1]
                self._session_state.set_plan(p.id, p.objective)
                from huginn.cognitive_engine import CognitiveState
                self._csm._state = CognitiveState.S4_CONSTRUCT
                logger.info("synced plan from PlanStore: %s", p.id)
        except Exception:
            logger.debug("PlanStore sync failed", exc_info=True)

    def _run_post_turn_reflection(self) -> None:
        """Run the rules-based reflector on this turn's tool results.

        Pure rules, no LLM, sub-millisecond.  Drives evolution triggers,
        CSM transitions, plan progress, and confirmation requests.
        Failures here never break the turn.
        """
        if not self._session_state.tool_results_this_turn:
            return

        for tr in self._session_state.tool_results_this_turn:
            try:
                reflection = self._reflector.reflect(
                    tool_name=tr.get("tool_name", "unknown"),
                    tool_result=tr,
                    session_state=self._session_state,
                )
            except Exception:
                logger.debug("reflection failed", exc_info=True)
                continue

            # Trigger evolution on failure / success signals.
            if reflection.should_evolve:
                try:
                    ev_engine = self._get_evolution_engine()
                    _content = tr.get("content", "")
                    ev_engine.logger.log_tool_call(
                        session_id=self._session_state.session_id
                        or "default",
                        tool_name=tr.get("tool_name", ""),
                        tool_input={},
                        result=_content
                        if reflection.tool_succeeded
                        else None,
                        error=None
                        if reflection.tool_succeeded
                        else str(_content),
                    )
                    if reflection.evolve_signal == "failure":
                        ev_engine.evolve_from_failures()
                    elif reflection.evolve_signal == "success":
                        ev_engine.evolve_from_successes()
                except Exception:
                    logger.debug("evolution trigger failed", exc_info=True)

            # Drive the cognitive state machine with the reflection result.
            try:
                sig_type = reflection.to_transition_signal()
                if sig_type:
                    from huginn.cognitive_engine import TransitionSignal as TS
                    self._csm.transition(TS(sig_type, {
                        "tool_name": tr.get("tool_name", ""),
                        "objective": self._session_state.active_plan_objective,
                        "step": str(self._session_state.active_plan_step_index + 1),
                        "result_summary": str(tr.get("content", ""))[:100],
                    }))
                    self._session_state.l1_coordinates = self._csm.l1_coordinates
                    self._session_state._cognitive_prompt = self._csm.get_attention_prompt()
            except Exception:
                logger.debug("CSM transition failed", exc_info=True)

            # Persist plan progress when a step is judged done.
            if (
                reflection.plan_step_completed
                and self._session_state.active_plan_id
            ):
                self._session_state.advance_step()
                try:
                    self.memory.store_plan_progress(
                        plan_id=self._session_state.active_plan_id,
                        objective=self._session_state.active_plan_objective,
                        step_index=self._session_state.active_plan_step_index,
                        status="in_progress",
                        l1_coordinates=self._session_state.l1_coordinates,
                    )
                except Exception:
                    logger.debug("plan progress store failed", exc_info=True)

            # If reflection says we need user input, set pending confirmation.
            if reflection.needs_user_input:
                self._session_state.request_confirmation(
                    reflection.confirm_type or "continue",
                    f"Tool '{tr.get('tool_name', 'unknown')}' reported issues. "
                    f"Continue or adjust approach?"
                )
                self._csm.request_confirmation(reflection.confirm_type or "continue")

        self._session_state.clear_turn_results()
