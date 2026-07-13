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

    def _append_reflection_sidecar(self, tool_result: dict, reflection: Any) -> None:
        """把反思结论写 sidecar 文件, 主上下文只引用结论不引用推理过程.

        避免自我污染: LLM 是 P(next|history), 反思文本进 history 会抬高
        P(next_支持_反思结论), 即便结论错. sidecar 让反思留痕但不污染主上下文.
        ponytail: JSONL 追加写, 不做索引. 升级: 按反思类型分文件 + grep 索引.
        """
        import json
        from datetime import datetime
        from pathlib import Path
        session_id = self._session_state.session_id or "default"
        sidecar_dir = Path.home() / ".huginn" / "reflections"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / f"{session_id}.jsonl"
        entry = {
            "ts": datetime.now().isoformat(),
            "tool_name": tool_result.get("tool_name", "unknown"),
            "tool_succeeded": reflection.tool_succeeded,
            "has_physics_errors": reflection.has_physics_errors,
            "has_physics_warnings": reflection.has_physics_warnings,
            "message": reflection.message,
            "should_switch_mode": reflection.should_switch_mode,
            "suggested_mode": reflection.suggested_mode,
        }
        with sidecar_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 首次写时暴露路径, 让用户/audit 知道 sidecar 在哪 (之前 write-only 无读者).
        # ponytail: 启动期一次 logger.info. 升级: /diagnostics 路由暴露 reader API.
        if not getattr(self, "_sidecar_path_announced", False):
            logger.info("reflection sidecar: %s", sidecar_path)
            self._sidecar_path_announced = True

    def load_reflection_sidecar(
        self, session_id: str | None = None, last_n: int = 50
    ) -> list[dict]:
        """读取 sidecar JSONL, 返回最近 last_n 条反思结论.

        给 audit / diagnostics / 用户查询用. 默认读当前 session, 最近 50 条.
        ponytail: 一次性 read + slice. 升级: 流式 read + 按 tool_name 过滤.
        """
        import json
        from pathlib import Path
        sid = session_id or self._session_state.session_id or "default"
        sidecar_path = Path.home() / ".huginn" / "reflections" / f"{sid}.jsonl"
        if not sidecar_path.exists():
            return []
        lines = sidecar_path.read_text(encoding="utf-8").strip().split("\n")
        out: list[dict] = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out[-last_n:] if last_n > 0 else out

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
                    # mode 切换 flag: S3_SWITCH / S6_FEEDBACK 时标记需要 compaction.
                    # streaming.py 下轮开头检查 flag, 触发 summarize_compact_messages.
                    # ponytail: flag 模式避开 async/sync 边界. 升级: CSM 直接 emit 事件.
                    from huginn.cognitive_engine import CognitiveState
                    if self._csm._state in (CognitiveState.S3_SWITCH, CognitiveState.S6_FEEDBACK):
                        self._needs_compaction = True
            except Exception:
                logger.debug("CSM transition failed", exc_info=True)

            # 反思 sidecar: 把反思结论写文件, 主上下文只引用结论不引用推理过程.
            # 避免自我污染: 反思文本进入 history 会抬高 P(next_支持_反思结论).
            # ponytail: 只在有实质内容时写. 升级: 按反思类型分文件.
            if reflection.message or reflection.has_physics_errors:
                try:
                    self._append_reflection_sidecar(tr, reflection)
                except Exception:
                    logger.debug("reflection sidecar write failed", exc_info=True)

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

            # 反思驱动的 mode 切换: should_switch_mode + suggested_mode 时切到目标 mode.
            # 之前算出该切但不切 → agent 永远停在当前 mode, 自适应闭环断裂.
            # ponytail: 1 行调用 set_mode, ValueError 静默 (suggested_mode 非法时降级).
            if reflection.should_switch_mode and reflection.suggested_mode:
                try:
                    self.set_mode(reflection.suggested_mode)
                    logger.info(
                        "reflection switched mode -> %s (tool=%s)",
                        reflection.suggested_mode,
                        tr.get("tool_name", "unknown"),
                    )
                except (ValueError, AttributeError):
                    logger.debug(
                        "set_mode(%s) failed — invalid suggested_mode",
                        reflection.suggested_mode,
                        exc_info=True,
                    )

        self._session_state.clear_turn_results()
