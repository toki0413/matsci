"""Evolution engine, summarizer factory, and post-turn reflection."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# v4 G17 续: 信号统一走 SignalHub, import 失败不阻断 v3 逻辑
try:
    from huginn.metacog.signal_hub import SignalHub
except ImportError:
    SignalHub = None  # type: ignore[assignment]

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

        ponytail: HUGINN_COMPACT_KIND 分流 local/remote/remote_v2 三条路径.
        升级路径是注册 PRE_COMPACT 钩子动态改 kind, 现在用环境变量静态配置.
        """
        import os

        kind = os.environ.get("HUGINN_COMPACT_KIND", "remote").lower()
        model = None
        if kind == "local":
            # 本地模型 (ollama/vllm), 隐私优先
            local = getattr(self, "_find_local_model", lambda: None)()
            if local is not None:
                model = local
            else:
                logger.warning("HUGINN_COMPACT_KIND=local but no local model found, fallback to remote")
        elif kind == "remote_v2":
            # 升级版远程 summarizer (例如更大的 summarization 模型)
            if self.model_router is not None:
                try:
                    model = self.model_router.select("summarize_v2", prefer_cheap=True)
                except Exception:
                    logger.warning("summarize_v2 unavailable, fallback to remote", exc_info=True)
        if model is None and self.model_router is not None:
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

        # ponytail: RCB/benchmark 场景 skip CSM transition — 无人工 subprocess,
        # CSM attention prompt 是 noise 还触发不必要 compaction. 升级: mode-aware.
        if os.environ.get("HUGINN_SKIP_CSM", "").lower() in ("1", "true", "yes"):
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
                    _new_rules = []
                    if reflection.evolve_signal == "failure":
                        _new_rules = ev_engine.evolve_from_failures()
                    elif reflection.evolve_signal == "success":
                        _new_rules = ev_engine.evolve_from_successes()
                    # G7: evolution 学新规则后让 CSM 重新探索 (新规则可能改路径)
                    if _new_rules:
                        try:
                            from huginn.cognitive_engine import TransitionSignal as _TS
                            self._csm.transition(_TS("evolution_rule_learned", {
                                "count": len(_new_rules),
                                "signal": reflection.evolve_signal,
                            }))
                        except Exception:
                            logger.debug("evolution CSM signal failed", exc_info=True)
                except Exception:
                    logger.debug("evolution trigger failed", exc_info=True)

            # Drive the cognitive state machine with the reflection result.
            try:
                sig_type = reflection.to_transition_signal()
                if sig_type:
                    from huginn.cognitive_engine import TransitionSignal as TS
                    new_state = self._csm.transition(TS(sig_type, {
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
                        # RCB 子集模式: CSM transition 走 (含 S7), 但不触发 compaction (Task 18)
                        self._needs_compaction = not os.environ.get("HUGINN_RCB_CSM_SUBSET")

                    # 哥德尔机闭环: S6 + 实质 gap → S7_SELF_MODIFY (打通 L232 flag 断层)
                    if (new_state == CognitiveState.S6_FEEDBACK
                            and self._has_substantive_gap(reflection)):
                        new_state = self._csm.transition(
                            TS("gap_found", {"gap": getattr(reflection, "message", "")})
                        )
                    # S7 状态: 调 meta critique 评估 proposal, accept→stable_principle / reject→rejection log
                    # S7 是 meta 状态, RCB 场景也不触发 compaction (与 S3/S6 不同)
                    if new_state == CognitiveState.S7_SELF_MODIFY:
                        self._needs_compaction = False
                        try:
                            from huginn.utils.async_bridge import run_async
                            run_async(self._handle_s7_self_modify(reflection, self._csm))
                        except Exception:
                            logger.warning("S7 self-modify handler failed", exc_info=True)
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

        # 新增: 信号驱动 CSM (G5 events / G6 belief_entropy). G7 在 evolution 调用点触发.
        # ponytail: 补充现有 reflection 规则信号, 不替代. try/except 静默, 不破坏 turn.
        try:
            self._check_event_signals(self._csm)
        except Exception:
            logger.debug("event signal check failed", exc_info=True)
        try:
            self._check_belief_entropy_signal(self._csm)
        except Exception:
            logger.debug("belief entropy signal check failed", exc_info=True)

        # 节流保存 session snapshot: 每 3 turn 一次, 让下次会话能恢复 _mode/_csm/_phase.
        # 之前 session resume 只恢复消息历史, mode/csm/phase 全丢. 放这里集中, 不污染 mutation 点.
        # ponytail: 节流 + try/except 静默. 升级: dirty flag + 增量 diff.
        if getattr(self, "_turn_count", 0) % 3 == 0:
            try:
                self._save_session_snapshot()
            except Exception:
                logger.debug("session snapshot save failed", exc_info=True)

        self._session_state.clear_turn_results()

    # ── S7 self-modification helpers ────────────────────────────────────
    # 哥德尔机闭环: 把 reflection gap → proposal, 调 meta critique 评估,
    # accept 写 stable_principles, reject 写 directive_rejections.jsonl.

    def _has_substantive_gap(self, reflection_result: Any) -> bool:
        """判断 reflection 是否有实质 gap (够触发 S7_SELF_MODIFY).

        ponytail: 复用 _extract_proposal_from_gap, 简单 bool 化. 升级: LLM 判 severity.
        """
        return bool(self._extract_proposal_from_gap(reflection_result))

    def _extract_proposal_from_gap(self, reflection_result: Any) -> str | None:
        """从 reflection 结果提取自修改提案.

        ponytail: 直接拼 message/failure_mode, ceiling 是 LLM 生成结构化 proposal
                  (含 motivation / patch / expected_utility_delta); 升级走专门 generator.
        """
        # ReflectionResult 没有 gap/failure_modes 字段, 用 message + 失败标志当 gap.
        msg = getattr(reflection_result, "message", "") or ""
        if getattr(reflection_result, "has_physics_errors", False):
            return f"address physics error: {msg}"
        if not getattr(reflection_result, "tool_succeeded", True):
            return f"avoid tool failure: {msg}"
        if getattr(reflection_result, "has_physics_warnings", False):
            return f"address physics warning: {msg}"
        return None

    def _load_recent_rejections(self, limit: int = 10) -> list[str]:
        """读最近 N 条 rejection 记录的 proposal 字段, 用于 meta critique 早期查重.

        ponytail: 全文 read + slice. 升级: 流式 tail + 索引 by similarity.
        """
        import json
        from pathlib import Path
        path = Path(".huginn/directive_rejections.jsonl")
        if not path.exists():
            return []
        proposals: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                proposals.append(rec.get("proposal", ""))
            except json.JSONDecodeError:
                continue
        return proposals[-limit:]

    def _write_rejection(self, proposal: str, reason: str) -> None:
        """追加一条 rejection 记录到 .huginn/directive_rejections.jsonl."""
        import json
        import time
        from pathlib import Path
        path = Path(".huginn/directive_rejections.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp": time.time(), "proposal": proposal, "reason": reason}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _get_system_prompt_summary(self) -> str:
        """返回当前 system_prompt 的摘要, 给 meta critique 当上下文.

        ponytail: 简单截断前 500 字符. ceiling 是 LLM 压缩成结构化摘要
                  (persona / tools / constraints 分段); 升级走 summarizer.
        """
        sp = getattr(self, "system_prompt", "") or ""
        return sp[:500]

    async def _handle_s7_self_modify(self, reflection_result: Any, csm: Any) -> None:
        """S7 状态: 把 gap 总结成 proposal, 调 meta critique 评估,
        accept→store_stable_principle; reject→directive_rejections.jsonl;
        处理完回 S1_DISCOVER.

        ponytail: gap→proposal 是字段拼接, ceiling 是 LLM 生成正式 proposal;
                  meta critique 失败默认 reject (保守, 不污染 stable_principles).
        """
        from huginn.cli.rcb_runner import adversarial_critique
        from huginn.memory import store_stable_principle

        # 1. 从 reflection gap 提取 proposal
        proposal = self._extract_proposal_from_gap(reflection_result)
        if not proposal:
            # 没 proposal 直接回 S1
            from huginn.cognitive_engine import TransitionSignal
            csm.transition(TransitionSignal("user_goal", {"goal": "continue"}))
            return

        # 2-3. 读 recent rejections + system_prompt 摘要
        recent_rejections = self._load_recent_rejections(limit=10)
        sys_prompt_summary = self._get_system_prompt_summary()

        # 4. 调 meta critique. 之前传 model=None 导致 client=None,
        # await client.ainvoke() 必报 AttributeError, S7 退化成全 reject 死循环.
        # 修: 传 self.model (agent 自己的 LLM), 失败仍走 except 默认 reject.
        try:
            verdict = await adversarial_critique(
                model=getattr(self, "model", None),
                mode="meta",
                proposal=proposal,
                system_prompt_summary=sys_prompt_summary,
                recent_rejections=recent_rejections,
            )
        except Exception as e:
            logger.warning("S7 meta critique invocation failed: %s, treating as reject", e)
            verdict = {
                "verdict": "reject",
                "reason": f"meta critique error: {e}",
                "expected_utility_delta": 0.0,
            }

        # 5. 处理 verdict: accept→stable_principle, reject→rejection log
        if verdict.get("verdict") == "accept":
            try:
                store_stable_principle(proposal, source="S7_self_modify")
                logger.info("S7 accepted proposal: %s", proposal[:80])
            except Exception:
                logger.warning("store_stable_principle failed", exc_info=True)
        else:
            try:
                self._write_rejection(proposal, verdict.get("reason", "unknown"))
                logger.info(
                    "S7 rejected proposal: %s reason: %s",
                    proposal[:80],
                    verdict.get("reason"),
                )
            except Exception:
                logger.warning("write_rejection failed", exc_info=True)

        # 6. 回 S1_DISCOVER (无论 accept/reject, S7 处理完都回 discovery)
        from huginn.cognitive_engine import TransitionSignal
        csm.transition(TransitionSignal("user_goal", {"goal": "continue after S7"}))


    def _check_event_signals(self, csm) -> None:
        """从 EventBus 拉关键事件转 CSM TransitionSignal (G5).

        context.overflow / compact.start / tool.error 连发都映射到
        context_overflow 信号, CSM resolve_transition 决定是否真切换.
        v4 G17 续: 信号构造优先走 SignalHub, 失败 fallback 到 v3 直构.
        """
        # ponytail: recent_events 轮询, 不订阅. last_ts 去重避免重复处理.
        # 升级: subscribe + 异步队列实时驱动; 缺点是跨 async 边界事件可能丢.
        try:
            from huginn.events import (
                EventBus, CONTEXT_OVERFLOW, COMPACT_START, TOOL_ERROR,
            )
        except ImportError:
            return
        try:
            bus = EventBus.shared()
            last_ts = getattr(self, "_last_event_check_ts", 0.0)
            overflow_events = [
                e for et in (CONTEXT_OVERFLOW, COMPACT_START)
                for e in bus.recent_events(n=5, event_type=et)
                if e.timestamp > last_ts
            ]
            tool_errors = [
                e for e in bus.recent_events(n=20, event_type=TOOL_ERROR)
                if e.timestamp > last_ts
            ]
            if overflow_events or len(tool_errors) >= 3:
                payload = {
                    "overflow_events": len(overflow_events),
                    "tool_error_count": len(tool_errors),
                }
                # 优先走 Hub: overflow 用 event_overflow, tool 连发用 event_tool_burst
                sig = None
                if SignalHub is not None:
                    source = "event_overflow" if overflow_events else "event_tool_burst"
                    sig = SignalHub.shared().route(source, payload)
                # ponytail: 保留 v3 逻辑骨架作 fallback, 只改信号构造走 Hub;
                # 升级路径是 SignalHub 直接调 csm.transition, 删 fallback.
                if sig is None:
                    from huginn.cognitive_engine import TransitionSignal as TS
                    sig = TS("context_overflow", payload)
                csm.transition(sig)
                all_ts = [e.timestamp for e in overflow_events] + \
                         [e.timestamp for e in tool_errors]
                if all_ts:
                    self._last_event_check_ts = max(all_ts)
        except Exception:
            logger.debug("event signal check failed", exc_info=True)

    def _check_belief_entropy_signal(self, csm) -> None:
        """高 belief_entropy 触发 CSM 重新评估 (G6).

        h_belief > 0.7 说明压缩丢信息太多, agent 迷糊了, 让 CSM 进 feedback.
        v4 G17 续: 信号构造优先走 SignalHub, 失败 fallback 到 v3 直构.
        v7 G59: 同步更新认知热机 T_hot (idea 池熵代理).
        """
        # ponytail: 阈值 0.7 硬编码, 跟 BeliefEntropyConfig.threshold_high 默认对齐.
        # 升级: 读 be.config.threshold_high, 别硬编码. 单次检查不看趋势.
        try:
            from huginn.utils.belief_entropy import get_belief_entropy
        except ImportError:
            return
        try:
            be = get_belief_entropy()
            history = be.get_history()
            if not history:
                return
            h_belief = history[-1]

            # v7 G59: 更新认知热机 T_hot (belief_entropy → idea 池熵)
            try:
                from huginn.metacog.cognitive_heat_engine import get_heat_engine
                get_heat_engine().update_T_hot(float(h_belief))
            except Exception:
                logger.debug("heat_engine.update_T_hot failed (non-fatal)", exc_info=True)

            if h_belief > 0.7:
                payload = {"h_belief": h_belief}
                # 优先走 Hub
                sig = None
                if SignalHub is not None:
                    sig = SignalHub.shared().route("belief_high", payload)
                # ponytail: 保留 v3 逻辑骨架作 fallback, 只改信号构造走 Hub;
                # 升级路径是 SignalHub 直接调 csm.transition, 删 fallback.
                if sig is None:
                    from huginn.cognitive_engine import TransitionSignal as TS
                    sig = TS("belief_high", payload)
                csm.transition(sig)
        except Exception:
            logger.debug("belief_entropy signal check failed", exc_info=True)
