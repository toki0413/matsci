"""EngineControlMixin — AutoloopEngine 的循环控制 + checkpoint 方法族.

从 engine.py 拆出 (P3 slim-down 续). 包含:
- 循环控制: _check_gate / _check_budget / _drain_side_questions / stop
- checkpoint: _wait_if_checkpoint_pending / _publish_checkpoint_event
- 状态持久化: _maybe_save_engine_state / _track_llm_usage / _distill_meta_trace
- 澄清交互: _get_clarification_manager / _maybe_clarify / _get_plan_store / _get_refine_model
- 事件总线: _get_event_bus / _dispatch_stage_event
- 偏差日志: _log_deviation

通过 self 访问 engine 状态. 方法体原样搬迁, 不改逻辑.

设计原则 (ponytail):
- 对 engine.py 模块级符号用方法内 lazy import, 避免 circular
- Mixin 不持有自己的状态, 全部走 self
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

# 控制阶段方法引用的 engine.py 模块级 import (均为叶子模块, 无 circular 风险)
from huginn.api.event import EventType, WorkflowStageEvent
from huginn.autoloop.budget import BudgetExhausted
from huginn.autoloop.phase_gate import PhaseGate, get_shared_phase_gate_state
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


class EngineControlMixin:
    """循环控制 + checkpoint 方法族. 通过 self 访问 engine 状态."""

    def _maybe_save_engine_state(
        self, *, force: bool = False, reason: str = "",
    ) -> None:
        """P15: 周期 / forced 落盘 engine_state + hypothesis_graph.

        - flag off (HUGINN_USE_PERSISTENCE=0) 时完全 no-op, 不碰磁盘.
        - force=True: 立刻 save (pivot / refute 等关键事件触发).
        - force=False: 周期 save, iteration % save_every == 0 才真写.
        - run_id 缺失 (run_cognitive 没启动) 时 no-op, 避免误写.

        ponytail: 失败只 log warning, 不抛 — save 失败不该阻塞主循环.
        ceiling: 单进程串行 save, 不加锁; 并发 run_id 隔离由 run_id 命名空间保证.
        """
        try:
            from huginn.runtime.engine_state import (
                save_engine_state,
                save_every_steps,
                use_persistence,
            )
            if not use_persistence():
                return
            run_id = getattr(self, "_run_id", None)
            if not run_id:
                return
            if not force:
                every = save_every_steps()
                if every <= 0:
                    return
                if self._iteration % every != 0:
                    return
            save_engine_state(self, run_id, self.workspace)
            if reason:
                logger.debug(
                    "engine_state saved (reason=%s, iter=%d, run_id=%s)",
                    reason, self._iteration, run_id,
                )
        except Exception:
            logger.warning(
                "_maybe_save_engine_state failed (non-fatal)", exc_info=True,
            )
        # Step 8: alignment_dataset 跟 engine_state 同节奏落盘
        self._save_alignment_dataset()


    async def _track_llm_usage(self, usage_meta) -> None:
        """P2-7: 从 langchain usage_metadata 累加 token/cost 到 _token_budget.

        usage_meta 是 langchain BaseMessage.usage_metadata (dict | None),
        含 input_tokens / output_tokens / total_tokens. 不同 provider 填充程度不一,
        缺字段按 0 处理. 超硬上限时先 force save engine_state (可 resume) 再抛.
        ponytail: cost 是粗估 ($3/M input + $15/M output, Claude-ish 混合),
        只作硬刹车 guardrail, 不作账单. 解析失败不阻塞主循环.
        """
        if not usage_meta:
            return
        try:
            input_tokens = int(usage_meta.get("input_tokens", 0) or 0)
            output_tokens = int(usage_meta.get("output_tokens", 0) or 0)
            total = input_tokens + output_tokens
            cost = (
                input_tokens / 1_000_000.0 * 3.0
                + output_tokens / 1_000_000.0 * 15.0
            )
            self._token_budget.update(total, cost)
            await self._maybe_run_budget_approval()
        except BudgetExhausted:
            # 超硬上限: 先保存进度 (可 resume), 再抛 — agent loop 优雅停止而非继续烧钱.
            self._maybe_save_engine_state(force=True, reason="budget_exhausted")
            raise
        except Exception:
            logger.debug("token budget tracking failed (non-fatal)", exc_info=True)

    async def _maybe_run_budget_approval(self) -> None:
        """预算软限制审批: off 不触发(auto 有限续/gui Inbox 人工批). abort -> 保存状态并优雅停.

        每次 LLM 调用后由 _track_llm_usage 触发。soft 不抛(不像硬刹车), 靠
        续投把 hard_limit 抬起来; 续满或用户拒绝则走 BudgetExhausted 停止。
        gui 模式经 Inbox 真挂起等用户 (复用 _await_human_decision_via_inbox),
        批准才续投, 拒绝则优雅停止 — 不静默降级为 auto。
        """
        budget = getattr(self, "_token_budget", None)
        if budget is None:
            return
        try:
            from huginn.budget_pause import get_approval_controller, get_approval_mode

            if get_approval_mode() != "gui":
                # auto / off 走同步裁决, 不与 Inbox 交互.
                ctl = get_approval_controller(budget)
                if not ctl.is_active():
                    return
                decision = ctl.on_tokens_used()
            else:
                # gui: 注入 Inbox 人工决定. 无 Inbox 可用时回退同步 callback 语义.
                human_decide = self._build_budget_human_decide()
                ctl = get_approval_controller(budget)
                decision = await ctl.on_tokens_used_async(human_decide)
            if decision == "renewed":
                logger.info(
                    "budget soft-limit renewed: tokens=%s/%s hard=%s",
                    budget.current_tokens, budget.soft_limit_tokens, budget.hard_limit_tokens,
                )
            elif decision == "abort":
                self._maybe_save_engine_state(force=True, reason="budget_renew_denied")
                raise BudgetExhausted("budget renewal denied by user/limit")
        except BudgetExhausted:
            raise
        except Exception:
            logger.debug("budget approval check failed (non-fatal)", exc_info=True)

    def _build_budget_human_decide(self):
        """构造预算续投的人工决定协程 -> 接 Inbox 真挂起.

        reason 经 _await_human_decision_via_inbox 生成带 quick-reply 的
        Inbox question, 用户在任意 surface 回答。回答含批准则续投, 否则中止。
        ponytail: 回调闭包直接抓 self, 由 engine 生命周期保证有效。
        """
        async def _human_decide(question: str, detail: str) -> bool:
            try:
                _step_id = int(getattr(self, "_iteration", 0) or 0)
                answer = await self._await_human_decision_via_inbox(
                    f"预算审批: {question}\n{detail}",
                    [
                        {"id": "approve", "label": "批准续投"},
                        {"id": "deny", "label": "停止 (保存并结束)"},
                    ],
                    _step_id,
                )
                if not answer:
                    # Inbox 不可用 / 超时 -> 默认不续投, 交硬刹车兜住.
                    return False
                low = str(answer).strip().lower()
                return "approve" in low or low.startswith("y") or "批准" in str(answer)
            except Exception:
                logger.debug("budget human decide failed (non-fatal)", exc_info=True)
                return False
        return _human_decide


    def _get_event_bus(self):
        """懒加载 EventBus. 没注册 handler 时返回 None, 调用方判空跳过."""
        if self._event_bus is not None:
            return self._event_bus
        try:
            from huginn.plugins.event_bus import EventBus

            self._event_bus = EventBus()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return None
        return self._event_bus


    async def _dispatch_stage_event(
        self,
        event_type: EventType,
        stage_name: str,
        duration_sec: float = 0.0,
        error: str | None = None,
    ) -> None:
        """向 EventBus 发一个 WorkflowStageEvent. 没总线或没 handler 时静默跳过."""
        bus = self._get_event_bus()
        if bus is None:
            return
        idx = (
            self._phase_order.index(stage_name) + 1
            if stage_name in self._phase_order
            else 0
        )
        event = WorkflowStageEvent(
            type=event_type,
            workflow_name="autoloop",
            stage_name=stage_name,
            stage_index=idx,
            duration_sec=duration_sec,
            error=error,
        )
        try:
            result = await bus.dispatch(event)
            if result.executed == 0 and result.failed == 0:
                logger.debug(
                    "stage event %s.%s had no handlers",
                    event_type.name,
                    stage_name,
                )
        except Exception:
            logger.warning(
                "error in _dispatch_stage_event: bus.dispatch failed", exc_info=True
            )


    def _check_gate(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> bool:
        """评估阶段转移门. 通过/已 override 返回 True; 阻断时把 feedback
        拼进 _speculator_hint (下轮 prompt 用) 并返回 False, caller 应
        continue 到下一轮迭代, 不推进到 to_phase.

        共享状态写一条记录进 history, 让 phase_tool 能查到最新门决策.
        """
        state = get_shared_phase_gate_state()
        # OAK 启发: trace_id 贯穿, 每个 gate 记录归属的 run
        tid = getattr(self, "_run_id", None) or ""
        parent_tid = getattr(self, "_parent_run_id", None)
        # override 优先: 已强制放行的转移直接记一条 approved, 不再评估
        if (from_phase, to_phase) in state.overrides:
            meta = state.override_meta.get((from_phase, to_phase), {})
            state.history.append(
                PhaseGate(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    status="approved",
                    required_evidence=self.phase_gate_hook.config.required_for(
                        from_phase, to_phase
                    ),
                    feedback="override 放行",
                    reviewer=meta.get("actor", "user"),
                    trace_id=tid,
                    parent_trace_id=parent_tid,
                )
            )
            state.pending_transition = (from_phase, to_phase)
            # override 同时清除 pending_human_review (用户已决策)
            if state.pending_human_review == (from_phase, to_phase):
                state.pending_human_review = None
            return True

        gate = self.phase_gate_hook.evaluate(from_phase, to_phase, evidence)
        gate.trace_id = tid
        gate.parent_trace_id = parent_tid
        state.history.append(gate)
        state.pending_transition = (from_phase, to_phase)

        if gate.is_blocked:
            fb = gate.feedback or (
                f"阶段转移 {from_phase}→{to_phase} 被阻断: 缺 {gate.missing_evidence}"
            )
            self._speculator_hint = (
                (self._speculator_hint + "\n" + fb).strip()
                if self._speculator_hint
                else fb
            )
            logger.info(
                "gate blocked %s→%s: missing %s",
                from_phase,
                to_phase,
                gate.missing_evidence,
            )
            return False

        # ── Human-in-the-loop checkpoint (LangGraph interrupt_before 模式) ──
        # 硬性证据已通过, 但用户配置了该转移需要人工审查. 设 pending_human_review
        # 并返回 False 让 engine 停在当前 phase. UI 层读到 phase_checkpoint 事件后
        # 展示 evidence 给用户, 用户通过 phase_tool override 或 submit_evidence + resume.
        if state.needs_human_checkpoint(from_phase, to_phase):
            if state.pending_human_review == (from_phase, to_phase):
                # 已经在等了, 不重复设 — 避免 dead loop
                return False
            state.pending_human_review = (from_phase, to_phase)
            logger.info(
                "human checkpoint pending %s→%s: awaiting user review",
                from_phase,
                to_phase,
            )
            # 记一条 pending 状态, phase_tool 查得到
            state.history.append(
                PhaseGate(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    status="pending",
                    required_evidence=self.phase_gate_hook.config.required_for(
                        from_phase, to_phase
                    ),
                    feedback=(
                        "⚠ 硬性检查点: 此转移不可超时自动放行, 必须人工确认. "
                        "请审查 evidence 后用 phase_tool override 显式放行, "
                        "或 submit_evidence 补充后 resume."
                        if state.is_hard_checkpoint(from_phase, to_phase)
                        else "等待人工 checkpoint 审查. 用 phase_tool override 放行, "
                        "或 submit_evidence 补充后 resume."
                    ),
                    reviewer="human_checkpoint",
                    trace_id=tid,
                    parent_trace_id=parent_tid,
                )
            )
            return False

        # 用户已审查完毕 (pending_human_review 被清除), 正常放行
        if state.pending_human_review == (from_phase, to_phase):
            state.pending_human_review = None
        return True


    async def _wait_if_checkpoint_pending(
        self, from_phase: str, to_phase: str, timeout: float = 600.0
    ) -> None:
        """等用户完成 checkpoint 审查.

        _check_gate 设了 pending_human_review 后, 调这个方法阻塞等.
        用户通过 phase_tool 加 override 后, 下一轮 _check_gate 走 override
        分支会清掉 pending 并放行. 所以这里同时盯 overrides 和 pending —
        任一变化就返回.

        timeout 到了还没决策, 强制清 pending 让下一轮放行, 避免无限阻塞.
        ponytail: 1s 轮询. 升级路径是 asyncio.Condition + phase_tool notify.
        """
        state = get_shared_phase_gate_state()
        key = (from_phase, to_phase)
        loop = asyncio.get_running_loop()
        is_hard = state.is_hard_checkpoint(from_phase, to_phase)
        # 硬门不设 deadline — 一直阻塞到用户显式 override, 不可超时偷偷放行
        deadline = loop.time() + timeout if not is_hard else float("inf")

        # 推送 checkpoint 等待事件到前端
        await self._publish_checkpoint_event(
            event_type="checkpoint_pending",
            from_phase=from_phase,
            to_phase=to_phase,
            is_hard=is_hard,
        )

        while state.pending_human_review == key and key not in state.overrides:
            if loop.time() > deadline:
                logger.warning(
                    "human checkpoint %s→%s timed out after %ss, force proceed",
                    from_phase,
                    to_phase,
                    timeout,
                )
                state.pending_human_review = None
                await self._publish_checkpoint_event(
                    event_type="checkpoint_timeout",
                    from_phase=from_phase,
                    to_phase=to_phase,
                    is_hard=is_hard,
                )
                return
            await asyncio.sleep(1.0)

        # checkpoint 已解决 (override 添加或 pending 被清除)
        await self._publish_checkpoint_event(
            event_type="checkpoint_resolved",
            from_phase=from_phase,
            to_phase=to_phase,
            is_hard=is_hard,
        )


    async def _publish_checkpoint_event(
        self,
        event_type: str,
        from_phase: str,
        to_phase: str,
        is_hard: bool = False,
    ) -> None:
        """推送 PhaseGate checkpoint 事件到 EventBus.

        之前传 dict 给 dispatch(event: Event) → AttributeError 被 except 吞掉,
        事件永远到不了 handler. 现在构造 base Event + data 载荷.
        """
        bus = self._get_event_bus()
        if bus is None:
            return
        try:
            from huginn.api.event import EventType, WorkflowStageEvent

            # 必须用 WorkflowStageEvent 子类: filter.on_workflow_stage_done 的
            # matcher (api/filter.py:251) 用 isinstance(event, WorkflowStageEvent)
            # 判定, 发基类 Event 即使 type 正确也会被 matcher 拒掉.
            ev = WorkflowStageEvent(
                type=EventType.ON_WORKFLOW_STAGE_DONE,
                plugin_name="phase_gate",
                workflow_name="phase_gate",
                stage_name=f"{from_phase}->{to_phase}",
                data={
                    "checkpoint_event": event_type,
                    "from_phase": from_phase,
                    "to_phase": to_phase,
                    "is_hard": is_hard,
                },
            )
            await bus.dispatch(ev)
        except Exception:
            logger.debug("checkpoint event publish failed", exc_info=True)


    def _check_budget(self, iteration: int, plan: dict[str, Any]) -> bool:
        """检查 plan 的 mode 是否在当前迭代预算允许范围内.

        通过返回 True (含 budget 未启用 / 已降级放行 / mode 允许三种情况).
        不通过时把"用哪个 mode 代替"的提示拼进 _speculator_hint, 下轮 prompt
        能看到, 返回 False 让 caller continue 到下一轮迭代.

        每个档位有 max_calls 次拒绝额度, 用尽后整条预算降级为放行, 避免
        LLM 反复提同样的 mode 把循环卡死.
        """
        if self._budget is None or self._budget_degraded:
            return True
        tier = self._budget.for_iteration(iteration)
        mode = plan.get("mode")
        if tier.allows(mode):
            # 这轮通过了就清掉该档位的拒绝计数, 下次重新数
            self._budget_rejects.pop(tier.label, None)
            return True

        rejects = self._budget_rejects.get(tier.label, 0) + 1
        self._budget_rejects[tier.label] = rejects
        if tier.max_calls is not None and rejects > tier.max_calls:
            # 拒绝额度用尽, 降级放行剩下的所有 mode, 不再卡
            self._budget_degraded = True
            logger.info(
                "budget degraded at iter %d: %s reject cap %s hit, allowing all modes",
                iteration,
                tier.label,
                tier.max_calls,
            )
            return True

        allowed = ", ".join(tier.allowed_modes) if tier.allowed_modes else "any"
        fb = (
            f"迭代 {iteration} 预算档位 {tier.label}: mode={mode} 不被允许, "
            f"可用: {allowed}. 请改用允许的 mode 重新规划."
        )
        self._speculator_hint = (
            (self._speculator_hint + "\n" + fb).strip() if self._speculator_hint else fb
        )
        logger.info(
            "budget rejected mode=%s at iter %d (tier %s, reject %d/%s)",
            mode,
            iteration,
            tier.label,
            rejects,
            tier.max_calls,
        )
        return False


    async def _drain_side_questions(self) -> int:
        """轮空时把 pending 侧边问题答掉. 返回答了几个.

        拿 shared SideChannel 的 pending 快照, 逐条调 model.ainvoke 出答案,
        再 channel.respond() 写回. 单条失败不阻塞其他条, 也不抛异常 ——
        侧边对话是次要任务, 不能影响主 loop.
        """
        if not self._side_channel_enabled:
            return 0
        from huginn.side_conversation import get_shared_side_channel

        channel = self._side_channel or get_shared_side_channel()
        pending = channel.drain()
        if not pending:
            return 0
        from langchain_core.messages import HumanMessage, SystemMessage

        # Side questions are low-priority — use a cheap model when available
        side_model = self.model
        router = getattr(self, "model_router", None) or getattr(
            getattr(self, "agent", None), "model_router", None
        )
        if router is not None:
            with contextlib.suppress(Exception):
                side_model = router.select("cheap", prefer_cheap=True) or self.model

        answered = 0
        for sq in pending:
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "You are answering a side question while the main "
                            "research loop is idle. Keep it concise and direct."
                        )
                    ),
                    HumanMessage(content=sq.question),
                ]
                response = await side_model.ainvoke(messages)
                answer = str(response.content).strip()
                if answer:
                    channel.respond(sq.id, answer)
                    answered += 1
                    logger.info("side answered %s: %s", sq.id, answer[:80])
            except Exception:
                # 单条失败不影响其他, 也不影响主 loop
                logger.warning("side failed to answer %s", sq.id, exc_info=True)
        return answered


    def _get_clarification_manager(self):
        """懒加载 ClarificationManager. 不可用时返回 None, 调用方判空跳过."""
        if self._clarification_mgr is not None:
            return self._clarification_mgr
        try:
            from huginn.interaction.clarification import get_clarification_manager

            self._clarification_mgr = get_clarification_manager()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return None
        return self._clarification_mgr


    def _get_plan_store(self):
        """懒加载 PlanStore. 不可用时返回 None, 调用方判空走老的纯 dict 路径."""
        if self._plan_store is not None:
            return self._plan_store
        try:
            from huginn.autoloop.plan_store import PlanStore

            self._plan_store = PlanStore()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return None
        return self._plan_store


    def _get_refine_model(self):
        """获取 refine 用的 LLM model, 优先用验证模型 (便宜档).

        没有就用 None, hypothesis_graph.refine_failed 会走 findings 模板拼接.
        """
        # ponytail: 直接用已有的 verification_model, 不需要额外的 llm_config 模块
        return getattr(self, "verification_model", None) or None


    async def _maybe_clarify(
        self,
        checkpoint: str,
        phase_result: Any,
        thread_id: str = "autoloop",
    ) -> str | None:
        """在关键决策点检查是否需要向用户提问.

        checkpoint 取值:
        - "plan": 计划生成后, 高成本 mode (workflow/DFT) 时确认
        - "validation_fail": 验证失败后, 连续 3+ 次时问方向

        返回用户回答的字符串, 或 None (无需提问 / manager 不可用 / 超时走默认).

        非阻塞设计: 没有 async event loop 时直接返回 None, 不强制阻塞.
        autoloop 在 async 上下文里跑, 所以正常路径能拿到回答.
        """
        mgr = self._get_clarification_manager()
        if mgr is None:
            return None

        # 构建上下文
        if checkpoint == "plan":
            plan = phase_result or {}
            mode = plan.get("mode", "")
            # 只对高成本 mode 提问 (workflow=DFT/MD, 通常是几小时)
            expensive_modes = ("workflow", "dft", "md", "vasp", "lammps")
            if mode.lower() not in expensive_modes:
                return None

            ctx = {
                "thread_id": thread_id,
                "question_type": "cost_confirm",
                "phase": "plan",
                "summary": f"mode={mode}, desc={plan.get('description', '')[:200]}",
                "tool": mode,
                "cost_estimate_hours": 1.0,  # workflow 类至少 1h
            }
        elif checkpoint == "validation_fail":
            if self._consecutive_failures < 3:
                return None
            ctx = {
                "thread_id": thread_id,
                "question_type": "validation_fail",
                "phase": "validate",
                "summary": str(phase_result)[:300],
                "consecutive_failures": self._consecutive_failures,
            }
        elif checkpoint == "plan_check_fail":
            # plan_check 连续失败 + 场景已知 -> 问用户方向.
            # 跟 validation_fail 同款: 不阻塞, 用户可以选 force_proceed.
            info = phase_result or {}
            consecutive = info.get("consecutive_fails", 0)
            if consecutive < 3 or info.get("scene") == "other":
                return None
            ctx = {
                "thread_id": thread_id,
                "question_type": "plan_check_fail",
                "phase": "plan",
                "summary": (
                    f"plan_check 连续 {consecutive} 次失败 "
                    f"(scene={info.get('scene', '?')}): "
                    f"{info.get('reason', '')[:200]}"
                ),
                "consecutive_fails": consecutive,
                "scene": info.get("scene", ""),
            }
        elif checkpoint == "hypothesize_align":
            # v11: FDE 对齐轮 — 首轮或 blind_spots/conflicts 时问用户假设方向.
            # 不阻塞, 60s timeout, 用户回答 append 到 _speculator_hint 自动流进 prompt.
            # ponytail: 复用现有 _maybe_clarify 管道, 不新建 ClarificationManager 子类.
            _info = phase_result if isinstance(phase_result, dict) else {}
            _is_first = self._iteration <= 1
            _has_signals = bool(
                _info.get("blind_spots") or _info.get("semantic_conflicts")
            )
            if not (_is_first or _has_signals):
                return None

            # 拼 recommended_directions: 优先用 cluster_by_dimension, 不足补 speculator
            _directions: list[str] = []
            try:
                _clusters = self.hypothesis_graph.cluster_by_dimension()
                for _dim, _nodes in list(_clusters.items())[:3]:
                    if _dim != "unknown" and _nodes:
                        _directions.append(f"{_dim}: {_nodes[0].statement[:80]}")
            except Exception:
                logger.debug("cluster directions skipped", exc_info=True)
            # 不足 3 个时补 speculator predictions (首轮自然走这条)
            while len(_directions) < 3:
                try:
                    _preds = getattr(self, "_speculator_predictions", []) or []
                    if _preds:
                        _directions.append(f"speculator: {str(_preds[len(_directions)])[:80]}")
                    else:
                        break
                except Exception:
                    logger.debug("best-effort op failed", exc_info=True)
                    break

            ctx = {
                "thread_id": thread_id,
                "question_type": "hypothesize_align",
                "phase": "hypothesize",
                "summary": (
                    f"目标: {self._objective or '?'}\n"
                    f"现场: {str(_info.get('summary', ''))[:200]}\n"
                    f"推荐方向:\n" + "\n".join(f"  - {d}" for d in _directions[:3])
                ),
                "recommended_directions": _directions[:3],
                "is_first_iteration": _is_first,
            }
        else:
            return None

        if not mgr.should_ask_contextual(ctx.get("question_type", ""), ctx):
            return None

        # 生成提问
        question, options, default = mgr.generate_question(ctx, model=None)

        try:
            answer = await mgr.ask(
                thread_id=thread_id,
                question=question,
                options=options,
                context=ctx.get("summary", ""),
                default_answer=default,
                timeout=60,  # 给用户足够时间回答
                metadata={
                    "question_type": ctx.get("question_type", ""),
                    "checkpoint": checkpoint,
                    "iteration": self._iteration,
                },
            )
            logger.info("clarify %s: %s", checkpoint, answer[:80])
            # v11: hypothesize_align 的回答 append 到 _speculator_hint,
            # 自动流进 _build_hypothesis_prompt 的 hint_block (零改动).
            if checkpoint == "hypothesize_align" and answer:
                self._speculator_hint += f"\n[FDE 对齐] 用户方向: {answer[:200]}\n"
            return answer
        except Exception:
            logger.warning("clarify %s failed", checkpoint, exc_info=True)
            return None


    def _distill_meta_trace(self, darwin_score: float, supported_ratio: float) -> None:
        """把本轮蒸馏成结构化科研要点, 追加到 .huginn/meta_trace.jsonl.

        Oxelra Meta-Trace 启示: 每步边界蒸馏 what attempted/found/evidence/
        limitation/artifact/next_hint. 不存 raw trace, 只存结构化要点.

        ponytail: 字段从 self.* 现有状态抽, 不调 LLM. ceiling 是 LLM 蒸馏.
                  文件路径跟 stable_principles 同目录 (.huginn/), 一行一个 JSON.
        """

        # 从 self.* 抽本轮关键信息 (都是上一轮 phase 写进去的)
        attempted = ""
        if getattr(self, "_last_hypothesis", None):
            attempted = str(self._last_hypothesis)[:300]

        found = ""
        limitations: list[str] = []
        if getattr(self, "_last_validation", None) and isinstance(self._last_validation, dict):
            v = self._last_validation
            found = str(v.get("result", ""))[:300]
            if v.get("errors"):
                limitations.append(str(v["errors"])[:200])

        evidence: list[str] = []
        try:
            for nd in self.hypothesis_graph.supported()[:3]:
                evidence.append(str(nd.statement)[:150])
        except Exception:
            logger.debug("supported evidence collect skipped", exc_info=True)

        artifacts: list[str] = []
        if getattr(self, "_last_execution_result", None) and isinstance(self._last_execution_result, dict):
            outs = self._last_execution_result.get("outputs") or self._last_execution_result.get("files")
            if isinstance(outs, list):
                artifacts = [str(f)[:150] for f in outs[:5]]
            elif isinstance(outs, str):
                artifacts = [outs[:150]]

        next_hint = (getattr(self, "_speculator_hint", "") or "")[-300:]

        entry = {
            "iteration": self._iteration,
            "ts": time.time(),
            "role": "autoloop",  # ponytail: 单 agent, role 固定; 升级多 agent 后填实际 role
            "attempted": attempted,
            "found": found,
            "evidence": evidence,
            "limitations": limitations,
            "artifacts": artifacts,
            "next_hint": next_hint,
            "darwin_score": round(darwin_score, 2),
            "supported_ratio": round(supported_ratio, 3),
        }

        # 写到 workspace 的 .huginn/meta_trace.jsonl (不存在就建)
        # ponytail: 不走 memory_manager, 直接写文件. 跟 directive_rejections 同模式.
        ws = getattr(self, "workspace_root", None) or Path.cwd()
        trace_path = Path(ws) / HUGINN_DIR_NAME / "meta_trace.jsonl"
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("meta_trace write failed (non-fatal)", exc_info=True)


    def stop(self) -> None:
        """Signal the loop to stop at the next safe point."""
        self._should_stop = True


    def _log_deviation(
        self,
        plan: dict[str, Any],
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """记录执行偏离计划的决策.

        借鉴 "Finding Your Unknowns" 的 implementation-notes.md 技术:
        agent 执行中发现需要换路径时, 记录 WHY — 不只记 WHAT (provenance 已做).

        触发条件:
        1. plan 有 expected_prediction 但 result 明显不符
        2. result 含 error/warning 字段
        3. _try_evolved_fix 被触发 (heuristic fix = 偏离了原 plan)

        存入 context['_deviation_log'] 供 _learn() 和 Feynman 使用.
        """
        deviations: list[dict[str, str]] = context.setdefault("_deviation_log", [])
        plan_mode = plan.get("mode", "unknown")
        plan_desc = plan.get("description", "")[:200]
        expected = plan.get("expected_prediction", "")

        # 检查 1: 有 error
        if isinstance(result, dict) and result.get("error"):
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "execution_error",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": f"Execution failed: {str(result['error'])[:200]}",
                    "expected": expected[:100] if expected else "(none)",
                }
            )

        # 检查 2: evolved fix 被使用
        if isinstance(context, dict) and context.get("_evolved_fix"):
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "heuristic_fix",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": "Applied evolved heuristic fix instead of following original plan",
                    "expected": expected[:100] if expected else "(none)",
                }
            )

        # 检查 3: result success=False
        if isinstance(result, dict) and result.get("success") is False:
            deviations.append(
                {
                    "iteration": str(self._iteration),
                    "type": "plan_mismatch",
                    "plan_mode": plan_mode,
                    "plan_desc": plan_desc,
                    "deviation": "Plan produced unsuccessful result, will need refinement",
                    "expected": expected[:100] if expected else "(none)",
                }
            )


