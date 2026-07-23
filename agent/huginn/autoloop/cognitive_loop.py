"""CognitiveLoop — rcb_runner 和 autoloop 的共享控制流内核.

核心问题: rcb_runner (6-step mini-loop) 和 autoloop (7-phase heavy loop) 是两条
独立代码线, 每次改 metacog 都要手动接两遍. Step 0 抽共享内核, 让优化改一处
影响两条路径.

设计原则 (ponytail):
- 不强行统一 6-step 和 7-phase — 它们任务粒度不同, 强行统一会引入抽象税
- 只统辖 4 个钩子: observe / decide / execute_action / reflect
- 两条路径各自实现钩子, CognitiveLoop 只负责编排
- 不绑定输出格式 — 通过 output_writer 接口注入 (RCBReportWriter / ProvenanceWriter)

向上兼容:
- rcb_runner: 6-step 逻辑作为 execute_action 的实现, observe/decide/reflect 退化为
  轻量钩子 (首轮 observe 读 INSTRUCTIONS.md, decide 固定 "execute", reflect 跑
  StepEvaluator + should_continue + detect_drift)
- autoloop: 7-phase 逻辑作为 execute_action 的实现, observe=perceive, decide=LLM
  选 next phase, reflect=metacog check

不做的事 (反模式警示):
- 不新建 7 个独立 phase 文件 — phase 方法保持原位
- 不做完整 state machine — decide() 输出字符串 action
- 不引入新依赖 — 用现有 LLM client
- 不一次全改 — 先抽骨架, 两条路径逐步迁移
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# 700 万步极限场景的 action_history 滑动窗口. 所有调用方只用尾部 (decider prompt
# 取 [-10:], cycle_detect O(n²) 在 1000 可接受, count/reversed/len 都 O(n)).
# ponytail: 环境变量覆盖, 极限模式可调大. 升级路径: deque + 专用 tail_n().
_MAX_ACTION_HIST = int(os.environ.get("HUGINN_ACTION_HIST_MAX", "1000"))


@dataclass
class LoopState:
    """CognitiveLoop 的可观测状态 — observe 填, decide 读, reflect 更新.

    不含 phase 内部状态 (那些留在 rcb_runner/autoloop 的 self 上).
    只含跨钩子共享的控制流状态.
    """
    iteration: int = 0
    max_iterations: int = 20
    should_stop: bool = False
    should_redirect: bool = False  # reflect 设 True, decide 看到 → 换 action
    last_action: str = ""  # 上次执行的 action (observe/decide/execute/learn/report/...)
    last_action_result: Any = None
    redirect_reason: str = ""  # reflect 写, decide 读
    # 死循环防护: action 历史 + 重复检测
    action_history: list[str] = field(default_factory=list)
    # 基模自主性: decide 返回的 rationale (供 reflect 评估)
    last_rationale: str = ""


@dataclass
class ActionDecision:
    """decide() 的返回 — LLM 或规则选的下一个 action."""
    action: str  # observe/hypothesize/plan/execute/validate/learn/pivot/skip/stop
    rationale: str = ""
    expected_outcome: str = ""
    force: bool = False  # True = 跳过 reflect 的 redirect 建议

# action 合法集合 — decide() 只能返回这些, 否则 reflect 标 redirect
# D3: report 不在 VALID_ACTIONS — _finalize_run 自动跑, LLM 选 report 等于
# 浪费一轮 (execute_fn 是 no-op). 升级路径: 如果要 LLM 主动触发 report,
# 改成 action="stop" + rationale="report ready".
VALID_ACTIONS = frozenset({
    "observe", "hypothesize", "plan", "execute", "validate",
    "learn", "pivot", "skip", "stop",
})


@dataclass
class ReflectionResult:
    """reflect() 的返回 — 评估上轮 action, 决定是否 redirect/stop."""
    should_continue: bool = True
    should_redirect: bool = False
    should_stop: bool = False
    redirect_reason: str = ""
    # 死循环检测: 连续重复 action 数
    repeated_action_count: int = 0
    # 评估意见 (供下轮 decide 看到)
    advice: str = ""


class CognitiveLoop:
    """共享控制流内核 — 编排 observe/decide/execute_action/reflect 四个钩子.

    使用方式:
        loop = CognitiveLoop(
            observe_fn=my_observe,
            decide_fn=my_decide,
            execute_fn=my_execute,
            reflect_fn=my_reflect,
            output_writer=my_writer,
            max_iterations=20,
        )
        result = await loop.run(initial_state)

    两条路径各自实现 4 个钩子, CognitiveLoop 不关心具体实现.
    """

    def __init__(
        self,
        observe_fn: Callable[[LoopState], Awaitable[dict[str, Any]]],
        decide_fn: Callable[[LoopState, dict[str, Any]], Awaitable[ActionDecision]],
        execute_fn: Callable[[LoopState, ActionDecision], Awaitable[Any]],
        reflect_fn: Callable[[LoopState, ActionDecision, Any], Awaitable[ReflectionResult]],
        output_writer: Any | None = None,
        max_iterations: int = 20,
        max_repeated_actions: int = 3,  # 死循环防护: 连续重复 action 超过此数 → stop
    ) -> None:
        self._observe = observe_fn
        self._decide = decide_fn
        self._execute = execute_fn
        self._reflect = reflect_fn
        self._output_writer = output_writer
        self._max_iterations = max_iterations
        self._max_repeated_actions = max_repeated_actions

    async def run(self, initial_state: LoopState | None = None) -> LoopState:
        """主循环: observe → decide → execute → reflect, 直到 should_stop 或 max_iter.

        不抛异常 — 钩子内部失败由钩子自己处理, CognitiveLoop 只负责编排.
        返回最终 LoopState, 调用方从 state.last_action_result 取最终产物.
        """
        state = initial_state or LoopState(max_iterations=self._max_iterations)
        state.max_iterations = self._max_iterations

        while state.iteration < state.max_iterations and not state.should_stop:
            state.iteration += 1
            logger.info("CognitiveLoop iter %d/%d", state.iteration, state.max_iterations)

            # 1. observe — 轻量环境扫描, 不强制 git diff
            try:
                observation = await self._observe(state)
            except Exception as e:
                logger.warning("observe failed: %s", e)
                observation = {}

            # 2. decide — 选下一个 action (LLM 自主或规则)
            try:
                decision = await self._decide(state, observation)
                if decision.action not in VALID_ACTIONS:
                    logger.warning(
                        "decide returned invalid action %r, defaulting to 'skip'",
                        decision.action,
                    )
                    decision.action = "skip"
            except Exception as e:
                logger.warning("decide failed: %s, defaulting to 'stop'", e)
                state.should_stop = True
                break

            # 死循环防护: 连续重复 action 超上限 → 强制 redirect, 超过 2 倍上限 → stop
            state.action_history.append(decision.action)
            # 700 万步极限场景防内存爆炸: 截断到滑动窗口. 所有调用方只用尾部
            # ([-10:], count, reversed, len), 截断前缀不影响语义. ponytail: 保留
            # list 不改 deque, 避免切片兼容性. 升级路径: deque + 专用 tail_n().
            if len(state.action_history) > _MAX_ACTION_HIST:
                del state.action_history[: -_MAX_ACTION_HIST]
            state.last_action = decision.action
            state.last_rationale = decision.rationale
            repeated = self._count_repeated_tail(state.action_history)
            if repeated >= self._max_repeated_actions and not decision.force:
                logger.warning(
                    "CognitiveLoop: %d consecutive '%s' actions, forcing redirect",
                    repeated, decision.action,
                )
                state.should_redirect = True
                state.redirect_reason = (
                    f"死循环防护: 连续 {repeated} 次 '{decision.action}', 换方向"
                )
                # 超过 2 倍上限仍未换方向 → 真死循环, 强制 stop
                if repeated >= self._max_repeated_actions * 2:
                    logger.error(
                        "CognitiveLoop: %d consecutive '%s' (2x limit), forcing stop",
                        repeated, decision.action,
                    )
                    state.should_stop = True
                    break

            # 3. execute_action — 跑 LLM 选的 action
            try:
                result = await self._execute(state, decision)
                state.last_action_result = result
            except Exception as e:
                logger.warning("execute_action '%s' failed: %s", decision.action, e)
                state.last_action_result = None
                # 失败不阻塞, 让 reflect 评估

            # 4. reflect — 评估上轮, 决定是否继续/换方向/停
            try:
                reflection = await self._reflect(state, decision, state.last_action_result)
                reflection.repeated_action_count = repeated
            except Exception as e:
                logger.warning("reflect failed: %s, defaulting to continue", e)
                reflection = ReflectionResult()

            state.should_stop = state.should_stop or reflection.should_stop
            state.should_redirect = reflection.should_redirect
            if reflection.should_redirect and reflection.redirect_reason:
                state.redirect_reason = reflection.redirect_reason

            # output_writer 钩子 — 每轮末尾写产物 (RCBench 写 report.md, autoloop 写 provenance)
            if self._output_writer is not None:
                try:
                    self._output_writer.write_step(
                        iteration=state.iteration,
                        action=decision.action,
                        result=state.last_action_result,
                        reflection=reflection,
                    )
                except Exception as e:
                    logger.debug("output_writer.write_step failed: %s", e)

            if decision.action == "stop":
                state.should_stop = True
                break

        return state

    @staticmethod
    def _count_repeated_tail(history: list[str]) -> int:
        """数 history 末尾连续重复的 action 数."""
        if not history:
            return 0
        last = history[-1]
        count = 0
        for a in reversed(history):
            if a == last:
                count += 1
            else:
                break
        return count


# === output_writer 接口 (供两条路径各自实现) ===

class OutputWriter:
    """output_writer 接口 — 两条路径各自实现 write_step.

    RCBench: RCBReportWriter 把 step 结果累积到 report/report.md
    Autoloop: ProvenanceWriter 把 step 结果写 provenance JSONL
    """

    def write_step(
        self,
        iteration: int,
        action: str,
        result: Any,
        reflection: ReflectionResult,
    ) -> None:
        raise NotImplementedError


# === AV4 路径 B: 共享元认知护航原语 ===
# rcb_runner mini-loop 和 autoloop run_cognitive 各自接了一遍 heat_engine /
# detect_drift / TaskMetrics / should_pause_for_decision, 代码独立. 这里抽共享
# 函数, 让两边调同一份, 避免 AV2/AV6/AV7/AV8 类型的偏移再发生.
# ponytail: 只抽无状态纯函数, 不引入 CognitiveLoop 子类, 不绑定 LLM.

def update_heat_engine_after_step(
    heat_engine: Any,
    step_eval: Any,
    prompt_len: int,
    idea_count: int,
    *,
    stable_principles_count: int = 1,
) -> None:
    """AV4: heat_engine 闭环共享函数 — StepEvaluation → T_hot/T_cold/kinematics.

    rcb_runner (AV8) 和 autoloop run_cognitive 都调这个, 避免 4 档映射逻辑两边各写一份.
    T_hot: evidence_quality 熵代理 (low=0.8 / medium=0.5 / high=0.2 / unknown=0.5).
    T_cold: on_track 秩序代理 (true=0.7 / false=0.2 / unsure=0.4).
    ponytail: 4 档离散映射, 天花板: 跳变不连续; 升级路径接连续 evidence_score.
    """
    if heat_engine is None:
        return
    try:
        _eq = (getattr(step_eval, "evidence_quality", "unknown") or "unknown").lower().strip()
        _t_hot_proxy = {"low": 0.8, "medium": 0.5, "high": 0.2}.get(_eq, 0.5)
        _ot = (getattr(step_eval, "on_track", "unsure") or "unsure").lower().strip()
        _t_cold_proxy = {"true": 0.7, "false": 0.2}.get(_ot, 0.4)
        heat_engine.update_T_hot(_t_hot_proxy)
        heat_engine.update_T_cold(_t_cold_proxy, darwin_score=0.0)
        heat_engine.update_kinematics(
            idea_count=idea_count,
            stable_principles_count=stable_principles_count,
            system_prompt_len=prompt_len,
        )
    except Exception as exc:
        logger.debug("update_heat_engine_after_step failed: %s", exc)


def update_drift_and_metrics(
    evals_history: list,
    step_eval: Any,
    task_metrics: Any,
    task_state: Any,
    workspace: Any,
    run_id: str,
    max_iterations: int,
) -> tuple[tuple | None, Any]:
    """AV4: detect_drift + TaskMetrics 滚动更新共享函数.

    返回 (drift_info, task_metrics). 失败时 drift_info=None, task_metrics 不变.
    rcb_runner (G62+G70) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: duck typing — step_eval 可以是 StepEvaluation 或 SimpleNamespace.
    """
    drift_info: tuple | None = None
    try:
        from huginn.metacog.target_chain import detect_drift as _detect_drift
        drift_info = _detect_drift(evals_history, window=3)
    except Exception as exc:
        logger.debug("AV4 detect_drift failed: %s", exc)

    try:
        from huginn.runtime.task_metrics import (
            TaskMetrics as _TM, load_metrics as _lm,
            update_metrics as _um, save_metrics as _sm,
        )
        if task_metrics is None:
            import time
            task_metrics = _lm(run_id, workspace) or _TM(
                task_id=run_id, total_steps=max_iterations * 6,
            )
            if task_state is None:
                from types import SimpleNamespace as _NS
                task_state = _NS(created_at=time.time())
        task_metrics = _um(task_metrics, step_eval, task_state=task_state)
        _sm(task_metrics, workspace)
    except Exception as exc:
        logger.debug("AV4 TaskMetrics update failed: %s", exc)

    return drift_info, task_metrics


def build_pmk_state(
    persona: Any,
    last_step_eval: Any,
    kb: Any,
    *,
    top_k: int = 2,
) -> dict[str, str] | None:
    """AV4: PMK 三路立场状态构建共享函数.

    persona: PersonaManager 返回的对象或 dict, 取 description.
    last_step_eval: 上一步 StepEvaluation, 取 pmk_feedback 中 Memory 段.
    kb: 知识库, 用 last_step_eval.attempted 查 top_k hits 取前 200 字.
    返回 {"persona","memory","kb"} 或 None (全空时).
    rcb_runner (P0-A) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 三段文本拼接, 不上 LLM 抽取; 升级路径接 LLM 立场抽取.
    """
    try:
        _persona_text = ""
        if persona is not None:
            _persona_text = str(
                getattr(persona, "description", None)
                or (persona.get("description") if isinstance(persona, dict) else "")
                or ""
            )
        _mem_text = ""
        if last_step_eval is not None:
            _pmk_fb = getattr(last_step_eval, "pmk_feedback", "") or ""
            for _seg in _pmk_fb.split(";"):
                _seg = _seg.strip()
                if _seg.lower().startswith("memory:"):
                    _mem_text = _seg[len("memory:"):].strip()
                    break
        _kb_text = ""
        if kb is not None and last_step_eval is not None:
            try:
                _kb_hits = kb.query(
                    getattr(last_step_eval, "attempted", "") or "", top_k=top_k)
                if _kb_hits:
                    _kb_text = " ".join(
                        str(h.get("content", "") if isinstance(h, dict) else h)
                        for h in _kb_hits[:top_k]
                    )[:200]
            except Exception:
                pass
        if _persona_text or _mem_text or _kb_text:
            return {
                "persona": _persona_text,
                "memory": _mem_text,
                "kb": _kb_text,
            }
    except Exception as exc:
        logger.debug("AV4 build_pmk_state failed: %s", exc)
    return None


def check_pause_decision(
    evals_history: list,
    target_chains: list,
    kb: Any,
    fired_intentions: list | None,
    pmk_state: dict[str, str] | None,
    grill_state: dict | None = None,
) -> tuple[bool, str, list]:
    """AV4: should_pause_for_decision 共享包装.

    返回 (pause, reason, opts). 失败时 (False, "", []).
    rcb_runner (G71) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 只调判定, 不做 lifecycle/resume — 那些动作两条路径不同, 留在 caller.

    grill_state (P0 grill-me) 由 caller 在 plan_check 阶段构造:
    {"has_grilled": bool, "ambiguity_score": float, "tier": str,
     "scene_tag": str, "plan_is_empty": bool}.
    """
    try:
        from huginn.runtime.task_lifecycle import (
            should_pause_for_decision as _spd,
        )
        _pause, _reason, _opts = _spd(
            evals_history, target_chains,
            kb_recall_empty=(kb is None),
            fired_intentions=fired_intentions or [],
            pmk_state=pmk_state,
            grill_state=grill_state,
        )
        return bool(_pause), str(_reason or ""), list(_opts or [])
    except Exception as exc:
        logger.debug("AV4 check_pause_decision failed: %s", exc)
        return False, "", []


# 已知 _validate dict 字段 (autoloop engine.py _validate 方法返回):
#   tests_passed: bool
#   constraints_satisfied: bool
#   benchmarks: dict (key: benchmark_name, val: {metric: value})
#   r_phys / physics_validation / physics_validation_error
#   thinking_collapse / thinking_collapse_error
#   test_output / math_validation / math_validation_error / math_evidence_error
#   generative_verify / generative_verify_error
#   reviewer_critique / reviewer_critique_error
#   grader_scores / grader_reward / grader_error
#   eval_summary / prediction_error
#   effort_floor_passed / effort_floor_deficits / failure_kind
#   emergent_complexity / literature_comparison / ...
_VALIDATION_ERROR_KEYS: tuple[str, ...] = (
    "thinking_collapse",
    "physics_validation_error",
    "math_validation_error",
    "math_evidence_error",
    "generative_verify_error",
    "reviewer_critique_error",
    "grader_error",
    "prediction_error",
    "test_output",
    "effort_floor_deficits",
)


def _validation_to_step_eval_fields(
    validation: dict,
    tests_ok: bool,
    execution_result: Any,
    *,
    step_id: int,
) -> dict:
    """P0.2: _validate dict → StepEvaluation 兼容字段映射.

    之前 reflect_fn 硬取 summary/result/errors 三个不存在的字段, 导致
    _step_eval.attempted/found/deviation 全空, PMK/drift/metrics 吃空数据.
    这里集中映射, 测试能直接验.

    attempted: execution_result 的 description (validation 里没有).
    found: tests_passed + benchmarks 关键指标摘要.
    deviation: 失败时收集所有 *_error / thinking_collapse / effort_floor_deficits.
    evidence_quality: tests_ok=high, 否则 low; 有 *_error 降级.
    structure_check: tests_ok=passed, 否则 failed.

    ponytail: dict 字段映射, 不上 schema 库. 升级路径: pydantic model.
    """
    # attempted: validation 里没有 attempted, 从 execution_result 拼
    # G6: PMK 的 K 查询看不到 visual_primitives → 加 visual 摘要让 KB 能召回视觉经验.
    _attempted = ""
    if isinstance(execution_result, dict):
        _attempted = str(
            execution_result.get("description")
            or execution_result.get("summary")
            or execution_result.get("result_type")
            or ""
        )[:200]
    # G6: 附 visual_primitives 摘要 (前 150 字), 让 PMK 的 K 查询能看到视觉内容.
    # ponytail: 直接拼字符串, 不上 schema. 升级路径: StepEvaluation 加 visual_context 字段.
    _vis = validation.get("visual_primitives") if isinstance(validation, dict) else None
    if _vis and isinstance(_vis, str):
        _attempted = (f"{_attempted} | Visual: {_vis[:150]}").strip(" |")

    # found: tests_passed 状态 + benchmarks 关键指标
    _found_parts: list[str] = []
    if tests_ok:
        _found_parts.append("tests_passed=True")
    else:
        _found_parts.append("tests_passed=False")
    _benches = validation.get("benchmarks") or {}
    if isinstance(_benches, dict):
        for _bn, _bv in list(_benches.items())[:3]:
            if isinstance(_bv, dict):
                _metric = _bv.get("metric") or _bv.get("value") or ""
                if _metric:
                    _found_parts.append(f"{_bn}={_metric}")
            else:
                _found_parts.append(f"{_bn}={_bv}")
    _found = "; ".join(_found_parts)[:200]

    # deviation: 失败时收集错误信号
    _dev_parts: list[str] = []
    if not tests_ok:
        for _k in _VALIDATION_ERROR_KEYS:
            _v = validation.get(_k)
            if _v:
                _dev_parts.append(f"{_k}: {str(_v)[:100]}")
    _deviation = "; ".join(_dev_parts)[:300]

    # evidence_quality: tests_ok=high, 否则 low
    _eq = "high" if tests_ok else "low"
    if _dev_parts:
        _eq = "low"

    return {
        "step_id": step_id,
        "attempted": _attempted,
        "found": _found,
        "on_track": "true" if tests_ok else "false",
        "evidence_quality": _eq,
        "deviation": _deviation,
        "structure_check": "passed" if tests_ok else "failed",
        "pmk_feedback": "",
        "tool_call_health": None,
        "target_chain_ref": None,
    }




class CognitiveLoopMixin:
    """cognitive loop 主循环方法族, 从 engine.py 下沉 (P3 slim-down). 通过 self 访问 engine 状态."""

    pass  # methods migrated from engine.py via P3 slim-down

# === 自检 ===

    def _run_phase(self, name: str, fn, *args) -> LoopPhase:
        """Run a synchronous phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 同步路径: 如果当前在 event loop 里, fire-and-forget 发开始事件.
        # 不 await, 因为 _run_phase 本身是同步的.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
            )
        except RuntimeError:
            logger.warning(
                "error in _run_phase: stage-start event dispatch skipped (no running loop)",
                exc_info=True,
            )
        # 包 telemetry span: 把 phase 级决策也记进轨迹, 回放时不止看 tool_call
        from huginn.telemetry import get_telemetry_collector

        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning(
                    "error in _run_phase: span metadata update failed", exc_info=True
                )
        phase.end_time = time.time()
        # fire-and-forget 发结束/失败事件
        try:
            loop = asyncio.get_running_loop()
            done_type = (
                EventType.ON_WORKFLOW_STAGE_DONE
                if phase.status == "completed"
                else EventType.ON_WORKFLOW_FAILED
            )
            loop.create_task(
                self._dispatch_stage_event(
                    done_type,
                    name,
                    duration_sec=phase.end_time - (phase.start_time or 0),
                    error=phase.error,
                )
            )
        except RuntimeError:
            logger.warning(
                "error in _run_phase: stage-done event dispatch skipped (no running loop)",
                exc_info=True,
            )
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 记下当前 phase, 让 _llm_chat 能注入 phase-aware thinking effort 指令.
        # ponytail: 隐式状态, 但 run() 是 single-threaded async, 无竞态.
        self._current_phase = name
        # C2: 追踪本 run 的 phase 序列, 供 trajectory_match 召回用.
        if not hasattr(self, "_current_run_phases"):
            self._current_run_phases = []
        self._current_run_phases.append(name)
        # 防止无界增长 (1000+ iter × 7 phase = 7000+), 只保留最近 50 个
        if len(self._current_run_phases) > 50:
            self._current_run_phases = self._current_run_phases[-50:]
        await self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
        from huginn.telemetry import get_telemetry_collector

        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = await fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning(
                    "error in _run_phase_async: span metadata update failed",
                    exc_info=True,
                )
        phase.end_time = time.time()
        if phase.status == "completed":
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_STAGE_DONE,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
            )
        else:
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_FAILED,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
                error=phase.error,
            )
        return phase

    def _render_report(self, data: dict[str, Any]) -> str:
        """Render a markdown report."""
        lines = [
            "# Huginn Autoloop Report",
            "",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            "",
            "## Phases",
            "",
            "| Phase | Status | Duration (s) | Error |",
            "|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(
                f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)

def _selfcheck() -> None:
    """CognitiveLoop 骨架自检 — 不依赖 LLM, 用 mock 钩子验证控制流."""
    import asyncio

    async def mock_observe(state: LoopState) -> dict[str, Any]:
        return {"step": state.iteration}

    async def mock_decide(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
        # 第 3 轮后选 stop, 验证主循环能退出
        if state.iteration >= 3:
            return ActionDecision(action="stop", rationale="done")
        return ActionDecision(action="execute", rationale="working")

    async def mock_execute(state: LoopState, decision: ActionDecision) -> Any:
        return f"result_iter_{state.iteration}"

    async def mock_reflect(
        state: LoopState, decision: ActionDecision, result: Any
    ) -> ReflectionResult:
        return ReflectionResult(should_continue=True)

    class MockWriter(OutputWriter):
        def __init__(self):
            self.steps = []
        def write_step(self, iteration, action, result, reflection):
            self.steps.append((iteration, action))

    async def run_test():
        writer = MockWriter()
        loop = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=mock_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            output_writer=writer,
            max_iterations=10,
        )
        state = await loop.run()
        assert state.iteration == 3, f"expected 3 iters, got {state.iteration}"
        assert state.should_stop, "should_stop should be True after 'stop' action"
        assert len(writer.steps) == 3, f"writer should have 3 steps, got {len(writer.steps)}"
        assert writer.steps[-1] == (3, "stop"), f"last step wrong: {writer.steps[-1]}"

        # 死循环防护测试: decide 总返回相同 action
        async def stuck_decide(state, obs):
            return ActionDecision(action="execute", rationale="stuck")
        writer2 = MockWriter()
        loop2 = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=stuck_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            output_writer=writer2,
            max_iterations=10,
            max_repeated_actions=3,
        )
        state2 = await loop2.run()
        # 死循环防护: 6 次 (2x limit) 后强制 stop, should_redirect 在过程中被设过
        assert state2.should_stop, "stuck loop should force stop at 2x limit"
        assert state2.iteration == 6, f"stuck loop should stop at 6 iters (2x limit=3), got {state2.iteration}"
        assert state2.should_redirect, "should_redirect should be set before force stop"
        assert "死循环防护" in state2.redirect_reason, f"redirect_reason wrong: {state2.redirect_reason}"

        # invalid action 测试
        async def bad_decide(state, obs):
            return ActionDecision(action="invalid_action", rationale="bad")
        loop3 = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=bad_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            max_iterations=2,
        )
        state3 = await loop3.run()
        # invalid action 被替换为 'skip', 不崩溃
        assert state3.iteration >= 1, "should run at least 1 iter"

        print("CognitiveLoop selfcheck OK (3/3: basic flow + stuck protection + invalid action)")

    asyncio.run(run_test())


if __name__ == "__main__":
    _selfcheck()
